#!/usr/bin/env python3
"""LT2: pre-registered A/B benchmark of the GWAYA low-tier optimizer on MBPP-sanitized.

Protocol (frozen in results/gwaya_low_tier/PREREGISTRATION.md BEFORE the first run):
  * Problems: first N task_ids (ascending) of the MBPP-sanitized *test* split.
  * Public test  = test_list[0] (shown to the model and used by the repair loop).
  * Hidden tests = test_list[1:] (never shown). A problem counts as solved only if the
    FINAL code passes ALL tests in the bwrap sandbox. The optimizer's own ``verified``
    flag is never used as the outcome.
  * Arms (same model, same seeds):
      A0  raw first sample, T=0.2, no verifier, no repair          (baseline)
      A2  best-of-3 ranked by the public test, no repair
      A3  best-of-3 + up to 3 repair rounds with bounded feedback  (the system under test)
      A4  A3 + 1 retrieved verified exemplar from MBPP train/validation/prompt splits
  * System output policy for A2/A3/A4: the optimizer's verified code if it found one,
    otherwise the A0 raw sample (a deployable "verified answer, else best effort" policy).
  * Gate: A3 - A0 >= +8.0 percentage points. Reported as-is, pass or fail.

Identical (prompt, temperature, max_tokens, seed) requests are served from an on-disk
cache so arms share their common generations and an interrupted run can resume. Cache
entries are real model outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gwaya.exemplar_store import ExemplarStore  # noqa: E402
from gwaya.generators import OllamaGenerator  # noqa: E402
from gwaya.low_tier_engine import (  # noqa: E402
    LowTierModelOptimizer,
    ModelTier,
    extract_code_block,
)
from gwaya.oracles import PythonCompilerOracle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("LowTierBenchmark")

OUT = ROOT / "results" / "gwaya_low_tier"
GATE_POINTS = 8.0
ARMS = ("A0", "A2", "A3", "A4")


# ── data ────────────────────────────────────────────────────────────────────────────────
def load_mbpp(mbpp_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load MBPP-sanitized from a local directory of parquet files (e.g. restored from the data lake) or the HF hub."""
    if mbpp_dir is not None:
        files = list(Path(mbpp_dir).glob("*.parquet")) or list((Path(mbpp_dir) / "sanitized").glob("*.parquet"))
    else:
        snap = Path(
            snapshot_download("google-research-datasets/mbpp", repo_type="dataset", allow_patterns=["sanitized/*"])
        )
        files = list((snap / "sanitized").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no MBPP parquet files found (mbpp_dir={mbpp_dir})")
    return {p.stem.split("-")[0]: pd.read_parquet(p) for p in files}


def problem_from_row(row: Any) -> dict[str, Any]:
    imports = "\n".join(row["test_imports"]) + ("\n" if len(row["test_imports"]) else "")
    tests = list(row["test_list"])
    return {
        "task_id": int(row["task_id"]),
        "goal": str(row["prompt"]),
        "public": imports + tests[0] + "\n",
        "all_tests": imports + "\n".join(tests) + "\n",
        "n_tests": len(tests),
        "reference": str(row["code"]),
    }


# ── cached generator ───────────────────────────────────────────────────────────────────
class CachedGenerator:
    """Deterministic per-problem seeds + disk cache of real Ollama responses."""

    def __init__(self, inner: OllamaGenerator, cache_path: Path, base_seed: int) -> None:
        self.inner, self.cache_path, self.base_seed, self.calls = inner, cache_path, base_seed, 0
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                rec = json.loads(line)
                self.cache[rec["k"]] = rec["v"]
        self.live_calls = 0

    def __call__(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512, **_: Any) -> str:
        seed = self.base_seed + self.calls
        self.calls += 1
        key = hashlib.sha256(
            f"{self.inner.model}|{prompt}|{temperature}|{max_tokens}|{seed}".encode()
        ).hexdigest()
        if key in self.cache:
            return self.cache[key]
        out = self.inner(prompt, temperature=temperature, max_tokens=max_tokens, seed=seed)
        self.live_calls += 1
        self.cache[key] = out
        with self.cache_path.open("a") as f:
            f.write(json.dumps({"k": key, "v": out}) + "\n")
        return out


# ── scoring ─────────────────────────────────────────────────────────────────────────────
def score_final(oracle: PythonCompilerOracle, code: str, prob: dict[str, Any]) -> dict[str, Any]:
    res = oracle.verify_with_test(extract_code_block(code, "python"), prob["all_tests"], timeout_s=15.0)
    iso = res.details.get("isolation", "")
    if "bwrap" not in str(iso):
        raise RuntimeError(f"Refusing to score without bwrap isolation (got {iso!r}); result would be untrusted")
    if res.details.get("reason") == "no_result":
        # The sandbox itself died (not the candidate): never count that as a wrong answer.
        raise RuntimeError(f"sandbox infrastructure failure while scoring: {res.details.get('output_tail', '')!r}")
    return {"solved": bool(res.success), "passed": res.details.get("passed", 0), "total": res.details.get("total", 0)}


def sandbox_selftest(oracle: PythonCompilerOracle) -> None:
    """Abort unless the sandbox demonstrably scores a correct solution as solved and a wrong one as failed.

    Without this a broken sandbox (e.g. bwrap unable to mount /proc inside a container) makes every
    candidate fail and looks like a legitimate 0 %. This is a control, not a benchmark result.
    """
    from gwaya.test_harness import isolation_available

    if not isolation_available():
        raise RuntimeError("bwrap isolation is not available in this environment; refusing to score (fail-closed)")
    good = oracle.verify_with_test("def f(x):\n    return x + 1\n", "assert f(1) == 2\nassert f(2) == 3\n", timeout_s=15.0)
    bad = oracle.verify_with_test("def f(x):\n    return x + 2\n", "assert f(1) == 2\nassert f(2) == 3\n", timeout_s=15.0)
    if not good.success or bad.success or good.details.get("passed") != 2:
        raise RuntimeError(
            f"sandbox self-test failed (good: success={good.success} err={good.error_message!r}; bad: success={bad.success})"
        )
    log.info("sandbox self-test passed (isolation=%s)", good.details.get("isolation"))


def make_optimizer(arm: str, gen: CachedGenerator) -> LowTierModelOptimizer:
    opt = LowTierModelOptimizer(tier=ModelTier.LOW, generator_fn=gen)
    opt.config.best_of_n_candidates = 3
    opt.config.max_repair_attempts = 1 if arm == "A2" else 3
    return opt


def run_problem(
    idx: int, prob: dict[str, Any], model: str, cache_path: Path, store: ExemplarStore, oracle: PythonCompilerOracle
) -> dict[str, Any]:
    base_seed = 1000 * (idx + 1)
    row: dict[str, Any] = {"task_id": prob["task_id"], "n_tests": prob["n_tests"]}
    raw_a0 = ""
    for arm in ARMS:
        gen = CachedGenerator(OllamaGenerator(model=model), cache_path, base_seed)
        t0 = time.perf_counter()
        info: dict[str, Any] = {}
        if arm == "A0":
            opt = make_optimizer(arm, gen)
            prompt = opt._build_base_prompt(prob["goal"], "python", "", prob["public"])
            code = raw_a0 = gen(prompt, temperature=0.2, max_tokens=opt.config.max_tokens_per_leaf)
        else:
            opt = make_optimizer(arm, gen)
            context = ""
            if arm == "A4":
                ex = store.retrieve_exemplars(prob["goal"], top_k=1)
                info["exemplar_task"] = ex[0]["problem_hash"] if ex else None
                if ex:
                    context = f"Similar solved example:\n# Task: {ex[0]['goal']}\n{ex[0]['code']}"
            res = opt.optimize_and_solve(prob["goal"], "python", context=context, test_spec=prob["public"])
            info.update(
                gate_verified=bool(res.verified),
                repair_rounds=res.repair_attempts,
                candidates=res.candidates_evaluated,
            )
            code = res.selected_code if res.verified else raw_a0
        row[arm] = {**score_final(oracle, code, prob), **info, "wall_s": round(time.perf_counter() - t0, 1)}
    return row


# ── statistics ─────────────────────────────────────────────────────────────────────────
def paired_stats(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    x = [int(r[a]["solved"]) for r in rows]
    y = [int(r[b]["solved"]) for r in rows]
    n = len(rows)
    delta = 100.0 * (sum(y) - sum(x)) / n
    rng = random.Random(0)
    boots = sorted(
        100.0 * sum(y[i] - x[i] for i in (rng.randrange(n) for _ in range(n))) / n for _ in range(10000)
    )
    gained = sum(1 for i in range(n) if y[i] and not x[i])
    lost = sum(1 for i in range(n) if x[i] and not y[i])
    k, m = min(gained, lost), gained + lost
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m) if m else 1.0
    return {
        "delta_points": round(delta, 2),
        "ci95": [round(boots[250], 2), round(boots[9749], 2)],
        "gained": gained,
        "lost": lost,
        "mcnemar_exact_p": round(p, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:1.5b")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out-dir", default=None, help="output directory (default results/gwaya_low_tier)")
    ap.add_argument("--mbpp-dir", default=None, help="directory with the MBPP-sanitized parquet files")
    args = ap.parse_args()

    global OUT
    if args.out_dir:
        OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_mbpp(Path(args.mbpp_dir) if args.mbpp_dir else None)
    test_ids = set(data["test"]["task_id"])
    ex_df = pd.concat([data["train"], data["validation"], data["prompt"]])
    assert not (test_ids & set(ex_df["task_id"])), "exemplar/test leakage"
    probs = [problem_from_row(r) for _, r in data["test"].sort_values("task_id").head(args.n).iterrows()]

    store = ExemplarStore(str(OUT / "mbpp_exemplars.json"))
    if not store.exemplars:
        log.info("building exemplar store from %d train/validation/prompt problems", len(ex_df))
        for _, r in ex_df.iterrows():
            store.add_exemplar(str(int(r["task_id"])), "train", str(r["prompt"]), str(r["code"]))

    oracle = PythonCompilerOracle()
    sandbox_selftest(oracle)
    cache_path = OUT / "generation_cache.jsonl"
    rows_path = OUT / "rows.jsonl"
    done = {json.loads(l)["task_id"]: json.loads(l) for l in rows_path.read_text().splitlines()} if rows_path.exists() else {}
    t_start = time.perf_counter()
    for i, prob in enumerate(probs):
        if prob["task_id"] in done:
            continue
        row = run_problem(i, prob, args.model, cache_path, store, oracle)
        done[prob["task_id"]] = row
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        log.info(
            "[%d/%d] task %d  " % (i + 1, len(probs), prob["task_id"])
            + "  ".join(f"{a}={'Y' if row[a]['solved'] else 'n'}" for a in ARMS)
            + f"  (elapsed {time.perf_counter() - t_start:.0f}s)"
        )

    rows = [done[p["task_id"]] for p in probs]
    summary: dict[str, Any] = {
        "model": args.model,
        "dataset": "mbpp-sanitized/test",
        "hardware": os.environ.get("GWAYA_HARDWARE", "unspecified"),
        "ollama_version": os.environ.get("GWAYA_OLLAMA_VERSION", "unspecified"),
        "n": len(rows),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "pass_at_1": {a: round(100.0 * sum(r[a]["solved"] for r in rows) / len(rows), 2) for a in ARMS},
        "paired_vs_A0": {a: paired_stats(rows, "A0", a) for a in ARMS if a != "A0"},
        "gate_verified_precision": {},
    }
    for a in ("A2", "A3", "A4"):
        ver = [r for r in rows if r[a]["gate_verified"]]
        summary["gate_verified_precision"][a] = {
            "coverage_pct": round(100.0 * len(ver) / len(rows), 2),
            "hidden_pass_given_verified_pct": round(100.0 * sum(r[a]["solved"] for r in ver) / len(ver), 2) if ver else None,
        }
    delta = summary["paired_vs_A0"]["A3"]["delta_points"]
    summary["gate"] = {"criterion": f"A3 - A0 >= +{GATE_POINTS} points", "delta_points": delta,
                       "verdict": "PASS" if delta >= GATE_POINTS else "FAIL"}
    summary["rows_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    (OUT / "gate_verdict.txt").write_text(summary["gate"]["verdict"])
    log.info("RESULT %s", json.dumps({k: summary[k] for k in ("pass_at_1", "paired_vs_A0", "gate")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
