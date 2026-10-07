#!/usr/bin/env python3
"""GWAYA v4 benchmark: does consensus verification + full-context repair + grounding help small local models?

Protocol is frozen in results/gwaya_v4/PREREGISTRATION.md before any ``--split test`` run.

Arms (same model, same seeds, shared generation cache):
  A0  raw first sample                                    (baseline, re-run in the same job)
  A3  v3 best-of-3 + 3 truncated-feedback repairs         (previous system)
  A5  v4 consensus self-tests, OLD truncated repair, no grounding
  A6  A5 + full-context repair
  A7  A6 + grounding                                      (system under test; the plan called it "A6")

Outcome for every arm = FINAL code passes ALL tests (public + hidden) in the bwrap sandbox.
"Delivered" = the arm claims the answer is verified; otherwise it abstains (code still scored, reported separately).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.run_benchmark import (  # noqa: E402
    CachedGenerator,
    load_mbpp,
    make_optimizer,
    paired_stats,
    problem_from_row,
    sandbox_selftest,
)

from gwaya.consensus_agent import ConsensusRepairAgent  # noqa: E402
from gwaya.generators import OllamaGenerator  # noqa: E402
from gwaya.grounding import grounding_flags, runtime_hallucination  # noqa: E402
from gwaya.low_tier_engine import extract_code_block  # noqa: E402
from gwaya.oracles import PythonCompilerOracle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("GwayaV4")

ARMS = ("A0", "A3", "A5", "A6", "A7")
G1_POINTS, G2_REL_REDUCTION, G3_PRECISION, G3_COVERAGE, G4_VRAM_GB = 3.0, 30.0, 92.0, 75.0, 8.0


def split_problem(prob: dict[str, Any]) -> tuple[str, str]:
    """(preamble of test imports, assert-only public test)."""
    lines = prob["public"].splitlines()
    pre = [ln for ln in lines if not ln.strip().startswith("assert")]
    asserts = [ln for ln in lines if ln.strip().startswith("assert")]
    return ("\n".join(pre) + "\n" if pre else ""), "\n".join(asserts) + "\n"


def score(
    oracle: PythonCompilerOracle, code: str, prob: dict[str, Any], preamble: str
) -> dict[str, Any]:
    code = extract_code_block(code, "python")
    res = oracle.verify_with_test(code, prob["all_tests"], timeout_s=15.0)
    if "bwrap" not in str(res.details.get("isolation", "")):
        raise RuntimeError("refusing to score without bwrap isolation")
    if res.details.get("reason") == "no_result":
        raise RuntimeError(
            f"sandbox infrastructure failure: {res.details.get('output_tail', '')!r}"
        )
    err = res.error_message or ""
    solved = bool(res.success)
    flags = grounding_flags(code, preamble)
    return {
        "solved": solved,
        "passed": res.details.get("passed", 0),
        "total": res.details.get("total", 0),
        "hallucinated": bool(flags) or (not solved and runtime_hallucination(err)),
        "flags": flags[:3],
    }


def run_problem(
    idx: int, prob: dict[str, Any], model: str, cache: Path, oracle: PythonCompilerOracle
) -> dict[str, Any]:
    base_seed = 1000 * (idx + 1)
    preamble, public = split_problem(prob)
    row: dict[str, Any] = {"task_id": prob["task_id"], "n_tests": prob["n_tests"]}
    raw_a0 = ""
    for arm in ARMS:
        inner = OllamaGenerator(model=model)
        gen = CachedGenerator(inner, cache, base_seed)
        t0 = time.perf_counter()
        info: dict[str, Any] = {}
        if arm == "A0":
            opt = make_optimizer(arm, gen)
            prompt = opt._build_base_prompt(prob["goal"], "python", "", prob["public"])
            code = raw_a0 = gen(prompt, temperature=0.2, max_tokens=opt.config.max_tokens_per_leaf)
            delivered = False
        elif arm == "A3":
            opt = make_optimizer(arm, gen)
            res = opt.optimize_and_solve(
                prob["goal"], "python", context="", test_spec=prob["public"]
            )
            delivered = bool(res.verified)
            code = res.selected_code if delivered else raw_a0
            info.update(repair_rounds=res.repair_attempts, candidates=res.candidates_evaluated)
        else:
            agent = ConsensusRepairAgent(
                gen,
                oracle=oracle,
                use_selftests=True,
                full_context_repair=arm in ("A6", "A7"),
                grounding=arm == "A7",
            )
            ans = agent.solve(prob["goal"], public_test=public, preamble=preamble)
            delivered, code = ans.verified, ans.code
            info.update(
                level=ans.level,
                repair_rounds=ans.rounds,
                candidates=ans.candidates,
                trusted_selftests=ans.trusted_selftests,
                flags_seen=ans.flags_seen[:5],
            )
        wall = time.perf_counter() - t0
        stats = inner.history
        row[arm] = {
            **score(oracle, code, prob, preamble),
            "delivered": delivered,
            "gen_tokens": sum(s.completion_tokens for s in stats),
            "gen_eval_s": round(sum(s.eval_ms for s in stats) / 1000.0, 2),
            "live_calls": gen.live_calls,
            "wall_s": round(wall, 1),
            **info,
        }
    return row


def arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    n = len(rows)
    d = [r for r in rows if r[arm]["delivered"]]
    wall = [r[arm]["wall_s"] for r in rows]
    hall_all = sum(1 for r in rows if r[arm]["hallucinated"])
    hall_del = sum(1 for r in d if r[arm]["hallucinated"])
    return {
        "pass_at_1": round(100.0 * sum(r[arm]["solved"] for r in rows) / n, 2),
        "coverage_pct": round(100.0 * len(d) / n, 2),
        "precision_pct": round(100.0 * sum(r[arm]["solved"] for r in d) / len(d), 2) if d else None,
        "hallucination_rate_all_pct": round(100.0 * hall_all / n, 2),
        "hallucination_rate_delivered_pct": round(100.0 * hall_del / len(d), 2) if d else None,
        "p50_wall_s_cache_shared": round(statistics.median(wall), 1),
        "mean_live_gen_tokens_cache_hits_excluded": round(
            sum(r[arm]["gen_tokens"] for r in rows) / n, 1
        ),
    }


def vram_peak_gb(out_dir: Path) -> float | None:
    f = next(
        (c for c in (out_dir / "vram_host.csv", out_dir.parent / "vram_host.csv") if c.exists()),
        None,
    )
    if f is None:
        return None
    samples: list[
        tuple[int, int]
    ] = []  # (epoch_s, MiB used) sampled on the HOST (nvidia-smi is absent in the container)
    for ln in f.read_text().splitlines():
        parts = ln.replace(" MiB", "").split(",")
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            samples.append((int(parts[0]), int(parts[1])))
    if len(samples) < 5:
        return None
    idle = min(m for _, m in samples[:3])
    win = out_dir / "window.txt"  # "start_epoch" of THIS model's run, written by container_entry.sh
    if win.exists():
        t0 = int(win.read_text().split()[0])
        samples = [(t, m) for t, m in samples if t >= t0] or samples
    return round((max(m for _, m in samples) - idle) / 1024.0, 2)  # MiB -> GiB above idle


def single_stream_tokens_per_s(model: str) -> float | None:
    """Interactive single-request speed (one warm-up call, then 3 timed calls), independent of --workers."""
    g = OllamaGenerator(model=model)
    try:
        g(
            "Write a Python function that reverses a string.",
            temperature=0.0,
            max_tokens=160,
            seed=1,
        )
        g.history.clear()
        for k in range(3):
            g(
                "Write a Python function that returns the n-th Fibonacci number.",
                temperature=0.0,
                max_tokens=160,
                seed=2 + k,
            )
    except Exception as exc:  # noqa: BLE001 - diagnostic, never part of the correctness result
        log.warning("single-stream speed probe failed: %s", exc)
        return None
    toks, secs = (
        sum(s.completion_tokens for s in g.history),
        sum(s.eval_ms for s in g.history) / 1000.0,
    )
    return round(toks / secs, 1) if secs else None


def gates(
    summary: dict[str, Any], hardware: str, vram: float | None, tok_s: float | None
) -> dict[str, Any]:
    arms = summary["arms"]
    g1 = summary["paired"]["A7_vs_A3"]["delta_points"]
    h0, h7 = (
        arms["A0"]["hallucination_rate_all_pct"],
        arms["A7"]["hallucination_rate_delivered_pct"],
    )
    red = None if (h7 is None or not h0) else round(100.0 * (h0 - h7) / h0, 1)
    p7, c7 = arms["A7"]["precision_pct"], arms["A7"]["coverage_pct"]
    return {
        "G1": {
            "criterion": f"A7 - A3 >= +{G1_POINTS} points",
            "value": g1,
            "verdict": "PASS" if g1 >= G1_POINTS else "FAIL",
        },
        "G2": {
            "criterion": f"delivered-A7 hallucination rate >= {G2_REL_REDUCTION}% lower (relative) than A0's",
            "a0_rate_pct": h0,
            "a7_delivered_rate_pct": h7,
            "relative_reduction_pct": red,
            "verdict": "PASS" if (red is not None and red >= G2_REL_REDUCTION) else "FAIL",
        },
        "G3": {
            "criterion": f"A7 precision >= {G3_PRECISION}% at coverage >= {G3_COVERAGE}%",
            "precision_pct": p7,
            "coverage_pct": c7,
            "verdict": "PASS"
            if (p7 is not None and p7 >= G3_PRECISION and c7 >= G3_COVERAGE)
            else "FAIL",
        },
        "G4": {
            "criterion": f"peak VRAM <= {G4_VRAM_GB} GB at 100% GPU (hardware: {hardware}); L4 is a proxy for a standard RTX card",
            "peak_vram_gb": vram,
            "tokens_per_s": tok_s,
            "verdict": (
                "NOT_MEASURED"
                if vram is None
                else ("PASS" if vram <= G4_VRAM_GB and "100% GPU" in hardware else "FAIL")
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:1.5b")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--n", type=int, default=0, help="0 = all problems of the split")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mbpp-dir", default=None)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="problems solved concurrently (needs OLLAMA_NUM_PARALLEL>=workers)",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_mbpp(Path(args.mbpp_dir) if args.mbpp_dir else None)
    if args.split == "test":
        df = data["test"]
    else:
        df = pd.concat([data["train"], data["validation"], data["prompt"]])
        assert not (set(df["task_id"]) & set(data["test"]["task_id"])), "dev/test leakage"
    df = df.sort_values("task_id")
    if args.n:
        df = df.head(args.n)
    probs = [problem_from_row(r) for _, r in df.iterrows()]

    oracle = PythonCompilerOracle()
    sandbox_selftest(oracle)
    single_stream = single_stream_tokens_per_s(args.model)
    cache, rows_path = out / "generation_cache.jsonl", out / "rows.jsonl"
    done = (
        {json.loads(ln)["task_id"]: json.loads(ln) for ln in rows_path.read_text().splitlines()}
        if rows_path.exists()
        else {}
    )
    t_start = time.perf_counter()
    todo = [(i, p) for i, p in enumerate(probs) if p["task_id"] not in done]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(run_problem, i, p, args.model, cache, oracle): (i, p) for i, p in todo}
        for fut in as_completed(futs):
            i, prob = futs[fut]
            row = (
                fut.result()
            )  # any sandbox/infra failure aborts the run (never counted as a wrong answer)
            done[prob["task_id"]] = row
            with rows_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            log.info(
                "[%d done/%d] task %d  %s  (elapsed %.0fs)",
                len(done),
                len(probs),
                prob["task_id"],
                "  ".join(
                    f"{a}={'Y' if row[a]['solved'] else 'n'}{'*' if row[a]['delivered'] else ''}"
                    for a in ARMS
                ),
                time.perf_counter() - t_start,
            )

    rows = [done[p["task_id"]] for p in probs]
    hardware = os.environ.get("GWAYA_HARDWARE", "unspecified")
    tokens = sum(r[a]["gen_tokens"] for r in rows for a in ARMS)
    secs = sum(r[a]["gen_eval_s"] for r in rows for a in ARMS)
    tok_s = round(tokens / secs, 1) if secs else None
    summary: dict[str, Any] = {
        "model": args.model,
        "split": args.split,
        "dataset": f"mbpp-sanitized/{args.split}",
        "n": len(rows),
        "hardware": hardware,
        "ollama_version": os.environ.get("GWAYA_OLLAMA_VERSION", "unspecified"),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip(),
        "arms": {a: arm_summary(rows, a) for a in ARMS},
        "paired": {f"{a}_vs_A0": paired_stats(rows, "A0", a) for a in ARMS if a != "A0"}
        | {f"{a}_vs_A3": paired_stats(rows, "A3", a) for a in ("A5", "A6", "A7")},
        "generation_tokens_per_s_under_load": tok_s,
        "single_stream_tokens_per_s": single_stream,
        "workers": args.workers,
    }
    summary["gates"] = gates(summary, hardware, vram_peak_gb(out), single_stream)
    summary["rows_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    (out / "gate_verdict.txt").write_text(
        "\n".join(f"{k}={v['verdict']}" for k, v in summary["gates"].items())
    )
    log.info(
        "RESULT %s", json.dumps({"arms": summary["arms"], "gates": summary["gates"]}, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
