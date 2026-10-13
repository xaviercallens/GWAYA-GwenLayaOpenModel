#!/usr/bin/env python3
"""Python training pool "O" from nvidia/OpenCodeInstruct (CC-BY-4.0): more verified Python tasks for Laya and the generator.

Why: pool P holds only 556 Python tasks (322 train / 65 calib); Laya's Python rows are too few to separate it from a
logistic regression on gate and log-prob features. OpenCodeInstruct ships LLM-written asserts per task (synthetic, so
tests can be weak or wrong: every row is re-verified here with the repo's python oracle and the source's own
`tests_execution_status` is NOT trusted).

Pipeline (no model is run; the network is used only to stream the dataset):
  1. stream rows; keep >= --min-tests asserts that the source marks "pass"; reference = first python block of `output`
  2. build MBPP-style rows (domain, task_id, prompt, checker_payload=all asserts, gate_payload=first assert, cluster)
  3. drop: prompt too long for Laya's 512-token window (--max-prompt-chars), exact/statement hash or 8-gram overlap with
     any held-out set (E, E', F, pool P, any --extra-holdout), duplicate prompts inside the pool
  4. sanity: reference must pass hidden AND gate checks in the bwrap oracle (build_pool_p.sanity_python)
  5. split by sha256(cluster) % 10: 0-7 train, 8 calib, 9 dev (dev is internal; E' and F stay the headline held-out sets)
Outputs in --out: tasks_O_{train,calib,dev}.jsonl, report.json (counts per drop reason, sha256 of each file, held-out
inputs used). Nothing here is a result; it is a data build.

  build_pool_py_oci.py --out $GWAYA_DATA_ROOT/pool_O --target 12000 --workers 8
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
import build_pool_p as P  # noqa: E402  (phash, stmt_of, sanity_python, clean, dump)

DATA = Path("/mnt/data/home/xavkal/gwaya-data")
DEFAULT_HOLDOUT = [DATA / "night/results/tasks_E_primary.d25.jsonl", DATA / "laya_data/tasks_Eprime.jsonl",
                   DATA / "laya_data/tasks_P_all.jsonl", DATA / "eval_sets_F_scored/tasks_F_pm.jsonl",
                   DATA / "eval_sets_F_raw/try_py/tasks_F_python.jsonl", DATA / "eval_sets_F_raw/prev4/tasks_F_python.jsonl"]
SHINGLE = 8
OVERLAP = 0.5


def shingles(text: str) -> set[int]:
    w = P.B.normalize(text).split()
    return {hash(tuple(w[i:i + SHINGLE])) for i in range(max(0, len(w) - SHINGLE + 1))} or {hash(tuple(w))}


class Holdout:
    def __init__(self, rows: list[dict]) -> None:
        self.full = {P.phash(r["prompt"]) for r in rows}
        self.stmt = {P.phash(P.stmt_of(r["prompt"])) for r in rows}
        self.index: dict[int, list[int]] = {}
        self.sizes: list[int] = []
        for i, r in enumerate(rows):
            s = shingles(r["prompt"])
            self.sizes.append(len(s))
            for h in s:
                self.index.setdefault(h, []).append(i)

    def why(self, prompt: str, task_text: str) -> str | None:
        if P.phash(prompt) in self.full:
            return "prompt_hash_in_holdout"
        if P.phash(task_text) in self.stmt or P.phash(P.stmt_of(prompt)) in self.stmt:
            return "statement_hash_in_holdout"
        hits: Counter = Counter()
        sh = shingles(task_text)
        for h in sh:
            for i in self.index.get(h, ()):
                hits[i] += 1
        if hits:
            i, c = hits.most_common(1)[0]
            if c / max(1, min(len(sh), self.sizes[i])) >= OVERLAP:
                return "ngram_overlap_with_holdout"
        return None


def first_python_block(out: str) -> str:
    from gwaya.low_tier_engine import extract_code_block
    return extract_code_block(out, "python")


def to_row(r: dict[str, Any], min_tests: int, max_chars: int) -> tuple[dict | None, str | None]:
    try:
        tests = json.loads(r["unit_tests"]) if isinstance(r["unit_tests"], str) else list(r["unit_tests"])
        status = json.loads(r["tests_execution_status"]) if isinstance(r["tests_execution_status"], str) \
            else list(r["tests_execution_status"])
    except (ValueError, TypeError, KeyError):
        return None, "bad_tests_json"
    good = [t.strip() for t, s in zip(tests, status) if isinstance(t, str) and s == "pass" and "assert" in t]
    good = list(dict.fromkeys(good))
    if len(good) < min_tests:
        return None, "too_few_passing_tests"
    for t in good:
        try:
            ast.parse(t)
        except SyntaxError:
            return None, "test_syntax_error"
    ref = first_python_block(r.get("output") or "")
    if not ref.strip() or "def " not in ref and "class " not in ref:
        return None, "no_reference_code"
    text = (r.get("input") or "").strip()
    if not text or len(text) > max_chars:
        return None, "prompt_too_long_or_empty"
    prompt = f"{text}\nYour code should pass this test:\n{good[0]}\nReply with the full Python function."
    cid = str(r["id"])
    return {"domain": "python", "task_id": f"py/OCI/{cid}", "prompt": prompt,
            "checker_payload": {"tests": "\n".join(good), "timeout_s": 30.0},
            "gate_payload": {"tests": good[0], "timeout_s": 10.0}, "cluster": f"OCI/{cid}",
            "_reference": ref, "_task_text": text}, None


def split_of(cluster: str) -> str:
    b = P.bucket_of(cluster)
    return "train" if b <= 7 else ("calib" if b == 8 else "dev")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=12000, help="stop streaming after this many pre-sanity candidates")
    ap.add_argument("--scan-limit", type=int, default=400000)
    ap.add_argument("--min-tests", type=int, default=3)
    ap.add_argument("--max-prompt-chars", type=int, default=1200)
    ap.add_argument("--extra-holdout", action="append", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rows-file", help="read rows from this JSONL instead of streaming (offline / tests)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    used, held = [], []
    for p in [*DEFAULT_HOLDOUT, *map(Path, a.extra_holdout)]:
        if p.is_file():
            held += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            used.append(str(p))
    if not used:
        print("no held-out file found: refusing to build (fail closed)", file=sys.stderr)
        return 2
    hold = Holdout(held)

    if a.rows_file:
        stream = (json.loads(l) for l in Path(a.rows_file).read_text().splitlines() if l.strip())
    else:
        from datasets import load_dataset
        stream = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)

    drops: Counter = Counter()
    cand, seen_prompts, scanned = [], set(), 0
    for r in stream:
        scanned += 1
        if scanned > a.scan_limit or len(cand) >= a.target:
            break
        row, why = to_row(r, a.min_tests, a.max_prompt_chars)
        if row is None:
            drops[why] += 1
            continue
        ph = P.phash(row["prompt"])
        if ph in seen_prompts:
            drops["duplicate_prompt_in_pool"] += 1
            continue
        why = hold.why(row["prompt"], row["_task_text"])
        if why:
            drops[why] += 1
            continue
        seen_prompts.add(ph)
        cand.append(row)

    kept, sdrops = P.run_sanity(cand, workers=a.workers)
    for d in sdrops:
        drops[d["reason"].split(":")[0]] += 1
    split: dict[str, list[dict]] = {"train": [], "calib": [], "dev": []}
    for r in kept:
        r = P.clean(r)
        r["split"] = split_of(r["cluster"])
        split[r["split"]].append(r)
    sha = {}
    for name, rows in split.items():
        rows.sort(key=lambda x: x["task_id"])
        sha[name] = P.dump(out / f"tasks_O_{name}.jsonl", rows)
    rep = {"source": "nvidia/OpenCodeInstruct (CC-BY-4.0), synthetic LLM-written tests re-verified here",
           "scanned": scanned, "candidates_pre_sanity": len(cand), "kept": len(kept),
           "n": {k: len(v) for k, v in split.items()}, "drops": dict(drops), "holdout_files": used,
           "holdout_rows": len(held), "params": {"min_tests": a.min_tests, "max_prompt_chars": a.max_prompt_chars,
                                                 "shingle": SHINGLE, "overlap": OVERLAP},
           "sha256": sha}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("scanned", "candidates_pre_sanity", "kept", "n", "drops")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
