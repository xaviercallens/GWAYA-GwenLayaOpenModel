#!/usr/bin/env python3
"""A18: build the miniF2F-lean4 test tasks and check that every statement elaborates with `sorry` (pre-generation sanity).

Writes <out>/tasks_lean.jsonl (domain, task_id, prompt, checker_payload) for the elaborating items and
<out>/lean_sanity.json (counts, excluded ids with the Lean error). Needs the pinned Mathlib project
($GWAYA_LEAN_MATHLIB_DIR or $GWAYA_DATA_ROOT/lean-mathlib/project).
  usage: build_minif2f_tasks.py --out results/gwenlaya_v4/lean_a18 [--workers 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gwaya.domains.loaders import row_to_task  # noqa: E402
from gwaya.lean_project import default_project_dir  # noqa: E402
from gwaya.oracles import Lean4CompilerOracle  # noqa: E402

HF_ID = "cat-searcher/minif2f-lean4"


def elaborates(oracle: Lean4CompilerOracle, payload: dict) -> tuple[bool, str]:
    src = f"{payload['header']}\n\n{payload['formal_statement']} by\n  sorry\n"
    res = oracle.verify_snippet(src, allow_sorry=True)
    return res.success, (res.error_message or res.stderr or "")[:400]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    from datasets import load_dataset
    root = os.environ.get("GWAYA_DATA_ROOT", os.path.expanduser("~/gwaya-data"))
    rows = load_dataset(HF_ID, split=args.split, cache_dir=os.path.join(root, "hf-datasets"))
    tasks = [row_to_task(HF_ID, dict(r), i) for i, r in enumerate(rows)]
    project = default_project_dir()
    if project is None:
        print("no pinned Mathlib project", file=sys.stderr)
        return 2
    oracle = Lean4CompilerOracle(timeout_s=300.0, project_dir=project)
    with ThreadPoolExecutor(args.workers) as ex:
        checks = list(ex.map(lambda t: elaborates(oracle, t.checker_payload), tasks))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = [t for t, (ok, _) in zip(tasks, checks) if ok]
    with (out / "tasks_lean.jsonl").open("w") as fh:
        for t in kept:
            fh.write(json.dumps({"domain": t.domain, "task_id": t.task_id, "prompt": t.prompt,
                                 "checker_payload": t.checker_payload}, ensure_ascii=False) + "\n")
    rep = {"dataset": HF_ID, "split": args.split, "n": len(tasks), "elaborating": len(kept),
           "excluded": [{"task_id": t.task_id, "error": e} for t, (ok, e) in zip(tasks, checks) if not ok],
           "mathlib": json.loads((Path(project) / "gwenlaya_lean_project.json").read_text()),
           "tasks_sha256": hashlib.sha256((out / "tasks_lean.jsonl").read_bytes()).hexdigest()}
    (out / "lean_sanity.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"{len(kept)}/{len(tasks)} statements elaborate; excluded {len(tasks) - len(kept)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
