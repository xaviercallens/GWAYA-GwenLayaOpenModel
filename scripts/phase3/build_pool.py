#!/usr/bin/env python3
"""A33: turn raw TPU generations (answer and sample kinds) into one candidate pool file, and score it locally.

  build_pool.py make  --raw-dir DIR --tiers t1,t2 --out pool.jsonl [--extra-pool old_pool.jsonl]
  build_pool.py score --tasks tasks.jsonl --pool pool.jsonl --out scored.jsonl [--workers 8]

Pool row: {tier, task_id, cand ("g" greedy | sample index), text, mean_logprob, min_logprob, completion_tokens, finish}.
Scoring = the project's checker on the hidden suite and on the visible gate payload (same as score_samples.py).
Hidden outcomes are labels for analysis only; nothing downstream selects with them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.pop("GWAYA_ALLOW_UNISOLATED", None)


def _row(tier: str, d: dict, cand) -> dict:
    lps = d.get("token_logprobs") or []
    return {"tier": tier, "task_id": d["task_id"], "cand": cand, "text": d.get("text") or "",
            "mean_logprob": d.get("mean_logprob"), "min_logprob": min(lps) if lps else None,
            "completion_tokens": d.get("completion_tokens"), "finish": d.get("finish_reason")}


def make(a) -> None:
    out = []
    for tier in a.tiers.split(","):
        for kind, prefix in (("answer", "raw_"), ("sample", "raw_sample_")):
            p = Path(a.raw_dir) / f"{prefix}{tier}.jsonl"
            if not p.exists():
                print("missing", p)
                continue
            for line in open(p):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("domain") != "python" or d.get("finish_reason") == "prompt_too_long":
                    continue
                out.append(_row(tier, d, "g" if kind == "answer" else int(d["sample"])))
    seen = set()
    with open(a.out, "w") as fh:
        for r in out:
            k = (r["tier"], r["task_id"], str(r["cand"]))
            if k not in seen:
                seen.add(k)
                fh.write(json.dumps(r) + "\n")
    print("pool rows", len(seen))


TASKS: dict = {}


def init(tasks_path: str) -> None:
    for line in open(tasks_path):
        d = json.loads(line)
        TASKS[d["task_id"]] = d


def work(row: dict) -> dict:
    from gwaya.domains.checkers import check
    from gwaya.domains.task import Task
    sys.path.insert(0, str(ROOT / "scripts" / "phase3"))
    from exec_tests import code_of
    d = TASKS[row["task_id"]]
    code = "```python\n" + code_of(row["text"]) + "\n```"
    hid = check(Task("python", d["task_id"], d["prompt"], dict(d["checker_payload"])), code).status
    gp = d.get("gate_payload")
    gate = check(Task("python", d["task_id"], d["prompt"], dict(gp)), code).status if gp else "none"
    return {**{k: row[k] for k in ("tier", "task_id", "cand")}, "hidden": hid, "gate": gate}


def score(a) -> None:
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["tier"], r["task_id"], str(r["cand"])))
    rows = [json.loads(line) for line in open(a.pool)]
    rows = [r for r in rows if (r["tier"], r["task_id"], str(r["cand"])) not in done]
    print("todo", len(rows), flush=True)
    with Pool(a.workers, initializer=init, initargs=(a.tasks,)) as p, open(a.out, "a") as fh:
        for i, r in enumerate(p.imap_unordered(work, rows, chunksize=4)):
            fh.write(json.dumps(r) + "\n")
            if i % 500 == 0:
                fh.flush()
                print(i, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--raw-dir", required=True)
    m.add_argument("--tiers", required=True)
    m.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--tasks", required=True)
    s.add_argument("--pool", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    make(a) if a.cmd == "make" else score(a)


if __name__ == "__main__":
    main()
