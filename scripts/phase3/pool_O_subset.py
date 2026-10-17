#!/usr/bin/env python3
"""A23/A33: the deterministic pool-O generation subset: train + calib rows sorted by sha256(task_id), first N.
Fixed before any pool-O label exists, so the generation budget is chosen without looking at outcomes.
  pool_O_subset.py --pool-dir $GWAYA_DATA_ROOT/pool_O --n 6000
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()
    d = Path(a.pool_dir)
    rows = []
    for split in ("train", "calib"):
        for line in open(d / f"tasks_O_{split}.jsonl"):
            r = json.loads(line)
            r["o_split"] = split
            rows.append(r)
    rows.sort(key=lambda r: hashlib.sha256(r["task_id"].encode()).hexdigest())
    sub = rows[: a.n]
    (d / f"tasks_O_subset{a.n}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sub))
    print({"available": len(rows), "subset": len(sub),
           "per_split": {s: sum(1 for r in sub if r["o_split"] == s) for s in ("train", "calib")}})


if __name__ == "__main__":
    main()
