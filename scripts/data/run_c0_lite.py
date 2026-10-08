#!/usr/bin/env python3
"""Driver for C0-lite stage 1: load primary E, run harness-sanity in parallel, cache to JSON.
Usage: run_c0_lite.py --out $NIGHT/datasets/items_sanity.json [--workers 3]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
import build_eval_manifest as B  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    items, meta = B.load_primary_e()
    print(len(items), "items", flush=True)
    t0 = time.time()
    reasons: dict[str, str | None] = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (tid, why) in enumerate(ex.map(B.sanity_one, items), 1):
            reasons[tid] = why
            if k % 50 == 0:
                print(k, f"{time.time() - t0:.0f}s", flush=True)
    for it in items:
        it["task_id"] = B.task_id_of(it)
        it["exclusion"] = reasons[it["task_id"]]
    Path(a.out).write_text(json.dumps({"meta": meta, "items": items}))
    print("done", f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
