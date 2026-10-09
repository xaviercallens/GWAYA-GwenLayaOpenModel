#!/usr/bin/env python
"""A13b sensitivity: router cost ratios with each run's first (compile warm-up) chunk of 192 tasks set to zero cost.

Same arrays and policy as scripts/analyze_laya.py (tau passed in, chosen on C there); programs for math stay charged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_cascades as AC  # noqa: E402
from scripts import analyze_laya as L  # noqa: E402
from scripts import analyze_study as A  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tiers", nargs=3, required=True)
    ap.add_argument("--e-tasks", required=True, type=Path)
    ap.add_argument("--e-study", nargs=3, required=True, type=Path, help="one scored dir per tier, same order as --tiers")
    ap.add_argument("--e-raw", nargs=3, required=True, type=Path, help="raw generation files per tier (file order = chunk order)")
    ap.add_argument("--router-preds", required=True, type=Path)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    rows = [r for d in a.e_study for r in L.read_jsonl(d / "rows.jsonl")]
    gens = [g for d in a.e_study for g in L.read_jsonl(d / "gens.jsonl")]
    tasks = sorted(L.read_jsonl(a.e_tasks), key=lambda t: (t["domain"], t["task_id"]))
    keys = [(t["domain"], t["task_id"]) for t in tasks]
    arr = [AC.tier_arrays(m, keys, rows, gens) for m in a.tiers]
    pr = {(p["key"][0], p["key"][1]): p["probs"] for p in L.read_jsonl(a.router_preds)}
    rp = np.array([pr[(t["task_id"].partition("/")[0], t["task_id"].partition("/")[2])] for t in tasks])
    nw = []
    for raw, x in zip(a.e_raw, arr):
        first = {(r["domain"], r["task_id"]) for r in L.read_jsonl(raw)[:192]}
        mask = np.array([k in first for k in keys])
        nw.append({**x, "cost": np.where(mask, 0.0, x["cost"])})
    pol, lad = L.run_policy(a.tau, nw, rp), AC.offline_cascade(nw)
    big = nw[-1]
    dom = np.array([t["domain"] for t in tasks]); clu = np.array([t["cluster"] for t in tasks])

    def st(idx):
        return (pol["cost"][idx].mean() / (big["cost"] + big["pot"])[idx].mean(), pol["cost"][idx].mean() / lad["cost"][idx].mean())
    pt = st(np.arange(len(keys)))
    b = A.cluster_boot(st, dom, clu, n_boot=a.n_boot, seed=0)
    out = {"tau": a.tau, "warmup_chunk_tasks": 192, "router_over_large_nowarm": A.pct_ci(pt[0], b[:, 0]),
           "router_over_ladder_nowarm": A.pct_ci(pt[1], b[:, 1])}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
