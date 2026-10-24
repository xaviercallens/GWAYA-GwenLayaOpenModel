#!/usr/bin/env python3
"""E1 (Python first, docs/PROPOSAL_ROUTING_DATASET_AND_STRATEGY.md 4.1): Python-positive routing rows from BigCodeBench (Apache-2.0).

Two views of the same task: ``visible`` = the full instruction (carries the ``def``/``import`` scaffold) and ``blind`` = only the
natural-language description before the scaffold. ``blind_nohit`` keeps the blind rows where the GWAYA keyword scorer finds no
keyword, i.e. the stratum a keyword router cannot see. Only ids and counts are written; prompts are rebuilt from the dataset.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
from gwaya.hivemind_expert import KEYWORDS, keyword_hit
from scripts.routing.route_eval import wilson

MARK = "You should write self-contained code starting with"


def hits(text: str) -> int:
    low = text.lower()
    return sum(keyword_hit(k, low) for k in KEYWORDS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", default="results/routing/python_stratum_counts.json")
    a = ap.parse_args()
    d = pd.read_parquet(a.parquet)
    vis = d["instruct_prompt"].tolist()
    blind = [t.split(MARK)[0].strip() for t in vis]
    hv = [hits(t) > 0 for t in vis]
    hb = [hits(t) > 0 for t in blind]
    n = len(d)
    res = {"n": n, "visible_keyword_hit": [sum(hv), n, wilson(sum(hv), n)],
           "blind_keyword_hit": [sum(hb), n, wilson(sum(hb), n)],
           "blind_nohit_ids": [i for i, h in zip(d["task_id"], hb) if not h]}
    res["blind_nohit"] = len(res["blind_nohit_ids"])
    Path(a.out).write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k != "blind_nohit_ids"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
