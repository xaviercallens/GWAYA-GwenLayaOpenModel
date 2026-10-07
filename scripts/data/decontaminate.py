#!/usr/bin/env python3
"""Decontaminate GwenLaya v4 train/calib sets against every eval set (all domains, CPU only).

Usage:
  decontaminate.py --index $GWAYA_DATA_ROOT/decontam/decontam_index.jsonl \
      --train train_sources.jsonl --out-dir $GWAYA_DATA_ROOT/decontam/run1

--index / --train are JSONL. Each line is either a dataset descriptor
  {"hf_id", "domain", "role", "path"?, "split"?, "config"?, "name"?, "row_count"?}
(path: local .jsonl/.json/.parquet/.arrow file, or a directory of HF-cache/save_to_disk .arrow
files; if absent, the path is looked up in experiments/DATA_MANIFEST.json), or an item record
  {"eval_set"|"dataset", "item_id", "domain", "fields": {role: text}, "ids"?: [...]}.

Never downloads. Fails closed: exit 2 (report only, no clean sets) when an eval set in the index is
not materialized, unless --allow-missing (the report then says complete=false).

Outputs in --out-dir: decontam_report.json (per eval set and per (train source, eval set) removal
counts, verification of every written file), flagged.jsonl, clean/<train set>.jsonl.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gwaya import decontaminate as dc  # noqa: E402

_DATA_ROOT = os.environ.get("GWAYA_DATA_ROOT", os.path.expanduser("~/gwaya-data"))
DEFAULT_INDEX = os.path.join(_DATA_ROOT, "decontam", "decontam_index.jsonl")
DEFAULT_OUT = os.path.join(_DATA_ROOT, "decontam", "run")
PLAN = ROOT / "experiments" / "plan.json"
MANIFEST = ROOT / "experiments" / "DATA_MANIFEST.json"


def _load_embedder(spec: str):
    mod, _, fn = spec.partition(":")
    if not fn:
        raise SystemExit("--embedder must be module:callable")
    return getattr(importlib.import_module(mod), fn)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=DEFAULT_INDEX, help="eval descriptors/items JSONL")
    ap.add_argument("--train", help="train/calib/dpo descriptors/items JSONL (omit for an eval-only load report)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--allow-missing", action="store_true", help="proceed when eval sets are not materialized")
    ap.add_argument("--no-manifest", action="store_true", help="do not resolve paths from DATA_MANIFEST.json")
    ap.add_argument("--no-plan-check", action="store_true",
                    help="do not require every role=eval dataset of experiments/plan.json in the index")
    ap.add_argument("--known-fields-only", action="store_true", help="ignore unknown string columns")
    ap.add_argument("--ngram-n", type=int, default=dc.NGRAM_N)
    ap.add_argument("--jaccard", type=float, default=dc.JACCARD_THRESHOLD)
    ap.add_argument("--ngram-max-eval-df", type=int, default=None,
                    help="ignore n-grams shared by more than this many eval items (default: never)")
    ap.add_argument("--embedder", help="module:callable(list[str]) -> list[list[float]] (off by default)")
    ap.add_argument("--embed-threshold", type=float, default=dc.EMBED_THRESHOLD)
    ap.add_argument("--no-dedup", action="store_true", help="skip APPS/TACO/CodeContests/LeetCode cross-dedup")
    a = ap.parse_args(argv)

    manifest = None if a.no_manifest or not MANIFEST.exists() else json.loads(MANIFEST.read_text())
    plan_ds = json.loads(PLAN.read_text()).get("datasets", []) if PLAN.exists() else []
    plan_domains = {d["hf_id"]: d["domain"] for d in plan_ds}
    expected = [] if a.no_plan_check else sorted({d["hf_id"] for d in plan_ds if d.get("role") == "eval"})
    cfg = dc.Config(ngram_n=a.ngram_n, jaccard_threshold=a.jaccard, ngram_max_eval_df=a.ngram_max_eval_df,
                    embedder=_load_embedder(a.embedder) if a.embedder else None,
                    embed_threshold=a.embed_threshold)
    if a.no_dedup:
        cfg.dedup_group = ()
    evals = dc.sources_from_jsonl(a.index, manifest, plan_domains)
    trains = dc.sources_from_jsonl(a.train, manifest, plan_domains) if a.train else []
    rep = dc.run(evals, trains, a.out_dir, cfg, allow_missing=a.allow_missing,
                 known_fields_only=a.known_fields_only,
                 index_sha256=dc.files_sha256([Path(a.index)]), expected_eval=expected)
    out = Path(a.out_dir) / "decontam_report.json"
    print(f"status: {rep['status']}")
    for name, st in rep["eval_status"].items():
        print(f"  eval {name}: {st['status']} n={st['n_items']}")
    for name, t in rep.get("train_sets", {}).items():
        print(f"  train {name}: in={t['n_in']} flagged={t['n_flagged']} kept={t['n_kept']} "
              f"verified_clean={t['verified_clean']} ({t['status']})")
    print(f"report: {out}")
    if rep["status"].startswith("aborted"):
        return 2
    return 0 if all(t["verified_clean"] for t in rep.get("train_sets", {}).values()) else 1


if __name__ == "__main__":
    sys.exit(main())
