#!/usr/bin/env python3
"""Build the verified SFT set (python, rust, lean4, math) by rejection sampling.

Only checker-VERIFIED outputs are kept (also AST-stub/grounding clean, deduplicated, <= 2 per
prompt, tasks whose tests accept a trivial baseline dropped). Rows are decontaminated against the
eval index before writing and the written file is re-checked. Rust can be added by verified
translation of the VERIFIED python rows (--rust-translate-from); Lean uses expert-iteration
rounds (--lean-round, repeatable).

  build_sft_dataset.py --dry-run                      # fake generator + fixtures, contacts nothing
  build_sft_dataset.py --tasks t.jsonl --decontam-index idx.jsonl \
      --ladder ollama@http://localhost:11434#qwen3.5:4b --ladder openai@http://vm:8000/v1#Qwen/Qwen3.5-9B

Outputs in --out-dir (default under $GWAYA_DATA_ROOT/builders, ~/gwaya-data if unset): sft.jsonl,
candidates.jsonl (cache for build_dpo_pairs.py --candidates), dropped.jsonl, flagged_decontam.jsonl,
build_report.json, DATASET_CARD.md. Exit: 0 ok, 1 re-check/provenance failure, 2 decontamination aborted.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from _common import (add_common_args, build_config, common_report, db, finish, load_tasks, maybe_translate,
                     out_dir, per_domain)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, "sft")
    a = ap.parse_args(argv)
    tasks = load_tasks(a)
    try:
        cfg = build_config(a, tasks, "sft")
    except db.DecontamAborted as exc:
        print(f"ABORT (fail closed): {exc}", file=sys.stderr)
        return 2
    drops: dict[str, list[str]] = {}
    translated, tstats = maybe_translate(a, cfg, tasks, tasks)
    kept = db.prefilter(tasks + translated, cfg, drops)
    states, info = db.run_sampling(kept, cfg)
    rows = db.select_sft(states, cfg, drops)
    report = common_report(a, cfg, drops, kept, info)
    cnt = Counter(r["domain"] for r in rows)
    ids: dict[str, set[str]] = {}
    for r in rows:
        ids.setdefault(r["domain"], set()).add(r["task_id"])
    report["per_domain"] = per_domain(cnt, states, ids)
    if tstats is not None:
        report["rust_translation"] = tstats
    if "lean4" in cfg.domains and any(s.bt.task.domain == "lean4" for s in states):
        n = cnt.get("lean4", 0)
        report["lean"] = {"rounds": {k: v for k, v in info["rounds"].items() if k.startswith("lean4")},
                          "accepted_proofs": n, "eval_only_rule_triggered": n < db.LEAN_MIN_ACCEPTED}
    report["generation_errors"] = sum(len(s.gen_errors) for s in states)
    return finish("sft", out_dir(a, "sft"), cfg, rows, report, drops, states)


if __name__ == "__main__":
    sys.exit(main())
