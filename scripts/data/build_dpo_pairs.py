#!/usr/bin/env python3
"""Build DPO pairs: same prompt, chosen = checker-VERIFIED, rejected = checker-FAILED.

UNVERIFIED verdicts and infrastructure failures (missing toolchain, checker error, timeouts,
unreachable endpoint) are never used as `rejected`. One pair per prompt; rejected prefers
confident-wrong over plain wrong over shallow (compile/stub/empty) and shallow pairs are capped at
25 % of the set. The chosen sample is picked by hash, not by length, and the chosen/rejected
length ratio (characters) is reported in build_report.json and the dataset card.

  build_dpo_pairs.py --dry-run
  build_dpo_pairs.py --candidates runs/sft/candidates.jsonl --tasks t.jsonl --decontam-index idx.jsonl
  build_dpo_pairs.py --tasks t.jsonl --decontam-index idx.jsonl --ladder ...   # samples itself

Outputs as build_sft_dataset.py, with dpo_pairs.jsonl. Exit: 0 ok, 1 re-check/provenance, 2 aborted.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from _common import (add_common_args, build_config, common_report, db, finish, load_tasks, maybe_translate,
                     out_dir, per_domain)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, "dpo")
    a = ap.parse_args(argv)
    tasks = load_tasks(a) if (a.tasks or a.dry_run) else []
    try:
        cfg = build_config(a, tasks, "dpo")
    except db.DecontamAborted as exc:
        print(f"ABORT (fail closed): {exc}", file=sys.stderr)
        return 2
    drops: dict[str, list[str]] = {}
    states, info, by_task, kept = None, {}, {}, []
    if a.candidates:
        for r in db.read_jsonl(a.candidates):
            r.pop("schema", None)
            by_task.setdefault(r["task_key"], []).append(db.Candidate(**r))
        bt_map = {t.key: t for t in tasks}
    else:
        translated, tstats = maybe_translate(a, cfg, tasks, tasks)
        kept = db.prefilter(tasks + translated, cfg, drops)
        states, info = db.run_sampling(kept, cfg)
        by_task = {s.bt.key: s.cands for s in states}
        bt_map = {t.key: t for t in kept}
    pairs, excl = db.build_pairs(by_task, bt_map, cfg, drops)
    report = common_report(a, cfg, drops, kept or list(bt_map.values()), info)
    cnt = Counter(p["domain"] for p in pairs)
    ids: dict[str, set[str]] = {}
    for p in pairs:
        ids.setdefault(p["domain"], set()).add(p["task_id"])
    report["per_domain"] = per_domain(cnt, states, ids)
    report["length_ratio"] = db.length_stats([p["length"]["ratio"] for p in pairs])
    report["rejected_categories"] = dict(Counter(p["provenance"]["rejected_category"] for p in pairs))
    report["excluded_from_rejected"] = excl
    report["candidates_source"] = a.candidates or "sampled in this run"
    if not a.candidates and a.rust_translate_from:
        report["rust_translation"] = tstats
    return finish("dpo", out_dir(a, "dpo"), cfg, pairs, report, drops, states)


if __name__ == "__main__":
    sys.exit(main())
