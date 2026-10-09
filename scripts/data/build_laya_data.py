#!/usr/bin/env python3
"""Turn per-tier study results into Laya training rows (docs/ANALYSIS_PLAN_E_TPU.md, addendum A13).

For each task and tier we need two things from a scored study directory (run_study.py --mode score, arms base and
gate_only): the hidden-test/answer correctness (`base` row `score`) and the executed gate's verdict (`gate_only`
row `gate`), plus the generation (text, mean and min token log-prob) from gens.jsonl.

  router rows      {"source","id","domain","prompt","tiers":{tier: 0|1}}           pre-generation head labels
  calibrator rows  {"source","id","domain","prompt","tier","candidate","correct",
                    "signals": {"gate","mean_logprob","min_logprob"}}               post-generation head

Only the signals that exist at answer time are used; correctness is the label and never a feature. `source` is the task
id prefix and `id` the task id, so train_laya.py's problem-grouped validation hash keeps all tiers of a task together.
Nothing here reads E' (the caller decides which task files go in).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def split_id(task_id: str) -> tuple[str, str]:
    head, _, rest = task_id.partition("/")
    return head, rest or task_id


def index_study(dirs: Iterable[Path]) -> dict[str, dict[str, dict[tuple[str, str], Any]]]:
    """-> {tier: {"correct": {key: bool}, "gate": {key: str}, "gen": {key: gens record}}}; later dirs win."""
    out: dict[str, dict[str, dict]] = {}
    for d in dirs:
        d = Path(d)
        for r in read_jsonl(d / "rows.jsonl"):
            t = out.setdefault(r["model"], {"correct": {}, "gate": {}, "gen": {}})
            k = (r["domain"], r["task_id"])
            if r["arm"] == "base":
                t["correct"][k] = r["score"] == "VERIFIED"
            elif r["arm"] == "gate_only":
                t["gate"][k] = str(r["gate"]).lower()
        for g in read_jsonl(d / "gens.jsonl"):
            if g.get("kind") == "pot" or g.get("temperature") not in (0.0, None):
                continue
            out.setdefault(g["model"], {"correct": {}, "gate": {}, "gen": {}})["gen"][(g["domain"], g["task_id"])] = g
    return out


def build(tasks: list[dict], study: dict, tiers: list[str], split_of: dict[tuple[str, str], str] | None = None
          ) -> tuple[list[dict], list[dict], dict[str, int]]:
    router, calib = [], []
    skipped = 0
    for t in tasks:
        k = (t["domain"], t["task_id"])
        source, tid = split_id(t["task_id"])
        labels = {}
        for tier in tiers:
            s = study.get(tier)
            if s is None or k not in s["correct"] or k not in s["gate"] or k not in s["gen"]:
                continue
            labels[tier] = int(s["correct"][k])
            g = s["gen"][k]
            lps = g.get("token_logprobs") or []
            calib.append({
                "source": source, "id": tid, "domain": t["domain"], "prompt": t["prompt"], "tier": tier,
                "candidate": g.get("text", ""), "correct": int(s["correct"][k]),
                "signals": {"gate": s["gate"][k], "mean_logprob": g.get("mean_logprob"),
                            "min_logprob": min(lps) if lps else None},
                **({"split": split_of[k]} if split_of and k in split_of else {}),
            })
        if labels:
            router.append({"source": source, "id": tid, "domain": t["domain"], "prompt": t["prompt"], "tiers": labels,
                           **({"split": split_of[k]} if split_of and k in split_of else {})})
        else:
            skipped += 1
    return router, calib, {"tasks": len(tasks), "router_rows": len(router), "calibrator_rows": len(calib),
                           "tasks_without_any_tier": skipped}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="+", required=True, type=Path)
    ap.add_argument("--study-dirs", nargs="+", required=True, type=Path, help="scored dirs with rows.jsonl + gens.jsonl; later wins")
    ap.add_argument("--tiers", nargs="+", required=True, help="served names, cheapest first")
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    tasks = [r for p in a.tasks for r in read_jsonl(p)]
    router, calib, counts = build(tasks, index_study(a.study_dirs), a.tiers)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "router.jsonl").write_text("".join(json.dumps(r) + "\n" for r in router))
    (a.out_dir / "calibrator.jsonl").write_text("".join(json.dumps(r) + "\n" for r in calib))
    (a.out_dir / "build_counts.json").write_text(json.dumps({**counts, "tiers": a.tiers}, indent=1))
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
