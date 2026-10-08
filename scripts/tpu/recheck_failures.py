#!/usr/bin/env python
"""A8 timeout audit: re-check every non-passing Python/Rust `base` row serially and report flips.

Parallel scoring (run_study --check-workers N) can turn a near-limit run into a spurious timeout under
CPU contention. A row that FAILED/UNVERIFIED in the parallel run but VERIFIES when re-run alone is a
contention artefact. This script re-runs those checks one at a time against the same stored answers and
reports (a) flips, (b) checks that time out even when run alone (genuine slowness), (c) per-reason counts.

  python scripts/tpu/recheck_failures.py --run DIR --tasks TASKS --model qwen3.5-4b-bf16 --out report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gwaya.domains.checkers import check  # noqa: E402
from scripts import run_study as rs  # noqa: E402


def recheck(run_dir: Path, tasks_path: Path, model: str, domains=("python", "rust")) -> dict:
    tasks = {(t.domain, t.task_id): t for t in rs.load_tasks_jsonl(tasks_path)}
    rows = [r for r in rs.ChainedLog(run_dir / "rows.jsonl").records if r["arm"] == "base" and r["model"] == model]
    gens = {(g["domain"], g["task_id"]): g for g in rs.ChainedLog(run_dir / "gens.jsonl").records
            if g["model"] == model and g.get("temperature") == 0.0}
    todo = [r for r in rows if r["domain"] in domains and r["score"] != "VERIFIED"]
    flips, alone_timeouts, reasons = [], [], Counter()
    for r in todo:
        res = check(tasks[(r["domain"], r["task_id"])], gens[(r["domain"], r["task_id"])]["text"])
        reason = (res.evidence.get("details") or {}).get("reason", res.evidence.get("reason"))
        reasons[f"{r['domain']}:{reason}"] += 1
        if res.status != r["score"]:
            flips.append({"domain": r["domain"], "task_id": r["task_id"], "was": r["score"], "now": res.status, "reason": reason})
        if reason == "timeout" or "timed out" in (res.evidence.get("error") or "").lower():
            alone_timeouts.append({"domain": r["domain"], "task_id": r["task_id"]})
    return {"model": model, "rechecked": len(todo), "flips": flips, "n_flips": len(flips),
            "timeouts_when_run_alone": alone_timeouts, "reasons": dict(reasons)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    rep = recheck(a.run, a.tasks, a.model)
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("model", "rechecked", "n_flips")}), "timeouts alone:", len(rep["timeouts_when_run_alone"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
