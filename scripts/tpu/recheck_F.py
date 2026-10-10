#!/usr/bin/env python
"""A21 timeout audit for the F python/math scores (same idea as scripts/tpu/recheck_failures.py + serial_overlay.py).

The parallel scoring run executes many sandboxes at once on a loaded machine, so a near-limit run can turn into a
spurious timeout. Every row that did NOT pass in that run is re-checked here, one check at a time, against the stored
answer: python scorer (hidden suite, arm `base`), python gate (visible test), math gate (program of thought) and math
scorer (gold answer). A verdict that changes is a contention artefact: the overlay file carries the corrected rows
(`later rows win` in analyze_F.py / build_F_laya_rows.py); the original hash-chained rows are never edited. Rows that
time out even when run alone are listed. PASSING rows are NOT re-run (contention cannot create a pass).

  python scripts/tpu/recheck_F.py --run DIR --tasks tasks_F_python.jsonl tasks_F_math.jsonl --model qwen3.5-4b-bf16 \
      --out OUT_DIR
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
from gwaya.domains.task import Task  # noqa: E402
from scripts import run_study as rs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--tasks", nargs="+", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    tasks = {}
    for p in a.tasks:
        for t in rs.load_tasks_jsonl(p):
            tasks[(t.domain, t.task_id)] = t
    rows = [r for r in rs.ChainedLog(a.run / "rows.jsonl").records if r.get("model") == a.model]
    base = {(r["domain"], r["task_id"]): r for r in rows if r["arm"] == "base"}
    gate = {(r["domain"], r["task_id"]): r for r in rows if r["arm"] == "gate_only"}
    gens = [g for g in rs.ChainedLog(a.run / "gens.jsonl").records if g["model"] == a.model]
    ans = {(g["domain"], g["task_id"]): g for g in gens if g.get("temperature") == 0.0 and g.get("kind") != "pot"}
    pot = {(g["domain"], g["task_id"]): g["text"] for g in gens if g.get("kind") == "pot"}
    dispatch = rs.GateDispatch(check)
    overlay, flips, alone, reasons = [], [], [], Counter()
    n = 0
    for k, t in sorted(tasks.items()):
        text = ans[k]["text"]
        if base[k]["score"] != "VERIFIED":  # scorer (hidden suite / gold answer)
            n += 1
            res = check(t, text)
            reasons[f"{k[0]}:score:{res.status}:{res.evidence.get('reason')}"] += 1
            if res.status != base[k]["score"]:
                flips.append({"what": "score", "domain": k[0], "task_id": k[1], "was": base[k]["score"], "now": res.status})
                overlay.append({**base[k], "score": res.status, "overlay": "serial_recheck"})
            if res.evidence.get("reason") == "timeout":
                alone.append({"what": "score", "domain": k[0], "task_id": k[1]})
        if gate[k]["gate"] != "VERIFIED":  # gate (python: first visible test; math: program of thought)
            payload = dict(t.checker_payload.get("__gate__", {}))
            if k[0] == "math":
                if k not in pot:
                    continue
                payload["pot"] = pot[k]
            n += 1
            res = dispatch(Task(t.domain, t.task_id, t.prompt, payload), text)
            reasons[f"{k[0]}:gate:{res.status}:{res.evidence.get('reason')}"] += 1
            if res.status != gate[k]["gate"]:
                flips.append({"what": "gate", "domain": k[0], "task_id": k[1], "was": gate[k]["gate"], "now": res.status})
                overlay.append({**gate[k], "gate": res.status, "answered": res.status == "VERIFIED", "overlay": "serial_recheck"})
            if res.evidence.get("reason") == "timeout":
                alone.append({"what": "gate", "domain": k[0], "task_id": k[1]})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"overlay_{a.model}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in overlay))
    rep = {"model": a.model, "rechecked": n, "n_flips": len(flips), "flips": flips,
           "timeouts_when_run_alone": alone, "reasons": dict(sorted(reasons.items()))}
    (a.out / f"recheck_{a.model}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("model", "rechecked", "n_flips")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
