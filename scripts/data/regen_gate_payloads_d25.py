#!/usr/bin/env python3
"""D25: rebuild the HumanEval+ gate payloads of the frozen task files without the hidden suite.

The v1 gate text for HumanEval+ kept the complete `inputs` / `results` literals (only the loop
header was sliced). This script rewrites ONLY the `gate_payload.tests` of `py/HumanEval/*` rows,
from that row's own `checker_payload.tests` (the full released test plus `assert check(f) is None`),
with the ast-based `build_eval_manifest.python_gate_tests_plus`. Prompts, checker payloads (hidden
checks), clusters, order, MBPP+, Rust and Math rows are copied byte-for-byte as JSON objects.
Rows whose gate cannot be rebuilt safely are EXCLUDED (fail closed) and listed in the report.

The input files are not modified: the outputs are written next to them with a `.d25` suffix, so the
night-run files (which the L2 generations used) keep their recorded sha256.

usage: regen_gate_payloads_d25.py --results $GWAYA_DATA_ROOT/night/results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
import build_eval_manifest as B  # noqa: E402


def rebuild(rows: list[dict]) -> tuple[list[dict], dict]:
    out, rep = [], {"rewritten": [], "excluded": [], "unchanged_rows": 0}
    for r in rows:
        if not r["task_id"].startswith("py/HumanEval/"):
            out.append(r)
            rep["unchanged_rows"] += 1
            continue
        full = r["checker_payload"]["tests"]
        gate = B.python_gate_tests_plus(full)
        if gate is None:
            rep["excluded"].append({"task_id": r["task_id"], "reason": "gate_not_rebuildable_fail_closed"})
            continue
        leaks = B.python_gate_leaks(gate, full)
        if leaks:
            rep["excluded"].append({"task_id": r["task_id"], "reason": f"leak_check_failed:{len(leaks)}"})
            continue
        r2 = dict(r)
        r2["gate_payload"] = {**r["gate_payload"], "tests": gate}
        out.append(r2)
        rep["rewritten"].append({"task_id": r["task_id"], "gate_chars_before": len(r["gate_payload"]["tests"]),
                                 "gate_chars_after": len(gate)})
    return out, rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    res = Path(a.results)
    report = {"deviation": "D25", "files": {}}
    for name in ("tasks_E_night.jsonl", "tasks_E_primary.jsonl"):
        src = res / name
        rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
        out, rep = rebuild(rows)
        dst = res / name.replace(".jsonl", ".d25.jsonl")
        text = "".join(json.dumps(r) + "\n" for r in out)
        dst.write_text(text)
        report["files"][name] = {
            "input_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "input_rows": len(rows),
            "output": dst.name, "output_sha256": hashlib.sha256(text.encode()).hexdigest(), "output_rows": len(out),
            "n_rewritten": len(rep["rewritten"]), "excluded": rep["excluded"], "rows_copied_unchanged": rep["unchanged_rows"]}
    (res / "gate_regen_d25_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rewritten"} for k, v in report["files"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
