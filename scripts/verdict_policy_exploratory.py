#!/usr/bin/env python
"""EXPLORATORY, not pre-registered: what the weak-gate verdict policy (GwenLaya min_visible_tests) would do on stored E data.

Read-only, CPU only, no generation and no gate re-execution. Three parts:
  1. depth_dial: from the stored aggregate k-gate verdicts of A15.2 (rust_gate_depth.json) and A17 (python_gate_depth.json).
     A k-gate sees exactly k visible tests, so a policy k_min demotes all of its VERIFIED outputs when k < k_min and none
     otherwise; the confident-wrong among the remaining VERIFIED is the stored k-gate's 1 - precision.
  2. stored_gate: per tier, the stored E gate verdicts (arm gate_only) against the hidden score (arm base), with the visible
     tests of each task's stored gate payload counted by gwaya.domains.strength (before and after the Rust counting fix).
  3. rust_count_histogram: visible_test_count over the E and R' Rust gate payloads (and, labelled, the hidden suites),
     before (raw assert-macro count) and after the strength.py fix.
Nothing here chooses a k_min; E is the evaluation set and must not set a default.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gwaya.domains.strength import _RUST_ASSERT, visible_test_count  # noqa: E402

K_MINS = (1, 2, 3, 4)


def _count_before(domain: str, tests: str | None) -> int | None:
    """visible_test_count as it was before this change (Rust: raw assert-macro count on unstripped text)."""
    if domain == "rust":
        return len(_RUST_ASSERT.findall(tests)) if tests else None
    return visible_test_count(domain, tests)


def _jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def depth_dial(path: Path, domain: str) -> dict[str, Any]:
    d = json.loads(path.read_text())
    ks = sorted({int(k.split(".")[0][1:]) for t in d["tiers"].values() for k in t if k.startswith("k")})
    out: dict[str, Any] = {}
    for tier, t in d["tiers"].items():
        rows = []
        for k in ks:
            cov, prec, cw = (t[f"k{k}.{m}"]["point"] for m in ("coverage", "precision", "confident_wrong"))
            for k_min in K_MINS:
                demoted = k < k_min
                rows.append({"gate_k": k, "k_min": k_min, "n_tasks": t["n"], "n_verified": round(cov * t["n"]),
                             "demoted_share_of_verified": 1.0 if demoted else 0.0,
                             "remaining_verified_share_of_tasks": 0.0 if demoted else cov,
                             "confident_wrong_among_remaining_verified": None if demoted else 1.0 - prec,
                             "confident_wrong_remaining_over_tasks": 0.0 if demoted else cw})
        out[tier] = rows
    return {"source": str(path.relative_to(ROOT)), "domain": domain, "ks": ks, "tiers": out}


def stored_gate(tasks: dict[str, dict[str, Any]], rows: list[dict[str, Any]], domain: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tier in sorted({r["model"] for r in rows if r["domain"] == domain and r["arm"] == "gate_only"}):
        hidden = {r["task_id"]: r["score"] == "VERIFIED" for r in rows if r["model"] == tier and r["arm"] == "base" and r["domain"] == domain}
        gate = {r["task_id"]: r["gate"] == "VERIFIED" for r in rows if r["model"] == tier and r["arm"] == "gate_only" and r["domain"] == domain}
        ids = sorted(i for i in gate if i in hidden and i in tasks)
        res: dict[str, Any] = {"n_tasks": len(ids), "n_verified": sum(gate[i] for i in ids),
                               "confident_wrong_among_verified_no_policy": _rate([not hidden[i] for i in ids if gate[i]])}
        for name, fn in (("after_fix", visible_test_count), ("before_fix", _count_before)):
            n_vis = {i: fn(domain, tasks[i]["gate_payload"].get("tests")) for i in ids}
            ver = [i for i in ids if gate[i]]
            res[name] = {"visible_tests_hist_verified": _hist(n_vis[i] for i in ver), "by_k_min": []}
            for k_min in K_MINS:
                keep = [i for i in ver if isinstance(n_vis[i], int) and n_vis[i] >= k_min]
                res[name]["by_k_min"].append({
                    "k_min": k_min, "demoted": len(ver) - len(keep),
                    "demoted_share_of_verified": _rate([i not in keep for i in ver]),
                    "remaining_verified": len(keep),
                    "confident_wrong_among_remaining_verified": _rate([not hidden[i] for i in keep]),
                    "confident_wrong_remaining_over_tasks": sum(not hidden[i] for i in keep) / len(ids) if ids else None})
        out[tier] = res
    return out


def _rate(xs: list[bool]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _hist(vals) -> dict[str, int]:
    c = collections.Counter("none" if v is None else str(v) for v in vals)
    return dict(sorted(c.items(), key=lambda kv: (kv[0] == "none", int(kv[0]) if kv[0] != "none" else 0)))


def rust_histograms(task_sets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, ts in task_sets.items():
        rust = [t for t in ts if t["domain"] == "rust"]
        for payload in ("gate_payload", "checker_payload"):
            before = [_count_before("rust", t[payload].get("tests")) for t in rust]
            after = [visible_test_count("rust", t[payload].get("tests")) for t in rust]
            out[f"{name}.{payload}"] = {"n_tasks": len(rust), "before_fix": _hist(before), "after_fix": _hist(after),
                                        "changed": sum(a != b for a, b in zip(before, after)),
                                        "has_test_attr": sum("#[test]" in (t[payload].get("tests") or "") for t in rust)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    dr = a.data_root
    e_tasks = _jsonl(dr / "night/results/tasks_E_primary.d25.jsonl")
    r_tasks = _jsonl(dr / "laya_data/tasks_Rust_fresh.jsonl")
    rows_e = _jsonl(dr / "etpu/tiers/rows.jsonl") + _jsonl(dr / "etpu/nine/rows.jsonl")
    rows_r = [r for t in ("2b", "4b", "9b") for r in _jsonl(dr / f"laya_data/rfresh_qwen3.5-{t}-bf16/rows.jsonl")]
    by_id = {t["task_id"]: t for t in e_tasks}
    r_by_id = {t["task_id"]: t for t in r_tasks}
    ed = ROOT / "results/gwenlaya_v4/edition"
    res = {
        "label": "EXPLORATORY, not pre-registered. Read-only on stored E/R' data; no generation, no gate re-execution. "
                 "Not evidence of any end-user confident-wrong reduction (users supply their own tests), and no k_min is "
                 "recommended: E is the evaluation set.",
        "metric_notes": {"confident_wrong_among_remaining_verified": "share of the VERIFIED outputs that survive the policy "
                         "and fail the hidden tests", "confident_wrong_remaining_over_tasks": "same errors divided by all tasks",
                         "demoted": "a demoted output goes to the calibrator / escalation path, it is not counted as wrong"},
        "inputs": {"e_tasks": "night/results/tasks_E_primary.d25.jsonl", "r_tasks": "laya_data/tasks_Rust_fresh.jsonl",
                   "e_rows": ["etpu/tiers/rows.jsonl", "etpu/nine/rows.jsonl"],
                   "r_rows": [f"laya_data/rfresh_qwen3.5-{t}-bf16/rows.jsonl" for t in ("2b", "4b", "9b")]},
        "depth_dial": {"python": depth_dial(ed / "python_gate_depth.json", "python"),
                       "rust": depth_dial(ed / "rust_gate_depth.json", "rust")},
        "stored_gate": {"E.python": stored_gate(by_id, rows_e, "python"), "E.rust": stored_gate(by_id, rows_e, "rust"),
                        "Rprime.rust": stored_gate(r_by_id, rows_r, "rust")},
        "rust_count_histogram": rust_histograms({"E": e_tasks, "Rprime": r_tasks}),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
