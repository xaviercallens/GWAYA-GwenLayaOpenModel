#!/usr/bin/env python3
"""Build the unified (task, tier) decision tables for the A26 trust loop.

dev_table.jsonl  = F python (407 x 2B/4B/9B) + E' python (169 MBPP x 2B/4B/9B)
test_table.jsonl = E python (HumanEval+ 156, MBPP+ 373) x every tier with stored generations
                   (2B, 4B, 9B in etpu/nine, Coder-1.5B). SEALED: this script only writes it,
                   never scores it.

Columns: set, benchmark, task_id, cluster, tier, correct (LABEL: hidden-check outcome),
gate (verified|failed|unverified|none; verdict of the VISIBLE first assert / gate payload,
taken from the 'gate_only' arm row), mean_logprob, min_logprob, completion_tokens,
finish_reason, prompt, candidate, n_visible_tests, laya_p, router_p, plus laya_src.
Nothing is imputed: absent fields are null.

Run: python3 scripts/loop/build_table.py
"""
import ast
import json
import os
import re
import sys
from collections import Counter

D = "/mnt/data/home/xavkal/gwaya-data"
OUT = f"{D}/trust_loop"
TIERS3 = ["qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16"]


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def n_visible(payload, declared=None):
    if declared is not None:
        return int(declared)
    t = (payload or {}).get("tests") or ""
    try:
        tree = ast.parse(t)
    except Exception:
        return len(re.findall(r"^\s*assert\b", t, re.M)) or None
    for node in ast.walk(tree):  # HumanEval+ style: inputs = [[...], ...]
        if isinstance(node, ast.Assign) and any(
            isinstance(x, ast.Name) and x.id == "inputs" for x in node.targets
        ) and isinstance(node.value, (ast.List, ast.Tuple)):
            return len(node.value.elts)
    n = sum(1 for node in tree.body if isinstance(node, ast.Assert))
    return n or None


def min_lp(tl):
    return min(tl) if tl else None


def tier_data(dirs, domain="python"):
    """(task_id, model) -> dict(correct, gate, gen) from base + gate_only rows and gens."""
    out = {}
    for d in dirs:
        for r in jl(f"{d}/rows.jsonl"):
            if r["domain"] != domain:
                continue
            e = out.setdefault((r["task_id"], r["model"]), {})
            if r["arm"] == "base":
                e["correct"] = 1 if r["score"] == "VERIFIED" else 0
            elif r["arm"] == "gate_only":
                e["gate"] = (r.get("gate") or "none").lower()
        for g in jl(f"{d}/gens.jsonl"):
            if g["domain"] != domain:
                continue
            out.setdefault((g["task_id"], g["model"]), {})["gen"] = g
    return out


def make_rows(set_name, tasks, td, tiers, bench_fn, laya, router, laya_src, notes):
    rows = []
    for t in tasks:
        for tier in tiers:
            e = td.get((t["task_id"], tier))
            if not e or "correct" not in e or "gen" not in e:
                notes["missing"][f"{set_name}/{tier}"] += 1
                continue
            g = e["gen"]
            tl = g.get("token_logprobs") or []
            if "gate" not in e:
                notes["no_gate"][f"{set_name}/{tier}"] += 1
            key = (t["task_id"], tier)
            rows.append({
                "set": set_name,
                "benchmark": bench_fn(t),
                "task_id": t["task_id"],
                "cluster": t.get("cluster") or t["task_id"],
                "tier": tier,
                "correct": e["correct"],
                "gate": e.get("gate", "none"),
                "mean_logprob": g.get("mean_logprob"),
                "min_logprob": min_lp(tl),
                "completion_tokens": g.get("completion_tokens"),
                "finish_reason": g.get("done_reason"),
                "prompt": t["prompt"],
                "candidate": g.get("text"),
                "n_visible_tests": n_visible(t.get("gate_payload"), t.get("n_gate_assertions")),
                "laya_p": laya.get(key),
                "router_p": router.get(key),
                "laya_src": laya_src if key in laya else None,
            })
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    notes = {"missing": Counter(), "no_gate": Counter()}

    # ---------------- F python ----------------
    f_tasks = [t for t in jl(f"{D}/eval_sets_F_scored/tasks_F_pm.jsonl") if t["domain"] == "python"]
    f_td = tier_data([f"{D}/eval_sets_F_scored/{t}" for t in TIERS3])
    cal = {}
    for p in jl(f"{D}/eval_sets_F_laya/out_x/out/calibrator/preds_F.jsonl"):
        if p["key"][0] == "py":
            cal[("py/" + p["key"][1], p["tier"])] = p["probs"][0]
    rtr = {}
    for p in jl(f"{D}/eval_sets_F_laya/out_x/out/router/preds_F.jsonl"):
        if p["key"][0] == "py":
            for tier, v in zip(TIERS3, p["probs"]):
                rtr[("py/" + p["key"][1], tier)] = v
    bf = {"python/LeetCodeDataset": "LeetCode", "python/FullStackBench": "FSB",
          "python/BigCodeBench": "BCB"}
    dev = make_rows("F", f_tasks, f_td, TIERS3, lambda t: bf[t["stratum"]], cal, rtr,
                    "laya_v1_calibrator_F", notes)

    # ---------------- E' python (MBPP) ----------------
    ep_tasks = [t for t in jl(f"{D}/laya_data/tasks_Eprime.jsonl") if t["domain"] == "python"]
    ep_td = tier_data([f"{D}/laya_data/eprime_{t}" for t in TIERS3])
    ecal = {}
    # only Laya v2 (layout) predictions exist for E'; no v1 and no router predictions.
    for p in jl(f"{D}/laya_layout/out_x/out/calibrator_v2/preds_eprime.jsonl"):
        if p["key"][0] == "py":
            ecal[("py/" + p["key"][1], p["tier"])] = p["probs"][0]
    dev += make_rows("Eprime", ep_tasks, ep_td, TIERS3, lambda t: "MBPP", ecal, {},
                     "laya_v2_layout_calibrator_eprime", notes)

    # ---------------- E python (sealed) ----------------
    e_tasks = [t for t in jl(f"{D}/night/results/tasks_E_primary.d25.jsonl") if t["domain"] == "python"]
    e_dirs = [f"{D}/etpu/tiers", f"{D}/etpu/nine", f"{D}/etpu/qwen2.5-coder-1.5b-bf16"]
    e_td = tier_data(e_dirs)
    e_tiers = sorted({m for (_, m) in e_td})
    test = make_rows("E", e_tasks, e_td, e_tiers,
                     lambda t: "HumanEval+" if "/HumanEval/" in t["task_id"] else "MBPP+",
                     {}, {}, None, notes)

    for name, rows in (("dev_table.jsonl", dev), ("test_table.jsonl", test)):
        with open(f"{OUT}/{name}", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # integrity (counts only; no metric is computed on test)
    dev_ids = {r["task_id"] for r in dev}
    test_ids = {r["task_id"] for r in test}
    summary = {
        "dev_rows": len(dev), "test_rows": len(test),
        "dev_by_set_tier": dict(Counter(f"{r['set']}|{r['tier']}" for r in dev)),
        "test_by_bench_tier": dict(Counter(f"{r['benchmark']}|{r['tier']}" for r in test)),
        "task_id_overlap_dev_test": sorted(dev_ids & test_ids)[:20],
        "n_overlap": len(dev_ids & test_ids),
        "missing": dict(notes["missing"]), "no_gate": dict(notes["no_gate"]),
    }
    with open(f"{OUT}/build_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
