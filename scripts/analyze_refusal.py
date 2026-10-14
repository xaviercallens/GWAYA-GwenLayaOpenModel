#!/usr/bin/env python3
"""Refusal / selective-accuracy analysis: does a score let the system decline tasks it would get wrong?

Descriptive (no hypothesis test). Input: rows with `correct` and `signals`, plus Laya predictions {key, tier, probs}.
For each tier and each scorer (laya P(correct), raw mean log-prob, gate-then-log-prob), answer the top-c share of tasks by
score and refuse the rest. Reported per coverage c: accuracy of the answered set, refusal precision (share of refused
tasks the tier really gets wrong) and recall of "too hard" tasks (wrong for EVERY tier in the file), each with a
task-level bootstrap 95% CI (10,000 resamples, numpy default_rng(0)). Python-only by default.

  analyze_refusal.py --rows calibrator_F.jsonl --preds preds_F.jsonl --domain python --out refusal_F.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COVERAGES = (1.0, 0.8, 0.6, 0.4)


def load(rows_path, preds_path, domain):
    probs = {}
    for l in Path(preds_path).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            probs[(tuple(r["key"]), r["tier"])] = float(r["probs"][0])
    out = {}
    for l in Path(rows_path).read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if domain and r.get("domain") != domain:
            continue
        sig = r.get("signals") or {}
        key = (r["source"], r["id"])
        out.setdefault(r["tier"], {})[key] = {
            "correct": int(str(r["correct"]) in ("1", "True", "true")),
            "laya": probs.get((key, r["tier"])),
            "logprob": sig.get("mean_logprob"),
            "gate": 1.0 if sig.get("gate") == "verified" else 0.0}
    return out


def scorers(d):
    lp = {k: v["logprob"] for k, v in d.items()}
    return {"laya": {k: v["laya"] for k, v in d.items()},
            "mean_logprob": lp,
            # gate verdict first, log-prob breaks ties (a conservative hand rule, not Laya)
            "gate_then_logprob": {k: v["gate"] * 1e3 + (lp[k] if lp[k] is not None else float("-inf"))
                                  for k, v in d.items()}}


def curve(keys, score, correct, hard, c):
    order = sorted(keys, key=lambda k: (-score[k], str(k)))
    n_ans = max(1, int(round(c * len(order))))
    ans, ref = order[:n_ans], order[n_ans:]
    acc = float(np.mean([correct[k] for k in ans]))
    prec = float(np.mean([1 - correct[k] for k in ref])) if ref else None
    rec = (sum(hard[k] for k in ref) / max(1, sum(hard.values()))) if hard else None
    return acc, prec, rec, ans, ref


def boot_ci(keys, fn, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(keys))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        v = fn([keys[i] for i in s])
        if v is not None and not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def analyse(data, n_boot=2000):
    tiers = sorted(data)
    common = set.intersection(*(set(data[t]) for t in tiers))
    keys = sorted(common)
    hard = {k: int(all(data[t][k]["correct"] == 0 for t in tiers)) for k in keys}
    res = {"n_tasks": len(keys), "tiers": tiers, "n_hard_all_tiers_wrong": sum(hard.values()), "by_tier": {}}
    for t in tiers:
        d = {k: data[t][k] for k in keys}
        correct = {k: d[k]["correct"] for k in keys}
        res["by_tier"][t] = {"base_accuracy": float(np.mean(list(correct.values()))), "scorers": {}}
        for name, sc in scorers(d).items():
            if any(v is None for v in sc.values()):
                continue
            rows = {}
            for c in COVERAGES:
                acc, prec, rec, _, _ = curve(keys, sc, correct, hard, c)

                def f(sample, c=c, sc=sc, correct=correct):
                    ss = {k: sc[k] for k in set(sample)}
                    order = sorted(sample, key=lambda k: -sc[k])
                    m = max(1, int(round(c * len(order))))
                    return float(np.mean([correct[k] for k in order[:m]]))
                rows[str(c)] = {"accuracy_answered": acc, "acc_ci95": boot_ci(keys, f, n_boot),
                                "refusal_precision": prec, "hard_recall": rec}
            res["by_tier"][t]["scorers"][name] = rows
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--domain", default="python")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    res = analyse(load(a.rows, a.preds, a.domain), a.n_boot)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for t, v in res["by_tier"].items():
        print(t, f"base={v['base_accuracy']:.3f}", {n: {c: round(x['accuracy_answered'], 3) for c, x in r.items()}
                                                  for n, r in v["scorers"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
