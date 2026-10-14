#!/usr/bin/env python3
"""A31 step 2: score the sealed E table ONCE with the frozen policy (scripts/loop/freeze_policy_D.py output).
Refuses to run if the output file already exists (read-once rule). Coder-1.5B is skipped: it has no gate on E."""
import json, math, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from freeze_policy_D import feats, wil
POL = "results/gwenlaya_v4/laya/frozen_policy_D"
TEST = "/mnt/data/home/xavkal/gwaya-data/trust_loop/test_table.jsonl"
OUT = "results/gwenlaya_v4/laya/sealed_E_D3_lr_core.json"
if os.path.exists(OUT): sys.exit("sealed test already scored: " + OUT)
P = pickle.load(open(f"{POL}/policy.pkl", "rb")); meta = json.load(open(f"{POL}/policy_thresholds.json"))
R = [json.loads(l) for l in open(TEST)]
R = [r for r in R if "coder" not in r["tier"]]
assert all(r["mean_logprob"] is not None and r["min_logprob"] is not None for r in R), "null logprob"
X = np.array([feats(r) for r in R], float); y = np.array([r["correct"] for r in R]).astype(int)
raw = P["model"].predict_proba(X)[:, 1]; sc = P["iso"].predict(raw) + 1e-6 * raw
tiers = np.array([r["tier"] for r in R]); bench = np.array([r["benchmark"] for r in R])
def summ(m):
    k = int(y[m].sum()); n = int(m.sum()); return dict(n_accept=n, coverage=n / len(m) if False else None, precision=k / n if n else None, wilson=wil(k, n))
def cut(mask):
    yy, ss = y[mask], sc[mask]; N = len(yy); out = dict(n=N, base_rate=float(yy.mean()))
    g = np.array([r["gate"] == "verified" for r in R])[mask]; kg = int(yy[g].sum()); ng = int(g.sum())
    out["gate_only"] = dict(n_accept=ng, coverage=ng / N, precision=kg / ng if ng else None, wilson=wil(kg, ng))
    out["at_frozen_threshold"] = {}
    for t, v in meta["thresholds"].items():
        a = ss >= v["threshold"]; k = int(yy[a].sum()); n = int(a.sum())
        out["at_frozen_threshold"][t] = dict(n_accept=n, coverage=n / N, precision=k / n if n else None, wilson=wil(k, n))
    o = np.argsort(-ss, kind="stable"); cs = np.cumsum(yy[o]); curve = {}
    for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        n = int(math.ceil(c * N)); curve[str(c)] = dict(n_accept=n, precision=float(cs[n - 1] / n), wilson=wil(int(cs[n - 1]), n))
    out["rank_curve"] = curve
    best = None
    for i in range(20, N + 1):
        if cs[i - 1] / i >= 0.84 and wil(int(cs[i - 1]), i)[0] >= 0.80: best = dict(n=i, coverage=i / N, precision=float(cs[i - 1] / i), wilson=wil(int(cs[i - 1]), i))
    out["A30_rank_curve_largest"] = best
    return out
res = dict(policy=meta, n_rows=len(R))
for t in sorted(set(tiers)):
    res[t] = {"ALL": cut(tiers == t), **{b: cut((tiers == t) & (bench == b)) for b in sorted(set(bench))}}
json.dump(res, open(OUT, "w"), indent=1)
for t in sorted(set(tiers)):
    for b, v in res[t].items():
        f = v["at_frozen_threshold"]["0.85"]; r5 = v["rank_curve"]["0.5"]
        print("   gate-only cov %.2f prec %s" % (v["gate_only"]["coverage"], v["gate_only"]["precision"]))
        print("%-22s %-10s n %d base %.3f | thr0.85 cov %.2f prec %s %s | cov0.5 prec %.3f %s | A30 rank %s" % (
            t, b, v["n"], v["base_rate"], f["coverage"], None if f["precision"] is None else round(f["precision"], 3),
            None if f["wilson"][0] is None else [round(x, 3) for x in f["wilson"]], r5["precision"], [round(x, 3) for x in r5["wilson"]],
            None if v["A30_rank_curve_largest"] is None else (round(v["A30_rank_curve_largest"]["coverage"], 2), round(v["A30_rank_curve_largest"]["precision"], 3))))
