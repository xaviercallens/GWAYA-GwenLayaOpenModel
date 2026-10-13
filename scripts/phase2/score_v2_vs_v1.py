#!/usr/bin/env python3
"""A28 comparison: frozen Laya v1 vs feedback-retrofitted v2 on the held-out F fold 0 (calibrator head). Local, no spend.
Paired bootstrap resamples source problems (all tiers of a task together), 2,000 resamples, seed 0."""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score

ROWS, V1, V2, OUT = sys.argv[1:5]
rows = {(tuple(r["key"]) if False else (r["source"], r["id"], r["tier"])): r for r in map(json.loads, open(ROWS))}
def preds(p): return {(k["key"][0], k["key"][1], k["tier"]): k["probs"][0] for k in map(json.loads, open(p))}
p1, p2 = preds(V1), preds(V2)
keys = [k for k in rows if k in p1 and k in p2]
y = np.array([int(rows[k]["correct"]) for k in keys]); dom = np.array([rows[k]["domain"] for k in keys])
s1 = np.array([p1[k] for k in keys]); s2 = np.array([p2[k] for k in keys])
gv = np.array([(rows[k].get("signals") or {}).get("gate") == "verified" for k in keys])
tasks = np.array([k[0] + "|" + k[1] for k in keys]); ut = np.unique(tasks)
def ece(p, yy, b=15):
    o = np.argsort(p); n = len(p); e = 0
    for c in np.array_split(o, b):
        if len(c): e += len(c) / n * abs(p[c].mean() - yy[c].mean())
    return e
def metrics(idx, s):
    out = {}
    for d in ("python", "math", "all"):
        m = idx if d == "all" else idx[dom[idx] == d]
        yy, ss = y[m], s[m]
        out[d + "_auroc"] = roc_auc_score(yy, ss) if 0 < yy.sum() < len(yy) else np.nan
        out[d + "_brier"] = float(np.mean((ss - yy) ** 2)); out[d + "_ece"] = ece(ss, yy)
    bad = idx[gv[idx] & (y[idx] == 0)]
    out["cw_catch"] = float(np.mean(s[bad] < np.median(s[idx]))) if len(bad) else np.nan
    out["n_cw"] = int(len(bad))
    return out
allidx = np.arange(len(y)); m1, m2 = metrics(allidx, s1), metrics(allidx, s2)
rng = np.random.default_rng(0); by = {t: np.where(tasks == t)[0] for t in ut}
diffs = {k: [] for k in m1}
for _ in range(2000):
    idx = np.concatenate([by[t] for t in rng.choice(ut, len(ut))])
    a, b = metrics(idx, s1), metrics(idx, s2)
    for k in m1: diffs[k].append(b[k] - a[k])
res = {"n_rows": len(y), "n_tasks": int(len(ut)), "v1": m1, "v2": m2,
       "delta_v2_minus_v1": {k: {"point": m2[k] - m1[k], "ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
                             for k, v in diffs.items() if k != "n_cw"}}
json.dump(res, open(OUT, "w"), indent=1, default=float)
for k, v in res["delta_v2_minus_v1"].items(): print("%-14s v1 %.3f v2 %.3f  d %+.3f [%+.3f,%+.3f]" % (k, m1[k], m2[k], v["point"], *v["ci95"]))
print("n", len(y), "tasks", len(ut), "confident-wrong", m1["n_cw"])
