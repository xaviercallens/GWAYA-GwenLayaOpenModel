#!/usr/bin/env python3
"""A31 step 1: freeze the downstream accept policy (D3 `lr|core`, named in A30) on DEV ONLY.
Never opens test_table.jsonl. Fits one logistic regression on all dev rows (F python + E' MBPP, 3 tiers), calibrates with
isotonic regression on inner out-of-fold scores (5 folds grouped by cluster, same fold hash as lever_D_stacked_calibrated.py),
and picks thresholds for targets 0.80 / 0.85 / 0.90 by the same rule (largest coverage with Wilson LB >= target).
Writes a pickle and a thresholds JSON; both are committed before the sealed test is read."""
import hashlib, json, math, pickle, sys, warnings
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
D = "/mnt/data/home/xavkal/gwaya-data/trust_loop"
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "results/gwenlaya_v4/laya/frozen_policy_D"
COLS = ["gate", "mlp", "minlp", "logtok", "trunc", "nvis"]
TARGETS = [0.80, 0.85, 0.90]

def wil(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - a) / d, (c + a) / d]

def feats(r):
    return [int(r["gate"] == "verified"), r["mean_logprob"], max(r["min_logprob"], -20), math.log1p(r["completion_tokens"]),
            int(r["finish_reason"] == "length"), r["n_visible_tests"]]

def mk(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
def fold_of(c, k=5): return int(hashlib.md5(("0|" + c).encode()).hexdigest(), 16) % k

def pick_thr(sc, yy, target):
    o = np.argsort(-sc, kind="stable"); cs = np.cumsum(yy[o]); n = len(yy); best = None
    for i in range(int(math.ceil(0.05 * n)), n + 1):
        if wil(cs[i - 1], i)[0] >= target: best = i
    if best is None:
        for i in range(int(math.ceil(0.05 * n)), n + 1):
            if cs[i - 1] / i >= target: best = i
    if best is None: best = int(math.ceil(0.5 * n))
    return float(sc[o[best - 1]]), best / n

if __name__ == "__main__":
    R = [json.loads(l) for l in open(f"{D}/dev_table.jsonl")]
    X = np.array([feats(r) for r in R], float); y = np.array([r["correct"] for r in R]); clus = [r["cluster"] for r in R]
    fo = np.array([fold_of(c) for c in clus]); idx = np.arange(len(y)); raw = np.zeros(len(y))
    for j in range(5):
        tr, te = idx[fo != j], idx[fo == j]
        raw[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, y)
    cal = iso.predict(raw) + 1e-6 * raw
    model = mk().fit(X, y)
    thr = {}
    for t in TARGETS:
        th, cov = pick_thr(cal, y, t); thr[str(t)] = dict(threshold=th, dev_oof_coverage=cov)
    import os; os.makedirs(OUTDIR, exist_ok=True)
    pickle.dump(dict(model=model, iso=iso, cols=COLS), open(f"{OUTDIR}/policy.pkl", "wb"))
    meta = dict(policy="D3 lr|core, one model for all tiers", cols=COLS, n_dev_rows=len(y), n_clusters=len(set(clus)),
                dev_base_rate=float(y.mean()), thresholds=thr, score="iso.predict(p)+1e-6*p, accept if score >= threshold",
                note="fit on dev only; test_table.jsonl never opened by this script")
    json.dump(meta, open(f"{OUTDIR}/policy_thresholds.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))
