#!/usr/bin/env python3
"""A33 Verifier V2: candidate features from executed synthesized tests, D4 selector, fixed-sequence certificate.

  verifier_v2.py dev   --tasks F.jsonl --pool pool.jsonl --scored scored.jsonl --exec exec.jsonl --tiers a,b --out dev.json
  verifier_v2.py sealed --dev-tasks F.jsonl --tasks G.jsonl ... --out sealed.json      (final stage only, reads G once)

Rule (A33, fixed before any phase-3 number): per tier, 5 folds grouped by cluster (hash as A26). In fold j the scorer
is fit on three folds, the calibration fold (j+1)%5 certifies a threshold by fixed-sequence Learn-then-Test (risk 0.10,
delta 0.10, thresholds on the calibrated probability {0.95, 0.925, 0.90, 0.875, 0.85} in that order, exact binomial,
stop at the first failure), and the test fold j is accepted at the certified threshold. V2 = candidate-level logistic
regression over the pool features, task score = max over candidates, selected candidate = argmax. B0 = D3 `lr|core`
on the greedy candidate only (gate, mean log-prob, min log-prob, log tokens, truncation, visible-test count).
Lever (rule a): mean certified test coverage of V2 exceeds B0's by >= 0.10 AND the paired cluster bootstrap CI of
the difference in correct accepted tasks at coverage 0.40 (ranked by out-of-fold score) is above zero.
Hidden outcomes are labels only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binom
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THRESHOLDS = [0.95, 0.925, 0.90, 0.875, 0.85]
ALPHA, DELTA = 0.10, 0.10
V2_COLS = ["gate", "mlp", "minlp", "logtok", "trunc", "frac_exp", "cl_share", "n_cl", "exc_frac", "is_greedy",
           "top_cluster", "agree_ref"]
B0_COLS = ["gate", "mlp", "minlp", "logtok", "trunc", "nvis"]


def wil(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - a) / d, (c + a) / d]


def fold_of(c, k=5):
    return int(hashlib.md5(("0|" + c).encode()).hexdigest(), 16) % k


def load(a):
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(a.tasks)}
    pool = defaultdict(dict)
    for l in open(a.pool):
        r = json.loads(l)
        pool[(r["tier"], r["task_id"])][str(r["cand"])] = r
    for l in open(a.scored):
        r = json.loads(l)
        c = pool.get((r["tier"], r["task_id"]), {}).get(str(r["cand"]))
        if c is not None:
            c["ok"] = int(r["hidden"] == "VERIFIED")
            c["gate_v"] = int(str(r["gate"]).upper() == "VERIFIED")
    for l in open(a.exec):
        r = json.loads(l)
        c = pool.get((r["tier"], r["task_id"]), {}).get(str(r["cand"]))
        if c is not None:
            c["out"], c["exp"], c["n_calls"] = r.get("out") or [], r.get("exp") or [], r.get("n_calls", 0)
    return tasks, pool


def features(tasks, pool, tiers, ref_tier=None):
    """Per (tier, task): list of candidate dicts with features. Label-free except `ok`."""
    groups = {}
    ref_mode = {}
    if ref_tier:
        for (tier, tid), cands in pool.items():
            if tier == ref_tier:
                vecs = Counter(json.dumps(c.get("out", [])) for c in cands.values() if c.get("out"))
                if vecs:
                    ref_mode[tid] = vecs.most_common(1)[0][0]
    for (tier, tid), cands in pool.items():
        if tier not in tiers or tid not in tasks:
            continue
        cs = [c for c in cands.values() if "ok" in c]
        if not cs or "g" not in cands or "ok" not in cands["g"]:
            continue
        vecs = Counter(json.dumps(c.get("out", [])) for c in cs)
        top = vecs.most_common(1)[0][0]
        nvis = tasks[tid].get("n_gate_assertions") or 1
        rows = []
        for c in cs:
            out, exp = c.get("out", []), c.get("exp", [])
            n = max(len(out), 1)
            v = json.dumps(out)
            lp = c.get("mean_logprob")
            rows.append({
                "cand": str(c["cand"]), "ok": c["ok"], "gate": c.get("gate_v", 0),
                "mlp": lp if lp is not None else -9.0, "minlp": max(c.get("min_logprob") or -20.0, -20.0),
                "logtok": math.log1p(c.get("completion_tokens") or 0), "trunc": int(c.get("finish") == "length"),
                "frac_exp": sum(1 for o, e in zip(out, exp) if o == e and not o.startswith("EXC")) / n if out else 0.0,
                "cl_share": vecs[v] / len(cs), "n_cl": len(vecs) / len(cs),
                "exc_frac": sum(1 for o in out if o.startswith("EXC")) / n if out else 1.0,
                "is_greedy": int(str(c["cand"]) == "g"), "top_cluster": int(v == top),
                "agree_ref": int(ref_mode.get(tid) == v) if ref_tier and ref_tier != tier else 0,
                "nvis": nvis})
        groups[(tier, tid)] = {"cluster": tasks[tid].get("cluster", tid), "cands": rows}
    return groups


def mk():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))


def task_scores(model, keys, groups, cols, greedy_only):
    sc, yy, sel = [], [], []
    for k in keys:
        cs = [c for c in groups[k]["cands"] if (c["cand"] == "g" or not greedy_only)]
        X = np.array([[c[x] for x in cols] for c in cs], float)
        p = model.predict_proba(X)[:, 1]
        i = int(np.argmax(p))
        sc.append(p[i]); yy.append(cs[i]["ok"]); sel.append(cs[i]["cand"])
    return np.array(sc), np.array(yy), sel


def fit(keys, groups, cols, greedy_only):
    X, y = [], []
    for k in keys:
        for c in groups[k]["cands"]:
            if c["cand"] == "g" or not greedy_only:
                X.append([c[x] for x in cols]); y.append(c["ok"])
    return mk().fit(np.array(X, float), np.array(y))


def certify(cal_p, cal_y):
    """Fixed-sequence LTT over THRESHOLDS on calibrated probabilities; returns the last certified threshold or None."""
    last = None
    for t in THRESHOLDS:
        acc = cal_p >= t
        n, e = int(acc.sum()), int((cal_y[acc] == 0).sum())
        if n == 0 or binom.cdf(e, n, ALPHA) > DELTA:
            break
        last = t
    return last


def run_cv(groups, tier, cols, greedy_only, K=5):
    keys = sorted(k for k in groups if k[0] == tier)
    fo = {k: fold_of(groups[k]["cluster"], K) for k in keys}
    oof_s, oof_y, res = {}, {}, []
    for j in range(K):
        cal_f = (j + 1) % K
        tr = [k for k in keys if fo[k] not in (j, cal_f)]
        ca = [k for k in keys if fo[k] == cal_f]
        te = [k for k in keys if fo[k] == j]
        m = fit(tr, groups, cols, greedy_only)
        cs, cy, _ = task_scores(m, ca, groups, cols, greedy_only)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cs, cy)
        thr = certify(iso.predict(cs), cy)
        ts, ty, _ = task_scores(m, te, groups, cols, greedy_only)
        tp = iso.predict(ts)
        acc = tp >= thr if thr is not None else np.zeros(len(ty), bool)
        res.append({"fold": j, "threshold": thr, "n_test": len(ty), "n_accept": int(acc.sum()),
                    "coverage": float(acc.mean()), "correct": int(ty[acc].sum())})
        for k, s_, y_ in zip(te, ts, ty):
            oof_s[k], oof_y[k] = s_, y_
    n_acc, n_cor = sum(r["n_accept"] for r in res), sum(r["correct"] for r in res)
    return {"folds": res, "mean_cert_coverage": float(np.mean([r["coverage"] for r in res])),
            "pooled_accept": n_acc, "pooled_precision": n_cor / n_acc if n_acc else None,
            "pooled_wilson": wil(n_cor, n_acc), "oof_score": {"|".join(k): float(v) for k, v in oof_s.items()},
            "oof_label": {"|".join(k): int(v) for k, v in oof_y.items()}}


def correct_at(scores, labels, keys, cov=0.40):
    o = sorted(keys, key=lambda k: -scores[k])
    n = int(math.ceil(cov * len(o)))
    return sum(labels[k] for k in o[:n])


def dev(a):
    tasks, pool = load(a)
    tiers = a.tiers.split(",")
    groups = features(tasks, pool, tiers, a.ref_tier)
    rng = np.random.default_rng(0)
    out = {"rule": __doc__.split("Rule (A33")[1].split("Hidden outcomes")[0].strip(), "tiers": {}}
    for tier in tiers:
        keys = [k for k in groups if k[0] == tier]
        if not keys:
            continue
        v2 = run_cv(groups, tier, V2_COLS, greedy_only=False)
        b0 = run_cv(groups, tier, B0_COLS, greedy_only=True)
        sk = ["|".join(k) for k in keys]
        clus = defaultdict(list)
        for k in keys:
            clus[groups[k]["cluster"]].append("|".join(k))
        cl = list(clus)
        d = []
        for _ in range(1000):
            pick = [kk for c in rng.choice(cl, len(cl)) for kk in clus[c]]
            d.append(correct_at(v2["oof_score"], v2["oof_label"], pick) - correct_at(b0["oof_score"], b0["oof_label"], pick))
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        oracle = float(np.mean([max(c["ok"] for c in groups[k]["cands"]) for k in keys]))
        greedy = float(np.mean([next(c["ok"] for c in groups[k]["cands"] if c["cand"] == "g") for k in keys]))
        gain = v2["mean_cert_coverage"] - b0["mean_cert_coverage"]
        out["tiers"][tier] = {
            "n_tasks": len(keys), "greedy_pass": greedy, "oracle_any": oracle,
            "V2": {k: v for k, v in v2.items() if not k.startswith("oof")},
            "B0": {k: v for k, v in b0.items() if not k.startswith("oof")},
            "paired_correct_diff_cov40": {"mean": float(np.mean(d)), "ci": [lo, hi],
                                          "v2": correct_at(v2["oof_score"], v2["oof_label"], sk),
                                          "b0": correct_at(b0["oof_score"], b0["oof_label"], sk)},
            "rule_a": {"coverage_gain": gain, "gain_ok": gain >= 0.10, "ci_ok": lo > 0, "lever": bool(gain >= 0.10 and lo > 0)}}
        t = out["tiers"][tier]
        print(f"{tier}: n {len(keys)} greedy {greedy:.3f} oracle {oracle:.3f} | V2 cert cov {v2['mean_cert_coverage']:.3f} "
              f"prec {v2['pooled_precision']} | B0 cert cov {b0['mean_cert_coverage']:.3f} prec {b0['pooled_precision']} | "
              f"dCorrect@0.40 {np.mean(d):.1f} [{lo:.0f},{hi:.0f}] lever {t['rule_a']['lever']}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev"])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--exec", required=True)
    ap.add_argument("--tiers", required=True)
    ap.add_argument("--ref-tier", default=None, help="tier whose modal output vector defines agree_ref (e.g. the 35B)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev(a)


if __name__ == "__main__":
    main()
