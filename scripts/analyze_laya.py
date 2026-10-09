#!/usr/bin/env python
"""A13/A13b: frozen-Laya evaluation on the fresh set E' (docs/ANALYSIS_PLAN_E_TPU.md).

Inputs are fixed by the plan: raw Laya scores for the calibration split C and for E' (train_laya.py --predict), the
labelled calibrator rows of C, E' and the training set (scripts/data/build_laya_data.py), and for the router policy the
scored tier arrays (accuracy, gate verdict, chip-second cost incl. math programs).

  1. isotonic map raw score -> P(correct), fitted on C only (pooled over tiers/domains), saved for the freeze file;
  2. confirmatory family (Holm over 2) on the primary tier: H1 = AURC of Laya vs raw mean log-prob (B3);
     H3 = selective risk at the gate's own coverage, Laya vs gate alone (B5);
  3. exploratory: Laya vs a logistic regression on hand features trained on the same rows; AUROC/ECE/Brier for all;
  4. router policy: tau chosen on C, evaluated on E' vs the static ladder and always-largest.
Bootstrap: cluster-stratified, 10,000 resamples, seed 0 (scripts/analyze_study.cluster_boot).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gwaya.metrics_selective import holm  # noqa: E402
from scripts import analyze_cascades as AC  # noqa: E402
from scripts import analyze_study as A  # noqa: E402

TAU_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
GATES = ("verified", "failed", "unverified")


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def raw_scores(preds: list[dict], col: int = 0) -> dict[tuple[str, str, str | None], float]:
    return {(p["key"][0], p["key"][1], p.get("tier")): p["probs"][col] for p in preds}


def fit_isotonic(raw: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    return {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}


def apply_isotonic(m: dict[str, list[float]], raw: np.ndarray) -> np.ndarray:
    return np.interp(raw, m["x"], m["y"])


def hand_features(rows: list[dict], tiers: list[str]) -> np.ndarray:
    out = []
    for r in rows:
        s = r.get("signals") or {}
        g = str(s.get("gate", "none")).lower()
        mlp = s.get("mean_logprob")
        mn = s.get("min_logprob")
        out.append([float(g == v) for v in GATES] + [float(mlp if mlp is not None else 0.0), float(max(mn, -20.0) if mn is not None else 0.0),
                    float(r["domain"] == "math")] + [float(r.get("tier") == t) for t in tiers[1:]])
    return np.asarray(out, dtype=float)


def fit_lr(train_rows: list[dict], tiers: list[str]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = hand_features(train_rows, tiers)
    y = np.array([r["correct"] for r in train_rows], dtype=int)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
    return lambda rows: lr.predict_proba(sc.transform(hand_features(rows, tiers)))[:, 1]


def topk_risk(score: np.ndarray, y: np.ndarray, k: int) -> float:
    """Risk of the k highest-scored answers; ties broken pessimistically (wrong first), as aurc."""
    if k <= 0:
        return float("nan")
    order = np.lexsort((y.astype(int), -score))[:k]
    return float((~y[order].astype(bool)).mean())


def metrics_block(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"auroc": A.auroc(p, y), "aurc": A.aurc(p, y), "ece15": A.ece_equal_mass(p, y), "brier": A.brier(p, y)}


def confirmatory(domain: np.ndarray, cluster: np.ndarray, y: np.ndarray, laya: np.ndarray, mlp: np.ndarray,
                 gate_ok: np.ndarray, lr: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    def stats(idx):
        yi, li, mi, gi, ri = y[idx], laya[idx], mlp[idx], gate_ok[idx], lr[idx]
        k = int(gi.sum())
        gate_risk = float((~yi[gi.astype(bool)].astype(bool)).mean()) if k else float("nan")
        return (A.aurc(li, yi) - A.aurc(mi, yi),                       # H1: lower AURC is better
                topk_risk(li, yi, k) - gate_risk,                      # H3: lower selective risk at gate coverage is better
                A.auroc(li, yi) - A.auroc(ri, yi), A.aurc(li, yi) - A.aurc(ri, yi))   # exploratory vs LR
    full = np.arange(len(y))
    point = stats(full)
    boot = A.cluster_boot(stats, domain, cluster, n_boot=n_boot, seed=seed)
    names = ["H1_dAURC_laya_minus_logprob", "H3_dRisk_at_gate_coverage_laya_minus_gate",
             "X_dAUROC_laya_minus_hand_lr", "X_dAURC_laya_minus_hand_lr"]
    res = {n: A.pct_ci(point[i], boot[:, i]) for i, n in enumerate(names)}
    # one-sided p in Laya's favour (difference < 0 for H1/H3), with +1 smoothing
    ps = []
    for i in (0, 1):
        v = boot[np.isfinite(boot[:, i]), i]
        ps.append((int((v >= 0).sum()) + 1) / (len(v) + 1))
    adj = holm(ps)
    res["H1_p_one_sided"], res["H3_p_one_sided"] = ps
    res["H1_p_holm"], res["H3_p_holm"] = adj
    res["H1_supported"] = bool(res[names[0]]["hi"] < 0 and adj[0] < 0.05)
    res["H3_supported"] = bool(res[names[1]]["hi"] < 0 and adj[1] < 0.05)
    res["gate_coverage"] = float(gate_ok.mean())
    return res


def run_policy(tau: float, arrays: list[dict[str, np.ndarray]], router_p: np.ndarray) -> dict[str, np.ndarray]:
    """Start at the cheapest tier whose router P >= tau (else the last), then cascade upward on the gate."""
    n, T = router_p.shape
    start = np.full(n, T - 1)
    for t in range(T - 1, -1, -1):
        start = np.where(router_p[:, t] >= tau, t, start)
    cor = np.zeros(n, bool); ans = np.zeros(n, bool); cost = np.zeros(n); done = np.zeros(n, bool)
    for t, a in enumerate(arrays):
        active = (~done) & (start <= t)
        cost += np.where(active, a["cost"] + a["pot"], 0.0)
        ok = active & a["gate_ok"]
        cor = np.where(ok, a["cor"], cor); ans |= ok; done |= ok
        if t == len(arrays) - 1:
            last = active & ~ok
            cor = np.where(last, a["cor"], cor)
    return {"cor": cor, "answered": ans, "cost": cost}


def choose_tau(arrays_c, router_c: np.ndarray, grid=TAU_GRID) -> tuple[float, dict]:
    base = AC.offline_cascade(arrays_c)
    target = base["cor"].mean()
    rows = {}
    for tau in grid:
        o = run_policy(tau, arrays_c, router_c)
        rows[tau] = {"acc": float(o["cor"].mean()), "cost": float(o["cost"].mean())}
    ok = [t for t in grid if rows[t]["acc"] >= target]
    tau = min(ok, key=lambda t: rows[t]["cost"]) if ok else max(grid)
    return tau, {"static_ladder_acc_C": float(target), "grid": {str(k): v for k, v in rows.items()}, "chosen": tau,
                 "any_meets_target": bool(ok)}


def router_eval(domain, cluster, arrays_e, router_e, tau, n_boot, seed) -> dict[str, Any]:
    pol = run_policy(tau, arrays_e, router_e)
    lad = AC.offline_cascade(arrays_e)
    big = arrays_e[-1]

    def stats(idx):
        out = []
        for c, a, k in ((pol["cor"], pol["answered"], pol["cost"]), (lad["cor"], lad["answered"], lad["cost"]),
                        (big["cor"], np.ones(len(big["cor"]), bool), big["cost"] + big["pot"])):
            cc = c[idx]; aa = a[idx]
            out += [cc.mean(), float((aa & ~cc).mean()), k[idx].mean()]
        return tuple(out) + (out[0] - out[3], out[2] / out[5], out[2] / out[8], out[0] - out[6])
    full = np.arange(len(domain))
    point = stats(full)
    boot = A.cluster_boot(stats, domain, cluster, n_boot=n_boot, seed=seed)
    names = ["router_acc", "router_cwr", "router_cost", "ladder_acc", "ladder_cwr", "ladder_cost", "large_acc", "large_cwr",
             "large_cost", "d_acc_router_minus_ladder", "cost_ratio_router_over_ladder", "cost_ratio_router_over_large",
             "d_acc_router_minus_large"]
    return {n: A.pct_ci(point[i], boot[:, i]) for i, n in enumerate(names)} | {"tau": tau}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tiers", nargs=3, required=True, help="served names, cheapest first")
    ap.add_argument("--primary", required=True, help="served name of the primary tier (the largest)")
    ap.add_argument("--train-cal-rows", required=True, type=Path)
    ap.add_argument("--c-cal-rows", required=True, type=Path)
    ap.add_argument("--e-cal-rows", required=True, type=Path)
    ap.add_argument("--c-cal-preds", required=True, type=Path)
    ap.add_argument("--e-cal-preds", required=True, type=Path)
    ap.add_argument("--c-router-preds", type=Path)
    ap.add_argument("--e-router-preds", type=Path)
    ap.add_argument("--c-tasks", required=True, type=Path)
    ap.add_argument("--e-tasks", required=True, type=Path)
    ap.add_argument("--c-study", nargs="+", type=Path)
    ap.add_argument("--e-study", nargs="+", type=Path)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-out", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    tiers = list(a.tiers)

    c_rows, e_rows, tr_rows = read_jsonl(a.c_cal_rows), read_jsonl(a.e_cal_rows), read_jsonl(a.train_cal_rows)
    cluster_of = {(t["domain"], t["task_id"]): t["cluster"] for f in (a.c_tasks, a.e_tasks) for t in read_jsonl(f)}

    def join(rows, preds_path):
        sc = raw_scores(read_jsonl(preds_path))
        return np.array([sc[(r["source"], r["id"], r["tier"])] for r in rows])

    raw_c, raw_e = join(c_rows, a.c_cal_preds), join(e_rows, a.e_cal_preds)
    y_c = np.array([r["correct"] for r in c_rows])
    iso = fit_isotonic(raw_c, y_c)
    a.map_out.parent.mkdir(parents=True, exist_ok=True)
    a.map_out.write_text(json.dumps({"isotonic_fitted_on": "C", "n": len(c_rows), **iso}, indent=1))
    laya_e = apply_isotonic(iso, raw_e)
    lr_fn = fit_lr(tr_rows, tiers)
    lr_e = lr_fn(e_rows)
    res: dict[str, Any] = {"meta": {"plan": "A13/A13b", "tiers": tiers, "primary": a.primary, "n_boot": a.n_boot, "seed": a.seed,
                                    "n_C_rows": len(c_rows), "n_Eprime_rows": len(e_rows), "n_train_rows": len(tr_rows)},
                           "per_tier": {}}
    for tier in tiers:
        idx = [i for i, r in enumerate(e_rows) if r["tier"] == tier]
        sub = [e_rows[i] for i in idx]
        y = np.array([r["correct"] for r in sub], bool)
        dom = np.array([r["domain"] for r in sub])
        clu = np.array([cluster_of[(r["domain"], f'{r["source"]}/{r["id"]}')] for r in sub])
        mlp = np.array([(r["signals"].get("mean_logprob") if r["signals"].get("mean_logprob") is not None else -20.0) for r in sub])
        gate = np.array([str(r["signals"].get("gate")).lower() == "verified" for r in sub])
        block = {"n": len(sub), "accuracy": float(y.mean()), "gate_coverage": float(gate.mean()),
                 "metrics": {"laya": metrics_block(laya_e[idx], y), "raw_logprob": metrics_block(np.exp(np.minimum(mlp, 0.0)), y),  # exp(mean log-prob) is the probability-like score; ECE/Brier need [0,1]
                             "gate": metrics_block(gate.astype(float), y), "hand_lr": metrics_block(lr_e[idx], y)}}
        block["inference"] = confirmatory(dom, clu, y, laya_e[idx], mlp, gate, lr_e[idx], a.n_boot, a.seed)
        res["per_tier"][tier] = block
    res["primary_inference"] = res["per_tier"][a.primary]["inference"]
    if a.c_router_preds and a.e_router_preds and a.c_study and a.e_study:
        rows_c = [r for d in a.c_study for r in read_jsonl(d / "rows.jsonl")]; gens_c = [g for d in a.c_study for g in read_jsonl(d / "gens.jsonl")]
        rows_e = [r for d in a.e_study for r in read_jsonl(d / "rows.jsonl")]; gens_e = [g for d in a.e_study for g in read_jsonl(d / "gens.jsonl")]
        def prep(tasks_path, rows, gens, preds_path):
            tasks = sorted(read_jsonl(tasks_path), key=lambda t: (t["domain"], t["task_id"]))
            keys = [(t["domain"], t["task_id"]) for t in tasks]
            arrays = [AC.tier_arrays(m, keys, rows, gens) for m in tiers]
            pr = {(p["key"][0], p["key"][1]): p["probs"] for p in read_jsonl(preds_path)}
            rp = np.array([pr[(t["task_id"].partition("/")[0], t["task_id"].partition("/")[2])] for t in tasks])
            return tasks, keys, arrays, rp
        tc, kc, arr_c, rp_c = prep(a.c_tasks, rows_c, gens_c, a.c_router_preds)
        te, ke, arr_e, rp_e = prep(a.e_tasks, rows_e, gens_e, a.e_router_preds)
        tau, info = choose_tau(arr_c, rp_c)
        dom = np.array([t["domain"] for t in te]); clu = np.array([t["cluster"] for t in te])
        res["router"] = {"tau_selection_on_C": info, "E_prime": router_eval(dom, clu, arr_e, rp_e, tau, a.n_boot, a.seed)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else o))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
