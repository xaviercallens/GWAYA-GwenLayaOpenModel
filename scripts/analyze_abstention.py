#!/usr/bin/env python
"""A11: cross-fitted calibrated abstention (docs/ANALYSIS_PLAN_E_TPU.md, addendum A11).

Predicts P(correct) of a tier's answer from stored outputs only (gate verdict, token log-prob statistics, length,
truncation, domain) with out-of-fold logistic regression, then evaluates (a) discrimination/calibration and (b)
target-risk operating points whose thresholds are chosen on training data only. Exploratory: not the registered GL
arm, no Laya, no separate calibration split. Every choice (features, C, folds, thresholds) is fixed in the plan
before the first fit; nothing is tuned.

Inputs: tiers rows/gens (+ optional overlay rows, e.g. serial corrections and the math-gate replay), tasks (clusters).
The full gens.jsonl (with per-token log-probs) is needed; it lives on the data disk, not in git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_study as A  # noqa: E402

DOMAINS = ("python", "rust", "math")
GATE_VOCAB = ("VERIFIED", "FAILED", "UNVERIFIED")
N_FOLDS = 5
INNER_FOLDS = 4
MIN_ANSWERED = 30
ALPHAS = (0.05, 0.10)
FEATURES_NON_GATE = ("mean_lp", "min_lp", "low10_lp", "log_tokens", "truncated", "dom_python", "dom_rust", "dom_math")
FEATURES_ALL = (*FEATURES_NON_GATE, "gate_verified", "gate_failed", "gate_unverified")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ── pure helpers (tested) ─────────────────────────────────────────────────────────────────────

def fold_of(cluster: str, n_folds: int = N_FOLDS) -> int:
    """Deterministic fold by source problem: int(sha256(cluster)[:8], 16) mod n_folds (no RNG)."""
    return int(hashlib.sha256(cluster.encode("utf-8")).hexdigest()[:8], 16) % n_folds


def logprob_features(token_lps: list[float] | None, completion_tokens: int, truncated: bool) -> dict[str, float]:
    lps = np.asarray([x for x in (token_lps or []) if x is not None], dtype=float)
    if len(lps) == 0:
        return {"mean_lp": float("nan"), "min_lp": float("nan"), "low10_lp": float("nan"),
                "log_tokens": math.log1p(max(completion_tokens, 0)), "truncated": float(truncated)}
    k = max(1, int(math.ceil(0.10 * len(lps))))
    return {"mean_lp": float(lps.mean()), "min_lp": float(lps.min()), "low10_lp": float(np.sort(lps)[:k].mean()),
            "log_tokens": math.log1p(max(completion_tokens, 0)), "truncated": float(truncated)}


def choose_threshold(p: np.ndarray, y: np.ndarray, alpha: float, min_n: int = MIN_ANSWERED) -> float:
    """Largest-coverage threshold t (answer iff p >= t) whose selective risk P(wrong | answered) <= alpha on (p, y),
    answering at least min_n tasks; +inf (answer nothing) if no prefix qualifies. Ties are kept together."""
    if len(p) == 0:
        return float("inf")
    order = np.argsort(-p, kind="stable")
    ps, wrong = p[order], (~y.astype(bool))[order].astype(float)
    risk = np.cumsum(wrong) / np.arange(1, len(p) + 1)
    best = None
    for k in range(min_n, len(p) + 1):
        if k < len(p) and ps[k - 1] == ps[k]:
            continue  # do not split a tie group
        if risk[k - 1] <= alpha:
            best = k
    return float(ps[best - 1]) if best is not None else float("inf")


def selective_stats(answered: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    """(coverage, selective risk P(wrong | answered)); risk is NaN when nothing is answered."""
    n = len(answered)
    k = int(answered.sum())
    return (k / n if n else float("nan")), (float((answered & ~correct).sum() / k) if k else float("nan"))


# ── data ──────────────────────────────────────────────────────────────────────────────────────

class Table:
    """Per-task arrays for one tier, aligned on the sorted task list."""

    def __init__(self, tasks: list[dict], rows: list[dict], gens: list[dict], model: str) -> None:
        tasks = sorted(tasks, key=lambda t: (t["domain"], t["task_id"]))
        self.keys = [(t["domain"], t["task_id"]) for t in tasks]
        self.dom = np.array([k[0] for k in self.keys])
        self.clu = np.array([t.get("cluster") or f"{t['domain']}:{t['task_id']}" for t in tasks])
        base = {(r["domain"], r["task_id"]): r for r in rows if r["arm"] == "base" and r["model"] == model}
        gate = {(r["domain"], r["task_id"]): r for r in rows if r["arm"] == "gate_only" and r["model"] == model}
        gen = {(g["domain"], g["task_id"]): g for g in gens
               if g["model"] == model and g.get("temperature") == 0.0 and g.get("kind") != "pot"}
        miss = [k for k in self.keys if k not in base or k not in gate or k not in gen]
        if miss:
            raise SystemExit(f"{model}: {len(miss)} tasks without base/gate_only rows or a generation, e.g. {miss[0]}")
        self.y = np.array([base[k]["score"] == "VERIFIED" for k in self.keys])
        self.gate = np.array([gate[k]["gate"] for k in self.keys])
        feats = [logprob_features(gen[k].get("token_logprobs"), int(gen[k].get("completion_tokens") or 0),
                                  gen[k].get("done_reason") == "length") for k in self.keys]
        X = {name: np.array([f[name] for f in feats], dtype=float) for name in ("mean_lp", "min_lp", "low10_lp", "log_tokens", "truncated")}
        for d in DOMAINS:
            X[f"dom_{d}"] = (self.dom == d).astype(float)
        for g in GATE_VOCAB:
            X[f"gate_{g.lower()}"] = (self.gate == g).astype(float)
        self.X = X
        self.n = len(self.keys)
        self.fold = np.array([fold_of(c) for c in self.clu])
        for name in ("mean_lp", "min_lp", "low10_lp"):  # a generation without tokens: fill with the training-free neutral value
            v = self.X[name]
            self.X[name] = np.where(np.isnan(v), np.nanmin(v) if np.isfinite(np.nanmin(v)) else -10.0, v)

    def matrix(self, names: tuple[str, ...], idx: np.ndarray) -> np.ndarray:
        return np.column_stack([self.X[n][idx] for n in names])


def fit_predict(T: Table, names: tuple[str, ...], train: np.ndarray, test: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(T.matrix(names, train))
    # default penalty is l2; C is fixed at 1.0 by the plan
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(T.matrix(names, train)), T.y[train])
    return clf.predict_proba(sc.transform(T.matrix(names, test)))[:, 1]


def oof_predictions(T: Table, names: tuple[str, ...]) -> np.ndarray:
    p = np.zeros(T.n)
    for f in range(N_FOLDS):
        test = np.where(T.fold == f)[0]
        train = np.where(T.fold != f)[0]
        p[test] = fit_predict(T, names, train, test)
    return p


def target_risk_answers(T: Table, names: tuple[str, ...], alpha: float) -> np.ndarray:
    """Answered mask on every task: per outer fold, the threshold is chosen on inner cross-fitted predictions of the
    training portion only; the model is fitted on the whole training portion and applied to the held-out fold."""
    answered = np.zeros(T.n, dtype=bool)
    for f in range(N_FOLDS):
        test = np.where(T.fold == f)[0]
        train = np.where(T.fold != f)[0]
        inner_p = np.zeros(len(train))
        inner_fold = np.array([fold_of(c, INNER_FOLDS) for c in T.clu[train]])
        for g in range(INNER_FOLDS):
            te_in = np.where(inner_fold == g)[0]
            tr_in = np.where(inner_fold != g)[0]
            inner_p[te_in] = fit_predict(T, names, train[tr_in], train[te_in])
        thr = choose_threshold(inner_p, T.y[train], alpha)
        answered[test] = fit_predict(T, names, train, test) >= thr
    return answered


# ── analysis ──────────────────────────────────────────────────────────────────────────────────

def analyze(T: Table, n_boot: int, seed: int) -> dict[str, Any]:
    preds = {"M1": oof_predictions(T, ("mean_lp",)),
             "M2": oof_predictions(T, FEATURES_NON_GATE),
             "M3": oof_predictions(T, FEATURES_ALL),
             "M0": (T.gate == "VERIFIED").astype(float)}
    operating = {}
    for alpha in ALPHAS:
        operating[f"M1@{alpha}"] = target_risk_answers(T, ("mean_lp",), alpha)
        operating[f"M3@{alpha}"] = target_risk_answers(T, FEATURES_ALL, alpha)
    operating["gate"] = T.gate == "VERIFIED"
    is_dom = {d: T.dom == d for d in DOMAINS}

    specs: list[tuple[str, Callable[[np.ndarray], float]]] = []

    def add(name: str, fn: Callable[[np.ndarray], float]) -> None:
        specs.append((name, fn))

    def sel(i: np.ndarray, d: str) -> np.ndarray:
        return i if d == "pooled" else i[is_dom[d][i]]

    for d in (*DOMAINS, "pooled"):
        for m, p in preds.items():
            add(f"disc.auroc.{m}.{d}", lambda i, p=p, d=d: A.auroc(p[sel(i, d)], T.y[sel(i, d)]))
            add(f"disc.aurc.{m}.{d}", lambda i, p=p, d=d: A.aurc(p[sel(i, d)], T.y[sel(i, d)]))
            if m != "M0":
                add(f"disc.brier.{m}.{d}", lambda i, p=p, d=d: A.brier(p[sel(i, d)], T.y[sel(i, d)].astype(float)))
                add(f"disc.ece.{m}.{d}", lambda i, p=p, d=d: A.ece_equal_mass(p[sel(i, d)], T.y[sel(i, d)]))
        for a, b in (("M3", "M0"), ("M3", "M1"), ("M2", "M1")):
            add(f"disc.d_auroc.{a}_minus_{b}.{d}",
                lambda i, a=a, b=b, d=d: A.auroc(preds[a][sel(i, d)], T.y[sel(i, d)]) - A.auroc(preds[b][sel(i, d)], T.y[sel(i, d)]))
            add(f"disc.d_aurc.{a}_minus_{b}.{d}",
                lambda i, a=a, b=b, d=d: A.aurc(preds[a][sel(i, d)], T.y[sel(i, d)]) - A.aurc(preds[b][sel(i, d)], T.y[sel(i, d)]))
        for name, ans in operating.items():
            add(f"op.cov.{name}.{d}", lambda i, ans=ans, d=d: selective_stats(ans[sel(i, d)], T.y[sel(i, d)])[0])
            add(f"op.risk.{name}.{d}", lambda i, ans=ans, d=d: selective_stats(ans[sel(i, d)], T.y[sel(i, d)])[1])

    full = np.arange(T.n)
    point = np.array([fn(full) for _, fn in specs], dtype=float)
    boot = A.cluster_boot(lambda idx: tuple(fn(idx) for _, fn in specs), T.dom, T.clu, n_boot=n_boot, seed=seed)
    out = {}
    for j, (name, _) in enumerate(specs):
        ci = A.pct_ci(float(point[j]), boot[:, j])
        out[name] = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in ci.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tiers", required=True, type=Path)
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--overlay", action="append", default=[], type=Path, help="rows.jsonl whose rows override (repeatable)")
    ap.add_argument("--models", nargs="+", default=["qwen3.5-4b-bf16", "qwen3.5-2b-bf16"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    tasks = read_jsonl(a.tasks)
    rows = read_jsonl(a.tiers / "rows.jsonl")
    for ov in a.overlay:
        rows.extend(read_jsonl(ov))
    gens = read_jsonl(a.tiers / "gens.jsonl")
    res = {"meta": {"plan": "docs/ANALYSIS_PLAN_E_TPU.md#A11", "n_folds": N_FOLDS, "inner_folds": INNER_FOLDS, "alphas": list(ALPHAS),
                    "min_answered": MIN_ANSWERED, "n_boot": a.n_boot, "seed": a.seed, "features_all": list(FEATURES_ALL),
                    "overlays": [str(o) for o in a.overlay],
                    "note": "CIs resample source-problem clusters of FIXED out-of-fold predictions; they do not include model-fit variability"}}
    for m in a.models:
        T = Table(tasks, rows, gens, m)
        res[m] = analyze(T, a.n_boot, a.seed)
        print(f"{m}: {len(res[m])} metrics")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
