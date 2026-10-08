#!/usr/bin/env python
"""Pre-registered analysis for GwenLaya v4 (docs/GWENLAYA_PREREGISTRATION.md sections 7-8).

Reads, never writes, the study outputs:
  rows.jsonl    scored rows from scripts/run_study.py (arm, domain, task_id, score, answered, gpu_s, ...)
  gens.jsonl    cached generations (cpu_seconds / gpu_s per call, token counts, finish reasons)
  tasks jsonl   E-night tasks (task_id -> source-problem cluster)
  scores jsonl  optional cross-fitted out-of-fold scores {task_id, arm, p_correct, correct[, tau_hi]}
  spend_ledger.jsonl

and writes papers/numbers_v4.json (every number with its source file), papers/tables_v4.tex and
papers/figures_v4/*.png. A hypothesis whose inputs are absent is reported as ``not_run`` with the
reason; no number is ever invented (unknown = null / TBD).

Conventions
  correct   = row.score == "VERIFIED" (hidden checks); FAILED and UNVERIFIED both count as not correct.
  answered  = row.answered (None = arm unavailable, e.g. B3 without logprobs or without tau).
  CWR       = P(answered and not correct) over ALL items; differences are GL minus comparator.
  cost      = CPU-seconds per call if every call has it (D3), else the row's gpu_s; the unit is recorded.
  bootstrap = percentile, 10,000 resamples, numpy default_rng(0) (not random.Random(0): see D22),
              resampling source-problem clusters within domain, all arms jointly.
  Holm      = over the kept family only (the hypotheses that actually ran).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gwaya.metrics_selective import holm, mcnemar_exact  # noqa: E402

NIGHT = Path(os.environ.get("GWAYA_DATA_ROOT", "/mnt/data/home/xavkal/gwaya-data")) / "night"
LEDGER = Path(os.environ.get("GWAYA_DATA_ROOT", "/mnt/data/home/xavkal/gwaya-data")) / "spend_ledger.jsonl"

# Hypothesis status for the night run, copied from docs/DEVIATIONS.md "Hypothesis status for this run".
STATUS = {
    "H1": "primary", "H3": "primary", "H4": "secondary", "H5": "secondary", "H6": "secondary",
    "H2": "exploratory", "H7": "exploratory", "H8": "exploratory_only_if_G1",
    "H9": "deferred", "H10": "dropped", "H11": "dropped", "H12": "dropped",
    "H13": "run", "H14": "only_if_E1", "H15": "offline_test",
}
ARM_NAMES = {"B1": "always_smallest", "B2": "always_largest", "B3": "raw_confidence",
             "B5": "gate_only", "GL": "gwenlaya"}


# ── numbers registry ───────────────────────────────────────────────────────────────

class Numbers:
    """Every number written to numbers_v4.json carries the file (and field) it was read or computed from."""

    def __init__(self) -> None:
        self.n: dict[str, dict[str, Any]] = {}
        self.hyp: dict[str, dict[str, Any]] = {}
        self.meta: dict[str, Any] = {}

    def add(self, key: str, value: Any, source: str) -> None:
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            value = None  # unknown stays unknown (JSON null), never NaN
        self.n[key] = {"value": value, "source": source}

    def add_ci(self, key: str, ci: dict[str, float], source: str) -> None:
        for k in ("point", "lo", "hi"):
            self.add(f"{key}.{k}", ci.get(k), source)

    def to_json(self) -> dict[str, Any]:
        return {"schema": "gwenlaya_v4.numbers/1", "numbers": self.n, "hypotheses": self.hyp, "meta": self.meta}


# ── loading ────────────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    if not Path(path).exists():
        return out
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    break  # torn tail from an interrupted writer
    return out


def verify_chain(records: list[dict[str, Any]]) -> bool:
    """Read-only check of the run_study hash chain."""
    from scripts.run_study import ChainedLog, chain_hash
    head = ChainedLog.GENESIS
    for r in records:
        if r.get("prev") != head or r.get("hash") != chain_hash(head, r):
            return False
        head = r["hash"]
    return True


def cluster_of(task_id: str, domain: str, task_clusters: dict[str, str]) -> str:
    if task_id in task_clusters:
        return task_clusters[task_id]
    return f"{domain}:{task_id}"  # each item its own cluster when no map is available


class Item:
    """Per-arm, per-item arrays aligned on a common task list."""


def build_arrays(rows: list[dict[str, Any]], arm_rows: dict[str, str | None], task_clusters: dict[str, str],
                 cost_fn: Callable[[dict[str, Any]], float | None]) -> dict[str, Any]:
    """arm_rows maps a logical arm label (GL, B1, ...) to (arm name); rows of other arms are ignored.
    Returns the common task ids, domains, clusters and per-arm numpy arrays."""
    by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for label, arm in arm_rows.items():
        by_arm[label] = {r["task_id"]: r for r in rows if r.get("arm") == arm}
    present = [lab for lab, d in by_arm.items() if d]
    if not present:
        return {"labels": [], "n": 0}
    common = sorted(set.intersection(*[set(by_arm[lab]) for lab in present]))
    dom = [by_arm[present[0]][t]["domain"] for t in common]
    cl = [cluster_of(t, d, task_clusters) for t, d in zip(common, dom)]
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for lab in present:
        rs = [by_arm[lab][t] for t in common]
        ans = [r.get("answered") for r in rs]
        arrays[lab] = {
            "correct": np.array([r.get("score") == "VERIFIED" for r in rs], dtype=bool),
            "answered_known": np.array([a is not None for a in ans], dtype=bool),
            "answered": np.array([bool(a) for a in ans], dtype=bool),
            "cost": np.array([cost_fn(r) if cost_fn(r) is not None else np.nan for r in rs], dtype=float),
            "escalated": np.array([len(r.get("tiers_invoked") or []) > 1 for r in rs], dtype=bool),
            "p": np.array([_conf(r) for r in rs], dtype=float),
            "gate": np.array([r.get("gate") or "" for r in rs], dtype=object),
            "scored": np.array([r.get("score") is not None for r in rs], dtype=bool),
        }
    return {"labels": present, "n": len(common), "task_ids": common, "domain": np.array(dom),
            "cluster": np.array(cl), "arr": arrays, "dropped_unpaired": {
                lab: len(by_arm[lab]) - len(common) for lab in present}}


def _conf(r: dict[str, Any]) -> float:
    if r.get("p_correct") is not None:
        return float(r["p_correct"])
    if r.get("confidence") is not None:  # B3: mean token logprob -> geometric-mean probability
        return float(math.exp(min(0.0, float(r["confidence"]))))
    return float("nan")


# ── metrics (numpy; cross-checked against gwaya.metrics_selective in the tests) ──────────

def acc(c: np.ndarray) -> float:
    return float(c.mean()) if len(c) else float("nan")


def cwr(ans: np.ndarray, cor: np.ndarray) -> float:
    return float((ans & ~cor).mean()) if len(ans) else float("nan")


def answered_acc(ans: np.ndarray, cor: np.ndarray) -> float:
    k = int(ans.sum())
    return float((ans & cor).sum() / k) if k else float("nan")


def cost_per_correct(ans: np.ndarray, cor: np.ndarray, cost: np.ndarray) -> float:
    good = int((ans & cor).sum())
    return float(np.nansum(cost) / good) if good else float("nan")


def ece_equal_mass(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    n = len(p)
    if n == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    k = min(n_bins, n)
    tot = 0.0
    for j in range(k):
        g = order[(j * n) // k:((j + 1) * n) // k]
        if len(g):
            tot += len(g) / n * abs(p[g].mean() - y[g].mean())
    return float(tot)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    """P(score of a correct item > score of a wrong one), ties count 1/2."""
    y = y.astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv = np.unique(s, return_inverse=True)
    cnt = np.bincount(inv)
    cum = np.cumsum(cnt) - cnt  # number of items with strictly smaller score
    avg_rank = cum[inv] + (cnt[inv] + 1) / 2.0
    return float((avg_rank[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def aurc(p: np.ndarray, y: np.ndarray) -> float:
    """Area under the risk-coverage curve; ties broken pessimistically (wrong first), as in
    gwaya.metrics_selective."""
    n = len(p)
    if n == 0:
        return float("nan")
    order = np.lexsort((y.astype(int), -p))  # descending confidence, wrong first within ties
    wrong = (~y.astype(bool))[order].astype(float)
    risk = np.cumsum(wrong) / np.arange(1, n + 1)
    return float(risk.mean())


def risk_coverage(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((y.astype(int), -p))
    wrong = (~y.astype(bool))[order].astype(float)
    n = len(p)
    return np.arange(1, n + 1) / n, np.cumsum(wrong) / np.arange(1, n + 1)


# ── bootstrap ──────────────────────────────────────────────────────────────────────

def cluster_boot(stat: Callable[[np.ndarray], Any], domain: np.ndarray, cluster: np.ndarray,
                 n_boot: int = 10000, seed: int = 0) -> np.ndarray:
    """Cluster-stratified bootstrap. ``stat`` receives an index array into the items (with repeats) and
    returns a float or a tuple of floats. Clusters are resampled with replacement within each domain;
    every arm shares the index array, so comparisons are paired."""
    rng = np.random.default_rng(seed)
    strata: list[list[np.ndarray]] = []
    for d in sorted(set(domain.tolist())):
        idx_d = np.where(domain == d)[0]
        cl_d = cluster[idx_d]
        groups = [idx_d[cl_d == c] for c in sorted(set(cl_d.tolist()))]
        strata.append(groups)
    out = []
    for _ in range(n_boot):
        parts = []
        for groups in strata:
            pick = rng.integers(0, len(groups), size=len(groups))
            parts.extend(groups[j] for j in pick)
        out.append(stat(np.concatenate(parts)))
    return np.asarray(out, dtype=float)


def pct_ci(point: float, vals: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_valid": 0}
    return {"point": float(point), "lo": float(np.percentile(v, 100 * alpha / 2)),
            "hi": float(np.percentile(v, 100 * (1 - alpha / 2))), "n_valid": int(len(v))}


def one_sided_p_below(vals: np.ndarray, null: float) -> float:
    """Bootstrap one-sided p for H: stat < null, = (1 + #{boot >= null}) / (B + 1)."""
    v = vals[np.isfinite(vals)]
    return float((1 + int((v >= null).sum())) / (len(v) + 1)) if len(v) else float("nan")


def one_sided_p_above(vals: np.ndarray, null: float) -> float:
    v = vals[np.isfinite(vals)]
    return float((1 + int((v <= null).sum())) / (len(v) + 1)) if len(v) else float("nan")


# ── cost lookup ────────────────────────────────────────────────────────────────────

def make_cost_fn(gens: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[Callable, str]:
    """Cost of a row = sum over the tiers it invoked of that tier's greedy call cost.
    CPU-seconds when every needed call has cpu_seconds, otherwise the row's own gpu_s."""
    idx: dict[tuple[str, str], dict[str, Any]] = {}
    for g in gens:
        if g.get("call_index", 0) == 0 and g.get("temperature", 0.0) == 0.0:
            idx[(g["model"], g["task_id"])] = g

    def models_of(r: dict[str, Any]) -> list[str]:
        return list(r.get("tiers_invoked") or [r.get("model")])

    use_cpu = bool(rows) and all(all((m, r["task_id"]) in idx and idx[(m, r["task_id"])].get("cpu_seconds") is not None
                                     for m in models_of(r)) for r in rows)
    unit = "cpu_seconds" if use_cpu else "gpu_seconds"

    def cost(r: dict[str, Any]) -> float | None:
        if use_cpu:
            return float(sum(idx[(m, r["task_id"])]["cpu_seconds"] for m in models_of(r)))
        v = r.get("gpu_s")
        return None if v is None else float(v)
    return cost, unit


# ── hypotheses ─────────────────────────────────────────────────────────────────────

def ci_dict(stat: Callable[[np.ndarray], float], D: dict[str, Any], n_boot: int, seed: int) -> tuple[dict, np.ndarray]:
    allidx = np.arange(D["n"])
    vals = cluster_boot(stat, D["domain"], D["cluster"], n_boot, seed)
    return pct_ci(float(stat(allidx)), vals), vals


def run_h1_h3(D: dict[str, Any], nb: int, seed: int, N: Numbers, src: str, unit: str) -> dict[str, dict]:
    res: dict[str, dict] = {}
    A = D.get("arr", {})
    n = D.get("n", 0)
    # H1: GL vs B3
    if "GL" in A and "B3" in A and A["GL"]["scored"].all() and A["B3"]["answered_known"].all():
        g, b = A["GL"], A["B3"]
        gcw, bcw = g["answered"] & ~g["correct"], b["answered"] & ~b["correct"]
        n_b3_only, n_gl_only = int((bcw & ~gcw).sum()), int((gcw & ~bcw).sum())
        p = mcnemar_exact(n_b3_only, n_gl_only, "greater")  # H1: B3 has more confident-wrong items
        d_ci, _ = ci_dict(lambda i: float(gcw[i].mean() - bcw[i].mean()), D, nb, seed)
        cov_ci, _ = ci_dict(lambda i: float(g["answered"][i].mean() - b["answered"][i].mean()), D, nb, seed)
        matched = bool(cov_ci["lo"] >= -0.03 and cov_ci["hi"] <= 0.03)
        res["H1"] = {"status": "ran", "n": n, "underpowered": n < 500, "delta_cwr": d_ci, "coverage_diff": cov_ci,
                     "coverage_matched": matched, "discordant_b3_only": n_b3_only, "discordant_gl_only": n_gl_only,
                     "p_raw": p, "cwr_gl": float(gcw.mean()), "cwr_b3": float(bcw.mean())}
        src_ = src + " (arms gwenlaya, raw_confidence)"
        N.add_ci("H1.delta_cwr", d_ci, src_)
        N.add_ci("H1.coverage_diff", cov_ci, src_)
        for k in ("n", "discordant_b3_only", "discordant_gl_only", "p_raw", "cwr_gl", "cwr_b3"):
            N.add(f"H1.{k}", res["H1"][k], src_)
    else:
        res["H1"] = {"status": "not_run", "reason": _why_h1(D)}
    # H3: GL vs B5, cost per correct answered item
    if "GL" in A and "B5" in A and A["GL"]["scored"].all() and A["B5"]["scored"].all() \
            and np.isfinite(A["GL"]["cost"]).all() and np.isfinite(A["B5"]["cost"]).all():
        g, b = A["GL"], A["B5"]

        def ratio(i):
            den = cost_per_correct(b["answered"][i], b["correct"][i], b["cost"][i])
            num = cost_per_correct(g["answered"][i], g["correct"][i], g["cost"][i])
            return num / den if den == den and den > 0 else float("nan")

        def accdiff(i):
            return answered_acc(g["answered"][i], g["correct"][i]) - answered_acc(b["answered"][i], b["correct"][i])

        allidx = np.arange(n)
        both = cluster_boot(lambda i: (ratio(i), accdiff(i), float(g["escalated"][i].mean())), D["domain"], D["cluster"], nb, seed)
        r_ci = pct_ci(float(ratio(allidx)), both[:, 0])
        a_ci = pct_ci(float(accdiff(allidx)), both[:, 1])
        e_ci = pct_ci(float(g["escalated"].mean()), both[:, 2])
        res["H3"] = {"status": "ran", "n": n, "underpowered": n < 500, "cost_unit": unit, "ratio": r_ci,
                     "answered_acc_diff": a_ci, "escalation_rate": e_ci,
                     "p_raw": one_sided_p_below(both[:, 0], 1.0),
                     "cost_per_correct_gl": cost_per_correct(g["answered"], g["correct"], g["cost"]),
                     "cost_per_correct_b5": cost_per_correct(b["answered"], b["correct"], b["cost"]),
                     "non_inferior": bool(a_ci["lo"] > -0.02)}
        src_ = src + f" (arms gwenlaya, gate_only; cost unit {unit})"
        N.add_ci("H3.ratio", r_ci, src_)
        N.add_ci("H3.answered_acc_diff", a_ci, src_)
        N.add_ci("H3.escalation_rate", e_ci, src_)
        for k in ("n", "p_raw", "cost_per_correct_gl", "cost_per_correct_b5"):
            N.add(f"H3.{k}", res["H3"][k], src_)
    else:
        res["H3"] = {"status": "not_run", "reason": "need scored rows with cost for arms gwenlaya and gate_only (largest tier)"}
    # Holm over the kept family
    ran = [h for h in ("H1", "H3") if res[h]["status"] == "ran"]
    if ran:
        adj = holm([res[h]["p_raw"] for h in ran])
        for h, a in zip(ran, adj):
            res[h]["p_holm"], res[h]["holm_m"] = a, len(ran)
            N.add(f"{h}.p_holm", a, f"holm over {ran}")
        if "H1" in ran:
            h = res["H1"]
            if not h["coverage_matched"]:
                h["verdict"] = "matching failed"
            elif h["p_holm"] < 0.05 and h["delta_cwr"]["hi"] < 0:
                h["verdict"] = "PASS (meaningful)" if h["delta_cwr"]["hi"] < -0.03 else "PASS"
            elif h["delta_cwr"]["lo"] > 0:
                h["verdict"] = "HARM"
            else:
                h["verdict"] = "not supported"
        if "H3" in ran:
            h = res["H3"]
            if h["p_holm"] < 0.05 and h["ratio"]["hi"] < 1 and h["non_inferior"]:
                h["verdict"] = "PASS (meaningful)" if h["ratio"]["hi"] <= 0.75 else "PASS"
            else:
                h["verdict"] = "not supported"
        for h in ran:
            if res[h]["underpowered"]:
                res[h]["verdict"] += " [underpowered: paired n < 500]"
    return res


def _why_h1(D: dict[str, Any]) -> str:
    A = D.get("arr", {})
    if "GL" not in A:
        return "no scored rows for arm gwenlaya"
    if "B3" not in A:
        return "no rows for arm raw_confidence"
    if not A["B3"]["answered_known"].all():
        return "raw_confidence has answered=None (no logprobs or no tau_b3)"
    return "rows are not scored"


def run_secondary(scores: list[dict[str, Any]], task_info: dict[str, tuple[str, str]], nb: int, seed: int,
                  N: Numbers, src: str) -> dict[str, dict]:
    """H4, H5, H6 from cross-fitted out-of-fold scores. Arms: laya, b3_ts, lr, sh."""
    res: dict[str, dict] = {h: {"status": "not_run", "reason": f"no cross-fitted scores file ({src})"} for h in ("H4", "H5", "H6")}
    if not scores:
        return res
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in scores:
        by[r["arm"]][r["task_id"]] = r
    arms = {a: d for a, d in by.items()}
    common = sorted(set.intersection(*[set(d) for d in arms.values()])) if arms else []
    if not common:
        return res
    dom = np.array([task_info.get(t, ("?", t))[0] if t in task_info else "?" for t in common])
    cl = np.array([task_info[t][1] if t in task_info else t for t in common])
    P = {a: np.array([d[t]["p_correct"] for t in common], dtype=float) for a, d in arms.items()}
    Y = {a: np.array([bool(d[t]["correct"]) for t in common]) for a, d in arms.items()}
    D = {"n": len(common), "domain": dom, "cluster": cl}
    ps: dict[str, float] = {}
    s = src
    if "laya" in P:
        y, p = Y["laya"], P["laya"]
        e_ci, _ = ci_dict(lambda i: ece_equal_mass(p[i], y[i].astype(float)), D, nb, seed)
        res["H4"] = {"status": "ran", "n": len(common), "ece_laya": e_ci}
        N.add_ci("H4.ece_laya", e_ci, s)
        N.add("H4.brier_laya", brier(p, y.astype(float)), s)
        N.add("H4.auroc_laya", auroc(p, y), s)
        N.add("H4.aurc_laya", aurc(p, y), s)
        if "b3_ts" in P:
            q, yq = P["b3_ts"], Y["b3_ts"]
            d_ci, vals = ci_dict(lambda i: ece_equal_mass(p[i], y[i].astype(float)) - ece_equal_mass(q[i], yq[i].astype(float)), D, nb, seed)
            res["H4"].update({"ece_b3_ts": ece_equal_mass(q, yq.astype(float)), "ece_diff": d_ci,
                              "p_raw": one_sided_p_below(vals, 0.0)})
            N.add("H4.ece_b3_ts", res["H4"]["ece_b3_ts"], s)
            N.add_ci("H4.ece_diff", d_ci, s)
            ps["H4"] = res["H4"]["p_raw"]
        else:
            res["H4"]["note"] = "no temperature-scaled B3 scores: the B3 comparison part of H4 is not run"
        # H6: per-domain selective error at tau_hi
        taus = {t: d["tau_hi"] for t, d in arms["laya"].items() if d.get("tau_hi") is not None}
        if len(taus) == len(common):
            tau = np.array([taus[t] for t in common])
            res["H6"] = {"status": "ran", "per_domain": {}}
            for d in sorted(set(dom.tolist())):
                m = (dom == d)
                sub = {"n": int(m.sum()), "domain": dom[m], "cluster": cl[m]}
                sel, cor = (p[m] >= tau[m]), y[m]

                def err(i, sel=sel, cor=cor):
                    k = sel[i].sum()
                    return float((sel[i] & ~cor[i]).sum() / k) if k else float("nan")
                ci, _ = ci_dict(err, sub, nb, seed)
                res["H6"]["per_domain"][d] = {**ci, "coverage": float(sel.mean()), "verdict": "FAIL" if ci["lo"] > 0.10 else "not refuted"}
                N.add_ci(f"H6.{d}.selective_error", ci, s)
                N.add(f"H6.{d}.coverage", float(sel.mean()), s)
    if "laya" in P and "lr" in P:
        pl, pr, yl = P["laya"], P["lr"], Y["laya"]
        d_ci, vals = ci_dict(lambda i: auroc(pl[i], yl[i]) - auroc(pr[i], yl[i]), D, nb, seed)
        res["H5"] = {"status": "ran", "n": len(common), "auroc_laya": auroc(pl, yl), "auroc_lr": auroc(pr, yl),
                     "auroc_diff": d_ci, "p_raw": one_sided_p_above(vals, 0.0)}
        N.add("H5.auroc_laya", res["H5"]["auroc_laya"], s)
        N.add("H5.auroc_lr", res["H5"]["auroc_lr"], s)
        N.add_ci("H5.auroc_diff", d_ci, s)
        ps["H5"] = res["H5"]["p_raw"]
        if "sh" in P:
            sci, _ = ci_dict(lambda i: auroc(P["sh"][i], Y["sh"][i]), D, nb, seed)
            res["H5"]["sh_auroc"] = sci
            res["H5"]["sh_leak_flag"] = bool(not (sci["lo"] <= 0.5 <= sci["hi"]))
            N.add_ci("H5.sh_auroc", sci, s)
    if ps:
        keys = sorted(ps)
        adj = holm([ps[k] for k in keys])
        for k, a in zip(keys, adj):
            res[k]["p_holm"], res[k]["holm_m"] = a, len(keys)
            N.add(f"{k}.p_holm", a, f"holm over family A {keys}")
        if "H4" in ps:
            h = res["H4"]
            h["verdict"] = "PASS" if (h["ece_laya"]["point"] <= 0.05 and h["p_holm"] < 0.05 and h["ece_diff"]["hi"] < 0) else "not supported"
        if "H5" in ps:
            h = res["H5"]
            h["verdict"] = "PASS" if (h["p_holm"] < 0.05 and h["auroc_diff"]["lo"] > 0) else "not supported"
            if h["auroc_diff"]["lo"] >= 0.02 and h["verdict"] == "PASS":
                h["verdict"] = "PASS (meaningful)"
    return res


def run_exploratory(D: dict[str, Any], nb: int, seed: int, N: Numbers, src: str) -> dict[str, dict]:
    res: dict[str, dict] = {}
    A = D.get("arr", {})
    if "GL" in A and "B5" in A and A["GL"]["scored"].all() and A["B5"]["scored"].all():
        g, b = A["GL"], A["B5"]
        gcw, bcw = g["answered"] & ~g["correct"], b["answered"] & ~b["correct"]
        ci, _ = ci_dict(lambda i: float(gcw[i].mean() - bcw[i].mean()), D, nb, seed)
        p = mcnemar_exact(int((gcw & ~bcw).sum()), int((bcw & ~gcw).sum()), "two-sided")
        res["H2"] = {"status": "ran_exploratory", "delta_cwr": ci, "mcnemar_p_unadjusted": p, "n": D["n"]}
        N.add_ci("H2.delta_cwr", ci, src + " (arms gwenlaya, gate_only; exploratory, unadjusted)")
        N.add("H2.mcnemar_p_unadjusted", p, src)
        if ci["lo"] <= 0 <= ci["hi"]:
            res["H2"]["statement"] = "Laya adds no confident-wrong reduction beyond the executed gate (CI includes 0)"
    else:
        res["H2"] = {"status": "not_run", "reason": "need scored rows for gwenlaya and gate_only"}
    # H7: VERIFIED precision per domain, from gate_only (largest tier) rows
    if "B5" in A and A["B5"]["scored"].all():
        b = A["B5"]
        ver = b["gate"] == "VERIFIED"
        res["H7"] = {"status": "ran_exploratory", "per_domain": {}}
        for d in sorted(set(D["domain"].tolist())):
            m = D["domain"] == d
            sel, cor = ver[m], b["correct"][m]
            sub = {"n": int(m.sum()), "domain": D["domain"][m], "cluster": D["cluster"][m]}

            def prec(i, sel=sel, cor=cor):
                k = sel[i].sum()
                return float((sel[i] & cor[i]).sum() / k) if k else float("nan")
            ci, _ = ci_dict(prec, sub, nb, seed)
            res["H7"]["per_domain"][d] = {**ci, "n_verified": int(sel.sum()), "flag_below_0.95": bool(ci["point"] < 0.95)}
            N.add_ci(f"H7.{d}.verified_precision", ci, src + " (arm gate_only, gate==VERIFIED vs hidden-check score)")
            N.add(f"H7.{d}.n_verified", int(sel.sum()), src)
    else:
        res["H7"] = {"status": "not_run", "reason": "need scored gate_only rows"}
    return res


def arm_summary(D: dict[str, Any], nb: int, seed: int, N: Numbers, src: str, unit: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    A = D.get("arr", {})
    for lab in D.get("labels", []):
        a = A[lab]
        if not a["scored"].all():
            continue
        row: dict[str, Any] = {"n": D["n"]}
        known = a["answered_known"].all()
        cor = a["correct"]
        row["accuracy"], _ = ci_dict(lambda i: float(cor[i].mean()), D, nb, seed)
        if known:
            ans = a["answered"]
            row["coverage"], _ = ci_dict(lambda i: float(ans[i].mean()), D, nb, seed)
            row["cwr"], _ = ci_dict(lambda i: cwr(ans[i], cor[i]), D, nb, seed)
            row["answered_accuracy"], _ = ci_dict(lambda i: answered_acc(ans[i], cor[i]), D, nb, seed)
            if np.isfinite(a["cost"]).all():
                row["cost_per_correct"], _ = ci_dict(lambda i: cost_per_correct(ans[i], cor[i], a["cost"][i]), D, nb, seed)
                row["cost_unit"] = unit
        p = a["p"]
        if np.isfinite(p).all():
            y = cor.astype(float)
            row["ece"], _ = ci_dict(lambda i: ece_equal_mass(p[i], y[i]), D, nb, seed)
            row["brier"], _ = ci_dict(lambda i: brier(p[i], y[i]), D, nb, seed)
            row["auroc"], _ = ci_dict(lambda i: auroc(p[i], cor[i]), D, nb, seed)
            row["aurc"], _ = ci_dict(lambda i: aurc(p[i], cor[i]), D, nb, seed)
        out[lab] = row
        for k, v in row.items():
            if isinstance(v, dict):
                N.add_ci(f"arm.{lab}.{k}", v, f"{src} (arm {ARM_NAMES.get(lab, lab)})")
        N.add(f"arm.{lab}.n", D["n"], src)
    return out


def generation_descriptives(gens: list[dict[str, Any]], N: Numbers, src: str) -> dict[str, Any]:
    cells: dict[tuple, dict[str, Any]] = {}
    for g in gens:
        c = cells.setdefault((g["model"], g["domain"]), {"n": 0, "tok": 0, "cpu": 0.0, "gpu": 0.0, "wall": 0.0, "fin": Counter(), "cpu_known": 0})
        c["n"] += 1
        c["tok"] += g.get("completion_tokens") or 0
        c["gpu"] += g.get("gpu_s") or 0.0
        c["wall"] += g.get("wall_s") or 0.0
        if g.get("cpu_seconds") is not None:
            c["cpu"] += g["cpu_seconds"]
            c["cpu_known"] += 1
        c["fin"][g.get("done_reason") or "?"] += 1
    out = {}
    for (m, d), c in sorted(cells.items()):
        k = f"{m}|{d}"
        out[k] = {"n": c["n"], "mean_completion_tokens": c["tok"] / c["n"], "cpu_seconds_total": c["cpu"] if c["cpu_known"] else None,
                  "server_reported_seconds_total": c["gpu"], "client_wall_seconds_total": c["wall"], "finish_reasons": dict(c["fin"])}
        N.add(f"gen.{k}.n", c["n"], src)
        N.add(f"gen.{k}.mean_completion_tokens", out[k]["mean_completion_tokens"], src)
        N.add(f"gen.{k}.cpu_seconds_total", out[k]["cpu_seconds_total"], src)
        N.add(f"gen.{k}.server_reported_seconds_total", c["gpu"], src + " (sum of gpu_s: server-reported total_ms)")
        N.add(f"gen.{k}.client_wall_seconds_total", c["wall"], src + " (sum of wall_s)")
        for fr, cnt in c["fin"].items():
            N.add(f"gen.{k}.finish.{fr}", cnt, src)
    return out


def ledger_h13(path: Path, N: Numbers) -> dict[str, Any]:
    rows = read_jsonl(path)
    known = [r for r in rows if r.get("usd_estimate") is not None]
    missing = [r for r in rows if r.get("usd_estimate") is None]
    total = float(sum(r["usd_estimate"] for r in known))
    res = {"status": "ran", "lines": len(rows), "usd_known_total": total, "lines_without_usd_estimate": len(missing),
           "seconds_without_estimate": float(sum(r.get("seconds", 0) or 0 for r in missing)),
           "note": "lines without usd_estimate are unpriced here (the TPU smoke tests, about 0.84 USD per docs/DEVIATIONS.md, "
                   "not read from this file); the verdict below uses the known total only and is a lower bound"}
    res["verdict"] = ("refuted (known total > 50)" if total > 50 else
                      "not refuted on the known lines (lower bound; unpriced lines exist)" if missing else
                      "not refuted" if total <= 40 else "planned cap exceeded")
    s = str(path)
    for k in ("lines", "usd_known_total", "lines_without_usd_estimate", "seconds_without_estimate"):
        N.add(f"H13.{k}", res[k], s)
    return res


# ── outputs: LaTeX and figures ──────────────────────────────────────────────────────

def tex(s: Any) -> str:
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("|", " ")


def fmt(v: Any, d: int = 3, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "TBD"
    return f"{100 * v:.1f}" if pct else f"{v:.{d}f}"


def fmt_ci(ci: dict | None, d: int = 3, pct: bool = False) -> str:
    if not ci:
        return "TBD"
    return f"{fmt(ci['point'], d, pct)} [{fmt(ci['lo'], d, pct)}, {fmt(ci['hi'], d, pct)}]"


def write_tables(path: Path, gen: dict, arms: dict, hyp: dict, unit: str) -> None:
    L = ["% generated by scripts/analyze_study.py; values come from papers/numbers_v4.json. TBD = unknown or not run."]
    L += ["\\begin{table}[t]\\centering\\small",
          "\\caption{Cached generations on the E-night set (greedy, one seed). Tokens are mean completion tokens.}",
          "\\begin{tabular}{llrrrl}\\toprule", "model & domain & n & mean tok. & CPU-s total & finish reasons \\\\\\midrule"]
    for k, v in gen.items():
        m, d = k.split("|")
        L.append(f"{tex(m)} & {tex(d)} & {v['n']} & {fmt(v['mean_completion_tokens'], 1)} & {fmt(v['cpu_seconds_total'], 0)} & "
                 f"{tex(', '.join(f'{a}:{b}' for a, b in v['finish_reasons'].items()))} \\\\")
    if not gen:
        L.append("\\multicolumn{6}{c}{no generations found} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    L += ["\\begin{table}[t]\\centering\\small",
          f"\\caption{{Arms on the paired E-night items (percent; 95\\% cluster bootstrap CI). Cost unit: {tex(unit)}. TBD = not run or not available.}}",
          "\\begin{tabular}{lccccc}\\toprule", "arm & acc. & coverage & CWR & ans. acc. & cost / correct \\\\\\midrule"]
    for lab, r in arms.items():
        L.append(f"{lab} & {fmt_ci(r.get('accuracy'), pct=True)} & {fmt_ci(r.get('coverage'), pct=True)} & {fmt_ci(r.get('cwr'), pct=True)} & "
                 f"{fmt_ci(r.get('answered_accuracy'), pct=True)} & {fmt_ci(r.get('cost_per_correct'), 1)} \\\\")
    if not arms:
        L.append("\\multicolumn{6}{c}{no scored rows yet: TBD} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    L += ["\\begin{table}[t]\\centering\\small", "\\caption{Hypothesis status for the night run. Verdicts follow the pre-registered rules; Holm over the kept family.}",
          "\\begin{tabular}{lll p{6.2cm}}\\toprule", "H & role & result & detail \\\\\\midrule"]
    for h in sorted(STATUS, key=lambda x: int(x[1:])):
        r = hyp.get(h, {"status": STATUS[h]})
        st = r.get("status", STATUS[h])
        res = r.get("verdict") or ("TBD" if st == "not_run" else st)
        det = r.get("reason") or ""
        if h == "H1" and st == "ran":
            det = f"$\\Delta$CWR {fmt_ci(r['delta_cwr'], pct=True)} pts; Holm p {fmt(r.get('p_holm'))}"
        if h == "H3" and st == "ran":
            det = f"ratio {fmt_ci(r['ratio'])}; Holm p {fmt(r.get('p_holm'))}"
        L.append(f"{h} & {tex(STATUS[h])} & {tex(res)} & {tex(det) if not det.startswith('$') else det} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    Path(path).write_text("\n".join(L))


def make_figures(outdir: Path, D: dict, arms: dict, scores_ctx: dict | None) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    A = D.get("arr", {})
    conf = [lab for lab in D.get("labels", []) if lab in A and A[lab]["scored"].all() and np.isfinite(A[lab]["p"]).all()]
    if conf:
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        for lab in conf:
            cov, risk = risk_coverage(A[lab]["p"], A[lab]["correct"])
            ax.plot(cov, risk, label=lab)
        ax.set_xlabel("coverage")
        ax.set_ylabel("selective risk")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "risk_coverage.png", dpi=160)
        plt.close(fig)
        made.append("risk_coverage.png")
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        for lab in conf:
            p, y = A[lab]["p"], A[lab]["correct"].astype(float)
            order = np.argsort(p, kind="stable")
            k = min(15, len(p))
            xs = [p[order[(j * len(p)) // k:((j + 1) * len(p)) // k]].mean() for j in range(k)]
            ys = [y[order[(j * len(p)) // k:((j + 1) * len(p)) // k]].mean() for j in range(k)]
            ax.plot(xs, ys, "o-", ms=3, label=lab)
        ax.set_xlabel("predicted P(correct)")
        ax.set_ylabel("observed accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "reliability.png", dpi=160)
        plt.close(fig)
        made.append("reliability.png")
    for key, fname, ylab, scale in (("cwr", "cwr_by_arm.png", "confident-wrong rate (%)", 100),
                                   ("cost_per_correct", "cost_per_correct.png", "cost per correct answered item", 1)):
        labs = [l for l, r in arms.items() if key in r]
        if labs:
            fig, ax = plt.subplots(figsize=(4.2, 3.4))
            pts = [arms[l][key]["point"] * scale for l in labs]
            err = [[arms[l][key]["point"] * scale - arms[l][key]["lo"] * scale for l in labs],
                   [arms[l][key]["hi"] * scale - arms[l][key]["point"] * scale for l in labs]]
            ax.bar(labs, pts, yerr=err, capsize=3)
            ax.set_ylabel(ylab)
            fig.tight_layout()
            fig.savefig(outdir / fname, dpi=160)
            plt.close(fig)
            made.append(fname)
    return made


# ── driver ─────────────────────────────────────────────────────────────────────────

def find_files(name: str, roots: list[Path]) -> list[Path]:
    seen = []
    for r in roots:
        if r.is_file() and r.name == name:
            seen.append(r)
        elif r.is_dir():
            seen += [f for f in sorted(r.rglob(name)) if "discarded" not in str(f.relative_to(r))]  # keep discarded caches out
    return seen


def analyze(rows_files: list[Path], gens_files: list[Path], tasks_file: Path | None, scores_file: Path | None,
            ledger: Path, n_boot: int, seed: int, out_numbers: Path, out_tables: Path, out_figs: Path) -> Numbers:
    N = Numbers()
    rows = [r for f in rows_files for r in read_jsonl(f)]
    gens = [g for f in gens_files for g in read_jsonl(f)]
    scores = read_jsonl(scores_file) if scores_file else []
    tasks = read_jsonl(tasks_file) if tasks_file else []
    task_clusters = {t["task_id"]: t["cluster"] for t in tasks if t.get("cluster")}
    task_info = {t["task_id"]: (t["domain"], t["cluster"]) for t in tasks if t.get("cluster")}
    chain_ok = {str(f): verify_chain(read_jsonl(f)) for f in list(rows_files) + list(gens_files)}
    N.meta.update({"n_boot": n_boot, "seed": seed, "rows_files": [str(f) for f in rows_files],
                   "gens_files": [str(f) for f in gens_files], "hash_chain_ok": chain_ok,
                   "bootstrap_rng": "numpy default_rng(seed); cluster-stratified by domain (D22)",
                   "hypothesis_roles": STATUS})
    N.add("rows.count", len(rows), ",".join(str(f) for f in rows_files) or "none")
    N.add("gens.count", len(gens), ",".join(str(f) for f in gens_files) or "none")
    gsrc = ",".join(str(f) for f in gens_files) or "none"
    gen = generation_descriptives(gens, N, gsrc)

    cost_fn, unit = make_cost_fn(gens, rows)
    # B2 = largest tier; B5 = gate_only restricted to that model.
    largest = None
    big = [r for r in rows if r.get("arm") == "always_largest"]
    if big:
        largest = Counter(r["model"] for r in big).most_common(1)[0][0]
    gate_rows = [r for r in rows if r.get("arm") != "gate_only" or r.get("model") == largest]
    D = build_arrays(gate_rows, ARM_NAMES, task_clusters, cost_fn)
    rsrc = ",".join(str(f) for f in rows_files) or "none"
    N.meta["largest_tier"], N.meta["cost_unit"] = largest, unit
    hyp: dict[str, dict] = {}
    arms: dict[str, dict] = {}
    if D["n"]:
        N.add("paired.n", D["n"], rsrc)
        for lab, k in D["dropped_unpaired"].items():
            N.add(f"paired.dropped_unpaired.{lab}", k, rsrc)
        for dd in sorted(set(D["domain"].tolist())):
            N.add(f"paired.n.{dd}", int((D["domain"] == dd).sum()), rsrc)
        arms = arm_summary(D, n_boot, seed, N, rsrc, unit)
        hyp.update(run_h1_h3(D, n_boot, seed, N, rsrc, unit))
        hyp.update(run_exploratory(D, n_boot, seed, N, rsrc))
    else:
        if rows:
            reason = ("scored rows exist but only for unregistered single arm(s) "
                      f"{sorted({r.get('arm') for r in rows})}: H1/H2/H3/H7 need paired registered arms (GL, B1, B2, B3, B5)")
        else:
            reason = "no scored rows (rows.jsonl absent or empty: generation done, scoring not yet run)"
        for h in ("H1", "H3", "H2", "H7"):
            hyp[h] = {"status": "not_run", "reason": reason}
        # descriptive single-arm summary (NOT a pre-registered hypothesis): raw confidence = exp(mean token logprob),
        # not cross-fitted, so ECE/Brier/AUROC/AURC here are uncalibrated-score descriptives
        extra = sorted({r.get("arm") for r in rows if r.get("arm")} - set(ARM_NAMES.values()))
        if extra:
            lp = {(g["task_id"], g["model"], g.get("seed")): g.get("mean_logprob") for g in gens}
            for r in rows:
                v = lp.get((r["task_id"], r["model"], r.get("seed")))
                if r.get("confidence") is None and v is not None:
                    r["confidence"] = v
            D = build_arrays(rows, {a: a for a in extra}, task_clusters, cost_fn)
            if D["n"]:
                N.add("single_arm.n", D["n"], rsrc)
                arms = arm_summary(D, n_boot, seed, N, rsrc + " + gens mean_logprob (raw confidence)", unit)
                N.meta["single_arm_note"] = ("descriptive only, not a pre-registered test; confidence = exp(mean_logprob) "
                                             "from gens.jsonl, uncalibrated and not cross-fitted")
    hyp.update(run_secondary(scores, task_info, n_boot, seed, N, str(scores_file) if scores_file else "no scores file"))
    hyp["H13"] = ledger_h13(ledger, N)
    for h, st in STATUS.items():
        if h not in hyp:
            hyp[h] = {"status": "not_run" if st not in ("dropped", "deferred") else st,
                      "reason": {"dropped": "dropped for this run (docs/DEVIATIONS.md D8)", "deferred": "deferred (D8)",
                                 "exploratory_only_if_G1": "needs G1 (no GPU slot)", "only_if_E1": "needs E1 (D12)",
                                 "offline_test": "offline kill/resume test, see tests/test_run_study.py"}.get(st, "not run")}
    N.hyp = hyp
    figs = make_figures(out_figs, D, arms, None)
    N.meta["figures"] = figs
    for p in (out_numbers, out_tables):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(out_numbers).write_text(json.dumps(N.to_json(), indent=1, sort_keys=True, default=str) + "\n")
    write_tables(out_tables, gen, arms, hyp, unit)
    return N


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="*", default=[str(ROOT / "results/gwenlaya_v4"), str(NIGHT / "cache")],
                    help="dirs searched for rows.jsonl and gens.jsonl")
    ap.add_argument("--tasks", default=str(NIGHT / "results/tasks_E_night.jsonl"))
    ap.add_argument("--scores", default=None, help="cross-fitted out-of-fold scores jsonl (optional)")
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-numbers", default=str(ROOT / "papers/numbers_v4.json"))
    ap.add_argument("--out-tables", default=str(ROOT / "papers/tables_v4.tex"))
    ap.add_argument("--out-figs", default=str(ROOT / "papers/figures_v4"))
    a = ap.parse_args(argv)
    roots = [Path(p) for p in a.results]
    scores = Path(a.scores) if a.scores else None
    if scores is None:
        cand = find_files("scores_crossfit.jsonl", roots)
        scores = cand[0] if cand else None
    N = analyze(find_files("rows.jsonl", roots), find_files("gens.jsonl", roots), Path(a.tasks), scores, Path(a.ledger),
                a.n_boot, a.seed, Path(a.out_numbers), Path(a.out_tables), Path(a.out_figs))
    ran = [h for h, r in N.hyp.items() if r.get("status", "").startswith("ran")]
    print(f"wrote {a.out_numbers} ({len(N.n)} numbers); hypotheses run: {ran or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
