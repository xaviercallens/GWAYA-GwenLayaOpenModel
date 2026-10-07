"""Selective-prediction and calibration metrics for GwenLaya (stdlib only, no numpy).

Conventions: ``p`` is a predicted P(correct) in [0, 1]; ``y`` is 1 when the answer was correct
against ground truth the gate never saw, else 0. "Answered" items are those whose verdict is
VERIFIED or LIKELY_CORRECT (docs/GWENLAYA_PREREGISTRATION.md section 1).

Nothing here reads or writes measured results; every function is a pure transformation of its
inputs. Ties in confidence are broken pessimistically (wrong first) so AURC is deterministic.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

_NAN = float("nan")


def _pairs(p: Sequence[float], y: Sequence[float]) -> tuple[list[float], list[float]]:
    p, y = [float(v) for v in p], [float(v) for v in y]
    if len(p) != len(y):
        raise ValueError("p and y must have equal length")
    if any(v != v or v < 0.0 or v > 1.0 for v in p):
        raise ValueError("p must lie in [0, 1] and contain no NaN")
    return p, y


def _mean(v: Sequence[float]) -> float:
    return math.fsum(v) / len(v) if len(v) else _NAN


def ece(p: Sequence[float], y: Sequence[float], n_bins: int = 15,
        strategy: str = "equal_mass") -> tuple[float, list[dict[str, float]]]:
    """Expected calibration error and the per-bin table.

    ``equal_mass`` (default, the pre-registered 15 equal-mass bins) splits the sorted
    predictions into n_bins groups of near-equal size; ``equal_width`` uses [0, 1] cells.
    """
    p, y = _pairs(p, y)
    n = len(p)
    if strategy not in ("equal_mass", "equal_width"):
        raise ValueError(f"unknown strategy {strategy!r}")
    if n == 0:
        return _NAN, []
    if strategy == "equal_mass":
        order = sorted(range(n), key=lambda i: p[i])
        k = min(n_bins, n)
        groups = [order[(j * n) // k:((j + 1) * n) // k] for j in range(k)]
    else:
        cells: dict[int, list[int]] = {}
        for i, v in enumerate(p):
            cells.setdefault(min(int(v * n_bins), n_bins - 1), []).append(i)
        groups = [cells[b] for b in sorted(cells)]
    total, bins = 0.0, []
    for g in groups:
        if not g:
            continue
        conf, acc = _mean([p[i] for i in g]), _mean([y[i] for i in g])
        total += len(g) / n * abs(conf - acc)
        bins.append({"n": len(g), "mean_p": conf, "accuracy": acc,
                     "lo": min(p[i] for i in g), "hi": max(p[i] for i in g)})
    return total, bins


def brier(p: Sequence[float], y: Sequence[float]) -> float:
    p, y = _pairs(p, y)
    return _mean([(a - b) ** 2 for a, b in zip(p, y)])


def auroc(score: Sequence[float], y: Sequence[float]) -> float:
    """P(score of a random correct item > score of a random wrong item); ties count 1/2.
    NaN when only one class is present."""
    s, y = [float(v) for v in score], [float(v) for v in y]
    if len(s) != len(y):
        raise ValueError("score and y must have equal length")
    n_pos = sum(1 for v in y if v > 0.5)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return _NAN
    order = sorted(range(len(s)), key=lambda i: s[i])
    rank_sum, i = 0.0, 0
    while i < len(order):  # average ranks for ties
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        rank_sum += avg * sum(1 for t in order[i:j + 1] if y[t] > 0.5)
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _sorted_correct(conf: Sequence[float], correct: Sequence[float]) -> list[float]:
    if len(conf) != len(correct):
        raise ValueError("conf and correct must have equal length")
    # confidence descending; among ties, wrong (0) first -> pessimistic and deterministic
    pairs = sorted(zip((float(c) for c in conf), (float(k) for k in correct)), key=lambda t: (-t[0], t[1]))
    return [k for _, k in pairs]


def risk_coverage_curve(conf: Sequence[float], correct: Sequence[float]) -> dict[str, list[float]]:
    """Risk (error rate among the top-k most confident) at coverage k/n for k = 1..n."""
    ks = _sorted_correct(conf, correct)
    n = len(ks)
    cov, risk, good = [], [], 0.0
    for k, v in enumerate(ks, start=1):
        good += v
        cov.append(k / n)
        risk.append(1.0 - good / k)
    return {"coverage": cov, "risk": risk}


def aurc(conf: Sequence[float], correct: Sequence[float]) -> float:
    """Area under the risk-coverage curve = mean cumulative risk (lower is better)."""
    return _mean(risk_coverage_curve(conf, correct)["risk"])


def e_aurc(conf: Sequence[float], correct: Sequence[float]) -> float:
    """Excess AURC over the oracle ordering (all correct items ranked first)."""
    if len(correct) == 0:
        return _NAN
    return aurc(conf, correct) - aurc(correct, correct)


def risk_at_coverage(conf: Sequence[float], correct: Sequence[float], coverage: float) -> float:
    ks = _sorted_correct(conf, correct)
    m = int(math.ceil(coverage * len(ks) - 1e-12))
    return 1.0 - _mean(ks[:m]) if m > 0 else _NAN


def coverage_at_risk(conf: Sequence[float], correct: Sequence[float], max_risk: float) -> float:
    """Largest coverage whose selective risk is <= max_risk (0.0 if none)."""
    r = risk_coverage_curve(conf, correct)
    best = 0.0
    for cov, risk in zip(r["coverage"], r["risk"]):
        if risk <= max_risk + 1e-12:
            best = cov
    return best


def confident_wrong_rate(p: Sequence[float], correct: Sequence[float], tau: float) -> float:
    """Hallucination rate at threshold tau: P(p >= tau and wrong) over ALL items."""
    p, y = _pairs(p, correct)
    return _mean([1.0 if (a >= tau and b < 0.5) else 0.0 for a, b in zip(p, y)])


def selective_summary(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Aggregate GwenLaya output records. Each record needs ``answered`` (bool), ``correct``
    (bool, ground truth), ``cost_s`` (GPU-seconds) and optionally ``escalated`` (bool).

    - coverage: answered / all
    - answered_accuracy: correct among answered (NaN if none answered)
    - cwr: answered and wrong, over all items (prereg confident-wrong rate)
    - acc_per_gpu_s: correct answered items per GPU-second spent over ALL items
    - gpu_s_per_correct: inverse (inf if nothing correct)
    - escalation_rate: share of items that were escalated
    """
    n = len(records)
    if n == 0:
        return {"n": 0, "coverage": _NAN, "answered_accuracy": _NAN, "cwr": _NAN,
                "acc_per_gpu_s": _NAN, "gpu_s_per_correct": _NAN, "escalation_rate": _NAN}
    ans = [bool(r["answered"]) for r in records]
    cor = [bool(r["correct"]) for r in records]
    cost = math.fsum(float(r.get("cost_s", 0.0)) for r in records)
    good = sum(a and c for a, c in zip(ans, cor))
    n_ans = sum(ans)
    return {
        "n": n,
        "coverage": n_ans / n,
        "answered_accuracy": good / n_ans if n_ans else _NAN,
        "cwr": sum(a and not c for a, c in zip(ans, cor)) / n,
        "acc_per_gpu_s": good / cost if cost > 0 else _NAN,
        "gpu_s_per_correct": cost / good if good else math.inf,
        "escalation_rate": sum(bool(r.get("escalated", False)) for r in records) / n,
    }


def _percentile(sorted_vals: list[float], q: float) -> float:
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap_ci(stat: Callable[..., float], *arrays: Sequence[Any], n_resamples: int = 10000,
                 seed: int = 0, alpha: float = 0.05,
                 strata: Sequence[Any] | None = None) -> dict[str, float]:
    """Percentile bootstrap CI of ``stat(*resampled_lists)``; items are resampled jointly
    (paired) and, if ``strata`` is given, within each stratum. Defaults match the pre-registered
    10,000 resamples with Random(0). Resamples whose statistic is NaN are dropped (count reported)."""
    if not arrays:
        raise ValueError("at least one array required")
    cols = [list(a) for a in arrays]
    n = len(cols[0])
    if any(len(c) != n for c in cols):
        raise ValueError("arrays must have equal length")
    point = float(stat(*cols))
    if n == 0:
        return {"point": point, "lo": _NAN, "hi": _NAN, "n_valid": 0}
    if strata is None:
        groups = [list(range(n))]
    else:
        if len(strata) != n:
            raise ValueError("strata length mismatch")
        by: dict[Any, list[int]] = {}
        for i, s in enumerate(strata):
            by.setdefault(s, []).append(i)
        groups = [by[k] for k in sorted(by, key=repr)]
    rng = random.Random(seed)
    vals = []
    for _ in range(n_resamples):
        idx = [g[rng.randrange(len(g))] for g in groups for _ in g]
        v = float(stat(*[[c[i] for i in idx] for c in cols]))
        if v == v:
            vals.append(v)
    if not vals:
        return {"point": point, "lo": _NAN, "hi": _NAN, "n_valid": 0}
    vals.sort()
    return {"point": point, "lo": _percentile(vals, 100 * alpha / 2),
            "hi": _percentile(vals, 100 * (1 - alpha / 2)), "n_valid": len(vals)}


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def mcnemar_exact(b: int, c: int, alternative: str = "two-sided") -> float:
    """Exact McNemar on discordant counts: b = A-only events, c = B-only events.
    ``greater`` tests b > c; ``less`` tests b < c."""
    n = b + c
    if n == 0:
        return 1.0

    def cdf(k: int) -> float:  # P(X <= k), X ~ Bin(n, 1/2)
        return sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n

    if alternative == "greater":
        return min(1.0, 1.0 - cdf(b - 1)) if b > 0 else 1.0
    if alternative == "less":
        return min(1.0, cdf(b))
    return min(1.0, 2.0 * cdf(min(b, c)))
