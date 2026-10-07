"""Offline toy-case tests for gwaya.metrics_selective."""
import math

import pytest

from gwaya import metrics_selective as m


def test_ece_perfect_and_worst():
    assert m.ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0], n_bins=2)[0] == 0.0
    val, bins = m.ece([1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0], n_bins=4)
    assert val == 1.0 and sum(b["n"] for b in bins) == 4


def test_ece_equal_mass_bins_and_width():
    p = [0.05 + 0.9 * i / 19 for i in range(20)]
    _, bins = m.ece(p, [0] * 20, n_bins=5)
    assert [b["n"] for b in bins] == [4] * 5
    val, _ = m.ece([0.5, 0.5], [1, 0], n_bins=10, strategy="equal_width")
    assert val == 0.0
    with pytest.raises(ValueError):
        m.ece([0.5], [1], strategy="x")
    with pytest.raises(ValueError):
        m.ece([1.5], [1])


def test_brier():
    assert m.brier([1, 0], [1, 0]) == 0.0
    assert m.brier([0.5, 0.5], [1, 0]) == 0.25


def test_auroc_cases():
    assert m.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert m.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert m.auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
    assert math.isnan(m.auroc([0.1, 0.2], [1, 1]))


def test_risk_coverage_aurc():
    conf, ok = [0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0]
    rc = m.risk_coverage_curve(conf, ok)
    assert rc["risk"] == [0.0, 0.0, pytest.approx(1 / 3), 0.5]
    assert m.aurc(conf, ok) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    assert m.e_aurc(conf, ok) == pytest.approx(0.0)
    assert m.e_aurc([0.1, 0.2, 0.8, 0.9], ok) > 0
    assert m.risk_at_coverage(conf, ok, 0.5) == 0.0
    assert m.risk_at_coverage(conf, ok, 1.0) == 0.5
    assert m.coverage_at_risk(conf, ok, 0.0) == 0.5


def test_ties_are_pessimistic():
    # same confidence: wrong item ranked first, so risk at coverage 0.5 is 1.0
    assert m.risk_at_coverage([0.5, 0.5], [1, 0], 0.5) == 1.0


def test_confident_wrong_rate():
    assert m.confident_wrong_rate([0.9, 0.9, 0.2, 0.95], [1, 0, 0, 0], 0.8) == 0.5


def test_selective_summary():
    recs = [dict(answered=True, correct=True, cost_s=1.0, escalated=False),
            dict(answered=True, correct=False, cost_s=3.0, escalated=True),
            dict(answered=False, correct=False, cost_s=2.0, escalated=True),
            dict(answered=False, correct=True, cost_s=2.0)]
    s = m.selective_summary(recs)
    assert s["coverage"] == 0.5 and s["answered_accuracy"] == 0.5 and s["cwr"] == 0.25
    assert s["acc_per_gpu_s"] == pytest.approx(1 / 8) and s["gpu_s_per_correct"] == 8.0
    assert s["escalation_rate"] == 0.5
    assert m.selective_summary([])["n"] == 0
    assert m.selective_summary([dict(answered=True, correct=False)])["gpu_s_per_correct"] == math.inf


def test_bootstrap_ci_deterministic_and_stratified():
    y = [1, 1, 1, 0, 0, 1, 0, 1]
    a = m.bootstrap_ci(lambda v: sum(v) / len(v), y, n_resamples=200, seed=0)
    b = m.bootstrap_ci(lambda v: sum(v) / len(v), y, n_resamples=200, seed=0)
    assert a == b and a["lo"] <= a["point"] <= a["hi"] and a["point"] == 0.625
    c = m.bootstrap_ci(lambda v: sum(v) / len(v), y, n_resamples=200, strata=[0] * 4 + [1] * 4)
    assert c["n_valid"] == 200
    const = m.bootstrap_ci(lambda v: sum(v) / len(v), [1] * 5, n_resamples=50)
    assert const["lo"] == const["hi"] == 1.0
    with pytest.raises(ValueError):
        m.bootstrap_ci(lambda a, b: 0.0, [1, 2], [1])


def test_holm_and_mcnemar():
    assert m.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert m.mcnemar_exact(0, 0) == 1.0
    assert m.mcnemar_exact(10, 0, "greater") == pytest.approx(1 / 1024)
    assert m.mcnemar_exact(5, 5) == 1.0
    assert m.mcnemar_exact(0, 10, "less") == pytest.approx(1 / 1024)


def test_prereg_alias_module():
    from gwaya import selective_metrics
    assert selective_metrics.ece is m.ece
