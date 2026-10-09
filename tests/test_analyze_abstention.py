"""Known-answer tests for scripts/analyze_abstention.py (A11)."""
import hashlib
import math

import numpy as np
import pytest

from scripts import analyze_abstention as AB


def test_fold_is_deterministic_hash_not_rng():
    c = "py/MBPP/271"
    assert AB.fold_of(c) == int(hashlib.sha256(c.encode()).hexdigest()[:8], 16) % 5
    assert AB.fold_of(c) == AB.fold_of(c) and 0 <= AB.fold_of(c, 4) < 4
    counts = np.bincount([AB.fold_of(f"cluster-{i}") for i in range(2000)], minlength=5)
    assert counts.min() > 300  # roughly balanced


def test_choose_threshold_known_answers():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1, 1, 1, 0, 1], dtype=bool)
    assert AB.choose_threshold(p, y, alpha=0.25, min_n=3) == 0.5  # k=5 has risk 0.2 <= 0.25 -> answer all
    assert AB.choose_threshold(p, y, alpha=0.10, min_n=3) == 0.7  # only the first 3 (risk 0)
    assert AB.choose_threshold(p, y, alpha=0.10, min_n=4) == float("inf")  # nothing qualifies with >= 4 answered
    assert AB.choose_threshold(np.array([]), np.array([], dtype=bool), 0.1) == float("inf")


def test_choose_threshold_does_not_split_ties():
    p = np.array([0.9, 0.5, 0.5, 0.5])
    y = np.array([1, 1, 0, 1], dtype=bool)
    # prefix k=2 would split the 0.5 tie group (and has risk 0); it must not be chosen -> k=1 or k=4
    thr = AB.choose_threshold(p, y, alpha=0.0, min_n=1)
    assert thr == 0.9  # k=1 (risk 0); k=4 has risk 0.25 > 0
    assert AB.choose_threshold(p, y, alpha=0.25, min_n=1) == 0.5  # whole tie group answered


def test_selective_stats():
    ans = np.array([True, True, False, True])
    cor = np.array([True, False, True, True])
    assert AB.selective_stats(ans, cor) == pytest.approx((0.75, 1 / 3))
    cov, risk = AB.selective_stats(np.zeros(3, dtype=bool), np.ones(3, dtype=bool))
    assert cov == 0.0 and math.isnan(risk)


def test_logprob_features():
    f = AB.logprob_features([-0.1, -0.3, -2.0, -0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1], 10, False)
    assert f["mean_lp"] == pytest.approx(-0.32) and f["min_lp"] == -2.0
    assert f["low10_lp"] == pytest.approx(-2.0)  # lowest 10% of 10 tokens = 1 token
    assert f["log_tokens"] == pytest.approx(math.log1p(10)) and f["truncated"] == 0.0
    e = AB.logprob_features([], 0, True)
    assert math.isnan(e["mean_lp"]) and e["truncated"] == 1.0


def synthetic(n_per_domain=120, seed=0):
    rng = np.random.default_rng(seed)
    tasks, rows, gens = [], [], []
    M = "tier"
    for d in ("python", "rust", "math"):
        for i in range(n_per_domain):
            tid = f"{d}{i}"
            tasks.append({"domain": d, "task_id": tid, "cluster": f"c-{d}-{i}"})
            correct = rng.random() < 0.55
            # informative log-probs; gate verdict correlated with correctness on code, never VERIFIED on math
            mean_lp = float(rng.normal(-0.2 if correct else -0.6, 0.15))
            gate = ("VERIFIED" if correct and rng.random() < 0.8 else "FAILED" if not correct and rng.random() < 0.6 else "UNVERIFIED") if d != "math" else "UNVERIFIED"
            rows.append({"arm": "base", "model": M, "domain": d, "task_id": tid, "score": "VERIFIED" if correct else "FAILED"})
            rows.append({"arm": "gate_only", "model": M, "domain": d, "task_id": tid, "gate": gate, "score": "VERIFIED" if correct else "FAILED"})
            gens.append({"model": M, "domain": d, "task_id": tid, "temperature": 0.0, "completion_tokens": 20,
                         "done_reason": "stop", "token_logprobs": [mean_lp + float(rng.normal(0, 0.1)) for _ in range(20)]})
    return tasks, rows, gens, M


def test_end_to_end_cross_fit_is_out_of_fold_and_informative():
    tasks, rows, gens, M = synthetic()
    T = AB.Table(tasks, rows, gens, M)
    p3 = AB.oof_predictions(T, AB.FEATURES_ALL)
    assert p3.shape == (T.n,) and np.all((p3 > 0) & (p3 < 1))
    assert AB.A.auroc(p3, T.y) > 0.85  # the gate and the log-probs carry real signal in this construction
    # out-of-fold: refitting on everything and predicting in-sample gives (weakly) higher accuracy than OOF
    allidx = np.arange(T.n)
    p_in = AB.fit_predict(T, AB.FEATURES_ALL, allidx, allidx)
    assert AB.A.brier(p_in, T.y.astype(float)) <= AB.A.brier(p3, T.y.astype(float)) + 1e-9


def test_target_risk_thresholds_use_only_training_data_and_meet_a_feasible_target():
    tasks, rows, gens, M = synthetic(n_per_domain=200, seed=1)
    T = AB.Table(tasks, rows, gens, M)
    ans = AB.target_risk_answers(T, AB.FEATURES_ALL, alpha=0.10)
    cov, risk = AB.selective_stats(ans, T.y)
    assert 0.1 < cov < 1.0 and risk < 0.2  # answers a meaningful share at a much lower error than the 45% base rate


def test_analyze_runs_and_reports_all_keys():
    tasks, rows, gens, M = synthetic(n_per_domain=60, seed=2)
    T = AB.Table(tasks, rows, gens, M)
    res = AB.analyze(T, n_boot=20, seed=0)
    for k in ("disc.auroc.M3.python", "disc.aurc.M0.math", "disc.d_auroc.M3_minus_M1.pooled", "op.cov.gate.rust",
              "op.risk.M3@0.1.pooled", "op.cov.M1@0.05.math"):
        assert k in res, k
    assert res["op.cov.gate.math"]["point"] == 0.0  # the synthetic math gate never verifies


def test_table_rejects_missing_generations():
    tasks, rows, gens, M = synthetic(n_per_domain=5)
    with pytest.raises(SystemExit):
        AB.Table(tasks, rows, gens[:-1], M)
