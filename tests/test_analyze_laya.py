import numpy as np

from scripts import analyze_laya as L


def test_topk_risk_breaks_ties_pessimistically():
    score = np.array([0.9, 0.9, 0.5, 0.1])
    y = np.array([True, False, True, False])
    assert L.topk_risk(score, y, 2) == 0.5  # the tied wrong item is counted first
    assert L.topk_risk(score, y, 3) == 1 / 3
    assert np.isnan(L.topk_risk(score, y, 0))


def test_isotonic_is_monotone_and_clips():
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9])
    y = np.array([0, 0, 1, 0, 1, 1, 1])
    m = L.fit_isotonic(raw, y)
    out = L.apply_isotonic(m, np.array([-1.0, 0.25, 0.5, 2.0]))
    assert (np.diff(out) >= -1e-12).all() and out[0] == min(m["y"]) and out[-1] == max(m["y"])


def _tier(cor, ok, cost):
    n = len(cor)
    return {"cor": np.array(cor, bool), "gate_ok": np.array(ok, bool), "cost": np.array(cost, float), "pot": np.zeros(n)}


def test_router_policy_skips_cheap_tiers_it_rejects():
    arrays = [_tier([1, 0], [1, 1], [1, 1]), _tier([1, 1], [1, 1], [5, 5])]
    rp = np.array([[0.9, 0.9], [0.1, 0.9]])  # second task: cheap tier rejected -> straight to the large tier
    o = L.run_policy(0.5, arrays, rp)
    assert o["cost"].tolist() == [1.0, 5.0] and o["cor"].tolist() == [True, True] and o["answered"].all()


def test_choose_tau_prefers_cheapest_that_keeps_ladder_accuracy():
    arrays = [_tier([1, 0, 0, 1], [1, 0, 0, 1], [1] * 4), _tier([1, 1, 0, 1], [1, 1, 0, 1], [4] * 4)]
    rp = np.array([[0.9, 0.9], [0.2, 0.9], [0.2, 0.9], [0.9, 0.9]])
    tau, info = L.choose_tau(arrays, rp)
    assert info["any_meets_target"] and tau in L.TAU_GRID and info["grid"][str(tau)]["acc"] >= info["static_ladder_acc_C"]


def test_confirmatory_detects_a_clearly_better_score():
    rng = np.random.default_rng(1)
    n = 400
    y = rng.random(n) < 0.6
    laya = np.where(y, rng.random(n) * 0.5 + 0.5, rng.random(n) * 0.5)      # near-perfect ranking
    mlp = rng.random(n)                                                     # uninformative
    gate = rng.random(n) < 0.5                                              # uninformative gate
    dom = np.array(["math"] * n)
    clu = np.array([f"c{i}" for i in range(n)])
    res = L.confirmatory(dom, clu, y, laya, mlp, gate, rng.random(n), n_boot=300, seed=0)
    assert res["H1_supported"] and res["H3_supported"]
    assert res["H1_dAURC_laya_minus_logprob"]["point"] < 0
