import numpy as np

from scripts import analyze_cascades as AC


def tier(cor, gate_ok, cost, pot=None):
    n = len(cor)
    return {"cor": np.array(cor, bool), "gate_ok": np.array(gate_ok, bool), "cost": np.array(cost, float),
            "pot": np.array(pot if pot is not None else [0.0] * n, float)}


def test_stops_at_first_verified_tier_and_charges_only_consulted_tiers():
    small = tier([1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 1, 1], pot=[0.5, 0.5, 0.5, 0.5])
    large = tier([1, 1, 0, 1], [1, 1, 0, 1], [4, 4, 4, 4], pot=[2, 2, 2, 2])
    out = AC.offline_cascade([small, large])
    assert out["answered"].tolist() == [True, True, False, True]
    assert out["escalated"].tolist() == [False, True, True, True]
    # task 0 stops at the small tier; the others pay small + large (programs included)
    assert out["cost"].tolist() == [1.5, 7.5, 7.5, 7.5]


def test_unverified_everywhere_takes_the_last_tiers_candidate_unanswered():
    small = tier([1, 1], [0, 0], [1, 1])
    large = tier([0, 1], [0, 0], [3, 3])
    out = AC.offline_cascade([small, large])
    assert out["answered"].tolist() == [False, False]
    assert out["cor"].tolist() == [False, True]  # the large tier's candidate, not the small one's


def test_three_tier_ladder_skips_tiers_once_verified():
    a, b, c = (tier([1, 0, 0], [1, 0, 0], [1, 1, 1]), tier([1, 1, 0], [1, 1, 0], [2, 2, 2]),
               tier([1, 1, 1], [1, 1, 1], [5, 5, 5]))
    out = AC.offline_cascade([a, b, c])
    assert out["cost"].tolist() == [1, 3, 8] and out["answered"].all()
