"""Offline tests for gwaya.gwenlaya (no GPU, network, model or toolchain)."""
import math

import random
import pytest

from gwaya import gwenlaya as g
from gwaya.domains.task import CheckResult


def fake_checker(status_by_resp):
    def _c(task, response):
        return CheckResult(status_by_resp.get(response, "UNVERIFIED"), {"resp": response})
    return _c


def tier(name, resp, calls=None):
    def gen(prompt, domain):
        if calls is not None:
            calls.append(name)
        if isinstance(resp, Exception):
            raise resp
        return resp
    return g.Tier(name, gen)


def cal(tau_hi=0.8, tau_lo=0.2):
    return g.CalibrationArtifact(calibrator=g.PlattCalibrator(1.0, 0.0).to_dict(), tau_hi=tau_hi, tau_lo=tau_lo)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def test_platt_fit_improves_calibration():
    rng = random.Random(0)
    s = [rng.uniform(0.01, 0.99) for _ in range(600)]
    y = [float(rng.random() < v ** 2) for v in s]  # miscalibrated scores
    c = g.PlattCalibrator().fit(s, y)
    p = c.predict(s)
    assert sum((a - b) ** 2 for a, b in zip(p, y)) < sum((a - b) ** 2 for a, b in zip(s, y))
    assert g.calibrator_from_dict(c.to_dict()).predict(0.5)[0] == pytest.approx(c.predict(0.5)[0])


def test_platt_degenerate_single_class():
    c = g.PlattCalibrator().fit([0.2, 0.9, 0.5], [1, 1, 1])
    assert 0.5 < c.predict(0.1)[0] < 1.0


def test_isotonic_monotone_and_pav():
    c = g.IsotonicCalibrator().fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    out = c.predict([0.0, 0.15, 0.25, 0.35, 0.5])
    assert all(b >= a - 1e-12 for a, b in zip(out, out[1:]))
    assert c.predict(0.2)[0] == pytest.approx(0.5) or c.predict(0.2)[0] <= 0.5
    assert g.calibrator_from_dict(c.to_dict()).predict(0.3)[0] == pytest.approx(c.predict(0.3)[0])
    assert g.IsotonicCalibrator().predict(0.3)[0] == 0.5


def test_select_calibrator_rules():
    rng = random.Random(1)
    s = [rng.random() for _ in range(50)]
    y = [float(rng.random() < v) for v in s]
    _, rep = g.select_calibrator(s, y)
    assert rep["chosen"] == "platt" and "isotonic" not in rep["cv_brier"]  # n < 100
    s2 = [rng.random() for _ in range(400)]
    y2 = [float(v > 0.5) for v in s2]  # step function: isotonic should beat Platt in CV Brier
    model, rep2 = g.select_calibrator(s2, y2)
    assert set(rep2["cv_brier"]) == {"platt", "isotonic"} and rep2["chosen"] == "isotonic"
    assert g.select_calibrator(s2, y2, "platt")[1]["forced"] is True
    assert g.select_calibrator(s2, y2, "isotonic")[0].kind == "isotonic"
    with pytest.raises(ValueError):
        g.select_calibrator(s2, y2, "bogus")


def test_thresholds_conformal_and_lo():
    p = [0.95] * 30 + [0.5] * 10 + [0.05] * 30
    y = [1.0] * 30 + [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] + [0.0] * 30
    assert g.choose_tau_hi(p, y, alpha=0.10, min_selected=20) == 0.95  # (0+1)/(30+1) <= .1
    assert g.choose_tau_hi(p, y, alpha=0.001, min_selected=20) == math.inf
    assert g.choose_tau_hi(p, y, alpha=0.5, min_selected=100) == math.inf
    assert g.choose_tau_lo(p, y, precision=0.9, min_selected=20) == 0.05
    assert g.choose_tau_lo(p, y, precision=0.9, min_selected=31) == -math.inf


def test_fit_calibration_roundtrip(tmp_path):
    rng = random.Random(2)
    s = [rng.random() for _ in range(300)]
    y = [rng.random() < v for v in s]
    art = g.fit_calibration(s, y, tau_route=0.3)
    g.save_calibrations(tmp_path / "c.json", {"python": art, "*": art})
    back = g.load_calibrations(tmp_path / "c.json")
    assert back["python"].sha256 == art.sha256 and back["python"].tau_route == 0.3
    inf = g.CalibrationArtifact(calibrator=art.calibrator, tau_hi=math.inf, tau_lo=-math.inf)
    g.save_calibrations(tmp_path / "i.json", {"*": inf})
    assert g.load_calibrations(tmp_path / "i.json")["*"].tau_hi == math.inf
    with pytest.raises(ValueError):
        g.fit_calibration([], [])


def test_verified_beats_p_and_reports_it():
    s = g.GwenLaya([tier("a", "ok")], scorer=lambda f: 0.5, calibrations={"*": cal()},
                   checker=fake_checker({"ok": "VERIFIED"}), clock=Clock())
    r = s.answer("q", "python", {"tests": "assert 1"})
    assert r["verdict"] == "VERIFIED" and r["answer"] == "ok" and r["tier"] == "a"
    assert r["p_correct"] == pytest.approx(0.5) and r["cost_s"] == 1.0
    assert set(r) >= {"answer", "verdict", "p_correct", "evidence", "tier", "cost_s"}


def test_uncalibrated_never_likely_correct():
    s = g.GwenLaya([tier("a", "x")], scorer=lambda f: 0.99, checker=fake_checker({}))
    r = s.answer("q")
    assert r["verdict"] == "ABSTAIN" and r["p_correct"] is None and r["answer"] is None
    assert r["evidence"]["calibrated"] is False


def test_likely_correct_and_likely_wrong_by_p():
    hi = g.GwenLaya([tier("a", "x")], scorer=lambda f: 0.9, calibrations={"*": cal()}, checker=fake_checker({}))
    r = hi.answer("q")
    assert r["verdict"] == "LIKELY_CORRECT" and r["answer"] == "x" and r["answered"] is True
    lo = g.GwenLaya([tier("a", "x")], scorer=lambda f: 0.1, calibrations={"*": cal()}, checker=fake_checker({}))
    r = lo.answer("q")
    assert r["verdict"] == "LIKELY_WRONG" and r["answer"] is None and r["answered"] is False
    mid = g.GwenLaya([tier("a", "x")], scorer=lambda f: 0.5, calibrations={"*": cal()}, checker=fake_checker({}))
    assert mid.answer("q")["verdict"] == "ABSTAIN"


def test_gate_failure_withholds_even_with_high_p():
    s = g.GwenLaya([tier("a", "bad")], scorer=lambda f: 0.99, calibrations={"*": cal()},
                   checker=fake_checker({"bad": "FAILED"}))
    r = s.answer("q")
    assert r["verdict"] == "LIKELY_WRONG" and r["answer"] is None


def test_escalation_to_verified_tier_and_cost_sum():
    calls = []
    s = g.GwenLaya([tier("small", "bad", calls), tier("big", "good", calls)],
                   checker=fake_checker({"bad": "FAILED", "good": "VERIFIED"}), clock=Clock())
    r = s.answer("q")
    assert calls == ["small", "big"] and r["verdict"] == "VERIFIED" and r["tier"] == "big"
    assert r["cost_s"] == 2.0 and r["evidence"]["escalated"] is True


def test_escalate_verdict_when_blocked():
    calls = []
    s = g.GwenLaya([tier("small", "bad", calls), tier("big", "good", calls)],
                   checker=fake_checker({"bad": "FAILED"}))
    r = s.answer("q", allow_escalation=False)
    assert r["verdict"] == "ESCALATE" and calls == ["small"] and r["evidence"]["next_tier"] == "big"
    b = g.GwenLaya([tier("small", "bad"), tier("big", "good")], checker=fake_checker({"bad": "FAILED"}),
                   budget_s=0.5, clock=Clock())
    r = b.answer("q")
    assert r["verdict"] == "ESCALATE" and r["evidence"]["escalation_blocked"] == "budget_exhausted"


def test_generator_error_is_undecided_not_crash():
    s = g.GwenLaya([tier("a", RuntimeError("down"))], checker=fake_checker({}))
    r = s.answer("q")
    assert r["verdict"] == "ABSTAIN" and "down" in r["evidence"]["trajectory"][0]["generation_error"]


def test_routing_abstain_and_select():
    router = lambda p, d: {"small": 0.2, "big": 0.9}
    calls = []
    s = g.GwenLaya([tier("small", "x", calls), tier("big", "good", calls)], router=router, tau_route=0.5,
                   checker=fake_checker({"good": "VERIFIED"}))
    r = s.answer("q")
    assert calls == ["big"] and r["verdict"] == "VERIFIED"
    none = g.GwenLaya([tier("small", "x", calls)], router=lambda p, d: {"small": 0.1}, tau_route=0.5,
                      checker=fake_checker({}))
    r = none.answer("q")
    assert r["verdict"] == "ABSTAIN" and r["tier"] is None and r["cost_s"] == 0.0
    broken = g.GwenLaya([tier("small", "good")], router=lambda p, d: 1 / 0, tau_route=0.5,
                        checker=fake_checker({"good": "VERIFIED"}))
    r = broken.answer("q")
    assert r["verdict"] == "VERIFIED" and "router_error" in r["evidence"]["routing"]


def test_scorer_out_of_range_is_ignored():
    s = g.GwenLaya([tier("a", "x")], scorer=lambda f: 7.0, calibrations={"*": cal()}, checker=fake_checker({}))
    r = s.answer("q")
    assert r["verdict"] == "ABSTAIN" and "scorer_error" in r["evidence"]["trajectory"][0]


def test_bad_domain_and_real_checker_fail_closed():
    s = g.GwenLaya([tier("a", "x")])
    with pytest.raises(ValueError):
        s.answer("q", "cobol")
    # real python checker with no tests in payload cannot verify anything
    r = g.GwenLaya([tier("a", "def f():\n    return 1\n")]).answer("q", "python", {})
    assert r["verdict"] == "ABSTAIN"
    assert r["evidence"]["trajectory"][0]["gate"]["status"] == "UNVERIFIED"


def test_payload_for_and_system_env(monkeypatch):
    assert g.payload_for("python", tests="assert 1") == {"tests": "assert 1"}
    assert g.payload_for("math", answer="4") == {"answer": "4"}
    assert g.payload_for("lean4", formal_statement=" ") == {}
    monkeypatch.setenv("GWENLAYA_TIERS", "m1, m2")
    monkeypatch.delenv("GWENLAYA_CALIBRATION", raising=False)
    sysm = g.system_from_env()
    assert [t.name for t in sysm.tiers] == ["m1", "m2"]
