"""Verdict policy of GwenLaya: weak-gate demotion (python/rust) and the Lean rules (offline: fake tiers, checker,
scorer and calibration; no toolchain, no sympy)."""
import pytest

from gwaya import gwenlaya as g
from gwaya.domains.task import CheckResult

STMT = "theorem t (n : Nat) : n + 0 = n"


def checker(by_resp):
    """by_resp: response -> (status, evidence)."""
    def _c(task, response):
        status, evidence = by_resp.get(response, ("UNVERIFIED", {}))
        return CheckResult(status, dict(evidence))
    return _c


def tier(name, resp):
    return g.Tier(name, lambda prompt, domain: resp)


def cal(tau_hi=0.8, tau_lo=0.2):
    return g.CalibrationArtifact(calibrator=g.PlattCalibrator(1.0, 0.0).to_dict(), tau_hi=tau_hi, tau_lo=tau_lo)


def system(tiers, by_resp, score=None, domain_cal="python", **kw):
    """score=None: no scorer/calibration. Platt(1, 0) is the identity, so p == score."""
    extra = {} if score is None else {"scorer": lambda f: score, "calibrations": {domain_cal: cal()}}
    return g.GwenLaya(tiers, checker=checker(by_resp), **extra, **kw)


def reasons(out):
    return [s.get("reason") for s in out["evidence"]["trajectory"]]


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(g.MIN_VISIBLE_TESTS_ENV, raising=False)


def test_policy_off_keeps_weak_verified():
    out = system([tier("4B", "a")], {"a": ("VERIFIED", {"visible_tests": 1})}).answer("q", "python")
    assert out["verdict"] == "VERIFIED" and out["answer"] == "a"
    assert out["evidence"]["policy"]["min_visible_tests"] == {}


def test_weak_gate_escalates_or_abstains_without_calibration():
    by = {"a": ("VERIFIED", {"visible_tests": 1}), "b": ("VERIFIED", {"visible_tests": 5})}
    one = system([tier("4B", "a")], by, min_visible_tests={"python": 3}).answer("q", "python")
    assert one["verdict"] == "ABSTAIN" and one["answer"] is None and reasons(one) == ["weak_gate"]
    st = one["evidence"]["trajectory"][0]
    assert st["gate_strength"] == {"visible_tests": 1, "min_required": 3}
    assert st["decision"] == "gate_unverified_and_p_unavailable_or_between_thresholds"
    assert one["evidence"]["policy"]["min_visible_tests"] == {"python": 3}
    two = system([tier("4B", "a"), tier("9B", "b")], by, min_visible_tests={"python": 3}).answer("q", "python")
    assert two["verdict"] == "VERIFIED" and two["tier"] == "9B" and reasons(two)[0] == "weak_gate"
    assert two["evidence"]["trajectory"][0]["escalated_to"] == "9B"
    blocked = system([tier("4B", "a"), tier("9B", "b")], by, min_visible_tests={"python": 3}).answer(
        "q", "python", allow_escalation=False)
    assert blocked["verdict"] == "ESCALATE" and blocked["evidence"]["next_tier"] == "9B"


def test_weak_gate_with_high_p_is_likely_correct_not_verified():
    out = system([tier("4B", "a")], {"a": ("VERIFIED", {"visible_tests": 1})}, score=0.9,
                 min_visible_tests={"python": 3}).answer("q", "python")
    assert out["verdict"] == "LIKELY_CORRECT" and out["answer"] == "a" and reasons(out) == ["weak_gate"]


def test_weak_gate_with_middle_p_is_undecided_then_escalates():
    by = {"a": ("VERIFIED", {"visible_tests": 1}), "b": ("UNVERIFIED", {})}
    out = system([tier("4B", "a"), tier("9B", "b")], by, score=0.5,
                 min_visible_tests={"python": 3}).answer("q", "python")
    t0 = out["evidence"]["trajectory"][0]
    assert t0["reason"] == "weak_gate" and t0["escalated_to"] == "9B"
    assert out["verdict"] == "ABSTAIN"


def test_weak_gate_is_wrong_only_through_tau_lo():
    out = system([tier("4B", "a")], {"a": ("VERIFIED", {"visible_tests": 1})}, score=0.1,
                 min_visible_tests={"python": 3}).answer("q", "python")
    assert out["verdict"] == "LIKELY_WRONG" and out["evidence"]["trajectory"][0]["decision"] == "p_at_or_below_tau_lo"


def test_enough_tests_is_verified_and_missing_count_is_weak():
    ok = system([tier("4B", "a")], {"a": ("VERIFIED", {"visible_tests": 3})},
                min_visible_tests={"python": 3}).answer("q", "python")
    assert ok["verdict"] == "VERIFIED"
    assert ok["evidence"]["trajectory"][0]["gate_strength"] == {"visible_tests": 3, "min_required": 3}
    for ev in ({}, {"visible_tests": None}, {"visible_tests": True}):
        out = system([tier("4B", "a")], {"a": ("VERIFIED", ev)}, min_visible_tests={"python": 3}).answer("q", "python")
        assert out["verdict"] == "ABSTAIN" and reasons(out) == ["weak_gate"]


def test_policy_applies_only_to_configured_domain():
    out = system([tier("4B", "a")], {"a": ("VERIFIED", {"visible_tests": 1})},
                 min_visible_tests={"python": 3}).answer("q", "rust")
    assert out["verdict"] == "VERIFIED"
    m = system([tier("4B", "a")], {"a": ("VERIFIED", {})}, min_visible_tests={"python": 3}).answer("q", "math")
    assert m["verdict"] == "VERIFIED"


def test_rust_weak_gate_escalates_from_2b_to_9b():
    by = {"r2": ("VERIFIED", {"visible_tests": 1}), "r9": ("VERIFIED", {"visible_tests": 2})}
    out = system([tier("2B", "r2"), tier("9B", "r9")], by, min_visible_tests={"rust": 2}).answer("q", "rust")
    assert out["verdict"] == "VERIFIED" and out["tier"] == "9B" and out["answer"] == "r9"
    assert out["evidence"]["trajectory"][0]["reason"] == "weak_gate" and out["evidence"]["escalated"] is True


def test_env_parsing(monkeypatch):
    monkeypatch.setenv(g.MIN_VISIBLE_TESTS_ENV, "3")
    assert system([tier("4B", "a")], {}).min_visible_tests == {"python": 3, "rust": 3}
    monkeypatch.setenv(g.MIN_VISIBLE_TESTS_ENV, " python=3, rust=2 ")
    s = system([tier("4B", "a")], {"a": ("VERIFIED", {"visible_tests": 2})})
    assert s.min_visible_tests == {"python": 3, "rust": 2}
    assert s.answer("q", "python")["verdict"] == "ABSTAIN" and s.answer("q", "rust")["verdict"] == "VERIFIED"
    monkeypatch.setenv(g.MIN_VISIBLE_TESTS_ENV, "rust=0")
    assert system([tier("4B", "a")], {}).min_visible_tests == {}
    monkeypatch.setenv(g.MIN_VISIBLE_TESTS_ENV, "")
    assert system([tier("4B", "a")], {}).min_visible_tests == {}
    for bad in ("three", "python=x", "python:3", "lean4=2", "python=3,python=4", "-1", "python=3,", "1.5"):
        monkeypatch.setenv(g.MIN_VISIBLE_TESTS_ENV, bad)
        with pytest.raises(ValueError):
            system([tier("4B", "a")], {})
    # an explicit mapping wins over the env and is validated the same way
    assert system([tier("4B", "a")], {}, min_visible_tests={"rust": 4}).min_visible_tests == {"rust": 4}
    with pytest.raises(ValueError):
        system([tier("4B", "a")], {}, min_visible_tests={"math": 2})


def test_lean_without_formal_statement_is_not_verified():
    out = system([tier("4B", "theorem easy : True := trivial")],
                 {"theorem easy : True := trivial": ("VERIFIED", {})}).answer("q", "lean4")
    assert out["verdict"] == "ABSTAIN" and out["answer"] is None and reasons(out) == ["lean_no_formal_statement"]
    blank = system([tier("4B", "p")], {"p": ("VERIFIED", {})}).answer("q", "lean4", {"formal_statement": "  "})
    assert blank["verdict"] != "VERIFIED"
    pol = out["evidence"]["policy"]["lean4"]
    assert pol == {"verified_requires_formal_statement": True, "likely_correct": False}


def test_lean_with_formal_statement_is_verified():
    out = system([tier("4B", "p")], {"p": ("VERIFIED", {})}).answer("q", "lean4", {"formal_statement": STMT})
    assert out["verdict"] == "VERIFIED" and out["answer"] == "p"


def test_lean_is_never_likely_correct():
    by = {"p": ("UNVERIFIED", {})}
    out = system([tier("4B", "p")], by, score=0.99, domain_cal="*").answer("q", "lean4", {"formal_statement": STMT})
    assert out["verdict"] == "ABSTAIN" and reasons(out) == ["lean_requires_kernel_verification"]
    nostmt = system([tier("4B", "p")], {"p": ("VERIFIED", {})}, score=0.99, domain_cal="*").answer("q", "lean4")
    assert nostmt["verdict"] == "ABSTAIN" and reasons(nostmt) == ["lean_requires_kernel_verification"]
    low = system([tier("4B", "p")], by, score=0.05, domain_cal="*").answer("q", "lean4", {"formal_statement": STMT})
    assert low["verdict"] == "LIKELY_WRONG"
    failed = system([tier("4B", "p")], {"p": ("FAILED", {})}).answer("q", "lean4", {"formal_statement": STMT})
    assert failed["verdict"] == "LIKELY_WRONG" and reasons(failed) == ["gate_refuted_candidate"]
    # the same calibration still gives LIKELY_CORRECT outside Lean
    py = system([tier("4B", "p")], by, score=0.99, domain_cal="*").answer("q", "python")
    assert py["verdict"] == "LIKELY_CORRECT"


def test_payload_for_min_visible_tests():
    assert g.payload_for("python", tests="assert 1") == {"tests": "assert 1"}
    assert g.payload_for("python", tests="assert 1", min_visible_tests=3) == {"tests": "assert 1", "min_visible_tests": 3}
    assert g.payload_for("rust", tests="", min_visible_tests=2) == {"min_visible_tests": 2}
    assert "min_visible_tests" not in g.payload_for("rust", tests="assert!(true);")
    for dom, k in (("math", 2), ("python", -1), ("python", "x")):
        with pytest.raises(ValueError):
            g.payload_for(dom, answer="4", tests="assert 1", min_visible_tests=k)
