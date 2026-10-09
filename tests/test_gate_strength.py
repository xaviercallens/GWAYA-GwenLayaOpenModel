import pytest

from gwaya.domains import checkers
from gwaya.domains.strength import visible_test_count
from gwaya.domains.task import CheckResult, Task


def test_counts_generated_suites_by_pairs_and_handwritten_by_asserts():
    gen = "inputs = [[1], [2], [3]]\nresults = [1, 2, 3]\nfor i, (a, b) in enumerate(zip(inputs, results)):\n    assert f(*a) == b\n"
    assert visible_test_count("python", gen) == 3
    assert visible_test_count("python", "assert f(1) == 1\nassert f(2) == 2\n") == 2
    assert visible_test_count("python", "def (") is None and visible_test_count("python", "") is None
    assert visible_test_count("rust", "fn main() { assert_eq!(f(1), 1); assert!(g()); assert_ne!(f(2), 3); }") == 3
    assert visible_test_count("math", "x") is None


def _task(domain, tests, **extra):
    return Task(domain, "t", "p", {"tests": tests, **extra})


def test_min_visible_tests_is_off_by_default_and_fails_closed_when_set(monkeypatch):
    monkeypatch.setitem(checkers._CHECKERS, "python", lambda t, r: CheckResult("VERIFIED", {"x": 1}))
    tests = "assert f(1) == 1\n"
    off = checkers.check(_task("python", tests), "r")
    assert off.status == "VERIFIED" and off.evidence["visible_tests"] == 1 and off.evidence["x"] == 1
    on = checkers.check(_task("python", tests, min_visible_tests=3), "r")
    assert on.status == "UNVERIFIED" and on.evidence["reason"] == "insufficient_visible_tests"
    assert on.evidence["visible_tests"] == 1 and on.evidence["required_visible_tests"] == 3
    enough = checkers.check(_task("python", "assert f(1)==1\nassert f(2)==2\nassert f(3)==3\n", min_visible_tests=3), "r")
    assert enough.status == "VERIFIED"


def test_unknown_count_fails_closed_when_a_minimum_is_required(monkeypatch):
    monkeypatch.setitem(checkers._CHECKERS, "python", lambda t, r: CheckResult("VERIFIED", {}))
    r = checkers.check(_task("python", "def (", min_visible_tests=1), "r")
    assert r.status == "UNVERIFIED" and r.evidence["visible_tests"] is None


def test_failed_and_other_domains_are_untouched(monkeypatch):
    monkeypatch.setitem(checkers._CHECKERS, "python", lambda t, r: CheckResult("FAILED", {"why": "wrong"}))
    r = checkers.check(_task("python", "assert f(1)==1", min_visible_tests=5), "r")
    assert r.status == "FAILED" and r.evidence["why"] == "wrong"
    monkeypatch.setitem(checkers._CHECKERS, "math", lambda t, r: CheckResult("VERIFIED", {"m": 1}))
    m = checkers.check(Task("math", "t", "p", {"answer": "4", "min_visible_tests": 9}), "r")
    assert m.status == "VERIFIED" and "visible_tests" not in m.evidence
