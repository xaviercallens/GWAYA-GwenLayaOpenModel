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


def test_rust_asserts_in_comments_and_strings_are_not_counted():
    src = ('fn main() {\n    // assert_eq!(f(9), 9);\n    /* assert!(g()); */\n'
           '    let s = "assert!(true)";\n    assert_eq!(f(1), 1);\n}\n')
    assert visible_test_count("rust", src) == 1


def test_rust_counts_test_functions_not_asserts():
    one = "#[test]\nfn t() {\n" + "".join(f"    assert_eq!(f({i}), {i});\n" for i in range(5)) + "}\n"
    assert visible_test_count("rust", one) == 1
    three = "".join(f"#[test]\nfn t{i}() {{ assert_eq!(f({i}), {i}); assert!(g()); }}\n" for i in range(3))
    assert visible_test_count("rust", three) == 3
    attrs = '#[test]\n#[should_panic(expected = "boom")]\npub fn p() { f(0); }\n#[test] fn q() { assert!(g()); }\n'
    assert visible_test_count("rust", attrs) == 2
    assert visible_test_count("rust", "// #[test]\nfn main() { assert!(a()); assert!(b()); }") == 2


def test_rust_fn_main_assert_list_still_counts_asserts():
    src = "fn main() {\n    assert_eq!(f(1), 1);\n    assert_ne!(f(2), 3);\n    assert!(g());\n    assert!(h(), \"msg\");\n}\n"
    assert visible_test_count("rust", src) == 4
