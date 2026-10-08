"""Offline tests for scripts/data/build_eval_manifest.py (pure helpers only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
import build_eval_manifest as B  # noqa: E402

RUST = """}

fn main() {
    let candidate = f;
    assert_eq!(candidate(vec![1.0, 2.0], 0.3), true);
    assert_eq!(candidate(String::from("a;b)"), 'x'), false);
    assert_eq!(candidate(3), 4);
}
"""


def test_rust_split_first_assert_only():
    sp = B.rust_split_tests(RUST)
    assert sp["n_asserts"] == 3
    assert sp["preamble"] == ["let candidate = f;"]
    g = sp["gate_tests"]
    assert g.count("assert_eq!") == 1 and "vec![1.0, 2.0]" in g and g.startswith("}")
    assert B.rust_split_tests(g)["n_asserts"] == 1


def test_rust_split_string_with_semicolon_and_paren():
    sp = B.rust_split_tests(RUST)
    assert 'String::from("a;b)")' in sp["asserts"][1]


def test_rust_unparsable_is_none():
    assert B.rust_split_tests("no main here") is None
    assert B.rust_split_tests("fn main() { let x = 1; }") is None


HE_TEST = '''
import numpy as np

def assertion(out, exp, atol):
    assert out == exp


def check(candidate):
    inputs = [[[1, 2, 3]], [['hidden_input_two', 'x']], [[987654321, 123456789]]]
    results = [6, 'hidden_result_two', [1111111110, 'tail']]
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        assertion(candidate(*inp), exp, 0)

assert check(f) is None
'''


def test_python_gate_keeps_only_first_pair_literals():
    g = B.python_gate_tests_plus(HE_TEST)
    assert "inputs = [[[1, 2, 3]]]" in g and "results = [6]" in g
    # D25: no hidden element (inputs[1:], results[1:]) may appear anywhere in the gate text
    for hidden in ("hidden_input_two", "987654321", "hidden_result_two", "1111111110", "'tail'"):
        assert hidden not in g
    assert B.python_gate_leaks(g, HE_TEST) == []
    assert B.python_gate_leaks(HE_TEST, HE_TEST) != []  # the v1 behaviour (full literals) is detected
    # still executable: the first pair passes with a correct candidate
    ns: dict = {"f": sum}
    exec(g.replace("assert check(f) is None", ""), ns)
    assert ns["check"](sum) is None


def test_python_gate_fails_closed():
    assert B.python_gate_tests_plus("assert True") is None
    assert B.python_gate_tests_plus("def check(c):\n    inputs = [1]\n") is None          # no results
    assert B.python_gate_tests_plus("def ref_func(x):\n    return x\n"
                                    "def check(c):\n    inputs = [[1], [2]]\n"
                                    "    for i, inp in enumerate(inputs):\n        c(*inp) == ref_func(*inp)\n") is None
    assert B.python_gate_tests_plus("def check(c):\n    inputs = make()\n    results = [1]\n") is None
    assert B.python_gate_tests_plus("def check(c):\n    inputs = [1, 2]\n    results = [1]\n") is None
    assert B.python_gate_tests_plus("def check(c:\n") is None


def test_clusters_pair_python_and_rust():
    assert B.cluster_of("rust", "HumanEval_0_has_close_elements") == B.cluster_of("python", "HumanEval/0")
    assert B.cluster_of("rust", "mbpp_3_is_not_prime") == B.cluster_of("python", "MBPP/3")
    assert B.cluster_of("math", "test/algebra/1.json") == "MATH-500/test/algebra/1.json"


def test_apportion_sums_and_proportional():
    a = B.apportion(80, B.PROPORTION)
    assert sum(a.values()) == 80
    assert a["python"] >= a["rust"] >= a["math"] - 1
    assert sum(B.apportion(7, B.PROPORTION).values()) == 7


def test_size_binding_stage():
    r = B.size_n_total({"L2": 2.7, "L3": 1.3}, 200.0, {"L2": 3.0, "L3": 4.0})
    assert r["binding_stage"] == "L3"
    assert r["n_total"] == int(0.85 * 4 * 3600 * 1.3 / 200.0)


def test_seeded_order_deterministic():
    ids = [f"x{i}" for i in range(20)]
    assert B.seeded_order(ids, tag="a") == B.seeded_order(list(reversed(ids)), tag="a")
    assert B.seeded_order(ids, tag="a") != B.seeded_order(ids, tag="b")


def test_item_sha_ignores_whitespace():
    assert B.item_sha("a  b", "c\n d") == B.item_sha("a b", "c d")


def test_power_reproduces_registered_table():
    # registered table (alpha 0.05/3): 6 pts (8%/2%) at n=500 -> 0.99; 2 pts (3%/1%) at n=244 -> 0.17
    assert abs(B.mcnemar_power(500, 0.08, 0.02, 0.05 / 3) - 0.99) < 0.01
    assert abs(B.mcnemar_power(244, 0.03, 0.01, 0.05 / 3) - 0.17) < 0.02
    assert B.mcnemar_power(50, 0.05, 0.01, 0.025) < B.mcnemar_power(500, 0.05, 0.01, 0.025)


def test_interleave_keeps_proportions_in_prefix():
    rows = {"a": [{"i": f"a{k}"} for k in range(10)], "b": [{"i": f"b{k}"} for k in range(5)]}
    pre = B.interleave(rows)[:6]
    assert sum(r["i"].startswith("a") for r in pre) == 4
