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


def test_python_gate_slices_first_pair():
    t = "inputs=[1,2]\nresults=[1,2]\nfor i,(inp,exp) in enumerate(zip(inputs, results)):\n    pass"
    assert "zip(inputs[:1], results[:1])" in B.python_gate_tests_plus(t)
    assert "enumerate(inputs[:1])" in B.python_gate_tests_plus("for i, inp in enumerate(inputs):\n pass")
    assert B.python_gate_tests_plus("assert True") is None


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
