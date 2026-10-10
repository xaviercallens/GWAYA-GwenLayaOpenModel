"""A15.3: opt-in visible-test selection for the Rust gate (gwaya/domains/rust_tests.py)."""
import pytest

from gwaya.domains import rust_tests as R
from gwaya.domains.checkers import check_rust
from gwaya.domains.task import Task

MAIN = """}

fn main() {
    let candidate = f;
    assert_eq!(candidate(0), 0);
    assert_eq!(candidate(12345), 54321); // longest ; { }
    assert!(candidate(1) == 1);
    assert_ne!(candidate(20), 2);
    assert_eq!(candidate(7), 7);
}
"""

BLOCKS = """fn square(x: i32) -> i32 {
    x * x
}
use std::collections::HashMap;
{
    let input = vec![];
    assert_eq!(map(input, square), vec![]);
}
{
    let s = r#"has } and { and "quotes""#;
    assert_eq!(s.len(), 26);
}
{
    let c = '}';
    assert_eq!(c, '}');
}
{
    // no assertion in this block
    map(vec![1], |_| 2);
}
{
    let long_name_for_input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    assert_eq!(map(long_name_for_input, |x| x), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
}
"""


def _units(tests):
    return R.suite_units(tests)[1]


def test_split_both_shapes():
    shape, u = R.suite_units(MAIN)
    assert shape == "main" and len(u) == 5 and "candidate(0)" in u[0]
    shape, u = R.suite_units(BLOCKS)
    assert shape == "blocks" and len(u) == 5 and 'r#"has }' in u[1] and "'}'" in u[2]


@pytest.mark.parametrize("rule,k,want", [
    ("first", 1, [0]), ("first", 3, [0, 1, 2]),
    ("first_last", 1, [4]), ("first_last", 2, [0, 4]), ("first_last", 3, [0, 1, 4]),
    ("spread", 1, [2]), ("spread", 2, [0, 4]), ("spread", 3, [0, 2, 4]),
])
def test_select_indices_positional(rule, k, want):
    assert R.select_indices(rule, 5, k) == want


def test_select_indices_longest_ties_by_file_order():
    assert R.select_indices("longest", 4, 2, [3, 9, 9, 1]) == [1, 2]
    assert R.select_indices("longest", 4, 1, [5, 5, 5, 5]) == [0]
    with pytest.raises(ValueError):
        R.select_indices("longest", 4, 5, [1, 1, 1, 1])
    with pytest.raises(ValueError):
        R.select_indices("random", 4, 1)


def test_select_visible_tests_main_keeps_preamble_and_order():
    g = R.select_visible_tests(MAIN, "first_last", 2)
    assert "let candidate = f;" in g and "candidate(0)" in g and "candidate(7)" in g
    assert g.count("assert") == 2 and g.index("candidate(0)") < g.index("candidate(7)")
    assert "12345" in R.select_visible_tests(MAIN, "longest", 1)
    assert "".join(R.select_visible_tests(MAIN, "first", 5).split()) == "".join(MAIN.split())


def test_select_visible_tests_blocks_keeps_items():
    g = R.select_visible_tests(BLOCKS, "first_last", 2)
    assert "fn square" in g and "use std::collections::HashMap;" in g
    assert len(_units(g)) == 2 and "vec![]" in g and "long_name_for_input" in g
    assert "long_name_for_input" in R.select_visible_tests(BLOCKS, "longest", 1)
    full = R.select_visible_tests(BLOCKS, "spread", 5)
    assert "".join(full.split()) == "".join(BLOCKS.split())


def test_payload_default_is_verbatim():
    assert R.gate_tests_from_payload({"tests": MAIN}) is MAIN
    assert R.gate_tests_from_payload({"tests": MAIN, "visible_test_selection": "first", "visible_tests_k": 1}).count("assert") == 1


def test_check_rust_bad_selection_is_unverified():
    t = Task("rust", "x", "", {"tests": MAIN, "visible_test_selection": "first", "visible_tests_k": 9})
    assert check_rust(t, "```rust\nfn f(x: i32) -> i32 { x }\n```").status == "UNVERIFIED"


@pytest.mark.skipif(not __import__("shutil").which("rustc"), reason="rustc not installed")
def test_check_rust_selection_changes_what_the_gate_sees():
    tests = "}\n\nfn main() {\n    let candidate = f;\n    assert_eq!(candidate(0), 0);\n    assert_eq!(candidate(2), 4);\n    assert_eq!(candidate(3), 9);\n}\n"
    wrong = "```rust\nfn f(x: i32) -> i32 { x }\n```"  # right on the degenerate first test only
    def st(**kw):
        return check_rust(Task("rust", "t", "", {"tests": tests, "timeout_s": 30.0, **kw}), wrong).status
    assert st(visible_test_selection="first", visible_tests_k=1) == "VERIFIED"
    assert st(visible_test_selection="first_last", visible_tests_k=2) == "FAILED"
    assert st() == "FAILED"  # no key: the full payload, verbatim
