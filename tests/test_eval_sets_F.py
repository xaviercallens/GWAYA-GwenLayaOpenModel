"""A21: offline tests of the F-set adapters (tiny synthetic rows; no dataset download)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
import build_eval_sets_F as F  # noqa: E402

LC_PRELUDE = ("import random\nfrom typing import *\nfrom collections import *\n\ninf = float('inf')\n\n"
              "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n"
              "        self.next = next\n\ndef list_node(values: list):\n    return None\n")
LC_ROW = {
    "task_id": "add-two", "question_id": 1, "estimated_date": "2024-09-01", "prompt": LC_PRELUDE,
    "problem_description": "Return a + b.\n\nExample 1:\n\nInput: a = 1, b = 2\nOutput: 3\n",
    "starter_code": "class Solution:\n    def add(self, a: int, b: int) -> int:\n        ",
    "completion": "class Solution:\n    def add(self, a: int, b: int) -> int:\n        return a + b\n",
    "entry_point": "Solution().add",
    "test": "def check(candidate):\n    assert candidate(a = 1, b = 2) == 3\n    assert candidate(a = 40, b = 2) == 42\n",
}


def test_leetcode_row_shape_and_gate_has_one_assert():
    r = F.leetcode_row(LC_ROW)
    assert r["domain"] == "python" and r["task_id"] == "py/LeetCodeDataset/add-two"
    assert "assert candidate(a=1, b=2) == 3" in r["gate_payload"]["tests"]
    assert "40" not in r["gate_payload"]["tests"] and "40" in r["checker_payload"]["tests"]
    assert r["gate_payload"]["tests"].rstrip().endswith("assert check(Solution().add) is None")
    # the hidden tests are not in the prompt
    assert "42" not in r["prompt"]
    assert "from typing import *" in r["prompt"] and "class Solution" in r["prompt"]


def test_leetcode_gate_visible_when_example_in_prompt():
    r = F.leetcode_row(LC_ROW)
    v = F.gate_visible_in_prompt(r["prompt"], r["gate_payload"]["tests"])
    assert v["visible"] is True  # a = 1, b = 2 -> output 3 appear in the statement's example; items shorter than 2 chars ignored
    hidden_first = F.gate_visible_in_prompt(r["prompt"], "assert f([7, 8, 9]) == 24\n")
    assert hidden_first["visible"] is False


def test_visibility_none_without_items():
    assert F.gate_visible_in_prompt("p", "assert True\n")["visible"] is None


def test_trim_check_keeps_setup_before_first_assert():
    src = "def check(candidate):\n    x = 5\n    assert candidate(x) == 1\n    assert candidate(6) == 2\n"
    out = F.trim_check_to_first_assert(src)
    assert "x = 5" in out and "candidate(6)" not in out


def test_fsb_row_filters():
    base = {"id": 7, "content": "Write f.", "canonical_solution": "def f(x):\n    return x\n",
            "labels": {"programming_language": "python", "execution_language": "python"},
            "test": {"asset": None, "code": "\n#<INSERT>\n\nassert f(1) == 1\n\nassert f(2) == 2\n"}}
    row, why = F.fsb_row(base)
    assert why == "" and row["task_id"] == "py/FullStackBench/7"
    assert F.count_asserts(row["gate_payload"]["tests"]) == 1
    assert F.count_asserts(row["checker_payload"]["tests"]) == 2
    bad = {**base, "test": {"asset": None, "code": "#<INSERT>\nimport pandas as pd\nassert f(1) == 1\n"}}
    assert F.fsb_row(bad)[1].startswith("test_needs_third_party")
    assert F.fsb_row({**base, "labels": {"programming_language": "go", "execution_language": "go_test"}})[1] \
        == "not_python_python"
    assert F.fsb_row({**base, "test": {"asset": None, "code": "assert f(1)==1"}})[1] == "no_insert_marker"
    assert F.fsb_row({**base, "test": {"asset": None, "code": "#<INSERT>\nprint(f(1))\n"}})[1] == "no_top_level_assert"
    # asserts only inside a function that is called later: a truncated gate would be a bare definition
    wrapped = {**base, "test": {"asset": None, "code": "#<INSERT>\ndef check(g):\n    assert g(1) == 1\n\ncheck(f)\n"}}
    assert F.fsb_row(wrapped)[1] == "asserts_only_inside_functions"


def test_bcb_row_converts_unittest_and_gate_keeps_first_method():
    test = ("import unittest\nclass TestCases(unittest.TestCase):\n"
            "    def test_a(self):\n        self.assertEqual(task_func(1), 2)\n"
            "    def test_b(self):\n        self.assertEqual(task_func(5), 6)\n")
    r = {"task_id": "BigCodeBench/9", "libs": "['math']", "test": test, "complete_prompt": "def task_func(x):\n    pass\n",
         "code_prompt": "def task_func(x):\n", "canonical_solution": "    return x + 1\n"}
    row, why = F.bcb_row(r, 9)
    assert why == "" and row["task_id"] == "py/BigCodeBench/9"
    g = row["gate_payload"]["tests"]
    assert "test_a" in g and "test_b" not in g and "test_b" in row["checker_payload"]["tests"]
    assert F.bcb_row({**r, "libs": "['pandas']"}, 9)[1].startswith("needs_third_party")
    assert F.count_asserts(row["checker_payload"]["tests"]) == 1


def test_stub_of_empties_bodies_but_keeps_interface():
    s = F.stub_of("import math\n\nclass Solution:\n    def f(self, a):\n        return a * 2\n")
    tree = ast.parse(s)
    fn = tree.body[1].body[0]
    assert fn.name == "f" and isinstance(fn.body[0], ast.Pass)


def test_math_rows():
    m = F.math_test_row("algebra", 3, {"problem": " Solve $x$. ", "solution": "so \\boxed{\\frac{1}{2}}.", "level": "Level 2"})
    assert m["task_id"] == "ma/MATHtest/algebra/3" and m["checker_payload"] == {"answer": "\\frac{1}{2}"}
    assert m["prompt"].endswith("Put the final answer in \\boxed{}.") and m["gate_payload"] == {}
    assert F.math_test_row("algebra", 4, {"problem": "p", "solution": "no box", "level": "1"}) is None
    g = F.gsm8k_test_row(0, {"question": "How many?", "answer": "work\n#### 1,234"})
    assert g["checker_payload"]["answer"] == "1234" and g["task_id"] == "ma/gsm8k-test/0"
    assert F.gsm8k_test_row(1, {"question": "q", "answer": "no marker"}) is None
    a = F.arena_row("aime2025", "MathArena/aime_2025", {"problem_idx": 4, "problem": "Find n.", "answer": 70})
    assert a["task_id"] == "ma/comp/aime2025/4" and a["checker_payload"]["answer"] == "70"
    assert F.sanity_math(g) is None


def test_stratified_sample_is_seeded_and_proportional():
    rows = [{"task_id": f"t{i}", "k": "a" if i < 30 else "b"} for i in range(40)]
    s1 = F.stratified_sample(rows, 8, 1, lambda r: r["k"])
    s2 = F.stratified_sample(rows, 8, 1, lambda r: r["k"])
    assert s1 == s2 and len(s1) == 8
    assert sum(r["k"] == "a" for r in s1) == 6
    assert F.stratified_sample(rows, 100, 1, lambda r: r["k"]) == sorted(rows, key=lambda r: r["task_id"])
    assert F.seeded_sample(rows, 5, 2, "x") == F.seeded_sample(rows, 5, 2, "x")


@pytest.mark.skipif(not __import__("gwaya.test_harness", fromlist=["x"]).isolation_available(),
                    reason="bwrap isolation unavailable")
def test_sanity_python_end_to_end_on_synthetic_leetcode_row():
    r = F.leetcode_row(LC_ROW)
    why, info = F.sanity_python(r)
    assert why is None and info["stub_status"] != "VERIFIED"
    # a wrong reference is rejected
    bad = dict(r, _reference=r["_reference"].replace("a + b", "a - b"))
    assert F.sanity_python(bad)[0].startswith("reference_fails_hidden")


def test_hidden_assert_shown_in_prompt_is_detected():
    prompt = "Fix it.\nTraceback:\n    assert(solve() == 7)\nAssertionError\n"
    assert F.hidden_asserts_in_prompt(prompt, "assert(solve() == 7)\n") == 1
    assert F.hidden_asserts_in_prompt("Write solve().", "assert(solve() == 7)\n") == 0
    row = {"task_id": "py/x/1", "prompt": prompt, "checker_payload": {"tests": "assert(solve() == 7)\n"},
           "_reference": "def solve():\n    return 7\n"}
    assert F.sanity_python(row)[0] == "hidden_test_shown_in_prompt"


def test_hidden_must_extend_gate():
    one = {"checker_payload": {"tests": "assert f(1) == 1\n"}, "gate_payload": {"tests": "assert f(1) == 1\n"}}
    two = {"checker_payload": {"tests": "assert f(1) == 1\nassert f(2) == 2\n"},
           "gate_payload": {"tests": "assert f(1) == 1\n"}}
    assert not F.hidden_extends_gate(one) and F.hidden_extends_gate(two)
    ut_h = "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): pass\n    def test_b(self): pass\n"
    ut_g = "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): pass\n"
    assert F.hidden_extends_gate({"_unittest": True, "checker_payload": {"tests": ut_h},
                                  "gate_payload": {"tests": ut_g}})
    assert not F.hidden_extends_gate({"_unittest": True, "checker_payload": {"tests": ut_g},
                                      "gate_payload": {"tests": ut_g}})


def test_informative_gram_rejects_alphabet_and_literal_runs():
    assert not F.informative_gram("a b c d e f g h i j k l m")
    assert not F.informative_gram("7 none none none none none none none none none none none none")
    assert F.informative_gram("find the number of positive integers whose sum of digits equals the target value")


def test_count_assertions_ignores_wrapper_and_counts_unittest_calls():
    src = "def check(candidate):\n    assert candidate(1) == 1\n    assert candidate(2) == 2\n\nassert check(f) is None\n"
    assert F.count_assertions(src) == 2
    ut = ("import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(f(1), 1)\n"
          "        self.assertTrue(f(2))\n\n_bcb_result = None\nassert _bcb_result.wasSuccessful()\n")
    assert F.count_assertions(ut) == 2


def test_decontaminate_rows_flags_overlap_and_keeps_fresh():
    from gwaya import decontaminate as dc
    stmt = ("A rectangle has a perimeter of 30 units and its length is twice its width. "
            "What is the area of the rectangle in square units?")
    seen = dc.Item(dataset="E", item_id="m1", domain="math", fields=[("prompt", stmt)])
    dec = dc.Decontaminator([seen])
    dup = {"domain": "math", "task_id": "ma/x/1", "source": "s", "prompt": "p", "_problem": stmt,
           "checker_payload": {"answer": "50"}}
    new = {"domain": "math", "task_id": "ma/x/2", "source": "s", "prompt": "p",
           "_problem": "How many distinct prime factors does 2310 have, counting each one once?",
           "checker_payload": {"answer": "5"}}
    kept, drops = F.decontaminate_rows(dec, [dup, new])
    assert [r["task_id"] for r in kept] == ["ma/x/2"] and drops[0]["task_id"] == "ma/x/1"
