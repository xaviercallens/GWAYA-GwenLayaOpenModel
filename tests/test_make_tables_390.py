"""Helpers of scripts/make_tables_390.py that decide what an E prompt shows of the tests (D63)."""
from scripts import make_tables_390 as M

PROMPT = ("Write a function to add.\nYour code should pass this test:\nassert add(2, 3)==5\n"
          "Reply with the full Python function.")


def test_gate_assert_in_prompt():
    assert M.gate_assert_in_prompt(PROMPT, "assert add(2, 3)==5")
    assert not M.gate_assert_in_prompt(PROMPT, "assert add(2, 4)==6")
    assert not M.gate_assert_in_prompt(PROMPT, "")          # no assert line: not counted as shown
    assert not M.gate_assert_in_prompt("no test here", "assert add(2, 3)==5")


def test_first_pair_matches():
    assert M.first_pair_matches("assert add(2, 3)==5", "[2, 3]") is True
    assert M.first_pair_matches("assert add(2, 3)==5", "[2, 4]") is False
    assert M.first_pair_matches("assert math.isclose(f(1), 2.0)", "[1]") is None   # other forms are 'not parsed'
    assert M.first_pair_matches("assert add(2, 3)==5", "[x]") is None
