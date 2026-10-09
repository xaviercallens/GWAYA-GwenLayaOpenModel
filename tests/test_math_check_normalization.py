"""Answer-equivalence normalization of the math scorer (post hoc, see docs/DEVIATIONS.md D62).

Every non-sympy case must pass without sympy: the `no_sympy` fixture hides it, so the exact path is what is
tested even on a host that has sympy. sympy-only cases use pytest.importorskip.
"""
from __future__ import annotations

import builtins

import pytest

from gwaya.domains.math_check import answers_equivalent, check_math, normalize_answer


@pytest.fixture
def no_sympy(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("sympy"):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)


# gold/candidate pairs observed in the audit of the stored E math generations
AUDIT_PAIRS = [
    (r"\frac14", "1/4"),
    (r"\frac 59", r"\frac{5}{9}"),
    (r"\frac9{19}", r"\frac{9}{19}"),
    (r"\dfrac{3}4", "3/4"),
    (r"\sqrt2", r"\sqrt{2}"),
    (r"6 \mbox{ cm}^2", "6"),
    (r"50\text{ cents}", "50"),
    ("(-1, 6)", "(-1,6)"),
    ("x=5", "5"),
    (r"x \in [-2,7]", "[-2, 7]"),
    ("4210_{5}", "4210_5"),
    (r"\begin{pmatrix} \frac{16}{49} \\ \frac{48}{49} \\ \frac{24}{49} \end{pmatrix}",
     r"\begin{pmatrix} 16/49 \\ 48/49 \\ 24/49 \end{pmatrix}"),
    (r"15 \frac{39}{40}", r"\frac{639}{40}"),
    (r"-2 \frac{1}{2}", "-5/2"),
    (r"137\frac{1}{2}", r"137 \frac{1}{2}"),  # the one right->wrong flip of the first re-score (D62)
    (r"137\ \frac{1}{2}", "275/2"),
    (r"5^{\text{th}}", "5"),
]


@pytest.mark.parametrize("cand,gold", AUDIT_PAIRS)
def test_audit_pairs_equivalent_without_sympy(no_sympy, cand, gold):
    assert answers_equivalent(cand, gold)[0] is True
    assert answers_equivalent(gold, cand)[0] is True


TRUE_NEGATIVES = [
    ("circle", "ellipse"),
    (r"[-\sqrt3,\sqrt3)", r"(-\sqrt3,\sqrt3)"),
    (r"[1,2]\cup[3,4]", r"[1,2)\cup[3,4)"),
    ("(0,36)", r"(0,9)\cup(9,36)"),
    ("(1,2)", "(1,2,3)"),
    ("(1,2)", "(2,1)"),
    (r"\begin{pmatrix} 1 \\ 2 \end{pmatrix}", r"\begin{pmatrix} 1 \\ 3 \end{pmatrix}"),
    (r"5\text{ million}", "5"),  # a multiplier word is not a unit
]


@pytest.mark.parametrize("cand,gold", TRUE_NEGATIVES)
def test_true_negatives_not_equivalent(cand, gold):
    assert answers_equivalent(cand, gold)[0] is not True
    assert answers_equivalent(gold, cand)[0] is not True
    assert check_math(rf"\boxed{{{cand}}}", gold).status != "VERIFIED"


def test_bracket_type_and_count_mismatch_are_false(no_sympy):
    assert answers_equivalent(r"[-\sqrt3,\sqrt3)", r"(-\sqrt3,\sqrt3)") == (False, "structured")
    assert answers_equivalent("[0,1]", "[0,1)")[0] is False
    assert answers_equivalent("(1,2)", "(1,2,3)")[0] is False


def test_union_is_decided_only_when_all_parts_match(no_sympy):
    assert answers_equivalent(r"(-\infty,0)\cup(0,\infty)", r"(-\infty, 0) \cup (0, \infty)")[0] is True
    assert answers_equivalent("(0,36)", r"(0,9)\cup(9,36)")[0] is None  # equal sets can be written differently
    assert answers_equivalent(r"[1,2]\cup[3,4]", r"[1,2)\cup[3,4)")[0] is None


def test_unparseable_stays_undecided(no_sympy):
    assert answers_equivalent("banana", "apple") == (None, "unparseable")
    assert answers_equivalent(r"\frac{x}{", "1")[0] is None
    r = check_math(r"so \boxed{\mathrm{some\ words}}", "12")
    assert r.status == "UNVERIFIED" and r.evidence["reason"] == "unparseable_answer"


def test_normalize_keeps_matrix_row_breaks_and_value_words():
    assert "\\\\" in normalize_answer(r"\begin{pmatrix} 1 \\ 2 \end{pmatrix}")
    assert normalize_answer(r"5\text{ million}") == "5 million"
    assert normalize_answer(r"\$1,000") == "1000"
    assert normalize_answer("x = y = 3") == "x = y = 3"  # only a single leading assignment is dropped


@pytest.mark.parametrize("cand,gold,exp", [
    ("1000001.0", "1000000", False),
    ("123456700.0", "123456789", False),
    ("3.9999999999999996", "4", True),
    ("864.000000000000", "864", True),
    ("0.3333", r"\frac{1}{3}", False),
    ("0.333333333", r"\frac{1}{3}", True),
    ("0.30000000000000004", "0.3", True),
    ("0.3000001", "0.3", False),
])
def test_numeric_tolerance(no_sympy, cand, gold, exp):
    assert answers_equivalent(cand, gold)[0] is exp


@pytest.mark.parametrize("cand,gold,exp", [
    ("3.14159", r"\pi", False),
    ("3.141593", r"\pi", True),
    ("3.141592", r"\pi", False),  # wrong last stated digit
    ("1.4142135623730951", r"\sqrt{2}", True),  # repr() of a double, as a program prints it
    ("1.4142", r"\sqrt2", False),
    (r"\sqrt2", r"\sqrt{2}", True),
    (r"2\sqrt{2}", r"\sqrt{8}", True),
    (r"15 \frac{39}{40}", r"\frac{639}{40}", True),
    (r"2\frac{3}{2}", "3", True),  # no space + improper fraction: still a product
])
def test_sympy_closed_forms(cand, gold, exp):
    pytest.importorskip("sympy")
    assert answers_equivalent(cand, gold)[0] is exp
