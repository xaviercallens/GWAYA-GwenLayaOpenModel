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


# ── review of c4e560d: fail-open paths that must stay closed ─────────────────────────────────

NEVER_EQUAL = [
    (r"2+3\text{i}", "5"),                      # the imaginary unit is not a unit
    (r"2\text{e}", "2"),                        # a constant is not a unit
    (r"3\text{ feet}", r"3\text{ inches}"),     # two different units
    (r"6\text{ cm}^2", r"6\text{ cm}"),         # different powers of one unit
    (r"30^\circ", r"30\text{ cm}"),
    (r"5\text{ dozen}", "5"),                   # multiplier words, singular and plural
    (r"3\text{ thousand}", "3"),
    (r"3\text{ thousands}", "3"),
    (r"2\text{ million}", "2"),
    (r"2\text{ millions}", "2"),
    (r"4\text{ hundred}", "4"),
    (r"4\text{ hundreds}", "4"),
    (r"5\text{ percent}", "5"),                 # the word percent is a multiplier, kept (undecided)
    (r"5\text{ m}", "5"),                       # single letters are never stripped
    (r"7\text{ students}", "7"),                # a count noun is not on the unit whitelist
    ("x=3", "y=3"),                             # different assigned variables
    (r"x \in [1,2]", r"y \in [1,2]"),
    ("0.0", r"\frac{1}{2000000}"),              # registered near-zero tolerance max(1e-9, 1e-6|gold|)
    ("0.0000001", "0"),
    ("0.000001", "0"),
    ("1000001.0", "1000000"),
]


@pytest.mark.parametrize("cand,gold", NEVER_EQUAL)
def test_never_equal(cand, gold):
    assert answers_equivalent(cand, gold)[0] is not True
    assert answers_equivalent(gold, cand)[0] is not True
    assert check_math(rf"\boxed{{{cand}}}", gold).status != "VERIFIED"


@pytest.mark.parametrize("cand,gold", NEVER_EQUAL)
def test_never_equal_without_sympy(no_sympy, cand, gold):
    assert answers_equivalent(cand, gold)[0] is not True
    assert answers_equivalent(gold, cand)[0] is not True


def test_unit_mismatch_is_undecided(no_sympy):
    assert answers_equivalent(r"3\text{ feet}", r"3\text{ inches}") == (None, "unit_mismatch")
    assert answers_equivalent(r"6\text{ cm}^2", r"6\text{ cm}") == (None, "unit_mismatch")


@pytest.mark.parametrize("cand,gold", [
    (r"6\text{ square centimeters}", r"6\text{ cm}^2"),   # same unit, different spelling
    (r"6\mbox{ cm}^{2}", r"6\text{ cm^2}"),
    (r"12\text{ sq ft}", r"12\text{ square feet}"),
    (r"8\text{ square units}", r"8\text{ units}^2"),
    (r"30^\circ", r"30\text{ degrees}"),
    (r"30^{\circ}", "30"),
    (r"3\text{ hours}", "3"),
    (r"3\text{ hr}", r"3\text{ hours}"),
    (r"40\text{ miles per hour}", r"40\text{ mph}"),
    (r"2\text{ dollars}", "2"),
    ("x=3", "x = 3"),                            # same variable on both sides
    (r"x \in [-2,7]", r"x\in[-2, 7]"),
    ("50\\%", "50"),                             # registered: the percent sign is dropped
])
def test_same_unit_or_variable_equivalent(no_sympy, cand, gold):
    assert answers_equivalent(cand, gold)[0] is True
    assert answers_equivalent(gold, cand)[0] is True


@pytest.mark.parametrize("cand,gold,exp", [
    ("0.0", r"\frac{1}{2000000}", False),
    ("0.0000001", "0", False),
    ("0.0000000001", "0", True),           # within the registered 1e-9 floor
    ("0.0", "0", True),
    ("0.0005000001", r"\frac{1}{2000}", True),   # 1e-10 <= max(1e-9, 5e-10)
    ("0.000501", r"\frac{1}{2000}", False),
    ("1000000.5", "1000000", False),        # registered would accept (1e-6*1e6 = 1); capped at 1e-6
])
def test_decimal_vs_exact_tolerance(no_sympy, cand, gold, exp):
    assert answers_equivalent(cand, gold)[0] is exp


def test_decimal_vs_closed_never_looser_than_registered():
    pytest.importorskip("sympy")
    assert answers_equivalent("0.0", r"\frac{\sqrt{2}}{2000000000}")[0] is False
    assert answers_equivalent("0.0000000", r"\sqrt{2}-\sqrt{2}")[0] is True  # gold simplifies to exact 0


def test_normalize_is_idempotent_and_keeps_unit_and_variable():
    for s in [r"6 \mbox{ cm}^2", "x = 5", r"x \in [-2,7]", r"30^\circ", r"\frac14", r"2+3\text{i}", r"5\text{ dozen}"]:
        n = normalize_answer(s)
        assert normalize_answer(n) == n
    assert normalize_answer("x=3") != normalize_answer("y=3")
    assert normalize_answer(r"3\text{ feet}") != normalize_answer(r"3\text{ inches}")


def test_run_study_consensus_keys_do_not_merge_different_answers():
    """scripts/run_study.py groups samples by normalize_answer (answer agreement); the D62 normalization
    may merge only notational variants of one answer."""
    from scripts.run_study import consensus_key

    def key(a: str) -> str:
        return consensus_key("math", rf"so \boxed{{{a}}}")

    different = [("x=3", "y=3"), (r"3\text{ feet}", r"3\text{ inches}"), (r"2+3\text{i}", "5"),
                 (r"5\text{ dozen}", "5"), (r"3\text{ thousands}", "3"), (r"6\text{ cm}^2", r"6\text{ cm}"),
                 ("0.0", r"\frac{1}{2000000}"), ("(1,2)", "(2,1)"), (r"30^\circ", r"30\text{ cm}")]
    for a, b in different:
        assert key(a) != key(b), (a, b)
    same = [(r"\frac14", r"\frac{1}{4}"), (r"\dfrac{3}{4}", r"\frac{3}{4}"), (r"\sqrt2", r"\sqrt{2}"),
            ("x = 3", "x=3"), (r"6\text{ cm}^2", r"6\mbox{ square centimeters}"), ("4210_{5}", "4210_5")]
    for a, b in same:
        assert key(a) == key(b), (a, b)
