"""Math final-answer extraction and equivalence. Fail-closed: an answer that cannot be
extracted or parsed is UNVERIFIED, never accepted. sympy is optional (lazy import); without
it only plain numbers, decimals and fractions are decided."""
from __future__ import annotations

import re
from fractions import Fraction

from gwaya.domains.task import CheckResult

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
_PLAIN_NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_SAFE_EXPR = re.compile(r"^[0-9A-Za-z+\-*/^().\s]*$")
_REL_TOL = 1e-6


def _last_boxed(text: str) -> str | None:
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i].strip()
    return None  # unbalanced braces


def extract_final_answer(text: str) -> tuple[str, str] | None:
    """Return (answer, method) using, in order: last \\boxed{}, last '####', last number."""
    if not text or not text.strip():
        return None
    boxed = _last_boxed(text)
    if boxed:
        return boxed, "boxed"
    if "####" in text:
        tail = text.rsplit("####", 1)[1].strip().splitlines()
        if tail and tail[0].strip():
            return tail[0].strip(), "hashes"
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1], "last_number"
    return None


def normalize_answer(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|!|,|;|:| )", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.rstrip(". ").strip()
    if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def _to_sympy_src(s: str) -> str | None:
    prev = None
    while prev != s:  # innermost-first so nested fractions work
        prev = s
        s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\pi", "pi")
    s = s.replace("{", "(").replace("}", ")").replace("^", "**")
    if "\\" in s or not _SAFE_EXPR.match(s) or not s.strip():
        return None
    return s


def _parse_exact(s: str):
    """Return (value, kind) with kind 'exact' (Fraction) | 'decimal' (Fraction from a decimal
    literal) | 'expr' (sympy expression), or None if unparseable."""
    s = normalize_answer(s)
    if _PLAIN_NUM.match(s):
        return Fraction(s), ("decimal" if "." in s else "exact")
    m = re.fullmatch(r"([+-]?)\\frac\{(\d+)\}\{(\d+)\}", s) or re.fullmatch(r"([+-]?)(\d+)/(\d+)", s)
    if m and int(m.group(3)) != 0:
        v = Fraction(int(m.group(2)), int(m.group(3)))
        return (-v if m.group(1) == "-" else v), "exact"
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            convert_xor, implicit_multiplication_application, parse_expr, standard_transformations)
    except ImportError:
        return None
    src = _to_sympy_src(s)
    if src is None:
        return None
    try:
        expr = parse_expr(
            src,
            local_dict={"sqrt": sympy.sqrt, "pi": sympy.pi},
            global_dict={"Integer": sympy.Integer, "Float": sympy.Float, "Symbol": sympy.Symbol,
                         "Rational": sympy.Rational},
            transformations=standard_transformations + (implicit_multiplication_application, convert_xor),
        )
    except Exception:
        return None
    return expr, "expr"


def answers_equivalent(candidate: str, gold: str) -> tuple[bool | None, str]:
    """(True|False|None, method). None means undecidable (unparseable): never a pass."""
    nc, ng = normalize_answer(candidate), normalize_answer(gold)
    if not nc or not ng:
        return None, "empty"
    pc, pg = _parse_exact(nc), _parse_exact(ng)
    if pc is None or pg is None:
        # exact normalized match to the reference is sound; a mismatch cannot be judged
        return (True, "string_exact") if nc == ng else (None, "unparseable")
    (vc, kc), (vg, kg) = pc, pg
    if kc != "expr" and kg != "expr":
        if kc == "decimal" or kg == "decimal":
            tol = max(1e-9, _REL_TOL * abs(float(vg)))
            return abs(float(vc) - float(vg)) <= tol, "numeric_tolerance"
        return vc == vg, "fraction_exact"
    import sympy
    try:
        a, b = sympy.sympify(vc), sympy.sympify(vg)
        diff = sympy.simplify(a - b)
        if diff == 0:
            return True, "sympy_simplify"
        if not (a.free_symbols or b.free_symbols):
            tol = max(1e-9, _REL_TOL * abs(float(b)))
            return abs(float(a) - float(b)) <= tol, "sympy_numeric"
        return False, "sympy_simplify"
    except Exception:
        return None, "sympy_error"


def check_math(response: str, gold: str) -> CheckResult:
    ev: dict = {"gold": gold}
    ext = extract_final_answer(response)
    if ext is None:
        ev["reason"] = "no_extractable_answer"
        return CheckResult("UNVERIFIED", ev)
    ans, how = ext
    ev.update(extracted=ans, extraction=how)
    ok, method = answers_equivalent(ans, gold)
    ev["method"] = method
    if ok is None:
        ev["reason"] = "unparseable_answer"
        return CheckResult("UNVERIFIED", ev)
    return CheckResult("VERIFIED" if ok else "FAILED", ev)
