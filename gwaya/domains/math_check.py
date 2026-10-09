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


def extract_final_answer(text: str, boxed_only: bool = False) -> tuple[str, str] | None:
    """Return (answer, method) using, in order: last \\boxed{}, last '####', last number.

    boxed_only=True is the registered scoring rule (D24): only the last \\boxed{} counts.
    """
    if not text or not text.strip():
        return None
    boxed = _last_boxed(text)
    if boxed:
        return boxed, "boxed"
    if boxed_only:
        return None
    if "####" in text:
        tail = text.rsplit("####", 1)[1].strip().splitlines()
        if tail and tail[0].strip():
            return tail[0].strip(), "hashes"
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1], "last_number"
    return None


_UNIT_TAIL = re.compile(r"^(.*\S)\s*\\(?:mbox|text|textnormal)\{\s*([^{}]*?)\s*\}(?:\^\{?[23]\}?)?$")
_NOT_UNITS = {"million", "billion", "trillion", "thousand", "hundred", "percent"}  # these change the value
_SPACING = re.compile(r"\\\\|\\(?:left|right|!|,|;|:| )")


def _braceless(s: str) -> str:
    """TeX one-token arguments: \\frac14, \\frac 59, \\frac9{19}, \\frac{9}2, \\sqrt2 -> braced forms."""
    s = re.sub(r"\\frac\s*(\{[^{}]*\})\s*(\d)", r"\\frac\1{\2}", s)
    s = re.sub(r"\\frac\s*(\d)\s*\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)
    return re.sub(r"\\sqrt\s*(\d)", r"\\sqrt{\1}", s)


def normalize_answer(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = _braceless(s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac"))
    s = re.sub(r"\^\{?\\(?:text|mbox|textnormal)\{(?:st|nd|rd|th)\}\}?", "", s)  # 5^{\text{th}}
    m = _UNIT_TAIL.match(s)  # trailing unit text: '6 \mbox{ cm}^2', '50\text{ cents}'
    if m and re.search(r"[A-Za-z]", m.group(2)) and m.group(2).lower() not in _NOT_UNITS:
        s = m.group(1)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = _SPACING.sub(lambda t: t.group(0) if t.group(0) == "\\\\" else "", s)  # '\\' = matrix row break, kept
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.rstrip(". ").strip()
    m = re.match(r"[A-Za-z]\s*(?:=|\\in(?![A-Za-z]))\s*(?=\S)", s)  # leading 'x =' / 'x \in'
    if m and "=" not in s[m.end():]:
        s = s[m.end():]
    s = re.sub(r"_\{(\d+)\}", r"_\1", s)  # base subscripts: 4210_{5} -> 4210_5
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
    # multi-letter words (e.g. "banana") are prose, not algebra: sympy would read them as free
    # symbols and return a spurious False instead of undecided
    if any(w not in ("sqrt", "pi") for w in re.findall(r"[A-Za-z]{2,}", s)):
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
    # mixed number '15 \frac{39}{40}'; without the space ('137\frac{1}{2}', as MATH writes it, or after '\ ' is
    # removed) only with a proper fraction, as a product would be written 2\cdot\frac{3}{2}
    m = re.fullmatch(r"([+-]?)(\d+)(\s*)\\frac\{(\d+)\}\{(\d+)\}", s)
    if m and int(m.group(5)) != 0 and (m.group(3) or int(m.group(4)) < int(m.group(5))):
        v = int(m.group(2)) + Fraction(int(m.group(4)), int(m.group(5)))
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


# ── structured answers: tuples, intervals, unions of intervals, matrices ─────────────────────

_OPEN, _CLOSE = "([{", ")]}"


def _split_top(s: str, sep: str) -> list[str] | None:
    """Split on `sep` at bracket depth 0; None if the brackets do not balance."""
    out, depth, start, i = [], 0, 0, 0
    while i < len(s):
        c = s[i]
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0 and s.startswith(sep, i):
            out.append(s[start:i])
            i += len(sep)
            start = i
            continue
        i += 1
    return out + [s[start:]] if depth == 0 else None


def _bracketed(s: str) -> tuple[str, str, list[str]] | None:
    """'(a, b)' / '[a, b)' -> (open, close, elements) when the first bracket closes at the end."""
    s = s.strip()
    if len(s) < 2 or s[0] not in "([" or s[-1] not in ")]":
        return None
    depth = 0
    for i, c in enumerate(s):
        depth += (c in _OPEN) - (c in _CLOSE)
        if depth == 0 and i < len(s) - 1:
            return None  # e.g. '(0,9)\cup(9,36)' or '(1)(2)'
    elems = _split_top(s[1:-1], ",")
    return (s[0], s[-1], elems) if elems and len(elems) >= 2 else None


def _structure(s: str):
    """('matrix', shape, cells) | ('seq', [(open, close, elements), ...]) or None (a scalar or unknown)."""
    m = re.fullmatch(r"\s*\\begin\{([pb]matrix)\}(.*)\\end\{\1\}\s*", s, flags=re.S)
    if m:
        rows = [r for r in m.group(2).split("\\\\") if r.strip()]
        cells = [[c for c in r.split("&")] for r in rows]
        if not cells or any(len(r) != len(cells[0]) for r in cells):
            return None
        return "matrix", (len(cells), len(cells[0])), [c for r in cells for c in r]
    parts = _split_top(s, "\\cup")
    if not parts:
        return None
    seq = [_bracketed(p) for p in parts]
    return ("seq", seq) if all(seq) else None


def _element(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    return re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", lambda m: (
        f"{m.group(1)}/{m.group(2)}" if re.fullmatch(r"-?\d+", m.group(1)) and m.group(2).isdigit()
        else m.group(0)), s)


def _elementwise(ec: list[str], eg: list[str]) -> bool | None:
    res = [_scalar_equivalent(_element(c), _element(g))[0] for c, g in zip(ec, eg)]
    return True if all(r is True for r in res) else (False if any(r is False for r in res) else None)


def _structured_equivalent(sc, sg) -> bool | None:
    """Element-wise comparison. Bracket types and element counts must match exactly, so [a,b) and (a,b)
    differ. A union is decided only when every interval matches in order; anything else about a union is
    undecided (equal sets can be written as different unions)."""
    if sc[0] != sg[0]:
        return None
    if sc[0] == "matrix":
        return _elementwise(sc[2], sg[2]) if sc[1] == sg[1] else False
    (pc, pg) = (sc[1], sg[1])
    if len(pc) == 1 and len(pg) == 1:
        (oc, cc, ec), (og, cg, eg) = pc[0], pg[0]
        if (oc, cc) != (og, cg) or len(ec) != len(eg):
            return False
        return _elementwise(ec, eg)
    if len(pc) != len(pg) or any((a[0], a[1], len(a[2])) != (b[0], b[1], len(b[2])) for a, b in zip(pc, pg)):
        return None
    return True if all(_elementwise(a[2], b[2]) is True for a, b in zip(pc, pg)) else None


# ── scalar equivalence and tolerances ─────────────────────────────────────────────────────────

_ABS_TOL_EXACT = Fraction(1, 10**6)  # a decimal against an exact integer/fraction: absolute only
_REL_TOL, _ABS_FLOOR = 1e-9, 1e-12   # everything else
_MIN_SIG_CLOSED = 7  # a decimal against a closed form (pi, sqrt 2): see _decimal_vs_closed


def _sig_digits(dec: str) -> tuple[int, int]:
    """(significant digits, digits after the point) of a plain decimal literal."""
    d = dec.lstrip("+-")
    ip, _, fp = d.partition(".")
    return len((ip + fp).lstrip("0")), len(fp)


def _decimal_vs_closed(dec: str, diff: float, value: float) -> bool:
    """A printed decimal matches a closed form only if it states >= 7 significant digits (3.14159 has 6
    and is rejected; 3.141593 is accepted) and agrees to its last stated digit: |diff| <= half a unit of
    that digit, floored at 2^-50 |value| so a program's repr() of a double still matches."""
    sig, ndp = _sig_digits(dec)
    return sig >= _MIN_SIG_CLOSED and abs(diff) <= max(0.5 * 10.0 ** -ndp, 2.0 ** -50 * abs(value))


def _scalar_equivalent(nc: str, ng: str) -> tuple[bool | None, str]:
    if not nc or not ng:
        return None, "empty"
    pc, pg = _parse_exact(nc), _parse_exact(ng)
    if pc is None or pg is None:
        # exact normalized match to the reference is sound; a mismatch cannot be judged
        return (True, "string_exact") if nc == ng else (None, "unparseable")
    (vc, kc), (vg, kg) = pc, pg
    if kc != "expr" and kg != "expr":
        if kc == kg == "decimal":
            return abs(vc - vg) <= max(Fraction(_ABS_FLOOR), Fraction(_REL_TOL) * abs(vg)), "numeric_tolerance"
        if "decimal" in (kc, kg):
            return abs(vc - vg) <= _ABS_TOL_EXACT, "numeric_tolerance"
        return vc == vg, "fraction_exact"
    import sympy
    try:
        a, b = sympy.sympify(vc), sympy.sympify(vg)
        diff = sympy.simplify(a - b)
        if diff == 0:
            return True, "sympy_simplify"
        if a.free_symbols or b.free_symbols:
            return False, "sympy_simplify"

        def cls(v, k: str) -> str:
            if k != "expr":
                return k
            return "float" if v.has(sympy.Float) else ("exact" if v.is_Rational else "closed")

        ca, cb = cls(a, kc), cls(b, kg)
        d = float(sympy.N(a - b, 30))
        if {ca, cb} == {"decimal", "exact"}:
            return abs(d) <= float(_ABS_TOL_EXACT), "sympy_numeric"
        if {ca, cb} == {"decimal", "closed"}:
            dec, other = (nc, b) if ca == "decimal" else (ng, a)
            return _decimal_vs_closed(normalize_answer(dec), d, float(sympy.N(other, 30))), "sympy_numeric"
        return abs(d) <= max(_ABS_FLOOR, _REL_TOL * abs(float(sympy.N(b, 30)))), "sympy_numeric"
    except Exception:
        return None, "sympy_error"


def answers_equivalent(candidate: str, gold: str) -> tuple[bool | None, str]:
    """(True|False|None, method). None means undecidable (unparseable): never a pass."""
    nc, ng = normalize_answer(candidate), normalize_answer(gold)
    if not nc or not ng:
        return None, "empty"
    sc, sg = _structure(nc), _structure(ng)
    if sc is not None and sg is not None:
        if nc == ng:
            return True, "string_exact"
        ok = _structured_equivalent(sc, sg)
        return ok, ("structured" if ok is not None else "structured_undecided")
    return _scalar_equivalent(nc, ng)


def check_math(response: str, gold: str, boxed_only: bool = True) -> CheckResult:
    """Fail-closed math check. By default only a final \\boxed{} answer is scored (registered rule,
    D24); boxed_only=False enables the '####' / last-number fallback for sensitivity analyses."""
    ev: dict = {"gold": gold, "boxed_only": boxed_only}
    ext = extract_final_answer(response, boxed_only=boxed_only)
    if ext is None:
        ev["reason"] = "no_boxed_answer" if boxed_only else "no_extractable_answer"
        return CheckResult("UNVERIFIED", ev)
    ans, how = ext
    ev.update(extracted=ans, extraction=how)
    ok, method = answers_equivalent(ans, gold)
    ev["method"] = method
    if ok is None:
        ev["reason"] = "unparseable_answer"
        return CheckResult("UNVERIFIED", ev)
    return CheckResult("VERIFIED" if ok else "FAILED", ev)
