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


def _braceless(s: str) -> str:
    """TeX one-token arguments: \\frac14, \\frac 59, \\frac9{19}, \\frac{9}2, \\sqrt2 -> braced forms."""
    s = re.sub(r"\\frac\s*(\{[^{}]*\})\s*(\d)", r"\\frac\1{\2}", s)
    s = re.sub(r"\\frac\s*(\d)\s*\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)
    return re.sub(r"\\sqrt\s*(\d)", r"\\sqrt{\1}", s)


# ── normalization ─────────────────────────────────────────────────────────────────────────────
# A trailing \text{..}/\mbox{..} is split off as a unit ONLY when it is on this whitelist of physical units
# (D62). Anything else -- a single letter (\text{i}), a constant, a multiplier word (dozen, thousand(s),
# million(s), hundred(s), percent), a count noun -- is left in place, so the answer stays unparseable and is
# undecided, never silently dropped. Units are canonicalized (singular word + ^2/^3) and kept: two answers
# that both carry a unit are compared only if the units are identical.
_UNIT_ALIASES = {
    "millimeter": "mm millimeter millimeters millimetre millimetres",
    "centimeter": "cm cms centimeter centimeters centimetre centimetres",
    "meter": "meter meters metre metres",
    "kilometer": "km kms kilometer kilometers kilometre kilometres",
    "inch": "in inch inches",
    "foot": "ft foot feet",
    "yard": "yd yds yard yards",
    "mile": "mi mile miles",
    "unit": "unit units",
    "second": "sec secs second seconds",
    "minute": "min mins minute minutes",
    "hour": "hr hrs hour hours",
    "day": "day days",
    "week": "week weeks",
    "month": "month months",
    "year": "yr yrs year years",
    "cent": "cent cents",
    "dollar": "dollar dollars",
    "degree": "deg degree degrees",
    "radian": "rad rads radian radians",
    "pound": "lb lbs pound pounds",
    "ounce": "oz ounce ounces",
    "gram": "gram grams",
    "kilogram": "kg kgs kilogram kilograms",
    "liter": "liter liters litre litres",
    "milliliter": "ml milliliter milliliters millilitre millilitres",
    "gallon": "gal gallon gallons",
    "mph": "mph",
}
_UNIT = {a: canon for canon, al in _UNIT_ALIASES.items() for a in al.split()}
_POWER_UNITS = {"millimeter", "centimeter", "meter", "kilometer", "inch", "foot", "yard", "mile", "unit"}
_UNIT_TAIL = re.compile(r"^(.*\S)\s*\\(?:mbox|text|textnormal|mathrm)\{([^{}]*)\}\s*(\^\s*\{?\s*([23])\s*\}?)?$")
_DEGREE_TAIL = re.compile(r"^(.*\S)\s*\^\s*(?:\\circ|\{\s*\\circ\s*\})$")
_DOLLAR_HEAD = re.compile(r"^((?:[A-Za-z]\s*=\s*)?[+-]?)\s*\\\$\s*(?=\S)")
_PERCENT_TAIL = re.compile(r"^(.*[^\s\\])\s*\\?%$")
_ASSIGN = re.compile(r"([A-Za-z])\s*(=|\\in(?![A-Za-z]))\s*(?=\S)")
_SPACING = re.compile(r"\\\\|\\(?:left|right|!|,|;|:| )")


def _canonical_unit(text: str, outer_power: str | None) -> str | None:
    """'cm' / ' square centimeters ' / 'units^2' / 'sq. ft.' / 'miles per hour' -> canonical unit, else None.
    Only whitelisted words; never a single letter, a constant or a multiplier word."""
    t = re.sub(r"\\\s|~", " ", text).strip().lower().rstrip(".").strip()
    power = outer_power
    m = re.fullmatch(r"(.*?)\s*\^\s*\{?\s*([23])\s*\}?", t)
    if m:
        if power:
            return None
        t, power = m.group(1), m.group(2)
    words = t.replace(".", " ").split()
    if len(words) >= 2 and words[0] in ("square", "sq", "cubic", "cu"):
        if power:
            return None
        power, words = ("2" if words[0] in ("square", "sq") else "3"), words[1:]
    if len(words) == 2 and words[1] in ("squared", "cubed"):
        if power:
            return None
        power, words = ("2" if words[1] == "squared" else "3"), words[:1]
    if words == ["miles", "per", "hour"]:
        words = ["mph"]
    if len(words) != 1 or len(words[0]) < 2 or words[0] not in _UNIT:
        return None
    canon = _UNIT[words[0]]
    if power and canon not in _POWER_UNITS:
        return None  # e.g. 'hours^2': not a unit we strip
    return canon + (f"^{power}" if power else "")


def _parts(s: str) -> tuple[str, str | None, tuple[str, str] | None]:
    """Normalize one answer into (value, unit, assignment). unit is a canonical whitelisted unit or None;
    assignment is (variable, operator) for a single leading 'x =' / 'x \\in', or None. The value contains
    neither, and _parts(value) == (value, None, None)."""
    s = s.strip().strip("$").strip()
    s = _braceless(s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac"))
    s = re.sub(r"\^\{?\\(?:text|mbox|textnormal)\{(?:st|nd|rd|th)\}\}?", "", s)  # 5^{\text{th}}
    s = s.rstrip(". ").strip()
    unit, raw_tail = None, ""
    m = _UNIT_TAIL.match(s)  # trailing unit text: '6 \mbox{ cm}^2', '50\text{ cents}'
    if m:
        unit = _canonical_unit(m.group(2), m.group(4))
        if unit is not None:
            s, raw_tail = m.group(1), s[len(m.group(1)):]
    if unit is None:
        m = _DEGREE_TAIL.match(s)  # '30^\circ' carries the unit degree
        if m:
            s, unit, raw_tail = m.group(1), "degree", s[len(m.group(1)):]
    # a leading '\$' is the unit dollar and a trailing '\%' the marker percent (D62): both are recorded, so
    # '5\text{ cents}' vs '\$5' and '0.5\%' vs '0.5\text{ dollars}' are unit mismatches, while '50\%' vs '50'
    # (one side only) stays True as in the registered scorer. With a second, different unit the unit text is
    # put back, so the answer is unparseable (undecided). '\$', '\%' elsewhere (inside tuples) are dropped
    # as in the registered scorer.
    marker = None
    m = _DOLLAR_HEAD.match(s)
    if m:
        s, marker = m.group(1) + s[m.end():], "dollar"
    else:
        m = _PERCENT_TAIL.match(s)
        if m:
            s, marker = m.group(1), "percent"
    if marker is not None:
        if unit is None or unit == marker:
            unit = marker
        else:
            s, unit = s + raw_tail, None
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = _SPACING.sub(lambda t: t.group(0) if t.group(0) == "\\\\" else "", s)  # '\\' = matrix row break, kept
    # the WORD 'percent' is not a unit and is kept, so '5\text{ percent}' vs '5' stays undecided
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.rstrip(". ").strip()
    assign = None
    m = _ASSIGN.match(s)  # leading 'x =' / 'x \in', only when nothing else is assigned
    if m and "=" not in s[m.end():]:
        assign, s = (m.group(1), "=" if m.group(2) == "=" else "\\in"), s[m.end():]
    s = re.sub(r"_\{(\d+)\}", r"_\1", s)  # base subscripts: 4210_{5} -> 4210_5
    if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s, unit, assign


def _render(value: str, unit: str | None, assign: tuple[str, str] | None) -> str:
    pre = "" if assign is None else assign[0] + ("=" if assign[1] == "=" else "\\in ")
    if unit == "percent":
        return pre + value + "\\%"
    if unit == "dollar":
        return pre + "\\$" + value
    return pre + value + (f"\\text{{ {unit}}}" if unit else "")


def normalize_answer(s: str) -> str:
    """Canonical string of an answer. It keeps the canonical unit and any leading assignment, so answers with
    different units or different assigned variables never share a string (scripts/run_study.py uses it as
    the answer-agreement key). Idempotent."""
    return _render(*_parts(s))


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
    s, unit, assign = _parts(s)
    if unit is not None or assign is not None:  # answers_equivalent splits these off; a leftover one is undecided
        return None
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
    """Whitespace-free element with simple fractions as a/b. A \\frac directly after a digit is a mixed
    number ('2\\frac{1}{2}', '2 \\frac{1}{2}'): the space is kept and that \\frac is not rewritten, so the
    element goes through the same mixed-number rule as a scalar (never '21/2')."""
    s = re.sub(r"(?<=\d)\s+(?=\\frac)", "\x00", s.strip())
    s = re.sub(r"\s+", "", s).replace("\x00", " ")
    return re.sub(r"(?<![\d ])\\frac\{([^{}]+)\}\{([^{}]+)\}", lambda m: (
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

_REG_REL, _REG_FLOOR = 1e-6, 1e-9   # registered rule: |diff| <= max(1e-9, 1e-6 |gold|)
_ABS_CAP = 1e-6                      # extra cap for a decimal vs an exact value when the gold is non-zero
_REL_TOL, _ABS_FLOOR = 1e-9, 1e-12   # everything else (decimal vs decimal, expression vs expression)
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


def _registered_tol(gold: float) -> float:
    return max(_REG_FLOOR, _REG_REL * abs(gold))


def _decimal_vs_exact(diff: float, gold: float) -> bool:
    """A decimal against an exact integer/fraction (either side may be the gold). The registered rule
    max(1e-9, 1e-6 |gold|) is the outer bound (so near zero only 1e-9 is allowed: '0.0' is not
    1/2000000); for a non-zero gold the tolerance is further capped at 1e-6 absolute, so '1000001.0' is
    not 1000000. Never looser than the registered rule."""
    tol = _registered_tol(gold)
    if gold != 0:
        tol = min(tol, _ABS_CAP)
    return abs(diff) <= tol


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
            return _decimal_vs_exact(float(vc - vg), float(vg)), "numeric_tolerance"
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
            return _decimal_vs_exact(d, float(sympy.N(b, 30))), "sympy_numeric"
        if {ca, cb} == {"decimal", "closed"}:
            dec, other = (nc, b) if ca == "decimal" else (ng, a)
            gold = float(sympy.N(b, 30))
            # significant-digit rule only for a non-zero gold, and never looser than the registered rule
            ok = abs(d) <= _registered_tol(gold) and (
                gold == 0 or _decimal_vs_closed(_parts(dec)[0], d, float(sympy.N(other, 30))))
            return ok, "sympy_numeric"
        return abs(d) <= max(_ABS_FLOOR, _REL_TOL * abs(float(sympy.N(b, 30)))), "sympy_numeric"
    except Exception:
        return None, "sympy_error"


def answers_equivalent(candidate: str, gold: str) -> tuple[bool | None, str]:
    """(True|False|None, method). None means undecidable (unparseable): never a pass.

    Units: if both sides carry a (whitelisted, canonical) unit they must be identical, else undecided; a
    unit on one side only is dropped ('6 \\text{ cm}^2' vs gold '6'). Assignment: a single leading
    'x =' / 'x \\in' is dropped when the other side has none or has the same variable and operator;
    'x=3' vs 'y=3' and 'x \\in S' vs 'x = S' keep both prefixes and are undecided. A leading '\\$' is the
    unit dollar and a trailing '\\%' the marker percent."""
    (nc, uc, ac), (ng, ug, ag) = _parts(candidate), _parts(gold)
    if not nc or not ng:
        return None, "empty"
    if uc is not None and ug is not None and uc != ug:
        return None, "unit_mismatch"
    if ac is not None and ag is not None and ac != ag:  # same variable AND same operator, else keep both
        nc, ng = _render(nc, None, ac), _render(ng, None, ag)
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
