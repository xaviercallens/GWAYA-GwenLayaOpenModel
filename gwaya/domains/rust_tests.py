"""Splitting a Rust hidden-test suite into visible-test units, and choosing which units the gate sees.

Two suite shapes occur (docs/ANALYSIS_PLAN_E_TPU.md A15.2, A15.3):
  * E (MultiPL-E): one `fn main() { ... }` whose top-level statements are a preamble (`let candidate = f;`) and
    assertions. A unit is ONE assertion statement; the preamble is always kept.
  * R' (Exercism, scripts/data/build_rust_pool.py): module-level items followed by plain `{ ... }` blocks, one per
    upstream `#[test]` function. A unit is ONE block (it may hold several assertions or none); items are always kept.

Selection rules at a budget of k units (n units, indices 0..n-1, k <= n; units always emitted in file order):
  first       0..k-1 (the stored gate is `first`, k = 1)
  first_last  0..k-2 plus n-1 (k = 1: the last unit alone)
  longest     the k units with the most non-whitespace characters (assertion macro argument text for E, the whole
              block for R'), ties broken by file order
  spread      k = 1: floor((n-1)/2); k >= 2: floor(i (n-1)/(k-1) + 1/2), i = 0..k-1

The gate uses this only when the payload opts in with `visible_test_selection` (+ `visible_tests_k`, default 1);
without that key the payload's tests are used verbatim, exactly as before.
"""
from __future__ import annotations

import re

RULES = ("first", "first_last", "longest", "spread")
_ASSERT_START = re.compile(r"assert(?:_eq|_ne)?!")
_LEADING_WS_COMMENTS = re.compile(r"\A(?:\s|//[^\n]*\n|/\*.*?\*/)*", re.S)


# ---------------------------------------------------------------------------------------------------------------
# E shape: statements of fn main (moved verbatim from scripts/rust_gate_depth.py, A15.2; behaviour unchanged)
# ---------------------------------------------------------------------------------------------------------------
def _scan(src: str, start: int, stop_at_close: bool) -> list[tuple[int, int]]:
    """Top-level statement spans (split at `;` outside (), [], {} and literals) of src[start:]."""
    spans, depth, i, s0, n = [], 0, start, start, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            i = src.find("\n", i)
            i = n if i < 0 else i
            continue
        if src.startswith("/*", i):
            i = src.find("*/", i)
            i = n if i < 0 else i + 2
            continue
        if c == '"':
            i += 1
            while i < n and src[i] != '"':
                i += 2 if src[i] == "\\" else 1
        elif c == "'":
            m = re.match(r"'(?:\\.[^']*|[^\\'])'", src[i:])
            if m:
                i += m.end() - 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0 and stop_at_close:
                break
            depth -= 1
        elif c == ";" and depth == 0:
            spans.append((s0, i + 1))
            s0 = i + 1
        i += 1
    return spans


def split_main(tests: str) -> tuple[str, list[str], list[str], str]:
    """-> (text before main's body, preamble statements, assertion statements, text after the body)."""
    m = re.search(r"fn\s+main\s*\(\s*\)\s*\{", tests)
    if not m:
        raise ValueError("no_main")
    spans = _scan(tests, m.end(), stop_at_close=True)
    if not spans:
        raise ValueError("no_statements")
    pre, asserts = [], []
    for a, b in spans:
        stmt = tests[a:b]
        body = _LEADING_WS_COMMENTS.sub("", stmt)
        (asserts if _ASSERT_START.match(body) else pre).append(stmt)
    end_body = spans[-1][1]
    return tests[:m.end()], pre, asserts, tests[end_body:]


def _main_units(tests: str) -> tuple[list[tuple[bool, str]], str, str]:
    """-> ([(is_unit, statement) in file order], head, tail) for the E shape."""
    m = re.search(r"fn\s+main\s*\(\s*\)\s*\{", tests)
    if not m:
        raise ValueError("no_main")
    spans = _scan(tests, m.end(), stop_at_close=True)
    if not spans:
        raise ValueError("no_statements")
    segs = []
    for a, b in spans:
        stmt = tests[a:b]
        segs.append((bool(_ASSERT_START.match(_LEADING_WS_COMMENTS.sub("", stmt))), stmt))
    return segs, tests[:m.end()], tests[spans[-1][1]:]


def assert_argument_text(stmt: str) -> str:
    """The text inside the outer parentheses of an assertion macro statement."""
    body = _LEADING_WS_COMMENTS.sub("", stmt)
    m = _ASSERT_START.match(body)
    if not m:
        return body
    rest = body[m.end():].lstrip()
    if not rest or rest[0] not in "([{":
        return rest
    end = _match_close(rest, 0)
    return rest[1:end] if end is not None else rest[1:]


# ---------------------------------------------------------------------------------------------------------------
# R' shape: module-level items and plain test blocks
# ---------------------------------------------------------------------------------------------------------------
def _skip_literal(src: str, i: int) -> int:
    """If src[i] starts a comment, string, raw string or char literal, return the index just after it; else i."""
    n = len(src)
    if src.startswith("//", i):
        j = src.find("\n", i)
        return n if j < 0 else j
    if src.startswith("/*", i):
        depth, j = 0, i
        while j < n:  # Rust block comments nest
            if src.startswith("/*", j):
                depth, j = depth + 1, j + 2
            elif src.startswith("*/", j):
                depth, j = depth - 1, j + 2
                if depth == 0:
                    return j
            else:
                j += 1
        return n
    m = re.match(r'b?r(#*)"', src[i:i + 300])
    if m and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
        close = '"' + m.group(1)
        j = src.find(close, i + m.end())
        return n if j < 0 else j + len(close)
    if src[i] == '"' or src.startswith('b"', i) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
        j = i + (2 if src[i] == "b" else 1)
        while j < n and src[j] != '"':
            j += 2 if src[j] == "\\" else 1
        return j + 1
    if src[i] == "'":
        m = re.match(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'\n])'", src[i:])
        if m:
            return i + m.end()
    return i


def _match_close(src: str, open_idx: int) -> int | None:
    """Index of the bracket closing src[open_idx] (one of ([{), skipping literals and comments."""
    depth, i, n = 0, open_idx, len(src)
    while i < n:
        j = _skip_literal(src, i)
        if j != i:
            i = j
            continue
        c = src[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _block_segments(tests: str) -> list[tuple[bool, str]]:
    """Top-level segments of an R'-shape suite in file order: (is_test_block, text). Text between segments
    (whitespace, comments) is attached to the following non-block item, or dropped if only whitespace."""
    segs: list[tuple[bool, str]] = []
    i, n, pending = 0, len(tests), ""
    while i < n:
        j = i
        while j < n:  # skip whitespace and comments
            if tests[j].isspace():
                j += 1
                continue
            k = _skip_literal(tests, j)
            if k != j and tests.startswith(("//", "/*"), j):
                j = k
                continue
            break
        pending += tests[i:j]
        i = j
        if i >= n:
            break
        if tests[i] == "{":
            end = _match_close(tests, i)
            if end is None:
                raise ValueError("unbalanced_braces")
            if pending.strip():
                segs.append((False, pending))
            pending = ""
            segs.append((True, tests[i:end + 1]))
            i = end + 1
            continue
        # a module-level item: ends at `;` at depth 0 or at the `}` that brings depth back to 0
        depth, j = 0, i
        while j < n:
            k = _skip_literal(tests, j)
            if k != j:
                j = k
                continue
            c = tests[j]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth < 0:
                    raise ValueError("unbalanced_braces")
                if depth == 0 and c == "}":
                    j += 1
                    break
            elif c == ";" and depth == 0:
                j += 1
                break
            j += 1
        segs.append((False, pending + tests[i:j]))
        pending, i = "", j
    if pending.strip():
        segs.append((False, pending))
    return segs


# ---------------------------------------------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------------------------------------------
def suite_units(tests: str) -> tuple[str, list[str]]:
    """-> (shape 'main' or 'blocks', unit texts in file order)."""
    if re.search(r"fn\s+main\s*\(\s*\)\s*\{", tests):
        segs, _, _ = _main_units(tests)
        return "main", [s for u, s in segs if u]
    return "blocks", [s for u, s in _block_segments(tests) if u]


def unit_length(shape: str, unit: str) -> int:
    text = assert_argument_text(unit) if shape == "main" else unit
    return len(re.sub(r"\s+", "", text))


def select_indices(rule: str, n: int, k: int, lengths: list[int] | None = None) -> list[int]:
    if rule not in RULES:
        raise ValueError(f"unknown visible_test_selection {rule!r}")
    if k < 1 or k > n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if rule == "first":
        idx = list(range(k))
    elif rule == "first_last":
        idx = list(range(k - 1)) + [n - 1]
    elif rule == "longest":
        if lengths is None or len(lengths) != n:
            raise ValueError("longest needs one length per unit")
        idx = sorted(range(n), key=lambda i: (-lengths[i], i))[:k]
    else:  # spread
        idx = [(n - 1) // 2] if k == 1 else [int(i * (n - 1) / (k - 1) + 0.5) for i in range(k)]
    return sorted(set(idx))


def select_visible_tests(tests: str, rule: str = "first", k: int = 1) -> str:
    """The gate suite that shows k units of `tests` chosen by `rule`; non-unit code is kept in place."""
    if re.search(r"fn\s+main\s*\(\s*\)\s*\{", tests):
        segs, head, tail = _main_units(tests)
        shape = "main"
    else:
        segs, head, tail = _block_segments(tests), "", ""
        shape = "blocks"
    units = [s for u, s in segs if u]
    keep = set(select_indices(rule, len(units), k, [unit_length(shape, s) for s in units]))
    out, ui = [], 0
    for is_unit, s in segs:
        if is_unit:
            if ui in keep:
                out.append(s)
            ui += 1
        else:
            out.append(s)
    if shape == "main":
        return head + "".join(out) + tail
    return "\n".join(x.strip("\n") for x in out) + "\n"


def gate_tests_from_payload(payload: dict) -> str | None:
    """The tests the gate runs: verbatim unless the payload opts in to a selection rule."""
    tests = payload.get("tests")
    rule = payload.get("visible_test_selection")
    if not tests or rule is None:
        return tests
    return select_visible_tests(tests, str(rule), int(payload.get("visible_tests_k", 1)))
