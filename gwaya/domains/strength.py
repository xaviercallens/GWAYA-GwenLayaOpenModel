"""Evidence strength of a code gate: how many visible tests stand behind a VERIFIED verdict.

Measured on the stored answers of E (docs/ANALYSIS_PLAN_E_TPU.md A15.2, A17): the more tests the gate sees, the lower its
confident-wrong rate (Python 4B: 13.9 percent with 1 test, 3.6 percent with 8; Rust 9B: 6.1 percent with 1, 1.4 percent with 3),
at a coverage cost. This module counts the tests and lets a caller require a minimum; the policy is OFF unless the task
payload carries ``min_visible_tests``. The count is a static lower-bound estimate of the executed checks and says nothing
about their quality.

Rust: comments and string literals are stripped first (an ``assert!`` inside ``// ...`` or ``"..."`` never runs); a spec
with ``#[test]`` functions counts one per test function (several asserts in one test are one test case, so this stays a
lower bound), otherwise the ``assert!``/``assert_eq!``/``assert_ne!`` macros are counted.
"""
from __future__ import annotations

import ast
import re

_RUST_ASSERT = re.compile(r"\bassert(?:_eq|_ne)?!\s*[\(\[{]")
# `#[test]`, optionally followed by more attributes and a visibility/async qualifier, then `fn`
_RUST_TEST_FN = re.compile(r"#\s*\[\s*test\s*\]\s*(?:#\s*\[[^\]]*\]\s*)*(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?fn\b")


def _python_count(src: str) -> int | None:
    try:
        mod = ast.parse(src)
    except SyntaxError:
        return None
    sizes: dict[str, list[int]] = {"inputs": [], "results": []}
    for node in ast.walk(mod):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in sizes:
                    sizes[t.id].append(len(node.value.elts))
    if len(sizes["inputs"]) == 1 and len(sizes["results"]) == 1 and sizes["inputs"][0] == sizes["results"][0]:
        return sizes["inputs"][0]  # generated suites: one case per input/result pair
    # hand-written suites: the assert statements (an assert inside a loop counts once, so this is a lower bound)
    return sum(isinstance(n, ast.Assert) for n in ast.walk(mod))


def _rust_count(src: str) -> int:
    from gwaya.oracles import _strip_rust_comments_and_strings
    code = _strip_rust_comments_and_strings(src)
    n_test_fns = len(_RUST_TEST_FN.findall(code))
    return n_test_fns if n_test_fns else len(_RUST_ASSERT.findall(code))


def visible_test_count(domain: str, tests: str | None) -> int | None:
    """Number of visible tests behind a gate payload (None when it cannot be determined)."""
    if not tests:
        return None
    if domain == "python":
        return _python_count(tests)
    if domain == "rust":
        return _rust_count(tests)
    return None
