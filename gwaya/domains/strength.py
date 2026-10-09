"""Evidence strength of a code gate: how many visible tests stand behind a VERIFIED verdict.

Measured on the stored answers of E (docs/ANALYSIS_PLAN_E_TPU.md A15.2, A17): the more tests the gate sees, the lower its
confident-wrong rate (Python 4B: 13.9 percent with 1 test, 3.6 percent with 8; Rust 9B: 6.1 percent with 1, 1.4 percent with 3),
at a coverage cost. This module counts the tests and lets a caller require a minimum; the policy is OFF unless the task
payload carries ``min_visible_tests``. The count is a static lower-bound estimate of the executed checks and says nothing
about their quality.
"""
from __future__ import annotations

import ast
import re

_RUST_ASSERT = re.compile(r"\bassert(?:_eq|_ne)?!\s*[\(\[{]")


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


def visible_test_count(domain: str, tests: str | None) -> int | None:
    """Number of visible tests behind a gate payload (None when it cannot be determined)."""
    if not tests:
        return None
    if domain == "python":
        return _python_count(tests)
    if domain == "rust":
        return len(_RUST_ASSERT.findall(tests))
    return None
