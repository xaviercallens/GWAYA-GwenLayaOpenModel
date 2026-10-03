"""
anse/gwaya/self_tests.py
========================
Self-generated tests for candidate selection (H1).

The model is asked for extra ``assert`` statements from the problem text and the one public example
only. Hidden tests are never an input. Generated asserts are noisy, so they are only trusted when
several independent candidates agree on them (consensus), the way CodeT-style selection works.
"""

from __future__ import annotations

import ast
import re

MAX_SELFTESTS = 8
_UNSAFE = re.compile(r"\b(import|open|exec|eval|compile|input|os|sys|subprocess|__\w+__)\b")


def build_selftest_prompt(goal: str, public_test: str | None) -> str:
    example = (
        f"Example test (already known to be correct):\n{public_test.strip()}\n"
        if public_test
        else ""
    )
    return (
        f"Task: {goal}\n{example}"
        f"Write {MAX_SELFTESTS} DIFFERENT additional assert statements that a correct implementation must satisfy. "
        "Use the same function name and calling convention as the example, plain literals only, one assert per line. "
        "Cover edge cases (empty, single element, negative, large). Output only assert lines."
    )


def parse_asserts(text: str, public_test: str | None = None) -> list[str]:
    """Extract unique, parseable, side-effect-free one-line asserts; never the public test itself."""
    body = re.sub(r"```[a-zA-Z]*", "", text)
    known = {ln.strip() for ln in (public_test or "").splitlines() if ln.strip()}
    out: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("assert ") or line in known or line in out or _UNSAFE.search(line):
            continue
        try:
            tree = ast.parse(line)
        except SyntaxError:
            continue
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Assert):
            out.append(line)
        if len(out) >= MAX_SELFTESTS:
            break
    return out


def consensus_asserts(
    asserts: list[str], pass_matrix: list[list[bool]], min_agree: int = 2
) -> list[str]:
    """Keep asserts that at least ``min_agree`` candidates pass.

    ``pass_matrix[c][a]`` is True when candidate ``c`` passes assert ``a``. An assert that no candidate
    (or only one) passes is more likely a wrong test than a unanimous bug, so it is not trusted.
    With fewer candidates than ``min_agree`` nothing is trusted: selection then rests on the public test.
    """
    if len(pass_matrix) < min_agree:
        return []
    return [
        a for i, a in enumerate(asserts) if sum(1 for row in pass_matrix if row[i]) >= min_agree
    ]
