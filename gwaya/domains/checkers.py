"""Per-domain checkers returning CheckResult{status: VERIFIED|FAILED|UNVERIFIED, evidence}.

python -> PythonCompilerOracle.verify_with_test (isolated harness)
rust   -> RustCompilerOracle.verify_with_test (executes the tests)
lean4  -> Lean4CompilerOracle (kernel check + `#print axioms` audit; statement must be preserved)
math   -> gwaya.domains.math_check (fail-closed extraction + equivalence)
"""
from __future__ import annotations

import re
from typing import Any, Callable

from gwaya.domains.math_check import check_math
from gwaya.domains.task import CheckResult, Task

_LEAN_ENV_MARKERS = ("unknown package", "unknown module prefix", "no such file or directory",
                     "object file", "unknown namespace 'mathlib'")


def _code_from(response: str, lang: str) -> str:
    from gwaya.low_tier_engine import extract_code_block
    return extract_code_block(response, lang)


def _from_oracle(res: Any, extra: dict[str, Any] | None = None) -> CheckResult:
    details = dict(getattr(res, "details", {}) or {})
    msg = res.error_message or ""
    ev: dict[str, Any] = {"compiler": res.compiler, "error": msg[:500], "details": details,
                          "latency_ms": res.latency_ms}
    ev.update(extra or {})
    if res.success:
        return CheckResult("VERIFIED", ev)
    if details.get("unverified") or msg.startswith("UNVERIFIED"):
        return CheckResult("UNVERIFIED", ev)
    return CheckResult("FAILED", ev)


def check_python(task: Task, response: str) -> CheckResult:
    tests = task.checker_payload.get("tests")
    if not tests:
        return CheckResult("UNVERIFIED", {"reason": "no_tests_in_payload"})
    code = _code_from(response, "python")
    if not code.strip():
        return CheckResult("FAILED", {"reason": "empty_response"})
    from gwaya.oracles import PythonCompilerOracle
    res = PythonCompilerOracle().verify_with_test(
        code, tests, timeout_s=float(task.checker_payload.get("timeout_s", 5.0)))
    ev = {}
    if res.details.get("total") == 0:
        # a harness that ran zero assertions proves nothing
        return CheckResult("UNVERIFIED", {"reason": "zero_tests_executed", "error": res.error_message})
    return _from_oracle(res, ev)


_RUST_ASSERT = re.compile(r"\bassert(?:_eq|_ne)?!")
_RUST_FN_WRAPPER = re.compile(r"(?:#\[test\]\s*)?\bfn\s+(main|\w+)\s*\(\s*\)\s*\{")


def _brace_body(src: str, open_idx: int) -> tuple[str, int] | None:
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[open_idx + 1:i], i + 1
    return None


def rust_test_body(tests: str) -> str:
    """The oracle pastes the spec inside its own `fn main`, so a spec that brings its own
    `fn main`/`#[test] fn` wrapper would define a nested function that never runs (a silent
    false pass). Unwrap such wrappers into plain blocks so the assertions really execute."""
    if "fn main" not in tests and "#[test]" not in tests:
        return tests
    blocks, pos = [], 0
    while True:
        m = _RUST_FN_WRAPPER.search(tests, pos)
        if not m:
            break
        got = _brace_body(tests, m.end() - 1)
        if got is None:
            break
        body, pos = got
        if m.group(1) == "main" or m.group(0).lstrip().startswith("#[test]"):
            blocks.append("{\n" + body + "\n}")
    return "\n".join(blocks)


def check_rust(task: Task, response: str) -> CheckResult:
    tests = task.checker_payload.get("tests")
    if not tests:
        return CheckResult("UNVERIFIED", {"reason": "no_tests_in_payload"})
    tests = rust_test_body(tests)
    if not _RUST_ASSERT.search(tests):
        return CheckResult("UNVERIFIED", {"reason": "no_assertions_in_tests"})
    code = _code_from(response, "rust")
    if not code.strip():
        return CheckResult("FAILED", {"reason": "empty_response"})
    from gwaya.oracles import RustCompilerOracle
    res = RustCompilerOracle().verify_with_test(
        code, tests, timeout_s=float(task.checker_payload.get("timeout_s", 10.0)))
    out = _from_oracle(res)
    if out.status == "FAILED" and "linker `cc` not found" in res.error_message:
        out = CheckResult("UNVERIFIED", {**out.evidence, "reason": "rust_linker_missing"})
    return out


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def check_lean4(task: Task, response: str) -> CheckResult:
    """The submitted source must contain the task's formal statement verbatim (whitespace-
    insensitive) so a model cannot pass by proving an easier theorem."""
    stmt = task.checker_payload.get("formal_statement")
    code = _code_from(response, "lean4")
    if not code.strip():
        return CheckResult("FAILED", {"reason": "empty_response"})
    if stmt and _squash(stmt) not in _squash(code):
        return CheckResult("FAILED", {"reason": "statement_not_preserved"})
    from gwaya.oracles import Lean4CompilerOracle
    res = Lean4CompilerOracle(timeout_s=float(task.checker_payload.get("timeout_s", 15.0))).verify_snippet(code)
    out = _from_oracle(res, {"axiom_audit": "#print axioms; non-standard axioms and sorry rejected"})
    low = (res.error_message + res.stderr + res.stdout).lower()
    if out.status == "FAILED" and any(m in low for m in _LEAN_ENV_MARKERS):
        # no pinned Mathlib project wired into the oracle yet: environment, not a refuted proof
        out = CheckResult("UNVERIFIED", {**out.evidence, "reason": "lean_environment_missing_dependency"})
    return out


def check_math_task(task: Task, response: str) -> CheckResult:
    gold = task.checker_payload.get("answer")
    if gold is None or str(gold).strip() == "":
        return CheckResult("UNVERIFIED", {"reason": "no_reference_answer"})
    return check_math(response, str(gold))


_CHECKERS: dict[str, Callable[[Task, str], CheckResult]] = {
    "python": check_python, "rust": check_rust, "lean4": check_lean4, "math": check_math_task,
}


def check(task: Task, response: str) -> CheckResult:
    """Dispatch on task.domain. Any checker exception becomes UNVERIFIED (fail-closed)."""
    try:
        return _CHECKERS[task.domain](task, response or "")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("UNVERIFIED", {"reason": "checker_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
