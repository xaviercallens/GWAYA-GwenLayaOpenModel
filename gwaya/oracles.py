"""
gwaya/oracles.py
================
Compiler-in-the-Loop Oracles for GWAYA v3 MCTS and Zero-Trust Verification.

Provides deterministic compiler oracles:
  - RustCompilerOracle: verifies Rust syntax, borrow checking, and typing via rustc / cargo check.
  - Lean4CompilerOracle: verifies Lean 4 formal proofs and syntax via lean CLI.
  - CppCompilerOracle: verifies C++ syntax and type correctness via g++ / clang++.
  - GoCompilerOracle: verifies Go syntax and type correctness via go build.
  - PythonCompilerOracle: verifies AST syntax and sandboxed execution.

All compiler oracles (Lean, Rust, C++, Go) are sandboxed via bwrap when available.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import secrets
import shutil
import subprocess  # nosec B404
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from gwaya.sandbox import run_in_sandbox

logger = logging.getLogger("GwayaOracles")


@dataclass
class OracleResult:
    """Outcome of a compiler oracle invocation."""
    success: bool
    compiler: str
    error_message: str = ""
    stdout: str = ""
    stderr: str = ""
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def terminal_reward(self) -> float:
        """Returns MCTS terminal reward: positive bonus for success, -1.0 for failure."""
        return 1.0 if self.success else -1.0

    @property
    def errors(self) -> list[str]:
        return [self.error_message] if self.error_message else []

    @property
    def status(self) -> str:
        return "PASS" if self.success else "REJECTED"


_RUST_PLACEHOLDER_MACRO = re.compile(r"\b(todo|unimplemented)!\s*\(")
# A function body consisting only of panic!(...) (optionally with a trailing semicolon).
_RUST_PANIC_ONLY_BODY = re.compile(
    r"\bfn\s+\w+\s*(?:<[^>{]*>)?\s*\([^{]*?\)\s*(?:->\s*[^{]+?)?\s*(?:where[^{]+)?\{\s*panic!\s*\([^{}]*\)\s*;?\s*\}"
)


def _strip_rust_comments_and_strings(code: str) -> str:
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', code)


def rust_placeholder_violations(code: str) -> list[str]:
    """Placeholder bodies that type-check but implement nothing (todo!, unimplemented!,
    or a function whose whole body is panic!). panic! inside real logic is allowed."""
    src = _strip_rust_comments_and_strings(code)
    found: list[str] = []
    for m in _RUST_PLACEHOLDER_MACRO.finditer(src):
        found.append(f"Rust placeholder macro {m.group(1)}!()")
    if _RUST_PANIC_ONLY_BODY.search(src):
        found.append("Rust function body is only panic!()")
    return found


class RustCompilerOracle:
    """
    Evaluates Rust code candidates using `rustc` metadata emission or `cargo check`.
    Flags syntax errors, borrow checker violations, and lifetime errors in <100ms.
    """

    def __init__(self, rustc_path: str | None = None, timeout_s: float = 5.0) -> None:
        self.rustc_path = rustc_path or shutil.which("rustc") or "rustc"
        self.timeout_s = timeout_s
        self.available = bool(shutil.which(self.rustc_path))

    def verify(self, code: str) -> OracleResult:
        return self.verify_snippet(code)

    def verify_snippet(self, code: str) -> OracleResult:
        """
        Compiles a Rust snippet as a library (`--emit=metadata`) to verify type soundness
        and borrow correctness without code generation overhead, in a sandboxed environment.
        """
        if not self.available:
            return OracleResult(
                success=False,
                compiler="rustc",
                error_message="UNVERIFIED: rustc toolchain not installed (fail-closed, E=1e6)",
                latency_ms=0.0,
                details={"unverified": True, "reason": "toolchain_missing"},
            )

        t0 = time.perf_counter()

        # LT0 / probe P10: `--emit=metadata` only type-checks, so a body that is just a
        # placeholder macro compiles. Reject those before calling rustc.
        stubs = rust_placeholder_violations(code)
        if stubs:
            return OracleResult(
                success=False,
                compiler="rustc",
                error_message="STUB_DETECTED: " + "; ".join(stubs),
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                details={"reason": "stub", "violations": stubs},
            )
        # Prepare code: wrap in lib if needed (no main required)
        code_to_compile = code
        has_fn = "fn " in code
        has_main = "fn main(" in code
        if not has_fn and not has_main:
            code_to_compile = f"pub fn _gwaya_check() {{\n{code}\n}}"

        source_file = "candidate.rs"
        try:
            cmd = [
                self.rustc_path,
                "--crate-type=lib",
                "--emit=metadata",
                "--out-dir",
                "/work/out",
                f"/work/{source_file}",
            ]
            success, stdout, stderr, timed_out = run_in_sandbox(
                cmd,
                timeout_s=self.timeout_s,
                mem_mb=2048,
                cpu_s=int(self.timeout_s) + 5,
                files={source_file: code_to_compile},
            )

            if timed_out:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return OracleResult(
                    success=False,
                    compiler="rustc",
                    error_message=f"Compilation timed out after {self.timeout_s}s",
                    latency_ms=round(latency_ms, 2),
                )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            err_msg = stderr.strip() if not success else ""

            return OracleResult(
                success=success,
                compiler="rustc",
                error_message=err_msg,
                stdout=stdout,
                stderr=stderr,
                latency_ms=round(latency_ms, 2),
                details={"returncode": 0 if success else 1},
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return OracleResult(
                success=False,
                compiler="rustc",
                error_message=f"Sandbox error: {str(exc)}",
                latency_ms=round(latency_ms, 2),
            )

    def verify_with_test(self, code: str, test_spec: str, timeout_s: float = 5.0) -> OracleResult:
        """
        Compiles and executes Rust candidate with test assertions.
        Test spec should be Rust code with assert!() calls.
        Success requires all assertions to pass and proper nonce-tagged result output.
        """
        from gwaya.ast_audit import ZeroStubAudit

        t0 = time.perf_counter()

        # Run ZeroStubAudit first
        audit_result = ZeroStubAudit.audit_rust_code(code)
        if not audit_result.is_clean:
            return OracleResult(
                success=False,
                compiler="rustc",
                error_message="STUB_DETECTED: " + "; ".join(audit_result.violations),
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                details={"reason": "stub", "violations": audit_result.violations},
            )

        if not self.available:
            return OracleResult(
                success=False,
                compiler="rustc",
                error_message="UNVERIFIED: rustc toolchain not installed (fail-closed, E=1e6)",
                latency_ms=0.0,
                details={"unverified": True, "reason": "toolchain_missing"},
            )

        # Count assertions in test spec
        test_count = test_spec.count("assert!")
        if test_count == 0:
            return OracleResult(
                success=False,
                compiler="rustc",
                error_message="INVALID_SPEC: test spec contains no assert!() calls",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                details={"reason": "no_asserts"},
            )

        nonce = secrets.token_hex(16)
        rust_program = _build_rust_test_program(code, test_spec, nonce)

        try:
            # Compile and run in a single bash command within the sandbox
            bash_cmd = f"{self.rustc_path} -o /work/test /work/main.rs && /work/test"
            success, stdout, stderr, timed_out = run_in_sandbox(
                ["/bin/bash", "-c", bash_cmd],
                timeout_s=timeout_s,
                mem_mb=2048,
                cpu_s=int(timeout_s) + 5,
                files={"main.rs": rust_program},
            )

            if timed_out:
                return OracleResult(
                    success=False,
                    compiler="rustc",
                    error_message=f"Rust compilation/execution timed out after {timeout_s}s",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"reason": "timeout", "total": test_count},
                )

            # Check if compilation/execution succeeded
            if not success:
                if "error" in stderr.lower() or "cannot find" in stderr.lower():
                    return OracleResult(
                        success=False,
                        compiler="rustc",
                        error_message=f"Compilation failed: {stderr.strip()[:500]}",
                        latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                        details={"reason": "compilation_failed"},
                    )
                # If it failed but looks like a runtime issue, check for panic
                return OracleResult(
                    success=False,
                    compiler="rustc",
                    error_message="Test assertion failed (panicked)",
                    stderr=stderr[:500] if stderr else "",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"reason": "assertion_failed"},
                )

            # Parse the nonce-tagged result from stdout
            result = _parse_rust_test_result(stdout, nonce)
            if not result:
                return OracleResult(
                    success=False,
                    compiler="rustc",
                    error_message="No nonce-tagged result found in test output",
                    stdout=stdout[-500:] if stdout else "",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"reason": "missing_result"},
                )

            passed = result.get("passed", 0)
            total = result.get("total", test_count)

            if passed == total and total == test_count:
                return OracleResult(
                    success=True,
                    compiler="rustc",
                    stdout=stdout[-500:] if stdout else "",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"passed": passed, "total": total, "reason": "all_tests_passed"},
                )
            else:
                failure_msg = result.get("first_failure", "Unknown assertion failure")
                return OracleResult(
                    success=False,
                    compiler="rustc",
                    error_message=f"Test assertion failed: {failure_msg[:200]}",
                    stdout=stdout[-500:] if stdout else "",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"passed": passed, "total": total, "reason": "assertion_failed"},
                )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return OracleResult(
                success=False,
                compiler="rustc",
                error_message=f"Sandbox error: {str(exc)}",
                latency_ms=round(latency_ms, 2),
            )


class Lean4CompilerOracle:
    def verify(self, code: str) -> OracleResult:
        return self.verify_snippet(code)

    """
    Evaluates Lean 4 candidates using `lean` CLI in a sandboxed environment.
    Enforces that theorems compile cleanly and verifies whether unacknowledged `sorry` exists.
    Prevents arbitrary code execution via #eval, #reduce, and other IO tactics.
    """

    def __init__(
        self,
        lean_path: str | None = None,
        timeout_s: float = 15.0,
        project_dir: str | os.PathLike | None = None,
        mem_mb: int = 2048,
    ) -> None:
        """
        project_dir: a pinned Lean+Mathlib lake project (see gwaya/lean_project.py). When given,
        the project's toolchain `lean` is used with LEAN_PATH pointing at its built oleans, the
        project is mounted read-only in the sandbox and /work is the only writable path. If the
        project is absent or unbuilt the oracle is unavailable (fail-closed UNVERIFIED), it never
        falls back to a Mathlib-less lean. Mathlib imports are slow: pass a larger timeout_s.
        """
        self.project = None
        self.mem_mb = mem_mb
        if project_dir is not None:
            from .lean_project import discover

            self.project = discover(project_dir)
            self.lean_path = str(self.project.lean_bin) if self.project else "lean"
            self.timeout_s = timeout_s
            self.available = self.project is not None
            return
        if lean_path:
            self.lean_path = lean_path
        else:
            # Search in PATH, then ELAN_HOME, then ~/.elan/bin
            elan_home = os.environ.get("ELAN_HOME", os.path.expanduser("~/.elan"))
            elan_paths = f"{elan_home}/bin"
            self.lean_path = (
                shutil.which("lean") or
                shutil.which("lean", path=elan_paths) or
                "lean"
            )
        self.timeout_s = timeout_s
        self.available = bool(shutil.which(self.lean_path)) or (os.path.isfile(self.lean_path) and os.access(self.lean_path, os.X_OK))

    STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
    _DECL_RE = re.compile(
        r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|partial|nonrec)\s+)*"
        r"(?:theorem|lemma|def|abbrev|instance|opaque)\s+([^\s(:{\[]+)",
        re.MULTILINE,
    )
    _AXIOMS_RE = re.compile(r"depends on axioms:\s*\[([^\]]*)\]")
    _ESCAPE_RE = re.compile(r"\b(sorry|admit|native_decide|axiom|unsafe|implemented_by|extern)\b")
    # Declarations without a name cannot be passed to `#print axioms`, so they are rejected (fail-closed).
    _UNAUDITABLE_RE = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)*(example\b|instance\s*[:\[{(])", re.MULTILINE)
    # Dangerous Lean constructs that could execute arbitrary code or IO
    _DANGEROUS_LEAN_RE = re.compile(
        r"(?:#eval|#reduce|run_cmd|run_elab|run_meta|"
        r"\belab\b|\bmacro\b|\bsyntax\b|"
        r"\binitialize\b|\bbuiltin_initialize\b|\bopen\s+Lean\b|"
        r"\bmacro_rules\b|\belab_rules\b|\bdeclare_syntax_cat\b|#exit\b)"
    )

    @classmethod
    def _lexical_flaws(cls, code: str) -> list[str]:
        """Defence-in-depth lexical scan (comments removed). The kernel axiom audit is authoritative."""
        stripped = re.sub(r"/-.*?-/", " ", code, flags=re.DOTALL)
        stripped = re.sub(r"--[^\n]*", " ", stripped)
        flaws = set(cls._ESCAPE_RE.findall(stripped))

        # Check for dangerous constructs that could execute arbitrary code
        if cls._DANGEROUS_LEAN_RE.search(stripped):
            dangerous = cls._DANGEROUS_LEAN_RE.findall(stripped)
            flaws.update(dangerous)

        if cls._UNAUDITABLE_RE.search(stripped):
            flaws.add("anonymous_declaration")
        return sorted(flaws)

    @classmethod
    def _axiom_probe(cls, code: str) -> str:
        """Append `#print axioms` for every named theorem so the kernel reports its axiom closure."""
        names = cls._DECL_RE.findall(code)
        return "".join(f"\n#print axioms {n}" for n in names) + ("\n" if names else "")

    @classmethod
    def _audit_count(cls, stdout: str) -> int:
        """Number of `#print axioms` answers in the output."""
        return len(re.findall(r"depends on axioms:|does not depend on any axioms", stdout))

    @classmethod
    def _disallowed_axioms(cls, stdout: str, allow_sorry: bool) -> list[str]:
        allowed = cls.STANDARD_AXIOMS | ({"sorryAx"} if allow_sorry else frozenset())
        found: set[str] = set()
        for group in cls._AXIOMS_RE.findall(stdout):
            found.update(a.strip() for a in group.split(",") if a.strip())
        return sorted(found - allowed)

    def _interpret_run(self, res: subprocess.CompletedProcess, allow_sorry: bool, latency_ms: float, expected_audits: int = 0) -> OracleResult:
        """Success requires exit status 0, an axiom closure within STANDARD_AXIOMS, and an answer
        for every `#print axioms` probe (a truncated run must not pass unaudited)."""
        success = res.returncode == 0
        err_msg = "" if success else (res.stderr.strip() or res.stdout.strip())
        bad_axioms = self._disallowed_axioms(res.stdout, allow_sorry) if success else []
        if success and self._audit_count(res.stdout) < expected_audits:
            success = False
            err_msg = f"Axiom audit incomplete: expected {expected_audits} #print axioms answers"
        elif bad_axioms:
            success = False
            err_msg = f"Proof depends on non-standard axioms {bad_axioms} (#print axioms)"
        return OracleResult(
            success=success,
            compiler="lean4",
            error_message=err_msg,
            stdout=res.stdout,
            stderr=res.stderr,
            latency_ms=round(latency_ms, 2),
            details={"returncode": res.returncode, "disallowed_axioms": bad_axioms},
        )

    def verify_snippet(self, code: str, allow_sorry: bool = False) -> OracleResult:
        """
        Runs Lean 4 on snippet in a sandboxed environment. Fails if lean exits non-zero
        or if `sorry` is present without allow_sorry=True. Prevents arbitrary code execution
        via #eval, #reduce, and other IO tactics.
        """
        t0 = time.perf_counter()

        forbidden = self._lexical_flaws(code)
        if forbidden and not (allow_sorry and set(forbidden) <= {"sorry", "admit"}):
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return OracleResult(
                success=False,
                compiler="lean4",
                error_message=f"Unsound or unauditable construct(s) {forbidden} detected in formal proof",
                latency_ms=round(latency_ms, 2),
                details={"flaw": "unauditable_construct", "forbidden": forbidden},
            )

        if not self.available:
            return OracleResult(
                success=False,
                compiler="lean4",
                error_message="UNVERIFIED: lean toolchain not installed (fail-closed, E=1e6)",
                latency_ms=0.0,
                details={"unverified": True, "reason": "toolchain_missing"},
            )

        source_file = "candidate.lean"
        source_content = code + self._axiom_probe(code)

        try:
            run_kwargs: dict = {}
            if self.project is not None:
                # Mathlib project mounted read-only; /work is the only writable path.
                cmd = [self.lean_path, "-M", str(self.mem_mb), f"/work/{source_file}"]
                run_kwargs = {
                    "ro_binds": list(self.project.ro_binds),
                    "env": {"LEAN_PATH": self.project.lean_path_env},
                }
            else:
                cmd = [self.lean_path, f"/work/{source_file}"]
            success, stdout, stderr, timed_out = run_in_sandbox(
                cmd,
                timeout_s=self.timeout_s,
                mem_mb=self.mem_mb,
                cpu_s=int(self.timeout_s) + 5,
                files={source_file: source_content},
                **run_kwargs,
            )

            if timed_out:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return OracleResult(
                    success=False,
                    compiler="lean4",
                    error_message=f"Lean 4 verification timed out after {self.timeout_s}s",
                    latency_ms=round(latency_ms, 2),
                )

            # For sandboxed execution, synthesize a CompletedProcess-like result
            class _SandboxResult:
                def __init__(self, success, stdout, stderr):
                    self.returncode = 0 if success else 1
                    self.stdout = stdout
                    self.stderr = stderr

            res = _SandboxResult(success, stdout, stderr)
            return self._interpret_run(
                res, allow_sorry, (time.perf_counter() - t0) * 1000.0, expected_audits=len(self._DECL_RE.findall(code))
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return OracleResult(
                success=False,
                compiler="lean4",
                error_message=f"Sandbox error: {str(exc)}",
                latency_ms=round(latency_ms, 2),
            )


class PythonCompilerOracle:
    def verify(self, code: str) -> OracleResult:
        return self.verify_snippet(code)

    """
    Evaluates Python candidates using AST validation and safe parsing.
    """

    def __init__(self) -> None:
        self.compiler_name = "python_ast"

    @staticmethod
    def _has_executable_code(tree: ast.AST) -> bool:
        """True if the module has at least one statement that is not a bare docstring/constant."""
        body = getattr(tree, "body", [])
        for stmt in body:
            is_constant_expr = isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            if not is_constant_expr:
                return True
        return False

    @staticmethod
    def _imported_names(node: ast.AST) -> list[str]:
        """Module names an import node refers to; relative imports keep their leading dots."""
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        if isinstance(node, ast.ImportFrom):
            if node.level:
                return ["." * node.level + (node.module or "")]
            return [node.module or ""]
        return []

    @classmethod
    def _unresolved_imports(cls, tree: ast.AST) -> list[str]:
        """Top-level module names imported by the snippet that cannot be found (phantom imports)."""
        import importlib.util

        missing: set[str] = set()
        for node in ast.walk(tree):
            for name in cls._imported_names(node):
                if name.startswith("."):
                    missing.add(name)  # relative imports cannot resolve in an isolated snippet
                    continue
                top = name.split(".")[0]
                if top and top != "__future__" and importlib.util.find_spec(top) is None:
                    missing.add(top)
        return sorted(missing)

    def verify_snippet(self, code: str) -> OracleResult:
        t0 = time.perf_counter()
        try:
            tree = ast.parse(code)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if not self._has_executable_code(tree):
                return OracleResult(
                    success=False,
                    compiler="python_ast",
                    error_message="EMPTY: candidate contains no executable code (fail-closed, E=1e6)",
                    latency_ms=round(latency_ms, 2),
                    details={"unverified": True, "reason": "empty_candidate"},
                )
            missing = self._unresolved_imports(tree)
            if missing:
                return OracleResult(
                    success=False,
                    compiler="python_ast",
                    error_message=f"UNRESOLVED_IMPORT: module(s) {missing} not installed or relative",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"reason": "unresolved_import", "modules": missing},
                )
            return OracleResult(
                success=True,
                compiler="python_ast",
                latency_ms=round(latency_ms, 2),
            )
        except SyntaxError as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return OracleResult(
                success=False,
                compiler="python_ast",
                error_message=f"SyntaxError at line {e.lineno}: {e.msg}",
                latency_ms=round(latency_ms, 2),
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return OracleResult(
                success=False,
                compiler="python_ast",
                error_message=str(e),
                latency_ms=round(latency_ms, 2),
            )

    def verify_with_test(self, code: str, test_spec: str, timeout_s: float = 5.0) -> OracleResult:
        """
        Executes the candidate and then the test spec in the isolated harness
        (`gwaya.test_harness`). Success requires a nonce-tagged harness record with every
        assert passing; the process exit code is never trusted. Without bwrap isolation the
        result is UNVERIFIED unless GWAYA_ALLOW_UNISOLATED=1 is set.
        """
        syntax_res = self.verify_snippet(code)
        if not syntax_res.success:
            return syntax_res

        from gwaya.test_harness import run_candidate_tests

        hres = run_candidate_tests(code, test_spec, timeout_s=timeout_s)
        details: dict[str, Any] = {
            "passed": hres.passed,
            "total": hres.total,
            "isolation": hres.isolation,
            "timed_out": hres.error.startswith("TIMEOUT"),
            **hres.details,
        }
        return OracleResult(
            success=hres.success,
            compiler="python_sandbox",
            error_message=hres.error,
            stdout=hres.stdout_tail,
            latency_ms=hres.duration_ms,
            details=details,
        )




class CppCompilerOracle:
    """
    Verifies C++ snippets via clang++ or g++ -fsyntax-only to ensure type/syntax correctness.
    Runs in a sandboxed environment to prevent arbitrary code execution.
    """
    def __init__(self, timeout_s: float = 10.0) -> None:
        self.timeout_s = timeout_s
        self.clang_path = shutil.which("clang++")
        self.gpp_path = shutil.which("g++")
        self.available = bool(self.clang_path or self.gpp_path)
        self.compiler_cmd = self.clang_path or self.gpp_path

    def verify(self, code: str) -> OracleResult:
        if not self.available:
            return OracleResult(
                success=False,
                compiler="cpp",
                error_message="UNVERIFIED: clang++/g++ not installed (fail-closed, E=1e6)",
                details={"unverified": True, "reason": "toolchain_missing"},
            )

        t0 = time.perf_counter()

        # Prevent empty or whitespace-only code
        if not code.strip():
            return OracleResult(
                success=False,
                compiler="cpp",
                error_message="STUB_DETECTED: Empty code",
                details={"reason": "stub"}
            )

        source_file = "candidate.cpp"
        try:
            cmd = [
                self.compiler_cmd,
                "-fsyntax-only",
                "-std=c++20",
                "-Wall",
                "-Werror=return-type",
                f"/work/{source_file}",
            ]
            success, stdout, stderr, timed_out = run_in_sandbox(
                cmd,
                timeout_s=self.timeout_s,
                mem_mb=2048,
                cpu_s=int(self.timeout_s) + 5,
                files={source_file: code},
            )

            if timed_out:
                return OracleResult(
                    success=False,
                    compiler="cpp",
                    error_message=f"C++ verification timed out after {self.timeout_s}s",
                )

            latency = (time.perf_counter() - t0) * 1000.0

            if success:
                return OracleResult(
                    success=True,
                    compiler="cpp",
                    stdout=stdout,
                    stderr=stderr,
                    latency_ms=round(latency, 2),
                )
            else:
                return OracleResult(
                    success=False,
                    compiler="cpp",
                    error_message=stderr.strip() or "Syntax error",
                    stdout=stdout,
                    stderr=stderr,
                    latency_ms=round(latency, 2),
                )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            return OracleResult(
                success=False,
                compiler="cpp",
                error_message=f"Sandbox error: {str(exc)}",
                latency_ms=round(latency, 2),
            )

    def verify_with_test(self, code: str, test_spec: str, timeout_s: float = 10.0) -> OracleResult:
        """
        Compiles and executes C++ candidate with test assertions.
        Test spec should be C++ code with assert() calls.
        Success requires all assertions to pass and proper nonce-tagged result output.
        """
        from gwaya.ast_audit import ZeroStubAudit

        t0 = time.perf_counter()

        # Run ZeroStubAudit first
        audit_result = ZeroStubAudit.audit_cpp_code(code)
        if not audit_result.is_clean:
            return OracleResult(
                success=False,
                compiler="cpp",
                error_message="STUB_DETECTED: " + "; ".join(audit_result.violations),
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                details={"reason": "stub", "violations": audit_result.violations},
            )

        if not self.available:
            return OracleResult(
                success=False,
                compiler="cpp",
                error_message="UNVERIFIED: clang++/g++ not installed (fail-closed, E=1e6)",
                latency_ms=0.0,
                details={"unverified": True, "reason": "toolchain_missing"},
            )

        # Build combined C++ program with candidate and test runner
        test_count = test_spec.count("assert(")
        if test_count == 0:
            return OracleResult(
                success=False,
                compiler="cpp",
                error_message="INVALID_SPEC: test spec contains no assert() calls",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                details={"reason": "no_asserts"},
            )

        nonce = secrets.token_hex(16)
        cpp_program = _build_cpp_test_program(code, test_spec, nonce)

        try:
            # Compile and run in a single bash command within the sandbox
            bash_cmd = f"{self.compiler_cmd} -std=c++20 -o /work/test /work/test.cpp && /work/test"
            success, stdout, stderr, timed_out = run_in_sandbox(
                ["/bin/bash", "-c", bash_cmd],
                timeout_s=timeout_s,
                mem_mb=2048,
                cpu_s=int(timeout_s) + 5,
                files={"test.cpp": cpp_program},
            )

            if timed_out:
                return OracleResult(
                    success=False,
                    compiler="cpp",
                    error_message=f"C++ compilation/execution timed out after {timeout_s}s",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"reason": "timeout", "total": test_count},
                )

            # Check if compilation/execution succeeded
            if not success:
                if "error" in stderr.lower() or "undefined" in stderr.lower():
                    return OracleResult(
                        success=False,
                        compiler="cpp",
                        error_message=f"Compilation failed: {stderr.strip()[:500]}",
                        latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                        details={"reason": "compilation_failed"},
                    )
                # If it failed but looks like a runtime issue, check for assertion
                return OracleResult(
                    success=False,
                    compiler="cpp",
                    error_message="Test assertion failed (Aborted)",
                    stderr=stderr[:500] if stderr else "",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"reason": "assertion_failed"},
                )

            # Parse the nonce-tagged result from stdout
            result = _parse_cpp_test_result(stdout, nonce)
            if not result:
                return OracleResult(
                    success=False,
                    compiler="cpp",
                    error_message="No nonce-tagged result found in test output",
                    stdout=stdout[-500:] if stdout else "",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"reason": "missing_result"},
                )

            passed = result.get("passed", 0)
            total = result.get("total", test_count)

            if passed == total and total == test_count:
                return OracleResult(
                    success=True,
                    compiler="cpp",
                    stdout=stdout[-500:] if stdout else "",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"passed": passed, "total": total, "reason": "all_tests_passed"},
                )
            else:
                failure_msg = result.get("first_failure", "Unknown assertion failure")
                return OracleResult(
                    success=False,
                    compiler="cpp",
                    error_message=f"Test assertion failed: {failure_msg[:200]}",
                    stdout=stdout[-500:] if stdout else "",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"passed": passed, "total": total, "reason": "assertion_failed"},
                )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            return OracleResult(
                success=False,
                compiler="cpp",
                error_message=f"Sandbox error: {str(exc)}",
                latency_ms=round(latency, 2),
            )


def _build_rust_test_program(candidate: str, test_spec: str, nonce: str) -> str:
    """Build a Rust program that combines candidate code with test execution and nonce output."""
    # For Rust, we'll execute the test spec and count successes
    # Simple approach: count assert! calls in spec as total tests
    total_asserts = test_spec.count("assert!")

    # Indent the test spec properly
    indented_test_spec = "\n".join("    " + line for line in test_spec.split("\n"))

    lines = [
        "#![allow(warnings)]",
        candidate,
        "",
        "fn main() {",
        "    let mut passed = 0;",
        f"    let total = {total_asserts};",
        "",
        "    // Run tests - any panic means failure",
        indented_test_spec,
        "",
        "    // All tests passed if we got here",
        "    passed = total;",
        "",
        '    // Output result with nonce',
        f'    println!("{{{{\\\"nonce\\\": \\\"{nonce}\\\", \\\"passed\\\": {{}}, \\\"total\\\": {{}}, \\\"stage\\\": \\\"complete\\\", \\\"first_failure\\\": \\\"\\\"}}}}", passed, total);',
        "}",
    ]
    return "\n".join(lines)


def _build_cpp_test_program(candidate: str, test_spec: str, nonce: str) -> str:
    """Build a C++ program that combines candidate code with test execution and nonce output."""
    # Count assert calls in test spec
    total_asserts = test_spec.count("assert(")

    return f"""
#include <iostream>
#include <iomanip>
#include <cassert>

{candidate}

int main() {{
    int passed = 0;
    int total = {total_asserts};

    try {{
        {test_spec}
        // All asserts passed
        passed = total;
    }} catch (...) {{
        // Assertion failed, passed stays 0
    }}

    // Output result with nonce
    std::cout << "{{";
    std::cout << "\\\"nonce\\\": \\\"{nonce}\\\", ";
    std::cout << "\\\"passed\\\": " << passed << ", ";
    std::cout << "\\\"total\\\": " << total << ", ";
    std::cout << "\\\"stage\\\": \\\"complete\\\", ";
    std::cout << "\\\"first_failure\\\": \\\"\\\"";
    std::cout << "}}" << std::endl;

    return 0;
}}
""".strip()


def _build_go_test_program(candidate: str, test_spec: str, nonce: str) -> str:
    """Build a Go program that combines candidate code with test execution and nonce output."""
    # For Go, count if statements or other test indicators
    # Simple approach: each "if " could be a test assertion
    total_tests = test_spec.count("if ")
    if total_tests == 0:
        total_tests = 1  # At least one test

    return f"""
package main

import (
    "fmt"
)

{candidate}

func main() {{
    passed := 0
    total := {total_tests}

    // Run tests
    defer func() {{
        if r := recover(); r != nil {{
            // panic means test failed
            passed = 0
        }} else {{
            // No panic means all tests passed
            passed = total
        }}

        // Output result with nonce
        fmt.Printf(`{{"nonce": "{nonce}", "passed": %d, "total": %d, "stage": "complete", "first_failure": ""}}` + "\\n", passed, total)
    }}()

    {test_spec}
}}
""".strip()


def _parse_rust_test_result(stdout: str, nonce: str) -> dict[str, Any] | None:
    """Parse Rust test result from stdout output."""
    for line in stdout.splitlines():
        line = line.strip()
        if not (line.startswith("{") and nonce in line):
            continue
        try:
            record = json.loads(line)
            if record.get("nonce") == nonce:
                return record
        except json.JSONDecodeError:
            continue
    return None


def _parse_cpp_test_result(stdout: str, nonce: str) -> dict[str, Any] | None:
    """Parse C++ test result from stdout output."""
    for line in stdout.splitlines():
        line = line.strip()
        if not (line.startswith("{") and nonce in line):
            continue
        try:
            record = json.loads(line)
            if record.get("nonce") == nonce:
                return record
        except json.JSONDecodeError:
            continue
    return None


def _parse_go_test_result(stdout: str, nonce: str) -> dict[str, Any] | None:
    """Parse Go test result from stdout output."""
    for line in stdout.splitlines():
        line = line.strip()
        if not (line.startswith("{") and nonce in line):
            continue
        try:
            record = json.loads(line)
            if record.get("nonce") == nonce:
                return record
        except json.JSONDecodeError:
            continue
    return None


class GoCompilerOracle:
    """
    Verifies Go snippets via 'go build' to ensure type/syntax correctness.
    Runs in a sandboxed environment to prevent arbitrary code execution.
    """
    def __init__(self, timeout_s: float = 10.0) -> None:
        self.timeout_s = timeout_s
        self.go_path = shutil.which("go")
        self.available = bool(self.go_path)

    def verify(self, code: str) -> OracleResult:
        if not self.available:
            return OracleResult(
                success=False,
                compiler="go",
                error_message="UNVERIFIED: go compiler not installed (fail-closed, E=1e6)",
                details={"unverified": True, "reason": "toolchain_missing"},
            )

        t0 = time.perf_counter()

        if not code.strip():
            return OracleResult(
                success=False,
                compiler="go",
                error_message="STUB_DETECTED: Empty code",
                details={"reason": "stub"}
            )

        code_to_compile = code
        # Go files need a package declaration.
        if "package " not in code:
            code_to_compile = f"package main\n\n{code}"

        source_file = "main.go"
        try:
            cmd = [
                self.go_path,
                "build",
                "-o",
                "/work/main",
                f"/work/{source_file}",
            ]
            success, stdout, stderr, timed_out = run_in_sandbox(
                cmd,
                timeout_s=self.timeout_s,
                mem_mb=2048,
                cpu_s=int(self.timeout_s) + 5,
                files={source_file: code_to_compile},
            )

            if timed_out:
                return OracleResult(
                    success=False,
                    compiler="go",
                    error_message=f"Go verification timed out after {self.timeout_s}s",
                )

            latency = (time.perf_counter() - t0) * 1000.0

            if success:
                return OracleResult(
                    success=True,
                    compiler="go",
                    stdout=stdout,
                    stderr=stderr,
                    latency_ms=round(latency, 2),
                )
            else:
                return OracleResult(
                    success=False,
                    compiler="go",
                    error_message=stderr.strip() or "Syntax error",
                    stdout=stdout,
                    stderr=stderr,
                    latency_ms=round(latency, 2),
                )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            return OracleResult(
                success=False,
                compiler="go",
                error_message=f"Sandbox error: {str(exc)}",
                latency_ms=round(latency, 2),
            )

    def verify_with_test(self, code: str, test_spec: str, timeout_s: float = 10.0) -> OracleResult:
        """
        Compiles and executes Go candidate with test code.
        Test spec should be Go code with test assertions (panic on failure).
        Success requires all test assertions to pass and proper nonce-tagged result output.
        """
        from gwaya.ast_audit import ZeroStubAudit

        t0 = time.perf_counter()

        # Run ZeroStubAudit first
        audit_result = ZeroStubAudit.audit_go_code(code)
        if not audit_result.is_clean:
            return OracleResult(
                success=False,
                compiler="go",
                error_message="STUB_DETECTED: " + "; ".join(audit_result.violations),
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                details={"reason": "stub", "violations": audit_result.violations},
            )

        if not self.available:
            return OracleResult(
                success=False,
                compiler="go",
                error_message="UNVERIFIED: go compiler not installed (fail-closed, E=1e6)",
                latency_ms=0.0,
                details={"unverified": True, "reason": "toolchain_missing"},
            )

        # Count assertions in test spec (each "if" with panic is a test)
        test_count = test_spec.count("if ")
        if test_count == 0:
            test_count = 1  # At least one test

        if not test_spec.strip():
            return OracleResult(
                success=False,
                compiler="go",
                error_message="INVALID_SPEC: test spec is empty",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                details={"reason": "no_asserts"},
            )

        nonce = secrets.token_hex(16)
        go_program = _build_go_test_program(code, test_spec, nonce)

        try:
            # Run the Go program directly with go run
            cmd = [
                self.go_path,
                "run",
                "/work/main.go",
            ]
            success, stdout, stderr, timed_out = run_in_sandbox(
                cmd,
                timeout_s=timeout_s,
                mem_mb=2048,
                cpu_s=int(timeout_s) + 5,
                files={"main.go": go_program},
            )

            if timed_out:
                return OracleResult(
                    success=False,
                    compiler="go",
                    error_message=f"Go execution timed out after {timeout_s}s",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"reason": "timeout", "total": test_count},
                )

            # Parse the nonce-tagged result from stdout
            result = _parse_go_test_result(stdout, nonce)
            if not result:
                return OracleResult(
                    success=False,
                    compiler="go",
                    error_message="No nonce-tagged result found in test output",
                    stdout=stdout[-500:] if stdout else "",
                    stderr=stderr[-500:] if stderr else "",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"reason": "missing_result"},
                )

            passed = result.get("passed", 0)
            total = result.get("total", test_count)

            if passed == total and total == test_count:
                return OracleResult(
                    success=True,
                    compiler="go",
                    stdout=stdout[-500:] if stdout else "",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"passed": passed, "total": total, "reason": "all_tests_passed"},
                )
            else:
                failure_msg = result.get("first_failure", "Unknown test failure")
                return OracleResult(
                    success=False,
                    compiler="go",
                    error_message=f"Test failure: {failure_msg}",
                    stdout=stdout[-500:] if stdout else "",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    details={"passed": passed, "total": total, "reason": "test_failed"},
                )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            return OracleResult(
                success=False,
                compiler="go",
                error_message=f"Sandbox error: {str(exc)}",
                latency_ms=round(latency, 2),
            )
