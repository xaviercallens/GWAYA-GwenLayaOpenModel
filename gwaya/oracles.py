"""
anse/gwaya/oracles.py
=====================
Compiler-in-the-Loop Oracles for GWAYA v3 MCTS and Zero-Trust Verification.

Provides deterministic compiler oracles:
  - RustCompilerOracle: verifies Rust syntax, borrow checking, and typing via rustc / cargo check.
  - Lean4CompilerOracle: verifies Lean 4 formal proofs and syntax via lean CLI.
  - PythonCompilerOracle: verifies AST syntax and sandboxed execution.
"""
from __future__ import annotations

import ast
import logging
import os
import re
import shutil
import subprocess  # nosec B404
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

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
    def verify(self, code: str) -> OracleResult:
        return self.verify_snippet(code)

    """
    Evaluates Rust code candidates using `rustc` metadata emission or `cargo check`.
    Flags syntax errors, borrow checker violations, and lifetime errors in <100ms.
    """

    def __init__(self, rustc_path: str | None = None, timeout_s: float = 5.0) -> None:
        self.rustc_path = rustc_path or shutil.which("rustc") or "rustc"
        self.timeout_s = timeout_s
        self.available = bool(shutil.which(self.rustc_path))

    def verify_snippet(self, code: str) -> OracleResult:
        """
        Compiles a Rust snippet as a library (`--emit=metadata`) to verify type soundness
        and borrow correctness without code generation overhead.
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

        with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False) as f:
            f.write(code_to_compile)
            temp_path = f.name

        with tempfile.TemporaryDirectory() as out_dir:
            try:
                cmd = [
                    self.rustc_path,
                    "--crate-type=lib",
                    "--emit=metadata",
                    "--out-dir",
                    out_dir,
                    temp_path,
                ]
                res = subprocess.run(  # nosec B603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                success = res.returncode == 0
                err_msg = res.stderr.strip() if not success else ""

                return OracleResult(
                    success=success,
                    compiler="rustc",
                    error_message=err_msg,
                    stdout=res.stdout,
                    stderr=res.stderr,
                    latency_ms=round(latency_ms, 2),
                    details={"returncode": res.returncode},
                )
            except subprocess.TimeoutExpired:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return OracleResult(
                    success=False,
                    compiler="rustc",
                    error_message=f"Compilation timed out after {self.timeout_s}s",
                    latency_ms=round(latency_ms, 2),
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return OracleResult(
                    success=False,
                    compiler="rustc",
                    error_message=str(exc),
                    latency_ms=round(latency_ms, 2),
                )
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)


class Lean4CompilerOracle:
    def verify(self, code: str) -> OracleResult:
        return self.verify_snippet(code)

    """
    Evaluates Lean 4 candidates using `lean` CLI.
    Enforces that theorems compile cleanly and verifies whether unacknowledged `sorry` exists.
    """

    def __init__(self, lean_path: str | None = None, timeout_s: float = 15.0) -> None:
        if lean_path:
            self.lean_path = lean_path
        else:
            self.lean_path = (
                shutil.which("lean") or
                shutil.which("lean", path="/mnt/data/home/xavkal/.elan/bin:/mnt/data/xdev-cache/home-cache/.elan/bin") or
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

    @classmethod
    def _lexical_flaws(cls, code: str) -> list[str]:
        """Defence-in-depth lexical scan (comments removed). The kernel axiom audit is authoritative."""
        stripped = re.sub(r"/-.*?-/", " ", code, flags=re.DOTALL)
        stripped = re.sub(r"--[^\n]*", " ", stripped)
        flaws = set(cls._ESCAPE_RE.findall(stripped))
        if cls._UNAUDITABLE_RE.search(stripped):
            flaws.add("anonymous_declaration")
        return sorted(flaws)

    @classmethod
    def _axiom_probe(cls, code: str) -> str:
        """Append `#print axioms` for every named theorem so the kernel reports its axiom closure."""
        names = cls._DECL_RE.findall(code)
        return "".join(f"\n#print axioms {n}" for n in names) + ("\n" if names else "")

    @classmethod
    def _disallowed_axioms(cls, stdout: str, allow_sorry: bool) -> list[str]:
        allowed = cls.STANDARD_AXIOMS | ({"sorryAx"} if allow_sorry else frozenset())
        found: set[str] = set()
        for group in cls._AXIOMS_RE.findall(stdout):
            found.update(a.strip() for a in group.split(",") if a.strip())
        return sorted(found - allowed)

    def _interpret_run(self, res: subprocess.CompletedProcess, allow_sorry: bool, latency_ms: float) -> OracleResult:
        """Success requires exit status 0 AND an axiom closure within STANDARD_AXIOMS."""
        success = res.returncode == 0
        err_msg = "" if success else (res.stderr.strip() or res.stdout.strip())
        bad_axioms = self._disallowed_axioms(res.stdout, allow_sorry) if success else []
        if bad_axioms:
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
        Runs Lean 4 on snippet. Fails if lean exits non-zero or if `sorry` is present
        without allow_sorry=True.
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
                details={"flaw": "unacknowledged_sorry", "forbidden": forbidden},
            )

        if not self.available:
            return OracleResult(
                success=False,
                compiler="lean4",
                error_message="UNVERIFIED: lean toolchain not installed (fail-closed, E=1e6)",
                latency_ms=0.0,
                details={"unverified": True, "reason": "toolchain_missing"},
            )

        with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as f:
            f.write(code + self._axiom_probe(code))
            temp_path = f.name

        try:
            cmd = [self.lean_path, temp_path]
            res = subprocess.run(  # nosec B603
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            return self._interpret_run(res, allow_sorry, (time.perf_counter() - t0) * 1000.0)
        except subprocess.TimeoutExpired:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return OracleResult(
                success=False,
                compiler="lean4",
                error_message=f"Lean 4 verification timed out after {self.timeout_s}s",
                latency_ms=round(latency_ms, 2),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return OracleResult(
                success=False,
                compiler="lean4",
                error_message=str(exc),
                latency_ms=round(latency_ms, 2),
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


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
        (`anse.gwaya.test_harness`). Success requires a nonce-tagged harness record with every
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
            
        with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            cmd = [
                self.compiler_cmd,
                "-fsyntax-only",
                "-std=c++20",
                "-Wall",
                "-Werror=return-type",
                temp_path,
            ]
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_s
            )
            latency = (time.perf_counter() - t0) * 1000.0
            
            if res.returncode == 0:
                return OracleResult(
                    success=True,
                    compiler="cpp",
                    stdout=res.stdout,
                    stderr=res.stderr,
                    latency_ms=round(latency, 2),
                )
            else:
                return OracleResult(
                    success=False,
                    compiler="cpp",
                    error_message=res.stderr.strip() or "Syntax error",
                    stdout=res.stdout,
                    stderr=res.stderr,
                    latency_ms=round(latency, 2),
                )
        except subprocess.TimeoutExpired:
            return OracleResult(
                success=False,
                compiler="cpp",
                error_message=f"C++ verification timed out after {self.timeout_s}s",
            )
        except Exception as exc:
            return OracleResult(
                success=False,
                compiler="cpp",
                error_message=str(exc),
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class GoCompilerOracle:
    """
    Verifies Go snippets via 'go build' to ensure type/syntax correctness.
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

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "main.go")
            with open(temp_path, "w") as f:
                f.write(code_to_compile)

            try:
                cmd = [
                    self.go_path,
                    "build",
                    "-o",
                    os.devnull,
                    temp_path,
                ]
                res = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout_s
                )
                latency = (time.perf_counter() - t0) * 1000.0
                
                if res.returncode == 0:
                    return OracleResult(
                        success=True,
                        compiler="go",
                        stdout=res.stdout,
                        stderr=res.stderr,
                        latency_ms=round(latency, 2),
                    )
                else:
                    return OracleResult(
                        success=False,
                        compiler="go",
                        error_message=res.stderr.strip() or "Syntax error",
                        stdout=res.stdout,
                        stderr=res.stderr,
                        latency_ms=round(latency, 2),
                    )
            except subprocess.TimeoutExpired:
                return OracleResult(
                    success=False,
                    compiler="go",
                    error_message=f"Go verification timed out after {self.timeout_s}s",
                )
            except Exception as exc:
                return OracleResult(
                    success=False,
                    compiler="go",
                    error_message=str(exc),
                )
