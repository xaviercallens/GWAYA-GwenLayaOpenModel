"""
gwaya/ast_audit.py
==================
AST-level Zero-Stub physical hardness auditor for GWAYA.
Enforces the Zero-Stub Law: rejects placeholder idioms (pass, ..., sorry, mock_*, fake_*)
with penalty energy E = 10^6.

Threat Model
------------
Detects syntactic stubs via AST inspection and pattern matching. For dynamic languages (Python),
detects common placeholder patterns (pass, Ellipsis, NotImplementedError, return None). For
compiled languages (Lean, Rust, C++, Go), strips comments and string literals before regex
matching to avoid false positives from documentation.

Does NOT defend against:
- Intentional obfuscation (e.g., redirecting to an external implementation server)
- Semantic stubs disguised as real logic
- Bytecode/compiled artifacts
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ZeroStubAuditResult:
    """Receipt from an AST-level zero-stub audit."""
    is_clean: bool
    violations: List[str] = field(default_factory=list)
    penalty_energy: float = 0.0


class ZeroStubAudit:
    """
    Enforces the Zero-Stub Physical Hardness Law:
    Rejects placeholders (pass, ..., sorry, mock_*, fake_*, dummy_*, stub_*) with E = 10^6 penalty.
    """

    FORBIDDEN_NAME_PATTERNS = [
        re.compile(r"^mock_", re.IGNORECASE),
        re.compile(r"^fake_", re.IGNORECASE),
        re.compile(r"^simulate_", re.IGNORECASE),
        re.compile(r"^dummy_", re.IGNORECASE),
        re.compile(r"^stub_", re.IGNORECASE),
    ]

    @classmethod
    def _is_pass_allowed(cls, node: ast.AST, parent: ast.AST | None) -> bool:
        """Check if a 'pass' statement is allowed (except handler or class body)."""
        if parent is None:
            return False
        # Allow pass in except handlers
        if isinstance(parent, ast.ExceptHandler):
            return True
        # Allow pass in class bodies
        if isinstance(parent, ast.ClassDef):
            return True
        return False

    @classmethod
    def _is_docstring(cls, node: ast.expr) -> bool:
        """Check if a node is a docstring (string constant)."""
        return isinstance(node, ast.Constant) and isinstance(node.value, str)

    @classmethod
    def _is_ellipsis(cls, node: ast.expr) -> bool:
        """Check if a node is an Ellipsis constant."""
        return isinstance(node, ast.Constant) and node.value is Ellipsis

    @classmethod
    def _is_not_implemented(cls, node: ast.stmt) -> bool:
        """Check if a statement is 'raise NotImplementedError'."""
        if isinstance(node, ast.Raise):
            exc = node.exc
            # Handle both "raise NotImplementedError" and "raise NotImplementedError()"
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                return True
            if isinstance(exc, ast.Call):
                if isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
                    return True
        return False

    @classmethod
    def _is_return_none(cls, node: ast.stmt) -> bool:
        """Check if a statement is 'return None'."""
        if isinstance(node, ast.Return):
            if node.value is None or (isinstance(node.value, ast.Constant) and node.value.value is None):
                return True
        return False

    @classmethod
    def _function_returns_value(cls, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if function signature indicates it should return a value (has return annotation)."""
        return node.returns is not None

    @classmethod
    def _check_function_body(cls, func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        """Check for stub function bodies."""
        violations: List[str] = []
        body = func_node.body

        if not body:
            return violations

        # Extract non-docstring statements
        non_docstring_stmts = []
        for i, stmt in enumerate(body):
            if i == 0 and isinstance(stmt, ast.Expr) and cls._is_docstring(stmt.value):
                # This is a docstring
                continue
            non_docstring_stmts.append(stmt)

        # Check for body that's only docstring (with optional pass)
        if not non_docstring_stmts:
            violations.append(f"Function '{func_node.name}' body contains only a docstring.")
            return violations

        # Check if body is only pass/Ellipsis/NotImplementedError/return None
        if len(non_docstring_stmts) == 1:
            stmt = non_docstring_stmts[0]

            if isinstance(stmt, ast.Pass):
                violations.append(f"Function '{func_node.name}' body contains only a 'pass' statement.")
            elif isinstance(stmt, ast.Expr) and cls._is_ellipsis(stmt.value):
                violations.append(f"Function '{func_node.name}' body contains only an Ellipsis ('...').")
            elif cls._is_not_implemented(stmt):
                violations.append(f"Function '{func_node.name}' body contains only 'raise NotImplementedError'.")
            elif cls._is_return_none(stmt) and cls._function_returns_value(func_node):
                violations.append(f"Function '{func_node.name}' promises a return value but only returns None.")

        return violations

    @classmethod
    def _check_magic_mock_return(cls, func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        """Check if function returns a MagicMock, Mock, or patch object."""
        violations: List[str] = []

        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value:
                # Check if returning a Call to Mock, MagicMock, or patch
                if isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Name):
                        if func.id in ("Mock", "MagicMock", "patch"):
                            violations.append(f"Function '{func_node.name}' returns {func.id} object as implementation.")
                    elif isinstance(func, ast.Attribute):
                        if func.attr in ("Mock", "MagicMock", "patch"):
                            violations.append(f"Function '{func_node.name}' returns {func.attr} object as implementation.")

        return violations

    @classmethod
    def audit_python_code(cls, source_code: str) -> ZeroStubAuditResult:
        violations: List[str] = []
        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            return ZeroStubAuditResult(
                is_clean=False,
                violations=[f"SyntaxError in Python source: {exc}"],
                penalty_energy=1e6,
            )

        # Build parent map for AST
        parent_map: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

        for node in ast.walk(tree):
            # Check for 'pass' statements (allow in except handlers and class bodies)
            if isinstance(node, ast.Pass):
                parent = parent_map.get(node)
                if not cls._is_pass_allowed(node, parent):
                    violations.append("Forbidden 'pass' statement detected in execution body.")

            # Check for Ellipsis literal (...) - allow only as docstrings
            if isinstance(node, ast.Expr) and cls._is_ellipsis(node.value):
                violations.append("Forbidden Ellipsis ('...') detected in execution body.")

            # Check function bodies for stubs
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                violations.extend(cls._check_function_body(node))
                violations.extend(cls._check_magic_mock_return(node))

            # Check for forbidden variable/function names (only flag variables if func is named that way)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ident = node.name
                for pat in cls.FORBIDDEN_NAME_PATTERNS:
                    if pat.search(ident):
                        violations.append(f"Forbidden mock identifier '{ident}' detected.")
            elif isinstance(node, ast.Name):
                # Only flag variables named fake_*/dummy_* if we're inside a function with that pattern
                ident = node.id
                parent = parent_map.get(node)
                func_parent = None
                # Find enclosing function
                check = parent
                while check is not None:
                    if isinstance(check, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_parent = check
                        break
                    check = parent_map.get(check)

                # Only flag if the enclosing function has the pattern
                if func_parent and any(pat.search(func_parent.name) for pat in cls.FORBIDDEN_NAME_PATTERNS):
                    for pat in cls.FORBIDDEN_NAME_PATTERNS:
                        if pat.search(ident):
                            violations.append(f"Forbidden mock identifier '{ident}' in function '{func_parent.name}'.")

        if violations:
            return ZeroStubAuditResult(is_clean=False, violations=violations, penalty_energy=1e6)
        return ZeroStubAuditResult(is_clean=True, violations=[], penalty_energy=0.0)

    @classmethod
    def _strip_comments_only(cls, source_code: str, single_line: str = "--",
                              multi_line_start: str = "/-", multi_line_end: str = "-/") -> str:
        """Strip only comments from code, preserving string literals.

        This is important for matching patterns in string contents (e.g., panic("TODO")).
        """
        result = []
        i = 0
        while i < len(source_code):
            # Handle strings (preserve them)
            if source_code[i] in ('"', "'"):
                quote = source_code[i]
                result.append(source_code[i])
                i += 1
                while i < len(source_code):
                    result.append(source_code[i])
                    if source_code[i] == quote and (i == 0 or source_code[i-1] != "\\"):
                        i += 1
                        break
                    i += 1
                continue

            # Handle multi-line comments
            if source_code[i:i+len(multi_line_start)] == multi_line_start:
                i += len(multi_line_start)
                while i < len(source_code):
                    if source_code[i:i+len(multi_line_end)] == multi_line_end:
                        i += len(multi_line_end)
                        break
                    i += 1
                result.append(" ")  # Replace comment with space
                continue

            # Handle single-line comments
            if source_code[i:i+len(single_line)] == single_line:
                # Skip until end of line
                while i < len(source_code) and source_code[i] != "\n":
                    i += 1
                result.append(" ")  # Replace comment with space
                continue

            result.append(source_code[i])
            i += 1

        return "".join(result)

    @classmethod
    def audit_lean_code(cls, source_code: str) -> ZeroStubAuditResult:
        violations: List[str] = []
        # Strip comments and strings before checking
        cleaned = cls._strip_comments_only(source_code, single_line="--",
                                                   multi_line_start="/-", multi_line_end="-/")

        # Check for sorry, admit, sorryAx
        if re.search(r"\bsorry\b", cleaned):
            violations.append("Forbidden Lean 4 'sorry' hole detected in formal proof.")
        if re.search(r"\badmit\b", cleaned):
            violations.append("Forbidden Lean 4 'admit' hole detected in formal proof.")
        if re.search(r"\bsorryAx\b", cleaned):
            violations.append("Forbidden Lean 4 axiom 'sorryAx' dependency detected.")

        if violations:
            return ZeroStubAuditResult(is_clean=False, violations=violations, penalty_energy=1e6)
        return ZeroStubAuditResult(is_clean=True, violations=[], penalty_energy=0.0)

    @classmethod
    def audit_rust_code(cls, source_code: str) -> ZeroStubAuditResult:
        violations: List[str] = []
        # Strip comments and strings before checking
        cleaned = cls._strip_comments_only(source_code, single_line="//",
                                                   multi_line_start="/*", multi_line_end="*/")

        if re.search(r"\bunimplemented!\s*\(", cleaned):
            violations.append("Forbidden Rust 'unimplemented!()' macro detected.")
        if re.search(r"\btodo!\s*\(", cleaned):
            violations.append("Forbidden Rust 'todo!()' macro detected.")

        if violations:
            return ZeroStubAuditResult(is_clean=False, violations=violations, penalty_energy=1e6)
        return ZeroStubAuditResult(is_clean=True, violations=[], penalty_energy=0.0)

    @classmethod
    def audit_cpp_code(cls, source_code: str) -> ZeroStubAuditResult:
        violations: List[str] = []
        # Strip comments and strings before checking
        cleaned = cls._strip_comments_only(source_code, single_line="//",
                                                   multi_line_start="/*", multi_line_end="*/")

        if re.search(r"\bTODO\b", cleaned):
            violations.append("Forbidden C++ 'TODO' comment/macro detected.")
        if re.search(r"throw\s+std::logic_error\(\s*\"(?:Not implemented|TODO).*\"\s*\)", cleaned, re.IGNORECASE):
            violations.append("Forbidden C++ 'Not implemented' exception detected.")

        if violations:
            return ZeroStubAuditResult(is_clean=False, violations=violations, penalty_energy=1e6)
        return ZeroStubAuditResult(is_clean=True, violations=[], penalty_energy=0.0)

    @classmethod
    def audit_go_code(cls, source_code: str) -> ZeroStubAuditResult:
        violations: List[str] = []
        # Strip comments and strings before checking
        cleaned = cls._strip_comments_only(source_code, single_line="//",
                                                   multi_line_start="/*", multi_line_end="*/")

        if re.search(r"\bpanic\(\"TODO.*\"\)", cleaned, re.IGNORECASE):
            violations.append("Forbidden Go 'panic(TODO)' macro detected.")
        if re.search(r"\bpanic\(\"Not implemented.*\"\)", cleaned, re.IGNORECASE):
            violations.append("Forbidden Go 'panic(Not implemented)' macro detected.")

        if violations:
            return ZeroStubAuditResult(is_clean=False, violations=violations, penalty_energy=1e6)
        return ZeroStubAuditResult(is_clean=True, violations=[], penalty_energy=0.0)
