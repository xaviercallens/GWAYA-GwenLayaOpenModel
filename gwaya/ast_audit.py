"""
gwaya/ast_audit.py
==================
AST-level Zero-Stub physical hardness auditor for GWAYA.
Enforces the Zero-Stub Law: rejects placeholder idioms (pass, ..., sorry, mock_*, fake_*)
with penalty energy E = 10^6.
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

        for node in ast.walk(tree):
            # Check for 'pass' statements (except empty class definition like custom Exception)
            if isinstance(node, ast.Pass):
                violations.append("Forbidden 'pass' statement detected in execution body.")

            # Check for Ellipsis literal (...)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value is Ellipsis:
                violations.append("Forbidden Ellipsis ('...') detected in execution body.")

            # Check for forbidden function or variable names
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Name)):
                ident = node.name if hasattr(node, "name") else node.id
                for pat in cls.FORBIDDEN_NAME_PATTERNS:
                    if pat.search(ident):
                        violations.append(f"Forbidden mock identifier '{ident}' detected.")

        if violations:
            return ZeroStubAuditResult(is_clean=False, violations=violations, penalty_energy=1e6)
        return ZeroStubAuditResult(is_clean=True, violations=[], penalty_energy=0.0)

    @classmethod
    def audit_lean_code(cls, source_code: str) -> ZeroStubAuditResult:
        violations: List[str] = []
        # Check for sorry, admit, sorryAx
        if re.search(r"\bsorry\b", source_code):
            violations.append("Forbidden Lean 4 'sorry' hole detected in formal proof.")
        if re.search(r"\badmit\b", source_code):
            violations.append("Forbidden Lean 4 'admit' hole detected in formal proof.")
        if re.search(r"\bsorryAx\b", source_code):
            violations.append("Forbidden Lean 4 axiom 'sorryAx' dependency detected.")

        if violations:
            return ZeroStubAuditResult(is_clean=False, violations=violations, penalty_energy=1e6)
        return ZeroStubAuditResult(is_clean=True, violations=[], penalty_energy=0.0)

    @classmethod
    def audit_rust_code(cls, source_code: str) -> ZeroStubAuditResult:
        violations: List[str] = []
        if re.search(r"\bunimplemented!\s*\(", source_code):
            violations.append("Forbidden Rust 'unimplemented!()' macro detected.")
        if re.search(r"\btodo!\s*\(", source_code):
            violations.append("Forbidden Rust 'todo!()' macro detected.")

        if violations:
            return ZeroStubAuditResult(is_clean=False, violations=violations, penalty_energy=1e6)
        return ZeroStubAuditResult(is_clean=True, violations=[], penalty_energy=0.0)

    @classmethod
    def audit_cpp_code(cls, source_code: str) -> ZeroStubAuditResult:
        violations: List[str] = []
        if re.search(r"\bTODO\b", source_code):
            violations.append("Forbidden C++ 'TODO' comment/macro detected.")
        if re.search(r"throw\s+std::logic_error\(\s*\"(?:Not implemented|TODO).*\"\s*\)", source_code, re.IGNORECASE):
            violations.append("Forbidden C++ 'Not implemented' exception detected.")

        if violations:
            return ZeroStubAuditResult(is_clean=False, violations=violations, penalty_energy=1e6)
        return ZeroStubAuditResult(is_clean=True, violations=[], penalty_energy=0.0)

    @classmethod
    def audit_go_code(cls, source_code: str) -> ZeroStubAuditResult:
        violations: List[str] = []
        if re.search(r"\bpanic\(\"TODO.*\"\)", source_code, re.IGNORECASE):
            violations.append("Forbidden Go 'panic(TODO)' macro detected.")
        if re.search(r"\bpanic\(\"Not implemented.*\"\)", source_code, re.IGNORECASE):
            violations.append("Forbidden Go 'panic(Not implemented)' macro detected.")

        if violations:
            return ZeroStubAuditResult(is_clean=False, violations=violations, penalty_energy=1e6)
        return ZeroStubAuditResult(is_clean=True, violations=[], penalty_energy=0.0)
