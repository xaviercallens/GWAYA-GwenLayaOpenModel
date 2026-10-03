"""
tests/test_ast_audit.py
=======================
Tests for GWAYA ZeroStubAudit AST-level physical hardness auditor.
"""
from gwaya.ast_audit import ZeroStubAudit


def test_python_clean_code_passes():
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    res = ZeroStubAudit.audit_python_code(code)
    assert res.is_clean is True
    assert res.violations == []
    assert res.penalty_energy == 0.0


def test_python_pass_statement_rejected():
    code = "def add(a, b):\n    pass\n"
    res = ZeroStubAudit.audit_python_code(code)
    assert res.is_clean is False
    assert any("pass" in v for v in res.violations)
    assert res.penalty_energy == 1e6


def test_python_ellipsis_literal_rejected():
    code = "def compute():\n    ...\n"
    res = ZeroStubAudit.audit_python_code(code)
    assert res.is_clean is False
    assert any("Ellipsis" in v for v in res.violations)
    assert res.penalty_energy == 1e6


def test_python_mock_identifiers_rejected():
    for name in ["mock_data", "fake_user", "simulate_call", "dummy_var", "stub_fn"]:
        code = f"def {name}():\n    return 42\n"
        res = ZeroStubAudit.audit_python_code(code)
        assert res.is_clean is False, f"Expected {name} to be rejected"
        assert res.penalty_energy == 1e6


def test_lean4_sorry_and_admit_rejected():
    sorry_code = "theorem t : 1 + 1 = 2 := by sorry"
    res_sorry = ZeroStubAudit.audit_lean_code(sorry_code)
    assert res_sorry.is_clean is False
    assert any("sorry" in v for v in res_sorry.violations)

    admit_code = "theorem t : 1 + 1 = 2 := by admit"
    res_admit = ZeroStubAudit.audit_lean_code(admit_code)
    assert res_admit.is_clean is False
    assert any("admit" in v for v in res_admit.violations)


def test_rust_unimplemented_and_todo_rejected():
    unimpl = "fn solve() -> i32 { unimplemented!() }"
    res_u = ZeroStubAudit.audit_rust_code(unimpl)
    assert res_u.is_clean is False
    assert any("unimplemented" in v for v in res_u.violations)

    todo = "fn solve() -> i32 { todo!(\"later\") }"
    res_t = ZeroStubAudit.audit_rust_code(todo)
    assert res_t.is_clean is False
    assert any("todo" in v for v in res_t.violations)
