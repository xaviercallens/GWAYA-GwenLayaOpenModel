"""
tests/test_mcp_server.py
========================
Unit tests for the GWAYA Model Context Protocol (MCP) server tools.
"""
import mcp_server


def test_mcp_system_status_returns_valid_structure():
    status = mcp_server.gwaya_system_status()
    assert "python" in status
    assert "rustc" in status
    assert "lean4" in status
    assert "bubblewrap_sandbox" in status
    assert "ollama" in status
    assert status["python"]["available"] is True


def test_mcp_audit_stubs_flags_placeholders():
    res = mcp_server.gwaya_audit_stubs("python", "def f():\n    pass\n")
    assert res["is_clean"] is False
    assert len(res["violations"]) > 0
    assert res["penalty_energy"] == 1e6

    clean = mcp_server.gwaya_audit_stubs("python", "def f():\n    return 42\n")
    assert clean["is_clean"] is True
    assert clean["violations"] == []
    assert clean["penalty_energy"] == 0.0


def test_mcp_verify_python_syntax():
    # Valid syntax
    valid = mcp_server.gwaya_verify_code("python", "def square(x):\n    return x * x\n")
    assert valid["success"] is True

    # Syntax error
    invalid = mcp_server.gwaya_verify_code("python", "def invalid_syntax(")
    assert invalid["success"] is False
    assert len(invalid["errors"]) > 0


def test_mcp_verify_python_with_tests():
    code = "def add(a, b):\n    return a + b\n"
    # Passing test spec
    pass_spec = "assert add(2, 3) == 5\n"
    res_pass = mcp_server.gwaya_verify_code("python", code, test_spec=pass_spec)
    assert res_pass["success"] is True

    # Failing test spec
    fail_spec = "assert add(2, 3) == 999\n"
    res_fail = mcp_server.gwaya_verify_code("python", code, test_spec=fail_spec)
    assert res_fail["success"] is False
