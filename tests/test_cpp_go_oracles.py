import pytest
from unittest import mock
import subprocess

from gwaya.oracles import CppCompilerOracle, GoCompilerOracle

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/clang++")
@mock.patch("gwaya.oracles.run_in_sandbox")
def test_cpp_oracle_success(mock_run_in_sandbox, mock_which):
    oracle = CppCompilerOracle()
    assert oracle.available is True

    # Mock run_in_sandbox to return (success, stdout, stderr, timed_out)
    mock_run_in_sandbox.return_value = (True, "", "", False)

    res = oracle.verify("int main() { return 0; }")
    assert res.success is True
    assert res.compiler == "cpp"

@mock.patch("gwaya.oracles.shutil.which", return_value=None)
def test_cpp_oracle_missing(mock_which):
    oracle = CppCompilerOracle()
    assert oracle.available is False
    res = oracle.verify("int main() { return 0; }")
    assert res.success is False
    assert "UNVERIFIED" in res.error_message

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/clang++")
def test_cpp_oracle_empty(mock_which):
    oracle = CppCompilerOracle()
    res = oracle.verify("   ")
    assert res.success is False
    assert "STUB_DETECTED" in res.error_message

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/clang++")
@mock.patch("gwaya.oracles.run_in_sandbox")
def test_cpp_oracle_error(mock_run_in_sandbox, mock_which):
    oracle = CppCompilerOracle()

    # Mock run_in_sandbox to return (success, stdout, stderr, timed_out)
    mock_run_in_sandbox.return_value = (False, "", "error: unknown type name 'foo'", False)

    res = oracle.verify("foo bar;")
    assert res.success is False
    assert "error: unknown type name" in res.error_message

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/clang++")
@mock.patch("gwaya.oracles.run_in_sandbox")
def test_cpp_oracle_timeout(mock_run_in_sandbox, mock_which):
    oracle = CppCompilerOracle()
    # Mock run_in_sandbox to return (success, stdout, stderr, timed_out)
    mock_run_in_sandbox.return_value = (False, "", "Execution timed out after 10.0s", True)

    res = oracle.verify("int main() { return 0; }")
    assert res.success is False
    assert "timed out" in res.error_message

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/go")
@mock.patch("gwaya.oracles.run_in_sandbox")
def test_go_oracle_success(mock_run_in_sandbox, mock_which):
    oracle = GoCompilerOracle()
    assert oracle.available is True

    # Mock run_in_sandbox to return (success, stdout, stderr, timed_out)
    mock_run_in_sandbox.return_value = (True, "", "", False)

    res = oracle.verify("package main\nfunc main() {}")
    assert res.success is True
    assert res.compiler == "go"

@mock.patch("gwaya.oracles.shutil.which", return_value=None)
def test_go_oracle_missing(mock_which):
    oracle = GoCompilerOracle()
    assert oracle.available is False
    res = oracle.verify("package main")
    assert res.success is False
    assert "UNVERIFIED" in res.error_message

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/go")
@mock.patch("gwaya.oracles.run_in_sandbox")
def test_go_oracle_no_package(mock_run_in_sandbox, mock_which):
    oracle = GoCompilerOracle()

    # Mock run_in_sandbox to return (success, stdout, stderr, timed_out)
    mock_run_in_sandbox.return_value = (True, "", "", False)

    # Missing package should trigger our fallback package insertion
    res = oracle.verify("func main() {}")
    assert res.success is True

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/go")
def test_go_oracle_empty(mock_which):
    oracle = GoCompilerOracle()
    res = oracle.verify("   ")
    assert res.success is False
    assert "STUB_DETECTED" in res.error_message

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/go")
@mock.patch("gwaya.oracles.run_in_sandbox")
def test_go_oracle_error(mock_run_in_sandbox, mock_which):
    oracle = GoCompilerOracle()

    # Mock run_in_sandbox to return (success, stdout, stderr, timed_out)
    mock_run_in_sandbox.return_value = (False, "", "undefined: foo", False)

    res = oracle.verify("package main\nfunc main() { foo() }")
    assert res.success is False
    assert "undefined: foo" in res.error_message

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/go")
@mock.patch("gwaya.oracles.run_in_sandbox")
def test_go_oracle_timeout(mock_run_in_sandbox, mock_which):
    oracle = GoCompilerOracle()
    # Mock run_in_sandbox to return (success, stdout, stderr, timed_out)
    mock_run_in_sandbox.return_value = (False, "", "Execution timed out after 10.0s", True)

    res = oracle.verify("package main")
    assert res.success is False
    assert "timed out" in res.error_message
