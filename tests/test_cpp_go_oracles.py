import pytest
from unittest import mock
import subprocess

from gwaya.oracles import CppCompilerOracle, GoCompilerOracle

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/clang++")
@mock.patch("gwaya.oracles.subprocess.run")
def test_cpp_oracle_success(mock_run, mock_which):
    oracle = CppCompilerOracle()
    assert oracle.available is True
    
    mock_res = mock.Mock()
    mock_res.returncode = 0
    mock_res.stdout = ""
    mock_res.stderr = ""
    mock_run.return_value = mock_res
    
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
@mock.patch("gwaya.oracles.subprocess.run")
def test_cpp_oracle_error(mock_run, mock_which):
    oracle = CppCompilerOracle()
    
    mock_res = mock.Mock()
    mock_res.returncode = 1
    mock_res.stderr = "error: unknown type name 'foo'"
    mock_run.return_value = mock_res
    
    res = oracle.verify("foo bar;")
    assert res.success is False
    assert "error: unknown type name" in res.error_message

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/clang++")
@mock.patch("gwaya.oracles.subprocess.run")
def test_cpp_oracle_timeout(mock_run, mock_which):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="clang++", timeout=10)
    oracle = CppCompilerOracle()
    res = oracle.verify("int main() { return 0; }")
    assert res.success is False
    assert "timed out" in res.error_message

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/go")
@mock.patch("gwaya.oracles.subprocess.run")
def test_go_oracle_success(mock_run, mock_which):
    oracle = GoCompilerOracle()
    assert oracle.available is True
    
    mock_res = mock.Mock()
    mock_res.returncode = 0
    mock_res.stdout = ""
    mock_res.stderr = ""
    mock_run.return_value = mock_res
    
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
@mock.patch("gwaya.oracles.subprocess.run")
def test_go_oracle_no_package(mock_run, mock_which):
    oracle = GoCompilerOracle()
    
    mock_res = mock.Mock()
    mock_res.returncode = 0
    mock_res.stderr = ""
    mock_run.return_value = mock_res
    
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
@mock.patch("gwaya.oracles.subprocess.run")
def test_go_oracle_error(mock_run, mock_which):
    oracle = GoCompilerOracle()
    
    mock_res = mock.Mock()
    mock_res.returncode = 1
    mock_res.stderr = "undefined: foo"
    mock_run.return_value = mock_res
    
    res = oracle.verify("package main\nfunc main() { foo() }")
    assert res.success is False
    assert "undefined: foo" in res.error_message

@mock.patch("gwaya.oracles.shutil.which", return_value="/bin/go")
@mock.patch("gwaya.oracles.subprocess.run")
def test_go_oracle_timeout(mock_run, mock_which):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="go", timeout=10)
    oracle = GoCompilerOracle()
    res = oracle.verify("package main")
    assert res.success is False
    assert "timed out" in res.error_message
