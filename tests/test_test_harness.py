import os
import sys
import tempfile
import pytest
import subprocess
from pathlib import Path
from unittest import mock

from gwaya.test_harness import (
    _bwrap_argv,
    _rlimits,
    _parse_record,
    _launch,
    isolation_available,
    run_candidate_tests,
    HarnessResult
)

def test_bwrap_argv_generation():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # Assuming bwrap path
        argv = _bwrap_argv("/usr/bin/bwrap", work)
        assert argv[0] == "/usr/bin/bwrap"
        assert "--ro-bind" in argv
        assert "--unshare-all" in argv
        assert "--die-with-parent" in argv

def test_rlimits():
    # Calling _rlimits factory
    func = _rlimits(cpu_s=1, mem_mb=100)
    # Just ensure it executes without crashing on platforms with/without resource module
    func()

def test_parse_record_valid():
    stdout = 'some text\n{"nonce": "123", "passed": 1, "total": 1, "stage": "run"}\nother'
    record = _parse_record(stdout, "123")
    assert record is not None
    assert record["passed"] == 1

def test_parse_record_invalid_json():
    stdout = '{"nonce": "123", "bad'
    record = _parse_record(stdout, "123")
    assert record is None

def test_parse_record_wrong_nonce():
    stdout = '{"nonce": "456", "passed": 1}'
    record = _parse_record(stdout, "123")
    assert record is None

@mock.patch("gwaya.test_harness._bwrap_path", return_value="/bin/bwrap")
@mock.patch("gwaya.test_harness._launch")
def test_run_candidate_tests_bwrap_mocked(mock_launch, mock_bwrap):
    # Mocking launch to return a success record
    nonce_mock = ""
    def fake_launch(argv, work, nonce, timeout_s, mem_mb):
        nonlocal nonce_mock
        nonce_mock = nonce
        return f'{{"nonce": "{nonce}", "passed": 1, "total": 1, "stage": "run"}}\n', False
    mock_launch.side_effect = fake_launch

    res = run_candidate_tests("def f(): pass", "assert True")
    assert res.success is True
    assert res.isolation == "bwrap"
    assert res.passed == 1

@mock.patch("gwaya.test_harness._bwrap_path", return_value=None)
def test_run_candidate_tests_no_isolation_allowed(mock_bwrap):
    # If GWAYA_ALLOW_UNISOLATED is not 1, should fail-closed
    with mock.patch.dict(os.environ, {"GWAYA_ALLOW_UNISOLATED": "0"}, clear=True):
        res = run_candidate_tests("def f(): pass", "assert True")
        assert res.success is False
        assert "UNVERIFIED" in res.error

@mock.patch("gwaya.test_harness.subprocess.run")
@mock.patch("gwaya.test_harness._bwrap_path", return_value="/bin/bwrap")
def test_isolation_available_true(mock_bwrap, mock_run):
    mock_proc = mock.Mock()
    mock_proc.returncode = 0
    mock_proc.stdout = "ok\n"
    mock_run.return_value = mock_proc
    
    assert isolation_available() is True

@mock.patch("gwaya.test_harness.subprocess.run")
@mock.patch("gwaya.test_harness._bwrap_path", return_value="/bin/bwrap")
def test_isolation_available_false_exception(mock_bwrap, mock_run):
    mock_run.side_effect = OSError("No bwrap")
    assert isolation_available() is False

@mock.patch("gwaya.test_harness.subprocess.Popen")
def test_launch_timeout(mock_popen):
    mock_proc = mock.Mock()
    mock_proc.pid = 1234
    
    def communicate(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="fake", timeout=1)
        
    mock_proc.communicate = communicate
    mock_popen.return_value = mock_proc
    
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        stdout, timed_out = _launch(["fake_cmd"], work, "nonce", 1.0, 128)
        assert timed_out is True

def test_run_candidate_tests_syntax_error():
    # Test invalid test spec
    res = run_candidate_tests("def f(): pass", "assert True = False")
    assert res.success is False
    assert "INVALID_SPEC" in res.error

def test_run_candidate_tests_no_asserts():
    # Test spec without asserts
    res = run_candidate_tests("def f(): pass", "x = 1")
    assert res.success is False
    assert "no assert statements" in res.error
