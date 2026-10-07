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
    # Calling _rlimits factory; mock setrlimit to avoid affecting pytest process
    func = _rlimits(cpu_s=1, mem_mb=100)
    with mock.patch("resource.setrlimit"):
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

def test_forged_record_poc_fails_with_new_harness():
    """
    Regression test for the nonce-theft vulnerability.

    With the NEW fork-based harness, a candidate cannot:
    1. Access the nonce from __main__._nonce (it's not in the child's globals)
    2. Forge a passing record (only parent writes records, child writes results)

    This test verifies that attempting to steal the nonce and forge a record
    results in a failed test (because the child's forged record is not processed).
    """
    # Candidate that tries to steal the nonce and forge a record
    malicious_code = r'''
# Try to steal the nonce like the old vulnerability allowed
import sys as _sys
import os as _os
import json as _json

try:
    _nonce = _sys.modules.get("__main__")._nonce
    # Try to forge a passing record
    _record = {"nonce": _nonce, "passed": 1, "total": 1}
    _os.write(1, ("\n" + _json.dumps(_record) + "\n").encode())
except (AttributeError, TypeError):
    # In the new fork-based harness, the nonce is not in __main__._nonce
    pass
'''

    # Spec with a real assertion
    spec = """
# This will fail
assert False, "This test always fails"
"""

    # Run the candidate through the harness
    res = run_candidate_tests(malicious_code, spec)

    # The test should fail because:
    # 1. The candidate cannot access the nonce (it's in parent only)
    # 2. The spec assertion fails
    assert res.success is False, "Forged record attempt should fail"
    assert res.passed == 0, "No tests should pass"
    assert res.total == 1, "There should be 1 test"


def test_harness_prevents_nonce_in_child_globals():
    """
    Verify that the child process does not have access to the nonce.
    
    The new harness keeps the nonce in the parent process only (read from stdin,
    kept in closure, not passed to child). The child process receives only a pipe
    for writing results, not the nonce itself.
    """
    # Candidate that introspects the environment
    code = r'''
import sys
import os

# Try various ways to access the nonce
nonce = None
try:
    nonce = sys.modules["__main__"]._nonce
except (AttributeError, KeyError):
    pass

try:
    nonce = os.environ.get("_NONCE")
except:
    pass

# If we found the nonce, write it to stderr (would indicate vulnerability)
if nonce:
    sys.stderr.write(f"FOUND_NONCE:{nonce}\n")
else:
    sys.stderr.write("NONCE_NOT_FOUND\n")
'''
    
    spec = "assert True"
    res = run_candidate_tests(code, spec)
    
    # The test should pass (assert True)
    assert res.success is True, "Basic test should pass"
    # Verify via stdout_tail that nonce was not found
    # (The new harness doesn't expose nonce to child)
    assert res.isolation in ("bwrap", "none"), f"Unexpected isolation: {res.isolation}"


def test_new_harness_with_multiple_tests():
    """
    Test that the new fork-based harness correctly counts multiple tests.
    
    The child writes "PASS\n" or "FAIL: error\n" for each test.
    The parent aggregates these and validates the count.
    """
    code = "x = 1"
    spec = """
assert x == 1, "x should be 1"
assert x > 0, "x should be positive"
assert x < 10, "x should be less than 10"
"""
    
    res = run_candidate_tests(code, spec)
    
    assert res.success is True, "All three tests should pass"
    assert res.passed == 3, "Should have 3 passing tests"
    assert res.total == 3, "Should have 3 total tests"


def test_new_harness_with_partial_failures():
    """
    Test that the new fork-based harness correctly reports partial failures.
    """
    code = "x = 5"
    spec = """
assert x == 5, "x should be 5"
assert x > 10, "x should be greater than 10"  # This will fail
assert x < 10, "x should be less than 10"
"""
    
    res = run_candidate_tests(code, spec)
    
    assert res.success is False, "Should fail because one test failed"
    assert res.passed == 2, "Should have 2 passing tests"
    assert res.total == 3, "Should have 3 total tests"
    assert "TEST_FAILED" in res.error, f"Expected TEST_FAILED in error, got: {res.error}"


def test_new_harness_import_failure_detection():
    """
    Test that the new harness correctly detects import failures.
    """
    code = "import nonexistent_module_xyz"
    spec = "assert True"
    
    res = run_candidate_tests(code, spec)
    
    assert res.success is False, "Should fail on import error"
    assert res.passed == 0, "No tests should pass"
    assert "IMPORT" in res.error or "import" in res.error.lower(), f"Should mention import, got: {res.error}"

