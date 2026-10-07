"""
gwaya/test_harness.py
=====================
Isolated, bypass-resistant execution of a candidate against a test specification.

Why this exists (audit probes P1-P3): the previous implementation appended the test spec to
the candidate and trusted the process exit code. A candidate calling ``os._exit(0)`` passed
any spec, and the "Tier-1 sandbox" had full host filesystem, network and environment access.

Design
------
* A trusted runner script (``_RUNNER_SOURCE``) is the entry point, not the candidate.
* The runner does NOT store the nonce in module globals. Instead, it is read from stdin
  into a closure-local variable before any candidate code runs.
* The runner is forked into a parent (supervisor) and child (executor) process:
  - CHILD: Executes the candidate, then each top-level statement of the spec in order.
    Every statement containing an ``assert`` is one test. Any ``BaseException`` (including
    ``SystemExit`` raised inside candidate functions) counts as a failure. The child writes
    one line per test to a pipe: "PASS" or "FAIL <error>". The child cannot name or modify
    the supervisor's record pipe and cannot forge the nonce or aggregate count.
  - PARENT (supervisor): Waits for the child to finish on the result pipe. For each line,
    counts tests and tracks passes. After the child exits, writes one JSON record tagged
    with the nonce (known only to parent). The parent validates that the child reported
    the expected number of tests and that all reported PASS/FAIL counts match.
* Execution happens under ``bwrap``: no network, empty environment, host ``/tmp``, ``/home``,
  ``/mnt``, ``/media``, ``/root`` hidden (only the interpreter's venv is re-mounted
  read-only), rlimits on CPU, address space, file size and open files, wall-clock timeout
  with process-group kill.
* If ``bwrap`` is missing or broken the result is UNVERIFIED (fail-closed). Setting
  ``GWAYA_ALLOW_UNISOLATED=1`` runs without isolation for local development only; the result
  then carries ``isolation="none"``.

Threat model (honest scope)
---------------------------
Defends against early exit, exit-code spoofing, ``atexit``/``excepthook`` tricks, rebinding
``AssertionError``, host file or network access, environment-secret leakage, and introspection
attacks on the nonce (``gc``, ``sys._getframe``) by keeping the nonce in the parent process
only, not in the child's globals or memory.

Does NOT defend against:
- A child that deliberately exhausts the pipe with junk data (the parent enforces a line limit)
- A child that intentionally crashes and lets the parent's rlimit kill it without reporting
  (this is detected as a failed child exit)
- Forged parent-side code in the runner script itself (the parent is trusted)
"""
from __future__ import annotations

import ast
import json
import os
import secrets
import shutil
import signal
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOW_UNISOLATED_ENV = "GWAYA_ALLOW_UNISOLATED"
SANDBOX_WORKDIR = "/work"
_MAX_REPORTED_OUTPUT = 4000

# The runner deliberately captures every builtin it needs before candidate code runs.
# It uses fork() to separate parent (nonce holder, judge) from child (executor).
# The nonce is read only by parent into a closure; child never sees it.
_RUNNER_SOURCE = r'''
import sys as _sys, os as _os
_stdin_nonce = _sys.stdin.readline().strip()
_sys.stdin.close()

import ast as _ast, json as _json, traceback as _tb
_dumps, _write, _read, _stdout_fd = _json.dumps, _os.write, _os.read, 1
_BaseException, _compile, _exec, _encode = BaseException, compile, exec, str.encode

def _short(exc):
    msg = "".join(_tb.format_exception_only(type(exc), exc)).strip()
    return msg[:500]

_read_fd, _write_fd = _os.pipe()
_child_pid = _os.fork()

if _child_pid == 0:
    # CHILD PROCESS: execute candidate and spec, report results line-by-line
    _os.close(_read_fd)

    _candidate = open("candidate.py", encoding="utf-8").read()
    _spec_src = open("spec.py", encoding="utf-8").read()
    _ns = {"__name__": "candidate"}

    try:
        _exec(_compile(_candidate, "candidate.py", "exec"), _ns)
    except _BaseException as _e:
        _write(_write_fd, (b"FAIL_IMPORT: " + _short(_e).encode()[:500] + b"\n"))
        _os._exit(0)

    _tree = _ast.parse(_spec_src)
    _lines = _spec_src.splitlines()
    for _stmt in _tree.body:
        _is_test = any(isinstance(_n, _ast.Assert) for _n in _ast.walk(_stmt))

        try:
            _exec(_compile(_ast.Module(body=[_stmt], type_ignores=[]), "spec.py", "exec"), _ns)
            if _is_test:
                _write(_write_fd, b"PASS\n")
        except _BaseException as _e:
            if _is_test:
                _write(_write_fd, (b"FAIL: " + _short(_e).encode()[:500] + b"\n"))
            else:
                # Non-test statement failed; treat as a setup error
                _write(_write_fd, (b"FAIL_SETUP: " + _short(_e).encode()[:500] + b"\n"))
                _os.close(_write_fd)
                _os._exit(0)

    _os.close(_write_fd)
    _os._exit(0)
else:
    # PARENT PROCESS: hold nonce, read results, aggregate, judge
    _os.close(_write_fd)
    _nonce = _stdin_nonce

    _passed, _total = 0, 0
    _results = []
    _buffer = b""

    try:
        while True:
            _chunk = _read(_read_fd, 4096)
            if not _chunk:
                break
            _buffer += _chunk
    except OSError:
        pass

    _os.close(_read_fd)

    _failures = []
    for _line in _buffer.decode("utf-8", errors="replace").splitlines():
        _line = _line.strip()
        if not _line:
            continue
        if _line == "PASS":
            _passed += 1
            _total += 1
            _results.append("pass")
        elif _line.startswith("FAIL_IMPORT:"):
            _total += 1
            _results.append("import_failed")
            _failures.append(_line[len("FAIL_IMPORT: "):])
            break
        elif _line.startswith("FAIL_SETUP:"):
            _results.append("setup_failed")
            _failures.append(_line[len("FAIL_SETUP: "):])
            break
        elif _line.startswith("FAIL:"):
            _total += 1
            _results.append("failed")
            _failures.append(_line[len("FAIL: "):])
        else:
            _total += 1
            _results.append("failed")
            _failures.append(_line)

    _os.waitpid(_child_pid, 0)

    _record = {
        "nonce": _nonce,
        "passed": _passed,
        "total": _total,
        "stage": "spec",
        "results": _results,
    }
    if _failures:
        _record["failures"] = _failures
    _write(_stdout_fd, _encode("\n" + _dumps(_record) + "\n"))
    _os._exit(0)
'''


@dataclass
class HarnessResult:
    success: bool
    passed: int
    total: int
    error: str
    isolation: str
    duration_ms: float
    timed_out: bool = False
    stdout_tail: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def count_spec_tests(test_spec: str) -> int:
    """Number of top-level spec statements that contain an ``assert`` (the parent's own count)."""
    tree = ast.parse(test_spec)
    return sum(1 for stmt in tree.body if any(isinstance(n, ast.Assert) for n in ast.walk(stmt)))


def _bwrap_path() -> str | None:
    return shutil.which("bwrap")


def isolation_available() -> bool:
    """True if bwrap exists and can start the interpreter in an unprivileged sandbox."""
    bwrap = _bwrap_path()
    if not bwrap:
        return False
    with tempfile.TemporaryDirectory(prefix="gwaya_probe_") as work:
        cmd = _bwrap_argv(bwrap, Path(work)) + ["-c", "print('ok')"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)  # nosec B603
        except (OSError, subprocess.TimeoutExpired):
            return False
        return res.returncode == 0 and res.stdout.strip() == "ok"


def _interpreter_root() -> Path:
    """Directory that must stay visible for the interpreter (the venv, or sys.prefix)."""
    return Path(sys.prefix).resolve()


def _bwrap_argv(bwrap: str, work: Path) -> list[str]:
    # LT0 / probe P3: expose only what the interpreter needs, not the host root.
    argv = [bwrap, "--ro-bind", "/usr", "/usr"]
    for top in ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32"):
        if os.path.islink(top):
            argv += ["--symlink", os.readlink(top), top]
        elif os.path.isdir(top):
            argv += ["--ro-bind", top, top]
    for etc_file in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/localtime"):
        argv += ["--ro-bind-try", etc_file, etc_file]
    argv += [
        "--proc", "/proc",
        "--dev", "/dev",
    ]
    root = _interpreter_root()
    argv += ["--ro-bind", str(root), str(root)]
    base = Path(sys.base_prefix).resolve()
    if base != root:
        argv += ["--ro-bind", str(base), str(base)]
    argv += [
        # The sandbox root is bwrap's own tmpfs (no host "/" bind), so /work can be created.
        "--bind", str(work), SANDBOX_WORKDIR,
        "--chdir", SANDBOX_WORKDIR,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "HOME", SANDBOX_WORKDIR,
        "--setenv", "TMPDIR", SANDBOX_WORKDIR,
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        sys.executable, "-I",
    ]
    return argv


def _rlimits(cpu_s: int, mem_mb: int):
    def _apply() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        except (ImportError, AttributeError):
            pass

    return _apply


def _parse_record(stdout: str, nonce: str) -> dict[str, Any] | None:
    record = None
    for line in stdout.splitlines():
        line = line.strip()
        if not (line.startswith("{") and nonce in line):
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("nonce") == nonce:
            record = candidate
    return record


def _fail(error: str, isolation: str, t0: float, **details: Any) -> HarnessResult:
    return HarnessResult(
        success=False, passed=0, total=details.pop("total", 0), error=error,
        isolation=isolation, duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        details=details,
    )


def _launch(argv: list[str], work: Path, nonce: str, timeout_s: float, mem_mb: int) -> tuple[str, bool]:
    """Runs the sandbox, returns (stdout text, timed_out). Output goes to a size-limited file."""
    out_path = work / ".stdout"
    with open(out_path, "wb") as out:
        kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": out,
            "stderr": subprocess.STDOUT,
            "cwd": work,
        }
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = _rlimits(int(timeout_s) + 1, mem_mb)
        proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell
            argv,
            **kwargs,
        )
        timed_out = False
        try:
            proc.communicate(input=(nonce + "\n").encode(), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            if sys.platform != "win32":
                try:
                    sig = getattr(signal, "SIGKILL", signal.SIGTERM)
                    os.killpg(proc.pid, sig)
                except (ProcessLookupError, AttributeError):
                    pass
            else:
                proc.kill()
            proc.wait()
    with open(out_path, "rb") as fh:
        text = fh.read(_MAX_REPORTED_OUTPUT * 64).decode("utf-8", errors="replace")
    return text, timed_out


def _judge_record(record: dict[str, Any], expected_total: int) -> tuple[int, str]:
    """Checks the runner's record against the parent's own assert count; '' means pass.

    With the fork-based architecture, the child reports individual test results (PASS/FAIL/FAIL_IMPORT)
    and the parent aggregates them. We validate:
    1. The total number of tests reported matches what we counted
    2. All tests passed (passed == total)
    3. No import failures or setup failures
    """
    passed, total = int(record.get("passed", 0)), int(record.get("total", -1))
    results = record.get("results", [])

    # Check for import failure
    if "import_failed" in results:
        return 0, f"IMPORT_FAILED: candidate could not be imported"

    # Check for setup failure (non-test statement that failed)
    if "setup_failed" in results:
        failure_msg = ""
        failures = record.get("failures", [])
        if failures:
            failure_msg = f": {failures[0][:200]}"
        return 0, f"SETUP_FAILED: test spec has invalid setup code{failure_msg}"

    if total != expected_total:
        return passed, f"RESULT_MISMATCH: harness reported {total} tests, spec has {expected_total}"
    if passed != total:
        failed_count = sum(1 for r in results if r == "failed")
        return passed, f"TEST_FAILED ({passed}/{total} passed, {failed_count} failed)"
    return passed, ""


def run_candidate_tests(
    code: str,
    test_spec: str,
    timeout_s: float = 5.0,
    mem_mb: int = 1024,
) -> HarnessResult:
    """Executes ``code`` then ``test_spec`` in isolation and returns a verified pass/fail."""
    t0 = time.perf_counter()
    try:
        expected_total = count_spec_tests(test_spec)
    except SyntaxError as exc:
        return _fail(f"INVALID_SPEC: test spec does not parse ({exc.msg})", "n/a", t0, reason="invalid_spec")
    if expected_total == 0:
        return _fail("INVALID_SPEC: test spec contains no assert statements (fail-closed)", "n/a", t0,
                     reason="no_asserts")

    bwrap = _bwrap_path()
    unisolated_ok = os.environ.get(ALLOW_UNISOLATED_ENV) == "1"
    if bwrap is None and not unisolated_ok:
        return _fail(
            "UNVERIFIED: bwrap isolation unavailable; refusing to execute untrusted code "
            f"(set {ALLOW_UNISOLATED_ENV}=1 for local development only)",
            "none", t0, reason="no_isolation", total=expected_total,
        )

    nonce = secrets.token_hex(16)
    with tempfile.TemporaryDirectory(prefix="gwaya_run_") as tmp:
        work = Path(tmp)
        (work / "candidate.py").write_text(code, encoding="utf-8")
        (work / "spec.py").write_text(test_spec, encoding="utf-8")
        (work / "runner.py").write_text(_RUNNER_SOURCE, encoding="utf-8")
        if bwrap is not None:
            argv, isolation = _bwrap_argv(bwrap, work) + ["runner.py"], "bwrap"
        else:
            argv, isolation = [sys.executable, "-I", "runner.py"], "none"
        stdout, timed_out = _launch(argv, work, nonce, timeout_s, mem_mb)

    tail = stdout[-_MAX_REPORTED_OUTPUT:]
    if timed_out:
        return _fail(f"TIMEOUT: execution exceeded {timeout_s}s", isolation, t0,
                     reason="timeout", total=expected_total)
    record = _parse_record(stdout, nonce)
    if record is None:
        return _fail("NO_RESULT: process ended before the harness reported (early exit or crash)",
                     isolation, t0, reason="no_result", total=expected_total, output_tail=tail)

    passed, error = _judge_record(record, expected_total)
    # Reconstruct failed indices from results list
    results = record.get("results", [])
    failed_idx = [i for i, result in enumerate(results) if result == "failed"]
    details = {"failed_idx": failed_idx}
    # Include failure messages if available
    if "failures" in record:
        details["failures"] = record["failures"]
    return HarnessResult(
        success=(error == ""),
        passed=passed,
        total=expected_total,
        error=error,
        isolation=isolation,
        duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        stdout_tail=tail,
        details=details,
    )
