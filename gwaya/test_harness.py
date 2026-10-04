"""
anse/gwaya/test_harness.py
==========================
Isolated, bypass-resistant execution of a candidate against a test specification.

Why this exists (audit probes P1-P3): the previous implementation appended the test spec to
the candidate and trusted the process exit code. A candidate calling ``os._exit(0)`` passed
any spec, and the "Tier-1 sandbox" had full host filesystem, network and environment access.

Design
------
* A trusted runner script (``_RUNNER_SOURCE``) is the entry point, not the candidate.
* The parent sends a random nonce on stdin. The runner reads it, then closes stdin, before
  any candidate code runs.
* The runner executes the candidate, then each top-level statement of the spec in order.
  Every statement containing an ``assert`` is one test. Any ``BaseException`` (including
  ``SystemExit`` raised inside candidate functions) counts as a failure.
* Only after all statements ran does the runner print one JSON record tagged with the nonce.
  The parent accepts a pass only if that record exists, carries the nonce, reports the
  number of tests the parent counted itself, and ``passed == total``. Exit code is ignored.
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
``AssertionError``, host file or network access, and environment-secret leakage. It does
NOT defend against a candidate that deliberately introspects the interpreter (``gc``,
``sys._getframe``) to steal the nonce and forge a record; that requires out-of-process
evaluation of results and is out of scope here.
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
_RUNNER_SOURCE = r'''
import sys as _sys
_nonce = _sys.stdin.readline().strip()
_sys.stdin.close()
import ast as _ast, json as _json, os as _os, traceback as _tb
_dumps, _write, _stdout_fd = _json.dumps, _os.write, 1
_BaseException, _compile, _exec, _len = BaseException, compile, exec, len

def _emit(record):
    record["nonce"] = _nonce
    _write(_stdout_fd, ("\n" + _dumps(record) + "\n").encode())

def _short(exc):
    msg = "".join(_tb.format_exception_only(type(exc), exc)).strip()
    return msg[:500]

_candidate = open("candidate.py", encoding="utf-8").read()
_spec_src = open("spec.py", encoding="utf-8").read()
_ns = {"__name__": "candidate"}
try:
    _exec(_compile(_candidate, "candidate.py", "exec"), _ns)
except _BaseException as _e:
    _emit({"passed": 0, "total": -1, "stage": "import", "first_failure": _short(_e)})
    _os._exit(0)

_tree = _ast.parse(_spec_src)
_passed, _total, _first, _failed = 0, 0, None, []
_lines = _spec_src.splitlines()
for _stmt in _tree.body:
    _is_test = any(isinstance(_n, _ast.Assert) for _n in _ast.walk(_stmt))
    _idx = _total
    _total += 1 if _is_test else 0
    _src = "\n".join(_lines[_stmt.lineno - 1:_stmt.end_lineno])
    try:
        _exec(_compile(_ast.Module(body=[_stmt], type_ignores=[]), "spec.py", "exec"), _ns)
        _passed += 1 if _is_test else 0
    except _BaseException as _e:
        if _is_test:
            _failed.append(_idx)
        if _first is None:
            _first = _src.strip()[:300] + " -> " + _short(_e)
        if not _is_test:
            _remaining = _tree.body[_tree.body.index(_stmt) + 1:]
            _total += sum(1 for _s in _remaining if any(isinstance(_n, _ast.Assert) for _n in _ast.walk(_s)))
            break
_emit({"passed": _passed, "total": _total, "stage": "spec", "first_failure": _first, "failed": _failed})
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
    """Checks the runner's record against the parent's own assert count; '' means pass."""
    passed, total = int(record.get("passed", 0)), int(record.get("total", -1))
    if record.get("stage") == "import":
        return passed, f"IMPORT_FAILED: {record.get('first_failure')}"
    if total != expected_total:
        return passed, f"RESULT_MISMATCH: harness reported {total} tests, spec has {expected_total}"
    if passed != total:
        return passed, f"TEST_FAILED ({passed}/{total} passed): {record.get('first_failure')}"
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
    return HarnessResult(
        success=(error == ""),
        passed=passed,
        total=expected_total,
        error=error,
        isolation=isolation,
        duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        stdout_tail=tail,
        details={"failed_idx": [int(i) for i in record.get("failed", [])]},
    )
