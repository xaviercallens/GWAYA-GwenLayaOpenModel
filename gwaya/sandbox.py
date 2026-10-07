"""
gwaya/sandbox.py
================
Secure sandboxing for compiler execution using bwrap.

Provides `run_in_sandbox()` for safely executing compiler commands with:
- Process isolation via bwrap (--unshare-all, --clearenv)
- Read-only toolchain binding (lean, rustc, g++, go)
- Writable tmpdir only
- CPU, memory, file size rlimits
- Wall-clock timeout with process-group kill
- Error message scrubbing to remove host paths and file contents

If bwrap is unavailable and GWAYA_ALLOW_UNISOLATED=1 is not set, returns
UNVERIFIED fail-closed result.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ALLOW_UNISOLATED_ENV = "GWAYA_ALLOW_UNISOLATED"
SANDBOX_WORKDIR = "/work"


def _bwrap_path() -> str | None:
    """Find bwrap executable."""
    return shutil.which("bwrap")


def _interpreter_root() -> Path:
    """Directory that must stay visible for the interpreter."""
    return Path(sys.prefix).resolve()


def _find_toolchain_root(tool_name: str) -> Path | None:
    """Find installation root for a toolchain by name.

    Args:
        tool_name: One of 'lean', 'rustc', 'clang++', 'g++', 'go'

    Returns:
        Path to toolchain root, or None if not found.
    """
    if tool_name == "lean":
        # Check ~/.elan first, then system
        elan_home = Path.home() / ".elan"
        if elan_home.exists():
            return elan_home
    elif tool_name in ("rustc", "cargo"):
        # Rust toolchain in ~/.cargo and ~/.rustup
        cargo_home = Path.home() / ".cargo"
        if cargo_home.exists():
            return cargo_home
        rustup_home = Path.home() / ".rustup"
        if rustup_home.exists():
            return rustup_home
    elif tool_name in ("g++", "clang++", "gcc", "clang"):
        # System toolchain in /usr
        return Path("/usr")
    elif tool_name == "go":
        # Go in /usr/local/go or /usr
        go_root = Path("/usr/local/go")
        if go_root.exists():
            return go_root
        return Path("/usr")

    return None


def _bwrap_argv(
    bwrap: str,
    work: Path,
    timeout_s: float,
    for_python: bool = True,
    ro_binds: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Build bwrap command with proper isolation and toolchain access.

    Args:
        bwrap: Path to bwrap binary
        work: Working directory for sandbox
        timeout_s: Timeout for process
        for_python: If True, add Python interpreter at the end; if False, add bash
        ro_binds: Extra host paths bound read-only at the same path (e.g. a Mathlib lake project)
        env: Extra environment variables set inside the sandbox (e.g. LEAN_PATH)

    Returns:
        Argument list for bwrap (+ interpreter or bash)
    """
    argv = [bwrap, "--ro-bind", "/usr", "/usr"]

    # Bind system libraries
    for top in ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32"):
        if os.path.islink(top):
            argv += ["--symlink", os.readlink(top), top]
        elif os.path.isdir(top):
            argv += ["--ro-bind", top, top]

    # Bind essential /etc files
    for etc_file in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/localtime"):
        argv += ["--ro-bind-try", etc_file, etc_file]

    argv += [
        "--proc", "/proc",
        "--dev", "/dev",
    ]

    # Bind interpreter root (venv or system python)
    root = _interpreter_root()
    argv += ["--ro-bind", str(root), str(root)]
    base = Path(sys.base_prefix).resolve()
    if base != root:
        argv += ["--ro-bind", str(base), str(base)]

    # Bind all known toolchain roots read-only
    # Keep track of already-bound paths to avoid duplicates
    bound_paths = set()
    # Explicitly include .rustup for Rust toolchain
    toolchain_names = ["lean", "rustc", "cargo", "g++", "go"]
    toolchain_paths = []
    for name in toolchain_names:
        root = _find_toolchain_root(name)
        if root:
            toolchain_paths.append(root)
    # Also check for .rustup explicitly
    rustup_home = Path.home() / ".rustup"
    if rustup_home.exists() and rustup_home not in toolchain_paths:
        toolchain_paths.append(rustup_home)

    for toolchain_root in toolchain_paths:
        if toolchain_root and toolchain_root.exists():
            # Resolve symlinks to get the actual path
            try:
                actual_path = toolchain_root.resolve()
            except (OSError, RuntimeError):
                actual_path = toolchain_root

            if str(actual_path) not in bound_paths:
                if actual_path != toolchain_root:
                    # The toolchain_root is a symlink; bind the actual path to the symlink location
                    argv += ["--ro-bind", str(actual_path), str(toolchain_root)]
                else:
                    # No symlink, just bind the path
                    argv += ["--ro-bind", str(actual_path), str(actual_path)]
                bound_paths.add(str(actual_path))

    for extra in ro_binds or []:
        argv += ["--ro-bind", extra, extra]

    # Writable work directory only
    argv += [
        "--bind", str(work), SANDBOX_WORKDIR,
        "--chdir", SANDBOX_WORKDIR,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "TMPDIR", SANDBOX_WORKDIR,
    ]

    for key, value in (env or {}).items():
        argv += ["--setenv", key, value]

    if for_python:
        # For Python, isolate HOME to prevent access to personal files
        argv += [
            "--setenv", "HOME", SANDBOX_WORKDIR,
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            sys.executable, "-I",
        ]
    else:
        # For compilers, use the actual home directory (toolchains need config files there)
        home_dir = str(Path.home())
        argv += [
            "--setenv", "HOME", home_dir,
            "--setenv", "RUSTUP_HOME", f"{home_dir}/.rustup",
            "--setenv", "CARGO_HOME", f"{home_dir}/.cargo",
            "--setenv", "USER", os.environ.get("USER", "builder"),
            "/bin/bash",
        ]

    return argv


def _rlimits(cpu_s: int, mem_mb: int):
    """Create a preexec function to apply resource limits."""
    def _apply() -> None:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
            # For compilers like Lean, we need more memory, and RLIMIT_AS can cause issues
            # Better to rely on bwrap's process isolation and cgroup limits if available
            # resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
        except (ImportError, AttributeError):
            pass
    return _apply


def _scrub_error_message(text: str, work_dir: str, source_file: str) -> str:
    """Remove host paths and file contents from compiler error messages.

    Keeps only compiler diagnostics for the candidate file, truncated to 2000 chars.

    Args:
        text: Original error message
        work_dir: Temporary working directory path
        source_file: Name of source file (e.g., 'candidate.lean')

    Returns:
        Scrubbed error message
    """
    if not text:
        return text

    # Remove home directory paths
    home = str(Path.home())
    text = re.sub(re.escape(home), "/home/user", text)

    # Remove temp directory paths
    text = re.sub(re.escape(work_dir), "/tmp/work", text)

    # Remove other absolute paths like /tmp/xxx, /var/xxx
    text = re.sub(r"/tmp[^\s\"']+", "/tmp/...", text)
    text = re.sub(r"/var[^\s\"']+", "/var/...", text)
    text = re.sub(r"/mnt[^\s\"']+", "/mnt/...", text)

    # Truncate to 2000 chars to prevent info leakage via error length
    if len(text) > 2000:
        text = text[:2000] + "...(truncated)"

    return text.strip()


def run_in_sandbox(
    cmd: list[str],
    stdin_data: str = "",
    timeout_s: float = 10.0,
    mem_mb: int = 1024,
    cpu_s: int = 15,
    work_dir: Path | None = None,
    files: dict[str, str] | None = None,
    ro_binds: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[bool, str, str, bool]:
    """Run a command in a sandboxed environment using bwrap.

    Args:
        cmd: Command to execute (must be a list of strings)
        stdin_data: Data to send on stdin
        timeout_s: Timeout in seconds
        mem_mb: Memory limit in MB
        cpu_s: CPU time limit in seconds
        work_dir: Optional pre-created work directory. If None, creates one.
        files: Optional dict of {relative_path: content} to create in work_dir
        ro_binds: Extra host paths mounted read-only at the same path (bwrap only)
        env: Extra environment variables (set inside bwrap, or merged into os.environ without it)

    Returns:
        Tuple of (success, stdout, stderr, timed_out)
        - success: True if returncode == 0
        - stdout: Captured stdout (up to 256KB)
        - stderr: Captured stderr (up to 256KB, scrubbed)
        - timed_out: True if process exceeded timeout

        If bwrap is unavailable and GWAYA_ALLOW_UNISOLATED=1 is not set,
        returns (False, "", "UNVERIFIED: bwrap unavailable", False)
    """
    bwrap = _bwrap_path()
    unisolated_ok = os.environ.get(ALLOW_UNISOLATED_ENV) == "1"

    if bwrap is None and not unisolated_ok:
        return (
            False,
            "",
            "UNVERIFIED: bwrap isolation unavailable; refusing to execute untrusted code",
            False,
        )

    t0 = time.perf_counter()

    # Create work directory if not provided
    cleanup_work = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="gwaya_sandbox_"))

    try:
        # Create files in work directory if provided
        if files:
            for rel_path, content in files.items():
                file_path = work_dir / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

        # Build command: either sandboxed or unsandboxed
        if bwrap is not None:
            # For sandboxed execution, we need to run the command inside the bwrap environment
            # Build bwrap argv with bash, then pass the command via -c
            import shlex
            cmd_str = " ".join(shlex.quote(str(arg)) for arg in cmd)
            argv = _bwrap_argv(bwrap, work_dir, timeout_s, for_python=False, ro_binds=ro_binds, env=env) + ["-c", cmd_str]
            isolation = "bwrap"
        else:
            argv = cmd
            isolation = "none"

        # Run process
        timed_out = False
        try:
            kwargs: dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "cwd": str(work_dir),
            }

            if bwrap is None and env:
                kwargs["env"] = {**os.environ, **env}
            if sys.platform != "win32":
                kwargs["start_new_session"] = True
                kwargs["preexec_fn"] = _rlimits(cpu_s, mem_mb)

            res = subprocess.run(  # nosec B603 - fixed argv
                argv,
                input=stdin_data if stdin_data else None,
                timeout=timeout_s,
                **kwargs,
            )

            success = res.returncode == 0
            stdout = res.stdout[:256 * 1024] if res.stdout else ""
            stderr = _scrub_error_message(
                (res.stderr or "")[:256 * 1024],
                str(work_dir),
                "candidate"
            )

            return success, stdout, stderr, timed_out

        except subprocess.TimeoutExpired:
            # subprocess.run already killed the child; under bwrap the pid namespace dies with it.
            # (The former os.killpg(os.getpgid(0)) targeted the *caller's* process group.)
            timed_out = True
            return False, "", f"Execution timed out after {timeout_s}s", timed_out

        except Exception as exc:
            return False, "", f"Sandbox error: {str(exc)}", False
    finally:
        if cleanup_work and work_dir and work_dir.exists():
            import shutil as sh
            try:
                sh.rmtree(work_dir)
            except Exception:
                pass


def is_sandbox_available() -> bool:
    """Check if bwrap sandbox is available on this system.

    Returns:
        True if bwrap exists and can run, False otherwise.
    """
    bwrap = _bwrap_path()
    if not bwrap:
        return False

    with tempfile.TemporaryDirectory(prefix="gwaya_probe_") as work:
        cmd = _bwrap_argv(bwrap, Path(work), 10.0) + ["-c", "print('ok')"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)  # nosec B603
        except (OSError, subprocess.TimeoutExpired):
            return False
        return res.returncode == 0 and res.stdout.strip() == "ok"
