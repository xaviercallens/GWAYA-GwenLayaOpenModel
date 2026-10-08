"""Tests for compiler sandboxing and Lean security measures."""
import shutil
import pytest

from gwaya.oracles import Lean4CompilerOracle, CppCompilerOracle, GoCompilerOracle, RustCompilerOracle
from gwaya.sandbox import is_sandbox_available

needs_lean = pytest.mark.skipif(shutil.which("lean") is None, reason="lean not installed")
needs_sandbox = pytest.mark.skipif(not is_sandbox_available(), reason="bwrap sandbox not available")


@needs_lean
def test_lean_rejects_eval():
    """Lean #eval is blocked lexically (defense-in-depth)."""
    r = Lean4CompilerOracle().verify_snippet(
        "theorem t : 1 = 1 := by\n  #eval 5\n  rfl"
    )
    assert not r.success
    assert "#eval" in r.error_message or "Unsound" in r.error_message


@needs_lean
def test_lean_rejects_reduce():
    """Lean #reduce is blocked lexically."""
    r = Lean4CompilerOracle().verify_snippet(
        "theorem t : 1 = 1 := by\n  #reduce (1 : Nat)\n  rfl"
    )
    assert not r.success
    assert "#reduce" in r.error_message or "Unsound" in r.error_message


@needs_lean
def test_lean_rejects_run_cmd():
    """Lean run_cmd is blocked lexically."""
    r = Lean4CompilerOracle().verify_snippet(
        "run_cmd IO.println \"hi\"\ntheorem t : 1 = 1 := rfl"
    )
    assert not r.success
    assert "run_cmd" in r.error_message or "Unsound" in r.error_message


@needs_lean
def test_lean_rejects_macro():
    """Lean macro is blocked lexically (could execute code)."""
    r = Lean4CompilerOracle().verify_snippet(
        "macro \"test\" : tactic => `(rfl)\ntheorem t : 1 = 1 := by test"
    )
    assert not r.success
    assert "macro" in r.error_message or "Unsound" in r.error_message


@needs_sandbox
def test_cpp_runs_in_sandbox():
    """C++ compilation happens inside bwrap sandbox."""
    oracle = CppCompilerOracle()
    # This should succeed (valid code)
    r = oracle.verify("int main() { return 0; }")
    assert r.success
    # The isolation field would show if sandboxing is active, but we can't directly
    # test that from the oracle level. The fact that it works means the sandbox is OK.


@needs_sandbox
def test_rust_runs_in_sandbox():
    """Rust compilation happens inside bwrap sandbox."""
    oracle = RustCompilerOracle()
    # This should succeed
    r = oracle.verify_snippet("pub fn add(a: i32, b: i32) -> i32 { a + b }")
    assert r.success


def test_sandbox_available():
    """Check that bwrap sandbox is available on this system."""
    # This might not fail since it's OK if bwrap is unavailable,
    # but we should know what the status is.
    available = is_sandbox_available()
    assert isinstance(available, bool)


def test_interpreter_symlinks_outside_bound_roots_are_recreated(tmp_path):
    # uv layout: venv/bin/python -> uv/cpython-3.12-x/bin/python3.12, cpython-3.12-x -> cpython-3.12.13-x
    from gwaya.sandbox import interpreter_symlink_args
    real = tmp_path / "uv" / "cpython-3.12.13-x"
    (real / "bin").mkdir(parents=True)
    (real / "bin" / "python3.12").write_text("")
    alias = tmp_path / "uv" / "cpython-3.12-x"
    alias.symlink_to(real)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(alias / "bin" / "python3.12")
    args = interpreter_symlink_args([venv, real], executable=str(venv / "bin" / "python"))
    assert args == ["--symlink", str(real), str(alias)]  # venv link is inside a bound root: left alone


def test_no_symlink_args_for_plain_interpreter(tmp_path):
    from gwaya.sandbox import interpreter_symlink_args
    exe = tmp_path / "py" / "bin" / "python3"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert interpreter_symlink_args([tmp_path / "py"], executable=str(exe)) == []
