"""Real-toolchain oracle tests (skip, never fake-pass, when lean/rustc are absent)."""
import shutil

import pytest

from gwaya.oracles import Lean4CompilerOracle, RustCompilerOracle

needs_lean = pytest.mark.skipif(shutil.which("lean") is None, reason="lean not installed")
needs_rust = pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc not installed")


@needs_lean
def test_lean_accepts_valid_proof():
    r = Lean4CompilerOracle().verify_snippet("theorem t (a b : Nat) : a + b = b + a := Nat.add_comm a b")
    assert r.success, r.error_message


@needs_lean
def test_lean_rejects_false_statement():
    assert not Lean4CompilerOracle().verify_snippet("theorem t : 1 = 2 := rfl").success


@needs_lean
@pytest.mark.parametrize("code", [
    "theorem t : 1 = 2 := by sorry",
    "axiom bad : False\ntheorem t : 1 = 2 := bad.elim",
    "theorem t : 2 + 2 = 4 := by native_decide",
])
def test_lean_rejects_escape_hatches(code):
    assert not Lean4CompilerOracle().verify_snippet(code).success


@needs_lean
def test_lean_axiom_audit_catches_lexical_bypass():
    # The "--" inside a string hid `axiom` from the old regex comment stripping (the `#print axioms` probe caught it);
    # the Lean-aware lexer (gwaya/lean_gate.strip_lean) now sees it directly.
    code = 'def s : String := "--" axiom bad : False\ntheorem t : 1 = 2 := bad.elim'
    r = Lean4CompilerOracle().verify_snippet(code)
    assert not r.success
    assert "axiom" in r.details.get("forbidden", []) or "bad" in r.details.get("disallowed_axioms", []), r.error_message


@needs_lean
@pytest.mark.parametrize("decl", [
    "private theorem t : 1 = 2 := bad.elim",
    "@[simp] theorem t : 1 = 2 := bad.elim",
    "def t : 1 = 2 := bad.elim",
    "example : 1 = 2 := bad.elim",
    "instance : Inhabited (1 = 2) := ⟨bad.elim⟩",
])
def test_lean_axiom_audit_covers_modifiers_and_defs(decl):
    code = 'def s : String := "--" axiom bad : False\n' + decl
    r = Lean4CompilerOracle().verify_snippet(code)
    assert not r.success, r.error_message


@needs_lean
def test_lean_namespace_closed_name_fails_closed():
    code = "namespace Foo\ntheorem t (a : Nat) : a = a := rfl\nend Foo"
    assert not Lean4CompilerOracle().verify_snippet(code).success


@needs_lean
def test_lean_comment_mentioning_axiom_is_accepted():
    code = "-- this proof uses no axiom\ntheorem t (a b : Nat) : a + b = b + a := Nat.add_comm a b"
    r = Lean4CompilerOracle().verify_snippet(code)
    assert r.success, r.error_message


def test_disallowed_axiom_parser():
    out = "'t' depends on axioms: [propext, Classical.choice, Lean.ofReduceBool]\n"
    assert Lean4CompilerOracle._disallowed_axioms(out, allow_sorry=False) == ["Lean.ofReduceBool"]
    assert Lean4CompilerOracle._disallowed_axioms("'t' depends on axioms: [sorryAx]", True) == []


@needs_rust
def test_rust_accepts_valid_and_rejects_borrow_error():
    o = RustCompilerOracle()
    assert o.verify_snippet("pub fn add(a: i32, b: i32) -> i32 { a + b }").success
    bad = "pub fn f() { let s = String::new(); let t = s; println!(\"{}\", s); }"
    assert not o.verify_snippet(bad).success
    assert not o.verify_snippet("pub fn f( {").success


def test_python_verify_with_test_sandboxed():
    from gwaya.oracles import PythonCompilerOracle
    oracle = PythonCompilerOracle()
    # Functional pass
    res_pass = oracle.verify_with_test("def square(x):\n    return x * x", "assert square(5) == 25")
    assert res_pass.success is True
    assert res_pass.compiler == "python_sandbox"

    # Functional failure (assertion error in sandbox)
    res_fail = oracle.verify_with_test("def square(x):\n    return x * 2", "assert square(5) == 25")
    assert res_fail.success is False
    assert "TEST_FAILED" in res_fail.error_message

