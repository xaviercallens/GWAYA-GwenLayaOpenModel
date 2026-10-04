"""
tests/test_rust_oracle.py
=========================
Unit tests for RustCompilerOracle on real toolchain.
Evaluates:
- Valid simple function compilation
- Generics with trait bounds
- Borrow checker move-after-use rejections
- Type mismatches
- Rust placeholder stubs (todo!(), unimplemented!(), panic!())
"""
from __future__ import annotations

import shutil
import pytest

from gwaya.oracles import RustCompilerOracle


@pytest.fixture
def oracle():
    return RustCompilerOracle()


def test_rust_oracle_available(oracle):
    if not shutil.which("rustc"):
        pytest.skip("rustc not installed on PATH")
    assert oracle.available is True


def test_rust_valid_simple_function(oracle):
    if not oracle.available:
        pytest.skip("rustc not available")
    code = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n"
    res = oracle.verify_snippet(code)
    assert res.success is True
    assert res.compiler == "rustc"
    assert res.latency_ms > 0


def test_rust_valid_generic_function(oracle):
    if not oracle.available:
        pytest.skip("rustc not available")
    code = """
pub fn largest<T: PartialOrd + Copy>(list: &[T]) -> Option<T> {
    if list.is_empty() {
        return None;
    }
    let mut largest = list[0];
    for &item in list.iter() {
        if item > largest {
            largest = item;
        }
    }
    Some(largest)
}
"""
    res = oracle.verify_snippet(code)
    assert res.success is True


def test_rust_rejects_borrow_checker_error(oracle):
    if not oracle.available:
        pytest.skip("rustc not available")
    code = """
pub fn borrow_error() {
    let s = String::from("hello");
    let t = s;
    println!("{}", s);
}
"""
    res = oracle.verify_snippet(code)
    assert res.success is False
    assert "borrow of moved value" in res.error_message or "use of moved value" in res.error_message or "moved" in res.error_message


def test_rust_rejects_type_mismatch(oracle):
    if not oracle.available:
        pytest.skip("rustc not available")
    code = """
pub fn add_wrong(a: i32, b: i32) -> i32 {
    "not a number"
}
"""
    res = oracle.verify_snippet(code)
    assert res.success is False
    assert "mismatched types" in res.error_message or "expected `i32`" in res.error_message


def test_rust_rejects_syntax_error(oracle):
    if not oracle.available:
        pytest.skip("rustc not available")
    code = "pub fn broken(a: i32 { a +"
    res = oracle.verify_snippet(code)
    assert res.success is False
    assert res.error_message != ""


@pytest.mark.parametrize("stub", [
    "pub fn calc() -> i32 { todo!() }",
    "pub fn calc() -> i32 { unimplemented!() }",
    "pub fn calc() -> i32 { panic!(\"stub\") }",
])
def test_rust_rejects_placeholder_stubs(oracle, stub):
    if not oracle.available:
        pytest.skip("rustc not available")
    res = oracle.verify_snippet(stub)
    assert res.success is False
    assert "STUB_DETECTED" in res.error_message
