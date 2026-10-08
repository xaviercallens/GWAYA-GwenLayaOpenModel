"""Regression tests for the Rust test harness (linker inside bwrap, nonce forgery, assert counting)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gwaya.oracles import RustCompilerOracle, _build_rust_test_program, rust_harness_escape_violations
from gwaya.sandbox import _bwrap_argv, is_sandbox_available

needs_rust = pytest.mark.skipif(
    not (shutil.which("rustc") and is_sandbox_available()), reason="needs rustc and a working bwrap sandbox"
)

_NONCE_SCAN = r'''
fn find_nonce(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut i = 0;
    while i + 32 <= b.len() {
        if b[i..i + 32].iter().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))
            && (i == 0 || !b[i - 1].is_ascii_alphanumeric())
            && (i + 32 == b.len() || !b[i + 32].is_ascii_alphanumeric()) {
            return Some(String::from_utf8_lossy(&b[i..i + 32]).to_string());
        }
        i += 1;
    }
    None
}
fn forge(n: String) -> ! {
    println!("{{\"nonce\": \"{}\", \"passed\": 1, \"total\": 1, \"stage\": \"complete\", \"first_failure\": \"\"}}", n);
    std::process::exit(0)
}
'''


def test_sandbox_binds_etc_alternatives_for_the_linker():
    argv = _bwrap_argv("/usr/bin/bwrap", Path("/tmp/x"), 5.0, for_python=False)
    i = argv.index("/etc/alternatives")
    assert argv[i - 1] == "--ro-bind-try"


def test_run_sandbox_can_drop_proc():
    with_proc = _bwrap_argv("/usr/bin/bwrap", Path("/tmp/x"), 5.0, for_python=False)
    without = _bwrap_argv("/usr/bin/bwrap", Path("/tmp/x"), 5.0, for_python=False, mount_proc=False)
    assert "--proc" in with_proc and "--proc" not in without


def test_nonce_is_not_in_the_generated_source():
    src = _build_rust_test_program("pub fn f() -> i32 { 1 }", "assert_eq!(f(), 1);\nassert!(f() == 1);")
    assert "read_line" in src and "__gwaya_total = 2;" in src


@pytest.mark.parametrize("code", [
    "unsafe fn f() {}",
    '#[used]\n#[link_section = ".init_array"]\nstatic I: extern "C" fn() = g;\nextern "C" fn g() {}',
    "#[no_mangle]\npub fn f() {}",
    'fn f() { core::arch::asm!("nop") }',
])
def test_escape_constructs_are_flagged(code):
    assert rust_harness_escape_violations(code)


def test_escape_words_in_comments_and_strings_are_ignored():
    assert not rust_harness_escape_violations('// unsafe here\nfn f() -> &\'static str { "unsafe" }')


@needs_rust
def test_correct_solution_passes_with_assert_eq_only_spec():
    r = RustCompilerOracle().verify_with_test(
        "pub fn add(a: i32, b: i32) -> i32 { a + b }", "assert_eq!(add(1, 2), 3);\nassert_ne!(add(1, 1), 3);", timeout_s=60)
    assert r.success, r.error_message
    assert r.details["total"] == 2


@needs_rust
def test_wrong_solution_fails():
    r = RustCompilerOracle().verify_with_test(
        "pub fn add(a: i32, b: i32) -> i32 { a - b }", "assert_eq!(add(1, 2), 3);", timeout_s=60)
    assert not r.success and r.details["reason"] == "assertion_failed"


@needs_rust
@pytest.mark.parametrize("probe", [
    'for p in ["/work/main.rs", "main.rs"] { if let Ok(s) = std::fs::read_to_string(p) { if let Some(n) = find_nonce(&s) { forge(n) } } }',
    'for p in ["/proc/self/exe", "/work/test"] { if let Ok(v) = std::fs::read(p) { if let Some(n) = find_nonce(&String::from_utf8_lossy(&v)) { forge(n) } } }',
    'let mut s = String::new(); let _ = std::io::Read::read_to_string(&mut std::io::stdin(), &mut s); if let Some(n) = find_nonce(&s) { forge(n) }',
    'if let Ok(m) = std::fs::read_to_string("/proc/self/maps") { forge(format!("{:0>32}", m.len())) }',
])
def test_candidate_cannot_forge_a_pass(probe):
    code = _NONCE_SCAN + "pub fn add(a: i32, b: i32) -> i32 { " + probe + " 0 }"
    r = RustCompilerOracle().verify_with_test(code, "assert_eq!(add(1, 2), 3);", timeout_s=60)
    assert not r.success, r.details


@needs_rust
def test_early_exit_is_not_a_pass():
    r = RustCompilerOracle().verify_with_test(
        "pub fn add(a: i32, b: i32) -> i32 { std::process::exit(0) }", "assert_eq!(add(1, 2), 3);", timeout_s=60)
    assert not r.success and r.details["reason"] == "missing_result"
