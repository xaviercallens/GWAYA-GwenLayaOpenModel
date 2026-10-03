"""WP1: oracles must fail closed. A missing toolchain is UNVERIFIED, never an approval."""

from gwaya.oracles import Lean4CompilerOracle, RustCompilerOracle


def _unavailable(cls, **kw):
    oracle = cls(**kw)
    oracle.available = False
    return oracle


def test_lean_missing_toolchain_is_rejected():
    res = _unavailable(Lean4CompilerOracle).verify_snippet("theorem t : 1 = 1 := rfl")
    assert res.success is False
    assert res.terminal_reward == -1.0
    assert res.details.get("unverified") is True
    assert "mock" not in res.error_message.lower()


def test_rust_missing_toolchain_is_rejected():
    res = _unavailable(RustCompilerOracle).verify_snippet("pub fn f() -> u8 { 1 }")
    assert res.success is False
    assert res.terminal_reward == -1.0
    assert res.details.get("unverified") is True
    assert "mock" not in res.error_message.lower()
