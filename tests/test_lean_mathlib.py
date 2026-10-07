"""Mathlib-aware Lean oracle: discovery, sandbox wiring, audit hardening.

Unit tests are offline and need no Lean. The integration class runs only when the pinned
project from scripts/setup_lean_mathlib.sh is built (and bwrap works) and skips otherwise.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gwaya import lean_project, oracles, sandbox
from gwaya.oracles import Lean4CompilerOracle
from gwaya.sandbox import _bwrap_argv, is_sandbox_available

TOOLCHAIN = "leanprover/lean4:v4.7.0"
MINIF2F_HEADER = (
    "import Mathlib.Algebra.BigOperators.Basic\nimport Mathlib.Data.Real.Basic\n"
    "import Mathlib.Data.Complex.Basic\nimport Mathlib.Data.Nat.Log\nimport Mathlib.Data.Complex.Exponential\n"
    "import Mathlib.NumberTheory.Divisors\nimport Mathlib.Data.ZMod.Defs\nimport Mathlib.Data.ZMod.Basic\n"
    "import Mathlib.Topology.Basic\nimport Mathlib.Data.Nat.Digits\n\n"
    "open BigOperators\nopen Real\nopen Nat\nopen Topology\n\n"
)


def _fake_project(tmp_path: Path, built: bool = True, with_toolchain: bool = True) -> Path:
    proj = tmp_path / "lm" / "project"
    proj.mkdir(parents=True)
    (proj / "lean-toolchain").write_text(TOOLCHAIN + "\n")
    (proj / "lake-manifest.json").write_text(
        json.dumps({"packages": [{"name": "mathlib", "rev": "abc123"}, {"name": "std", "rev": "def"}]})
    )
    lib = proj / ".lake" / "packages" / "mathlib" / ".lake" / "build" / "lib"
    lib.mkdir(parents=True)
    (proj / ".lake" / "packages" / "std" / ".lake" / "build" / "lib").mkdir(parents=True)
    if built:
        (lib / "Mathlib.olean").write_bytes(b"")
    if with_toolchain:
        tc = tmp_path / "lm" / "elan" / "toolchains" / "leanprover--lean4---v4.7.0" / "bin"
        tc.mkdir(parents=True)
        (tc / "lean").write_text("#!/bin/sh\n")
    return proj


def test_discover_records_pins(tmp_path):
    p = lean_project.discover(_fake_project(tmp_path))
    assert p is not None
    assert p.toolchain == TOOLCHAIN and p.mathlib_rev == "abc123"
    assert p.lean_bin.name == "lean"
    assert any(x.endswith("mathlib/.lake/build/lib") for x in p.lean_path)
    assert str(p.project_dir) in p.ro_binds


def test_discover_none_when_absent_or_unbuilt(tmp_path, monkeypatch):
    assert lean_project.discover(tmp_path / "nope") is None
    assert lean_project.discover(_fake_project(tmp_path / "a", built=False)) is None
    assert lean_project.discover(_fake_project(tmp_path / "b", with_toolchain=False)) is None
    monkeypatch.delenv(lean_project.PROJECT_ENV, raising=False)
    monkeypatch.setattr(lean_project, "DEFAULT_ROOT", str(tmp_path / "missing"))
    assert lean_project.default_project_dir() is None


def test_absent_project_is_fail_closed_not_plain_lean(tmp_path):
    o = Lean4CompilerOracle(project_dir=tmp_path / "nope")
    assert not o.available
    r = o.verify_snippet(MINIF2F_HEADER + "theorem t : 1 = 1 := rfl")
    assert not r.success and r.details.get("unverified")


def test_project_mode_passes_readonly_bind_env_and_timeout(tmp_path, monkeypatch):
    proj = _fake_project(tmp_path)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return True, "'t' does not depend on any axioms\n", "", False

    monkeypatch.setattr(oracles, "run_in_sandbox", fake_run)
    o = Lean4CompilerOracle(project_dir=proj, timeout_s=77.0, mem_mb=4096)
    r = o.verify_snippet(MINIF2F_HEADER + "theorem t : 1 = 1 := rfl")
    assert r.success
    assert seen["cmd"][0].endswith("/bin/lean") and seen["cmd"][1:3] == ["-M", "4096"]
    assert seen["kw"]["timeout_s"] == 77.0
    assert seen["kw"]["env"]["LEAN_PATH"] == o.project.lean_path_env
    assert str(o.project.project_dir) in seen["kw"]["ro_binds"]
    assert "theorem t" in seen["kw"]["files"]["candidate.lean"]
    assert "#print axioms t" in seen["kw"]["files"]["candidate.lean"]


def test_lexical_bans_still_apply_in_project_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(oracles, "run_in_sandbox", lambda *a, **k: pytest.fail("must not run"))
    o = Lean4CompilerOracle(project_dir=_fake_project(tmp_path))
    for bad in ("#eval 1", "run_cmd pure ()", "macro_rules | `(foo) => `(1)", "#exit", "axiom a : False"):
        r = o.verify_snippet(MINIF2F_HEADER + f"{bad}\ntheorem t : 1 = 1 := rfl")
        assert not r.success, bad


def test_missing_axiom_audit_is_rejected(tmp_path, monkeypatch):
    """Exit 0 with no `#print axioms` answer (truncated run) must not verify."""
    monkeypatch.setattr(oracles, "run_in_sandbox", lambda *a, **k: (True, "", "", False))
    o = Lean4CompilerOracle(project_dir=_fake_project(tmp_path))
    r = o.verify_snippet("theorem t : 1 = 1 := rfl")
    assert not r.success and "audit incomplete" in r.error_message


def test_bwrap_argv_extra_binds_are_readonly_and_env_is_set(tmp_path):
    argv = _bwrap_argv("bwrap", tmp_path, 5.0, for_python=False, ro_binds=["/x/proj"], env={"LEAN_PATH": "/x/lib"})
    i = argv.index("/x/proj")
    assert argv[i - 1] == "--ro-bind" and argv[i + 1] == "/x/proj"
    j = argv.index("LEAN_PATH")
    assert argv[j - 1] == "--setenv" and argv[j + 1] == "/x/lib"
    assert argv.index("--bind") > i  # only /work is writable


def test_timeout_does_not_kill_caller(monkeypatch, tmp_path):
    monkeypatch.setenv(sandbox.ALLOW_UNISOLATED_ENV, "1")
    monkeypatch.setattr(sandbox, "_bwrap_path", lambda: None)
    ok, _, err, timed_out = sandbox.run_in_sandbox(["sleep", "5"], timeout_s=0.5)
    assert timed_out and not ok  # and we are still alive to assert it


_PROJ = lean_project.discover()
needs_mathlib = pytest.mark.skipif(
    _PROJ is None or not is_sandbox_available(), reason="pinned Lean+Mathlib project or bwrap not available"
)


@needs_mathlib
class TestRealMathlib:
    ST = (
        "theorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h))"
        " (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := "
    )

    @pytest.fixture
    def oracle(self):
        return Lean4CompilerOracle(project_dir=_PROJ.project_dir, timeout_s=300.0)

    def test_valid_proof_verifies(self, oracle):
        r = oracle.verify_snippet(MINIF2F_HEADER + self.ST + "by\n  subst h₂ h₃\n  rw [h₁]; norm_num")
        assert r.success, r.error_message

    def test_wrong_and_sorry_rejected(self, oracle):
        assert not oracle.verify_snippet(MINIF2F_HEADER + self.ST + "by norm_num").success
        assert not oracle.verify_snippet(MINIF2F_HEADER + self.ST + "sorry").success

    def test_statement_with_sorry_elaborates_when_allowed(self, oracle):
        assert oracle.verify_snippet(MINIF2F_HEADER + self.ST + "sorry", allow_sorry=True).success

    def test_project_is_read_only_inside_sandbox(self, oracle):
        target = _PROJ.project_dir / "pwned.txt"
        ok, _, _, _ = sandbox.run_in_sandbox(
            ["touch", str(target)], ro_binds=list(_PROJ.ro_binds), timeout_s=20.0
        )
        assert not ok and not target.exists()

    def test_per_check_timeout(self):
        o = Lean4CompilerOracle(project_dir=_PROJ.project_dir, timeout_s=0.5)
        r = o.verify_snippet(MINIF2F_HEADER + "theorem t : 1 = 1 := rfl")
        assert not r.success and "timed out" in r.error_message
