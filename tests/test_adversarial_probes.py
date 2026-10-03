"""
Adversarial regression tests for the GWAYA verifier (audit probes P1-P5, P10, and Lean 4 soundness).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from gwaya.low_tier_engine import LowTierModelOptimizer, ModelTier, VerificationLevel
from gwaya.oracles import Lean4CompilerOracle, PythonCompilerOracle, RustCompilerOracle
from gwaya.test_harness import isolation_available

logging.disable(logging.WARNING)

needs_isolation = pytest.mark.skipif(
    not isolation_available(), reason="bwrap isolation not available on this machine"
)


@pytest.fixture
def oracle() -> PythonCompilerOracle:
    return PythonCompilerOracle()


# ── P1/P2: a candidate must not be able to skip the test specification ─────────

@needs_isolation
@pytest.mark.parametrize(
    "candidate",
    [
        "import os\nos._exit(0)\n",
        "import sys\nsys.exit(0)\n",
        "import atexit, os\natexit.register(lambda: os._exit(0))\nraise SystemExit(0)\n",
        "import builtins\nbuiltins.AssertionError = type('X', (Exception,), {})\n",
    ],
    ids=["os_exit", "sys_exit", "atexit_exit", "rebind_assertionerror"],
)
def test_p1_p2_test_spec_cannot_be_skipped(oracle, candidate):
    res = oracle.verify_with_test(candidate, "assert False, 'spec must fail'\n")
    assert res.success is False


@needs_isolation
def test_p1_exit_inside_called_function_is_a_failure(oracle):
    code = "import sys\ndef add(a, b):\n    sys.exit(0)\n"
    res = oracle.verify_with_test(code, "assert add(2, 3) == 5\n")
    assert res.success is False


@needs_isolation
def test_honest_candidate_still_passes_with_partial_credit_reported(oracle):
    code = "def add(a, b):\n    return a + b\n"
    res = oracle.verify_with_test(code, "assert add(2, 3) == 5\nassert add(0, 0) == 0\n")
    assert res.success is True
    assert res.details.get("passed") == 2
    assert res.details.get("total") == 2


@needs_isolation
def test_partial_failure_reports_counts_accurately(oracle):
    code = "def add(a, b):\n    return a + b if a > 0 else 0\n"
    res = oracle.verify_with_test(code, "assert add(2, 3) == 5\nassert add(0, 5) == 5\n")
    assert res.success is False
    assert res.details.get("passed") == 1
    assert res.details.get("total") == 2


# ── P3: sandbox isolation must prevent host filesystem tampering ──────────────

@needs_isolation
@pytest.mark.parametrize("path", ["/etc/hostname", "/etc/passwd", "/var/log", "/opt"])
def test_p3_candidate_cannot_see_host_system_paths(oracle, path):
    code = f"import os\nSEEN = os.path.exists({path!r})\n"
    res = oracle.verify_with_test(code, "assert SEEN\n")
    assert res.success is False


# ── P4: fake / non-existent package imports must be rejected ──────────────────

@needs_isolation
def test_p4_fake_package_imports_rejected(oracle):
    code = "import totally_fake_pkg_xyz_12345\ndef solve(): return 1\n"
    res = oracle.verify_with_test(code, "assert solve() == 1\n")
    assert res.success is False


def test_p4_real_imports_still_accepted(oracle):
    assert oracle.verify("import math\nfrom collections import deque\nx = math.pi\n").success


# ── P5: without user tests the optimizer must not claim VERIFIED ───────────────

def test_p5_no_test_spec_is_well_formed_not_verified():
    opt = LowTierModelOptimizer(tier=ModelTier.LOW, generator=None)
    res = opt.optimize_and_solve(goal="Implement Dijkstra shortest path")
    assert res.is_verified is False


# ── P10: Rust placeholder bodies are rejected before type-checking ────────────

@pytest.mark.parametrize(
    "code",
    [
        "fn solve() -> i32 { unimplemented!() }",
        "fn solve() -> i32 { todo!() }",
        "fn solve() -> i32 { todo!(\"reason\") }",
    ],
)
def test_rust_compiler_oracle_flags_placeholders(code):
    rust = RustCompilerOracle()
    res = rust.verify(code)
    assert res.success is False
    assert any("placeholder" in err.lower() or "unimplemented" in err.lower() or "todo" in err.lower() for err in res.errors)


# ── Lean 4 Soundness & Axiom Leakage Checks ───────────────────────────────────

def test_lean4_oracle_rejects_sorry_and_admit():
    lean = Lean4CompilerOracle()
    res_sorry = lean.verify("theorem fake_proof : 1 = 2 := by sorry")
    assert res_sorry.success is False
    assert any("sorry" in err.lower() for err in res_sorry.errors)

    res_admit = lean.verify("theorem fake_proof : 1 = 2 := by admit")
    assert res_admit.success is False
    assert any("admit" in err.lower() for err in res_admit.errors)
