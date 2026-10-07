"""
tests/gwaya/test_low_tier_engine.py
===================================
Tests for GWAYA v3 Low-Tier Model Optimizer and Self-Repair Loop.
"""
from __future__ import annotations

import os
import pytest

from gwaya.low_tier_engine import LowTierModelOptimizer, ModelTier


@pytest.fixture(scope="module", autouse=True)
def allow_unisolated():
    """Allow unisolated execution for this test module."""
    old_val = os.environ.get("GWAYA_ALLOW_UNISOLATED")
    os.environ["GWAYA_ALLOW_UNISOLATED"] = "1"
    yield
    if old_val is None:
        os.environ.pop("GWAYA_ALLOW_UNISOLATED", None)
    else:
        os.environ["GWAYA_ALLOW_UNISOLATED"] = old_val


def test_low_tier_optimizer_rejects_stubs():
    optimizer = LowTierModelOptimizer(tier=ModelTier.LOW)
    # Stub with 'pass'
    res = optimizer.evaluate_candidate("def solve():\n    pass", domain="python")
    assert res.is_valid is False
    assert res.energy == 1_000_000.0
    assert any("pass" in v for v in res.violations)


def test_low_tier_optimizer_accepts_clean_code():
    optimizer = LowTierModelOptimizer(tier=ModelTier.LOW)
    res = optimizer.evaluate_candidate("def solve():\n    x = 10\n    return x * 2", domain="python")
    assert res.is_valid is True
    assert res.energy < 100.0
    assert len(res.violations) == 0


def test_low_tier_repair_loop_succeeds_on_second_attempt():
    def mock_generator(prompt: str) -> str:
        if "REPAIR FEEDBACK" in prompt:
            # Repaired code generated after receiving feedback
            return "def solve():\n    return 42"
        # First attempt: lazy stub
        return "def solve():\n    pass"

    optimizer = LowTierModelOptimizer(tier=ModelTier.LOW, generator_fn=mock_generator)
    opt_res = optimizer.optimize_and_solve(goal="Implement integer solver", domain="python")

    # LT0: no test spec -> well-formed, not verified
    assert opt_res.verified is False
    assert opt_res.level == "well_formed"
    assert opt_res.repair_attempts >= 2
    assert "return 42" in opt_res.selected_code
    assert opt_res.energy < 100.0
    assert opt_res.proof_token is not None


def test_low_tier_fail_closed_when_exhausted():
    def always_stub_generator(prompt: str) -> str:
        return "def solve():\n    ..."

    optimizer = LowTierModelOptimizer(tier=ModelTier.LOW, generator_fn=always_stub_generator)
    opt_res = optimizer.optimize_and_solve(goal="Mission critical task", domain="python")

    assert opt_res.verified is False
    assert opt_res.energy == 1_000_000.0
    assert opt_res.telemetry["status"] == "UNVERIFIED"
    assert "UNVERIFIED" in opt_res.selected_code


def test_low_tier_extracts_markdown_code_blocks():
    from gwaya.low_tier_engine import extract_code_block
    markdown_output = (
        "Here is the requested solution:\n"
        "```python\n"
        "def total(xs):\n"
        "    return sum(xs)\n"
        "```\n"
        "I hope this helps!"
    )
    cleaned = extract_code_block(markdown_output, domain="python")
    assert cleaned == "def total(xs):\n    return sum(xs)"

    # Evaluation accepts fenced markdown without syntax error
    optimizer = LowTierModelOptimizer(tier=ModelTier.LOW)
    res = optimizer.evaluate_candidate(markdown_output, domain="python")
    assert res.is_valid is True
    assert "def total" in res.code
    assert "```" not in res.code


def test_low_tier_functional_test_verification():
    optimizer = LowTierModelOptimizer(tier=ModelTier.LOW)

    # Correct code passing the test spec
    passing_code = "def add(a, b):\n    return a + b"
    test_spec = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
    res_pass = optimizer.evaluate_candidate(passing_code, domain="python", test_spec=test_spec)
    assert res_pass.is_valid is True
    assert res_pass.energy < 100.0

    # Failing code that passes syntax but fails functional assertion
    failing_code = "def add(a, b):\n    return a - b"
    res_fail = optimizer.evaluate_candidate(failing_code, domain="python", test_spec=test_spec)
    assert res_fail.is_valid is False
    assert res_fail.energy == 1_000_000.0
    assert "TEST_FAILED" in res_fail.oracle_error


def test_low_tier_test_driven_self_repair_loop():
    def mock_learning_generator(prompt: str, **kwargs) -> str:
        if "REPAIR FEEDBACK" in prompt:
            # Model receives test failure feedback and fixes the logic
            return "def multiply(a, b):\n    return a * b"
        # First attempt: has a logic bug
        return "def multiply(a, b):\n    return a + b"

    optimizer = LowTierModelOptimizer(tier=ModelTier.LOW, generator_fn=mock_learning_generator)
    spec = "assert multiply(3, 4) == 12"
    opt_res = optimizer.optimize_and_solve(
        goal="Implement multiplication function", domain="python", test_spec=spec
    )

    assert opt_res.verified is True
    assert opt_res.repair_attempts == 2
    assert "return a * b" in opt_res.selected_code
    assert opt_res.energy < 100.0


def test_low_tier_atomic_context_bounding():
    received_kwargs = {}

    def kwargs_capturing_generator(prompt: str, **kwargs) -> str:
        received_kwargs.update(kwargs)
        return "def solve():\n    return 1"

    optimizer = LowTierModelOptimizer(tier=ModelTier.LOW, generator_fn=kwargs_capturing_generator)
    huge_context = "A" * 5000  # Exceeds max_tokens_per_leaf * 4 (600 * 4 = 2400)
    opt_res = optimizer.optimize_and_solve(goal="Solve", context=huge_context)

    # LT0: no test spec -> well-formed, not verified
    assert opt_res.verified is False
    assert opt_res.level == "well_formed"
    assert opt_res.telemetry.get("atomic_decomposition") is True
    assert received_kwargs.get("temperature") == 0.8
    assert received_kwargs.get("max_tokens") == 600


def test_adaptive_temperature_schedule_widens_for_low_tier():
    optimizer_low = LowTierModelOptimizer(tier=ModelTier.LOW)
    # Attempt 1: canonical 0.2, then 0.8
    assert optimizer_low._get_candidate_temperature(0, attempt=1) == 0.2
    assert optimizer_low._get_candidate_temperature(1, attempt=1) == 0.8

    # Attempt 2: widens exploration for SLMs
    temp_round2 = optimizer_low._get_candidate_temperature(0, attempt=2)
    assert temp_round2 > 0.2  # 0.35
    assert optimizer_low._get_candidate_temperature(1, attempt=2) > temp_round2

    # Attempt 3: widens further
    temp_round3 = optimizer_low._get_candidate_temperature(0, attempt=3)
    assert temp_round3 > temp_round2


def test_repair_prompt_contains_full_code_context():
    optimizer = LowTierModelOptimizer(tier=ModelTier.LOW)
    long_code = "def process_data(items):\n" + "\n".join(f"    x_{i} = items[{i}] * 2" for i in range(25)) + "\n    return x_0\n"
    assert len(long_code) > 400

    cand = optimizer.evaluate_candidate(long_code, domain="python")
    prompt = optimizer._build_repair_prompt("Goal: process data", cand, attempt=1)

    # Verify code was not cut off at 200 characters
    assert "x_10" in prompt
    assert "CRITICAL: Fix the specific error" in prompt


