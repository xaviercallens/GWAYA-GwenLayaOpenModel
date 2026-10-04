"""
tests/test_cascade_router.py
============================
Unit tests for GWAYA Speculative Cascade Router.
"""
from __future__ import annotations

import os

os.environ["GWAYA_ALLOW_UNISOLATED"] = "1"

import pytest

from gwaya.cascade_router import GwayaCascadeRouter, CascadeResult
from gwaya.low_tier_engine import VerificationLevel


def test_cascade_returns_tier1_on_early_success():
    def mock_factory(model_name: str):
        if "0.5b" in model_name:
            # Tier 1 succeeds immediately
            return lambda prompt, **kwargs: "def add(a, b):\n    return a + b"
        raise AssertionError("Should not escalate to larger models when Tier 1 succeeded")

    ladder = ["qwen2.5-coder:0.5b", "qwen2.5-coder:1.5b", "qwen2.5-coder:7b"]
    router = GwayaCascadeRouter(ladder=ladder, generator_factory=mock_factory)

    test_spec = "assert add(2, 3) == 5"
    result = router.route_and_solve(goal="Implement add", test_spec=test_spec)

    assert result.verified is True
    assert result.final_model == "qwen2.5-coder:0.5b"
    assert result.escalated is False
    assert result.tiers_evaluated == 1
    assert result.effective_tokens_per_s == 226.9
    assert result.telemetry.get("saved_tiers") == 2
    assert "add" in result.selected_code


def test_cascade_escalates_to_tier2_when_tier1_fails():
    def mock_factory(model_name: str):
        if "0.5b" in model_name:
            # Tier 1 fails (lazy stub)
            return lambda prompt, **kwargs: "def add(a, b):\n    pass"
        elif "1.5b" in model_name:
            # Tier 2 succeeds
            return lambda prompt, **kwargs: "def add(a, b):\n    return a + b"
        raise AssertionError("Should not escalate to Tier 3 when Tier 2 succeeded")

    ladder = ["qwen2.5-coder:0.5b", "qwen2.5-coder:1.5b", "qwen2.5-coder:7b"]
    router = GwayaCascadeRouter(ladder=ladder, generator_factory=mock_factory)

    test_spec = "assert add(2, 3) == 5"
    result = router.route_and_solve(goal="Implement add", test_spec=test_spec)

    assert result.verified is True
    assert result.final_model == "qwen2.5-coder:1.5b"
    assert result.escalated is True
    assert result.tiers_evaluated == 2
    assert len(result.trajectory) == 2
    assert result.trajectory[0]["verified"] is False
    assert result.trajectory[1]["verified"] is True


def test_cascade_escalates_to_tier3_for_hard_problems():
    def mock_factory(model_name: str):
        if "0.5b" in model_name:
            # Tier 1 fails
            return lambda prompt, **kwargs: "def solve(x):\n    return x"
        elif "1.5b" in model_name:
            # Tier 2 fails
            return lambda prompt, **kwargs: "def solve(x):\n    return x + 1"
        else:
            # Tier 3 succeeds
            return lambda prompt, **kwargs: "def solve(x):\n    return x * 10"

    ladder = ["qwen2.5-coder:0.5b", "qwen2.5-coder:1.5b", "qwen2.5-coder:7b"]
    router = GwayaCascadeRouter(ladder=ladder, generator_factory=mock_factory)

    test_spec = "assert solve(5) == 50"
    result = router.route_and_solve(goal="Solve hard scaling problem", test_spec=test_spec)

    assert result.verified is True
    assert result.final_model == "qwen2.5-coder:7b"
    assert result.escalated is True
    assert result.tiers_evaluated == 3
    assert result.trajectory[0]["model"] == "qwen2.5-coder:0.5b"
    assert result.trajectory[1]["model"] == "qwen2.5-coder:1.5b"
    assert result.trajectory[2]["model"] == "qwen2.5-coder:7b"


def test_cascade_fails_closed_when_all_tiers_exhausted():
    def mock_factory(model_name: str):
        # All tiers return stubs
        return lambda prompt, **kwargs: "def solve(x):\n    ..."

    ladder = ["qwen2.5-coder:0.5b", "qwen2.5-coder:1.5b"]
    router = GwayaCascadeRouter(ladder=ladder, generator_factory=mock_factory)

    result = router.route_and_solve(goal="Impossible task", test_spec="assert solve(1) == 1")

    assert result.verified is False
    assert result.energy == 1_000_000.0
    assert result.telemetry.get("ladder_exhausted") is True
    assert result.level == VerificationLevel.UNVERIFIED
