"""Tests for gwaya.lite (VRAM estimates, tier choice, launch line) and gwaya.hivemind_expert (routing score, escalation)."""
from __future__ import annotations

import pytest

from gwaya.assistant import AssistResult
from gwaya.hivemind_expert import HIVEMIND_SPEC, GwayaCodeExpert, score
from gwaya.lite import HIVEMIND_VRAM_CEILING_MIB, LITE_TIERS, estimate_vram_mib, pick_tier, server_command


def test_7b_fits_hivemind_ceiling_at_4k_and_8k():
    assert estimate_vram_mib(LITE_TIERS["7b"], 4096) <= HIVEMIND_VRAM_CEILING_MIB
    assert estimate_vram_mib(LITE_TIERS["7b"], 8192) <= HIVEMIND_VRAM_CEILING_MIB


def test_estimate_grows_with_ctx_and_q8_cache_is_smaller():
    t = LITE_TIERS["7b"]
    assert estimate_vram_mib(t, 8192) > estimate_vram_mib(t, 4096)
    assert estimate_vram_mib(t, 8192, kv_fp16=False) < estimate_vram_mib(t, 8192)
    with pytest.raises(ValueError):
        estimate_vram_mib(t, 100)


def test_pick_tier_by_budget():
    assert pick_tier(6656).name == "qwen2.5-coder:7b"
    assert pick_tier(3000).name == "qwen2.5-coder:3b"
    assert pick_tier(2000).name == "qwen2.5-coder:1.5b"
    assert pick_tier(1000) is None


def test_server_command_is_local_and_quoted():
    cmd = server_command(LITE_TIERS["3b"], "/m/my model.gguf", port=9000, ctx=8192, kv_fp16=False)
    assert "--host 127.0.0.1" in cmd and "--port 9000" in cmd and "'/m/my model.gguf'" in cmd
    assert "q8_0" in cmd and "--reasoning-budget 0" in cmd


def test_score_matches_registry_scale():
    assert score("tell me a poem") == 1.0
    assert score("fix this python def with a traceback") == 1.0 + 2.5 * 3
    assert HIVEMIND_SPEC["lora_rank_r"] == 16 and "def " in HIVEMIND_SPEC["keywords"]


class _Client:
    def __init__(self, name):
        self.name = name


def _factory(verdicts):
    def make(client, max_rounds=3):
        class A:
            def solve(self, task, tests=""):
                return AssistResult(verdicts[client.name], f"code from {client.name}")
        return A()
    return make


def test_small_first_escalates_only_on_failed():
    ex = GwayaCodeExpert(_Client("s"), _Client("l"), small_first=True,
                         assistant_factory=_factory({"s": "FAILED", "l": "VERIFIED"}))
    r = ex.handle("t", tests="assert 1")
    assert (r.verdict, r.tier, r.escalated, len(r.attempts)) == ("VERIFIED", "l", True, 2)
    ex = GwayaCodeExpert(_Client("s"), _Client("l"), small_first=True,
                         assistant_factory=_factory({"s": "VERIFIED", "l": "FAILED"}))
    r = ex.handle("t", tests="assert 1")
    assert (r.tier, r.escalated) == ("s", False)


def test_no_escalation_without_tests_or_when_unverified_and_default_is_large():
    ex = GwayaCodeExpert(_Client("s"), _Client("l"), small_first=True,
                         assistant_factory=_factory({"s": "UNVERIFIED", "l": "VERIFIED"}))
    assert ex.handle("t", tests="").tier == "l"  # no tests: small tier cannot be judged, go straight to large
    assert ex.handle("t", tests="assert 1").tier == "s"  # UNVERIFIED does not escalate
    ex = GwayaCodeExpert(_Client("s"), _Client("l"), assistant_factory=_factory({"s": "FAILED", "l": "FAILED"}))
    r = ex.handle("t", tests="assert 1")
    assert r.tier == "l" and not r.escalated and r.verdict == "FAILED"
