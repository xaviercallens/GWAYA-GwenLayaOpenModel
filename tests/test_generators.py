"""Unit tests for gwaya.generators (LT1). The live test needs GWAYA_OLLAMA_TESTS=1."""
from __future__ import annotations

import os

import pytest

from gwaya.generators import GeneratorUnavailableError, OllamaGenerator
from gwaya.low_tier_engine import (
    LowTierModelOptimizer,
    ModelTier,
    VerificationLevel,
    extract_code_block,
)


class _FakeOllama(OllamaGenerator):
    def __init__(self, responses: list[str], **kw):
        super().__init__(**kw)
        self.responses = list(responses)
        self.payloads: list[dict] = []

    def _post(self, path, payload):
        self.payloads.append(payload)
        return {
            "response": self.responses.pop(0) if self.responses else "",
            "eval_count": 20,
            "eval_duration": 2_000_000_000,
            "prompt_eval_count": 50,
            "total_duration": 3_000_000_000,
            "done_reason": "stop",
        }


def test_payload_uses_raw_prefill_stop_and_seed():
    gen = _FakeOllama(["def f():\n    return 1"], base_seed=7)
    out = gen("Write f", temperature=0.4, max_tokens=64)
    p = gen.payloads[0]
    assert p["raw"] is True and p["stream"] is False
    assert p["prompt"].endswith("<|im_start|>assistant\n```python\n")
    assert "```" in p["options"]["stop"]
    assert p["options"] == {**p["options"], "temperature": 0.4, "num_predict": 64, "seed": 7}
    assert extract_code_block(out) == "def f():\n    return 1"


def test_seeds_differ_per_call_and_explicit_seed_wins():
    gen = _FakeOllama(["a = 1", "a = 2", "a = 3"], base_seed=100)
    gen("x")
    gen("x")
    gen("x", seed=5)
    assert [p["options"]["seed"] for p in gen.payloads] == [100, 101, 5]


def test_stats_recorded():
    gen = _FakeOllama(["a = 1"])
    gen("x")
    assert gen.last_stats is not None
    assert gen.last_stats.completion_tokens == 20
    assert gen.last_stats.tokens_per_s == pytest.approx(10.0)
    assert len(gen.history) == 1


def test_domain_fence_tag():
    gen = _FakeOllama(["theorem t : 1 = 1 := rfl"], domain="lean4")
    out = gen("prove 1 = 1")
    assert gen.payloads[0]["prompt"].endswith("```lean\n")
    assert out.startswith("```lean\n")


def test_missing_response_raises():
    class Bad(OllamaGenerator):
        def _post(self, path, payload):
            return {"error": "model not found"}

    with pytest.raises(GeneratorUnavailableError):
        Bad()("x")


def test_unreachable_host_raises():
    gen = OllamaGenerator(host="http://127.0.0.1:9", timeout_s=2)
    assert gen.available() is False
    with pytest.raises(GeneratorUnavailableError):
        gen("x")


def test_plugs_into_low_tier_optimizer_signature():
    gen = _FakeOllama(["def add(a, b):\n    return a + b"] * 3)
    opt = LowTierModelOptimizer(tier=ModelTier.LOW, generator_fn=gen)
    res = opt.optimize_and_solve(goal="add two numbers", domain="python")
    assert res.level == VerificationLevel.WELL_FORMED
    assert gen.payloads[0]["options"]["num_predict"] == opt.config.max_tokens_per_leaf


@pytest.mark.skipif(os.environ.get("GWAYA_OLLAMA_TESTS") != "1", reason="set GWAYA_OLLAMA_TESTS=1 for live Ollama")
def test_live_ollama_smoke():
    gen = OllamaGenerator()
    if not gen.available():
        pytest.skip("Ollama model not available")
    out = gen("Write a Python function add(a, b) that returns a + b.", temperature=0.0, max_tokens=64)
    code = extract_code_block(out)
    assert "def add" in code
    assert gen.last_stats is not None and gen.last_stats.completion_tokens <= 64
