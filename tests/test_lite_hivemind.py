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


def test_lite_bench_stats():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from lite_bench import mcnemar_exact, summarize, wilson
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(5, 0) == pytest.approx(0.0625)
    lo, hi = wilson(50, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61
    rows = [{"a0_pass": False, "a3_pass": True, "verdict": "VERIFIED", "gen_s": 1.0, "completion_tokens": 10},
            {"a0_pass": True, "a3_pass": True, "verdict": "FAILED", "gen_s": 3.0, "completion_tokens": 20}]
    s = summarize(rows)
    assert (s["gained"], s["lost"], s["verified_hidden_precision"], s["mean_gen_s"]) == (1, 0, 1.0, 2.0)


def test_estimate_covers_t4_measurements():
    measured = {("3b", 4096): 2191, ("3b", 8192): 2339, ("7b", 4096): 4663, ("7b", 8192): 4891}  # nvidia-smi, T4, b11854
    for (k, ctx), mib in measured.items():
        est = estimate_vram_mib(LITE_TIERS[k], ctx)
        assert mib <= est <= mib + 250, (k, ctx, est, mib)


def test_score_is_whole_word():
    from gwaya.hivemind_expert import keyword_hit
    assert score("define a classification of subclasses and an important matter") == 1.0
    assert score("python def import pytest") == 1.0 + 2.5 * 4
    assert keyword_hit(".py", "open main.py now") and not keyword_hit("def ", "define it")


def test_route_eval_harness_metrics():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "routing"))
    from route_eval import compare, evaluate, load_rows, predictions
    rows = load_rows(Path(__file__).resolve().parents[1] / "scripts" / "routing" / "routing_fixture_synthetic.jsonl")
    assert len(rows) == 27 and sum(r["domain"] == "none" for r in rows) == 10
    always_py = lambda p: (["Python_Code_GWAYA"], {"Python_Code_GWAYA": 1.0})  # noqa: E731
    abstain = lambda p: ([], {})  # noqa: E731
    ev = evaluate(predictions(always_py, rows), rows)
    assert ev["top1_in_domain"][:2] == [5, 17] and ev["abstain_off"][:2] == [0, 10]
    ev = evaluate(predictions(abstain, rows), rows)
    assert ev["abstain_off"][:2] == [10, 10] and ev["abstain_in"] == [17, 17]
    c = compare(predictions(always_py, rows), predictions(abstain, rows), rows)
    assert c["b_only_right"] == 10 and c["a_only_right"] == 5


def test_tier_cost_model():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "routing"))
    from route_eval import cascade_cost, evaluate_tiers
    costs = [1.0, 2.0, 10.0]
    assert cascade_cost(0, 0, costs) == 1.0 and cascade_cost(2, 0, costs) == 10.0  # over-tier pays the high tier
    assert cascade_cost(0, 2, costs) == 13.0 and cascade_cost(1, 2, costs) == 12.0  # under-tier pays every failed attempt
    rows = [{"tier": 0}, {"tier": 1}, {"tier": 2}, {"prompt": "no label"}]
    ev = evaluate_tiers([0, 0, 2, 1], rows, costs)
    assert ev["n_tiered"] == 3 and ev["exact"] == round(2 / 3, 4) and ev["under"] == round(1 / 3, 4) and ev["over"] == 0.0
    assert ev["oracle_cost"] == round((1 + 2 + 10) / 3, 4) and ev["mean_cost"] == round((1 + 3 + 10) / 3, 4)


def test_discrimination_band_is_computed():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "routing"))
    from route_eval import discrimination
    rows = [{"domain": "A"}, {"domain": "A"}, {"domain": "B"}, {"domain": "none"}]
    arms = {"x": ["A", "A", "B", "none"], "y": ["A", "none", "none", "none"]}  # pooled in-domain: 4 right of 6
    d = discrimination(arms, rows, {"all": [True] * 4, "blind": [False, True, True, False]})
    assert d["all"]["pooled_top1"] == round(4 / 6, 4) and d["all"]["in_band"] is True
    assert d["blind"]["pooled_top1"] == 0.5 and d["blind"]["n"] == 4 and d["blind"]["in_band"] is True
    dead = discrimination({"y": ["none"] * 4, "z": ["none"] * 4}, rows, {"s": [True] * 4})
    assert dead["s"]["pooled_top1"] == 0.0 and dead["s"]["in_band"] is False  # all arms wrong: cannot rank them
