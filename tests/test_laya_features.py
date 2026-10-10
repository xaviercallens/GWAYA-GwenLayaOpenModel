"""Pins the Laya input layout (gwaya.laya_features) byte-for-byte; scripts/train_laya.py must re-export the same objects."""
import importlib.util
import sys
from pathlib import Path

import pytest

from gwaya import laya_features as F

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("train_laya_for_features", ROOT / "scripts" / "train_laya.py")
T = importlib.util.module_from_spec(spec)
sys.modules["train_laya_for_features"] = T
spec.loader.exec_module(T)

ROW = {"source": "ma", "id": "gsm8k/1", "domain": "python", "prompt": "Write f.\nassert f(1) == 2", "tier": "qwen3.5-4b-bf16",
       "candidate": "```python\ndef f(x):\n    return x + 1\n```", "correct": 1,
       "signals": {"gate": "verified", "mean_logprob": -0.05, "min_logprob": -1.25}}


def test_constants_are_the_frozen_layout():
    assert F.GATE_VOCAB == ("verified", "failed", "unverified", "compile_error", "none")
    assert F.SIGNAL_KEYS == ("tests_passed_frac", "selftest_agreement", "mean_logprob", "min_logprob", "repair_rounds")
    assert F.N_FEATURES == 16 and F.CAND_CHARS == 2000


def test_rendered_text_is_byte_identical_to_the_frozen_format():
    assert F.render_router_text(ROW) == "[domain] python\n[prompt]\nWrite f.\nassert f(1) == 2"
    assert F.render_calibrator_text(ROW) == (
        "[domain] python [tier] qwen3.5-4b-bf16\n[signals] gate=verified mean_logprob=-0.05 min_logprob=-1.25\n"
        "[prompt]\nWrite f.\nassert f(1) == 2\n[candidate]\n```python\ndef f(x):\n    return x + 1\n```")
    assert F.render_calibrator_text_only(ROW) == (
        "[domain] python [tier] qwen3.5-4b-bf16\n[prompt]\nWrite f.\nassert f(1) == 2\n[candidate]\n"
        "```python\ndef f(x):\n    return x + 1\n```")
    long = dict(ROW, candidate="x" * 2500, signals={})
    assert F.render_calibrator_text(long).endswith("[candidate]\n" + "x" * 2000)
    assert "[signals] \n" in F.render_calibrator_text(long)


def test_gate_features_values():
    v = F.gate_features(ROW["signals"])
    # gate one-hot (5) + other, then (value, missing) for tests_passed_frac, selftest_agreement, mean, min, repair
    assert v == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 1.0, -0.005, 0.0, -0.125, 0.0, 0.0, 1.0])
    assert len(v) == F.N_FEATURES
    w = F.gate_features({"gate": "weird", "mean_logprob": float("nan"), "min_logprob": True})
    assert w[5] == 1.0 and sum(w[:5]) == 0.0 and w[10:14] == [0.0, 1.0, 0.0, 1.0]
    assert F.gate_features(None)[4] == 1.0  # gate defaults to "none"


def test_training_script_reexports_the_same_objects():
    for name in ("GATE_VOCAB", "SIGNAL_KEYS", "N_FEATURES", "CAND_CHARS", "gate_features", "render_router_text",
                 "render_calibrator_text", "render_calibrator_text_only"):
        assert getattr(T, name) is getattr(F, name), name
    ex = T.make_examples([ROW], "calibrator", ["t"])[0]
    assert ex["text"] == F.render_calibrator_text(ROW) and ex["feats"] == F.gate_features(ROW["signals"])
