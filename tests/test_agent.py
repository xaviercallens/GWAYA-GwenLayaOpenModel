import json
import pytest
import os

from gwaya import agent as agent_mod
from gwaya.test_harness import isolation_available


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


def test_model_ladder_by_vram():
    assert agent_mod.recommend_model(None) == "qwen2.5-coder:1.5b"  # CPU only -> smallest
    assert agent_mod.recommend_model(4.0) == "qwen2.5-coder:3b"
    assert agent_mod.recommend_model(8.0) == "qwen2.5-coder:7b"
    assert agent_mod.recommend_model(24.0) == "qwen2.5-coder:14b"
    assert agent_mod.recommend_model(2.0) == "qwen2.5-coder:1.5b"


def test_detect_vram_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(agent_mod.shutil, "which", lambda _: None)
    assert agent_mod.detect_vram_gib() == (None, None)


def test_doctor_reports_fields(monkeypatch):
    monkeypatch.setattr(agent_mod, "detect_vram_gib", lambda: ("NVIDIA GeForce RTX 3060", 12.0))
    d = agent_mod.doctor()
    assert d["gpu"].startswith("NVIDIA") and d["recommended_model"] == "qwen2.5-coder:14b"
    assert set(d) >= {"ollama", "installed_models", "sandbox_bwrap"}


def test_ask_is_fail_closed_without_sandbox(monkeypatch):
    monkeypatch.delenv("GWAYA_ALLOW_UNISOLATED", raising=False)
    monkeypatch.setattr(agent_mod, "isolation_available", lambda: False)
    with pytest.raises(RuntimeError, match="fail-closed"):
        agent_mod.ask("double x", None, "m", 1, 1)


def test_cli_verified_and_unverified(monkeypatch, capsys):
    class Fake:
        def __init__(self, model, **_):
            self.model, self.codes = (
                model,
                {"ok": "def f(x):\n    return x * 2\n", "bad": "def f(x):\n    return x + 5\n"},
            )

        def __call__(self, prompt, temperature=0.2, max_tokens=512, **_):
            return f"```python\n{self.codes[self.model]}\n```"

    monkeypatch.setattr(agent_mod, "OllamaGenerator", Fake)
    args = [
        "ask",
        "double x",
        "--test",
        "assert f(2) == 4",
        "--rounds",
        "1",
        "--candidates",
        "2",
        "--json",
    ]
    assert agent_mod.main([*args, "--model", "ok"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "VERIFIED"
    assert agent_mod.main([*args, "--model", "bad"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "UNVERIFIED" and out["verified"] is False
