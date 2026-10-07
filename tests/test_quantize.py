"""Offline tests for scripts/quantize.py (no torch, llama.cpp, network or GPU)."""
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("quantize", ROOT / "scripts" / "quantize.py")
q = importlib.util.module_from_spec(spec)
sys.modules["quantize"] = q
spec.loader.exec_module(q)


def test_plan_steps_and_order(tmp_path):
    p = q.build_plan("Qwen/Qwen3.5-4B", "adapter", tmp_path, imatrix_text="train.txt", awq=True)
    ids = [s["id"] for s in p["steps"]]
    assert ids[:3] == ["merge", "convert_f16", "imatrix"]
    assert "quantize_q8_0" in ids and "quantize_q4_K_M" in ids and ids[-1] == "awq"
    cmd = next(s for s in p["steps"] if s["id"] == "quantize_q4_K_M")["cmd"]
    assert "--imatrix" in cmd and cmd[-1] == "q4_K_M"
    assert "f16" in next(s for s in p["steps"] if s["id"] == "convert_f16")["cmd"]
    assert p["eval_quants"] == ["bf16", "q8_0", "q4_K_M"]


def test_27b_merge_refused_but_base_quantize_allowed(tmp_path):
    with pytest.raises(ValueError, match="merge refused"):
        q.build_plan("Qwen/Qwen3.8-27B", "adapter", tmp_path)
    assert q.build_plan("Qwen/Qwen3.8-27B", None, tmp_path)["steps"][0]["id"] == "convert_f16"
    with pytest.raises(ValueError):
        q.build_plan("Qwen/Qwen3.5-4B", None, tmp_path, quants=["q2_K"])


def test_cli_plan_runs_nothing(tmp_path, capsys):
    out = tmp_path / "o"
    assert q.main(["plan", "--base", "Qwen/Qwen3.5-4B", "--adapter", "a", "--out-dir", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["base"] == "Qwen/Qwen3.5-4B"
    assert not out.exists()


def test_execute_with_fake_runner_hashes_and_modelfile(tmp_path):
    p = q.build_plan("Qwen/Qwen3.5-4B", None, tmp_path, name="t")

    def fake(cmd, check):
        Path(cmd[cmd.index("--outfile") + 1] if "--outfile" in cmd else cmd[-2]).write_bytes(b"x" * 10)

    m = q.execute(p, runner=fake, system="s")
    assert m["files"]["t-q4_K_M.gguf"]["bytes"] == 10
    assert m["files"]["t-q8_0.gguf"]["sha256"] == q.sha256_file(tmp_path / "t-q8_0.gguf")
    assert "FROM ./t-q4_K_M.gguf" in (tmp_path / "Modelfile.q4_K_M").read_text()
    m2 = q.execute(p, runner=lambda *a, **k: pytest.fail("rerun"))
    assert "quantize_q8_0" in m2["steps_skipped_existing"]


def test_install_uses_runner_and_rejects_root_path(tmp_path):
    calls = []
    q.install_llama_cpp(tmp_path / "llama.cpp", runner=lambda c, check: calls.append(c[0]))
    assert calls == ["git", "cmake", "cmake"]
    with pytest.raises(ValueError):
        q.check_path(Path("/home/someone/x"))


def _synth(n, shift, seed):
    r = random.Random(seed)
    s = [r.random() for _ in range(n)]
    return {"scores": s, "correct": [1.0 if r.random() < min(max(v - shift, 0.02), 0.98) else 0.0 for v in s]}


def test_transfer_reports_both_and_flags_F():
    rep = q.transfer_report(_synth(300, 0.0, 1), _synth(300, 0.3, 2), _synth(300, 0.3, 3), method="platt")
    assert {"T_transfer", "R_refit"} <= rep.keys() and rep["F_full_retrain"] == "not_run"
    assert not rep["eval_is_target_c"]
    for k in ("T_transfer", "R_refit"):
        assert 0 <= rep[k]["ece"] <= 1 and rep[k]["n"] == 300
    # synthetic shifted target: refit should not be worse on Brier than transfer
    assert rep["R_refit"]["brier"] <= rep["T_transfer"]["brier"] + 1e-9


def test_calibration_metrics_perfect():
    m = q.calibration_metrics([1.0, 0.0], [1.0, 0.0])
    assert m["brier"] == 0 and m["ece"] == 0
