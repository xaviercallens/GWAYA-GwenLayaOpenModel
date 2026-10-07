"""Offline tests for scripts/train_lora.py and scripts/train_laya.py (no torch, network or GPU)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


tl = _load("train_lora")
ty = _load("train_laya")
PLAN = tl.load_plan()

QWEN35_CFG = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
              "vision_config": {"depth": 2}, "text_config": {"model_type": "qwen3_5_text", "hidden_size": 2560,
                                                              "num_hidden_layers": 32}}  # synthetic fixture


# ---- train_lora ------------------------------------------------------------------------------

def test_recipes_agree_with_plan_and_prereg():
    prereg = (ROOT / "docs" / "GWENLAYA_PREREGISTRATION.md").read_text()
    for name, r in tl.RECIPES.items():
        if r.get("stage"):
            st = tl.stage_of(PLAN, r["stage"])
            assert r["model"] in st["models"], name
        if r["model"] != "Qwen/Qwen2.5-Coder-1.5B-Instruct":
            assert r["model"].split("/")[1] in prereg
    assert tl.RECIPES["qwen3.5-4b-dpo"]["beta"] == 0.1
    assert tl.RECIPES["qwen3.8-27b-sft"]["r"] == 8 and tl.RECIPES["qwen3.8-27b-sft"]["seq"] == 1024


def test_gpu_dtype_and_fit():
    assert tl.resolve_recipe("qwen3.5-4b-sft", "t4", PLAN)["compute_dtype"] == "float16"
    assert tl.resolve_recipe("qwen3.5-9b-sft", "l4", PLAN)["compute_dtype"] == "bfloat16"
    with pytest.raises(ValueError):
        tl.resolve_recipe("qwen3.5-9b-sft", "t4", PLAN)
    with pytest.raises(ValueError):
        tl.resolve_recipe("nope", "l4", PLAN)


def test_vram_and_steps_estimates():
    cfg = tl.resolve_recipe("qwen3.8-27b-sft", "l4", PLAN)
    v = tl.estimate_vram(cfg)
    assert v["weights_4bit_gib_lower_bound"] == round(27.8e9 * 0.5 / 1024 ** 3, 2)
    assert v["activations_gib"] == "TBD"
    assert tl.estimate_steps(None, cfg)["total_steps"] == "TBD"
    assert tl.estimate_steps(1000, cfg)["total_steps"] == 63  # ceil(1000/16) * 1 epoch


def test_detect_qwen35_config_fixture(tmp_path):
    f = tmp_path / "config.json"
    f.write_text(json.dumps(QWEN35_CFG))
    a = tl.detect_text_backbone(json.loads(f.read_text()))
    assert a["multimodal"] and a["load_text_config_only"]
    assert a["text_model_type"] == "qwen3_5_text" and a["hidden_size"] == 2560
    plain = tl.detect_text_backbone({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "hidden_size": 1536})
    assert not plain["multimodal"] and not plain["load_text_config_only"]


def test_lora_targets_exclude_vision_and_routers():
    names = ["model.language_model.layers.0.self_attn.q_proj", "model.language_model.layers.0.mlp.gate_proj",
             "model.language_model.layers.0.mlp.gate", "model.visual.blocks.0.attn.qkv", "lm_head",
             "model.language_model.layers.0.linear_attn.in_proj_qkv"]
    t = tl.discover_lora_targets(names)
    assert "model.language_model.layers.0.mlp.gate_proj" in t and "model.language_model.layers.0.linear_attn.in_proj_qkv" in t
    assert not any(x in " ".join(t) for x in ("visual", "lm_head")) and "model.language_model.layers.0.mlp.gate" not in t
    with pytest.raises(ValueError):
        tl.discover_lora_targets(["lm_head"])


def test_validate_rows():
    ok = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
    tl.validate_rows([ok], "sft")
    with pytest.raises(ValueError):
        tl.validate_rows([{"messages": [{"role": "user", "content": "a"}]}], "sft")
    with pytest.raises(ValueError):
        tl.validate_rows([dict(ok, provenance={"dry_run": True})], "sft")
    tl.validate_rows([dict(ok, provenance={"dry_run": True})], "sft", allow_dry_run=True)
    dpo = {"prompt": [{}], "chosen": [{}], "rejected": [{}], "provenance": {"chosen": {"dry_run": True}}}
    with pytest.raises(ValueError):
        tl.validate_rows([dpo], "dpo")
    assert tl.sft_to_prompt_completion(ok) == {"prompt": [ok["messages"][0]], "completion": [ok["messages"][1]]}


def test_save_schedule():
    t = [0.0]
    s = tl.SaveSchedule(200, 900, clock=lambda: t[0])
    assert not s.should_save(100)
    assert s.should_save(200) and not s.should_save(250)
    t[0] = 1000
    assert s.should_save(260)  # time-based
    assert not s.should_save(260)


def test_resume_picks_latest_committed_only(tmp_path):
    for n, committed in ((100, True), (200, True), (300, False)):
        d = tmp_path / f"checkpoint-{n}"
        d.mkdir()
        if committed:
            tl.mark_committed(d)
    assert tl.latest_committed(tmp_path).name == "checkpoint-200"
    assert tl.latest_committed(tmp_path / "missing") is None


class FakeRun:
    def __init__(self, ls_out=""):
        self.calls, self.ls_out = [], ls_out

    def __call__(self, cmd):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=self.ls_out if cmd[2] == "ls" else "")


def test_gcs_prefix_guard():
    with pytest.raises(ValueError):
        tl.GcsSync("gs://socrateai-datalake-gen-lang-client-0625573011/other/x")
    with pytest.raises(ValueError):
        tl.GcsSync("gs://elsewhere/gwenlaya_v4/x")
    with pytest.raises(ValueError):
        tl.check_gcs_uri(tl.ALLOWED_GCS_PREFIX_DEFAULT + "../x")
    assert tl.main(["--recipe", "qwen3.5-4b-sft", "--gpu", "l4", "--print-plan", "--gcs-sync", "gs://bad/x"]) == 2


def test_gcs_push_marker_last_and_pull_newer(tmp_path):
    base = tl.ALLOWED_GCS_PREFIX_DEFAULT + "adapters/run1"
    run = FakeRun()
    sync = tl.GcsSync(base, runner=run)
    ck = tmp_path / "checkpoint-200"
    ck.mkdir()
    tl.mark_committed(ck)
    sync.push(ck)
    assert run.calls[0][2] == "rsync" and run.calls[-1][2] == "cp" and run.calls[-1][-1].endswith("checkpoint-200/COMMITTED")
    remote = FakeRun(f"{base}/checkpoint-100/COMMITTED\n{base}/checkpoint-400/COMMITTED\n")
    s2 = tl.GcsSync(base, runner=remote)
    assert s2.latest_remote_committed() == "checkpoint-400"
    got = tl.find_resume(tmp_path, s2)
    assert got == tmp_path / "checkpoint-400"
    assert tl.find_resume(tmp_path, tl.GcsSync(base, runner=FakeRun(f"{base}/checkpoint-100/COMMITTED\n"))).name == "checkpoint-200"


def test_print_plan_imports_no_heavy_libs(tmp_path):
    f = tmp_path / "sft.jsonl"
    f.write_text("\n".join(json.dumps({"messages": []}) for _ in range(40)) + "\n")
    code = ("import sys, runpy; sys.argv=['x','--recipe','qwen3.5-4b-sft','--gpu','l4','--print-plan','--train-file',%r];"
            "\ntry: runpy.run_path(%r, run_name='__main__')\nexcept SystemExit: pass\n"
            "assert not any(m in sys.modules for m in ('torch','transformers','peft','trl')), 'heavy import'"
            % (str(f), str(ROOT / "scripts" / "train_lora.py")))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["steps"]["n_examples"] == 40 and plan["steps"]["total_steps"] == 6  # ceil(40/16)=3 x 2 epochs


# ---- train_laya ------------------------------------------------------------------------------

def test_auroc_ece_brier_known_values():
    assert ty.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert ty.auroc([0.5, 0.5], [0, 1]) == 0.5
    assert ty.auroc([0.1, 0.2], [1, 1]) is None
    assert ty.ece_equal_mass([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0], bins=2) == 0.0
    assert ty.ece_equal_mass([0.9, 0.9], [0, 0], bins=1) == pytest.approx(0.9)
    assert ty.brier([1.0, 0.0], [1, 1]) == 0.5


def test_slice_filter_and_val_grouping():
    rows = [{"source": "s", "id": str(i), "prompt": "p", "tiers": {"a": 1}} for i in range(400)]
    tr, va, c = ty.select_rows(rows)
    assert all(ty.split_bucket(r["source"], r["id"]) in (1, 2) for r in tr + va)
    assert c["dropped_outside_slice"] > 0 and c["train"] + c["val"] + c["dropped_outside_slice"] == 400
    assert not {r["id"] for r in tr} & {r["id"] for r in va}
    tr2, va2, c2 = ty.select_rows(rows, enforce_slice=False)
    assert c2["dropped_outside_slice"] == 0 and len(tr2) + len(va2) == 400


def test_gate_features_missing_flags():
    full = ty.gate_features({"gate": "verified", "tests_passed_frac": 0.5, "mean_logprob": -1.0})
    assert len(full) == ty.N_FEATURES and full[0] == 1.0
    empty = ty.gate_features(None)
    assert len(empty) == ty.N_FEATURES and empty[ty.N_FEATURES - 1] == 1.0  # last signal missing flag
    assert ty.gate_features({"gate": "weird"})[len(ty.GATE_VOCAB)] == 1.0  # 'other'


def test_make_examples_router_and_calibrator():
    tiers = ["a", "b"]
    r = ty.make_examples([{"source": "s", "id": "1", "domain": "python", "prompt": "p", "tiers": {"b": 1}},
                          {"source": "s", "id": "2", "prompt": "p", "tiers": {}}], "router", tiers)
    assert len(r) == 1 and r[0]["labels"] == [0.0, 1.0] and r[0]["mask"] == [0.0, 1.0] and "python" in r[0]["text"]
    c = ty.make_examples([{"source": "s", "id": "1", "prompt": "p", "tier": "a", "candidate": "x" * 5000, "correct": 1,
                           "signals": {"gate": "failed"}}, {"source": "s", "id": "2", "prompt": "p"}], "calibrator", tiers)
    assert len(c) == 1 and c[0]["labels"] == [1.0] and len(c[0]["text"]) < 2300 and len(c[0]["feats"]) == ty.N_FEATURES


def test_shuffle_labels_is_permutation_and_deterministic():
    ex = [{"key": i, "text": "", "feats": [], "labels": [float(i % 2)], "mask": [1.0]} for i in range(20)]
    a, b = ty.shuffle_labels(ex), ty.shuffle_labels(ex)
    assert a == b and sorted(e["labels"][0] for e in a) == sorted(e["labels"][0] for e in ex)
    assert [e["labels"] for e in a] != [e["labels"] for e in ex]


def test_timing_projection_decision():
    p = ty.project_timing(1.0, 10000, 8, 3)
    assert p["total_steps"] == 3750 and p["run_on"] == "cpu"
    assert ty.project_timing(20.0, 10000, 8, 3)["run_on"].startswith("l4")


def test_laya_print_plan(tmp_path, capsys):
    rows = [{"source": "s", "id": str(i), "prompt": "p", "tiers": {"qwen3.5:2b": i % 2}} for i in range(200)]
    f = tmp_path / "r.jsonl"
    f.write_text("\n".join(map(json.dumps, rows)))
    assert ty.main(["--mode", "router", "--data", str(f), "--print-plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["selection"]["dropped_outside_slice"] > 0 and plan["projected_hours"].startswith("TBD")
