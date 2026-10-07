"""Offline tests for the staged GwenLaya launcher (deploy/run_on_gcp.sh --stage, deploy/stage_launch.py)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

import stage_launch as sl  # noqa: E402

PLAN = sl.load_plan()
SCRIPT = ROOT / "deploy" / "run_on_gcp.sh"
GPU_STAGES = [s["id"] for s in PLAN["stages"] if s.get("where") in ("l4", "t4")]


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    """PATH with a gcloud that logs every call; dry-run must never call it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    g = bin_dir / "gcloud"
    g.write_text(f'#!/bin/bash\necho "$*" >> {tmp_path}/calls.log\nexit 0\n')
    g.chmod(0o755)
    e = os.environ.copy()
    e["PATH"] = f"{bin_dir}:{e['PATH']}"
    e.pop("GWAYA_CONFIRM_SPEND", None)
    return e


def run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([str(SCRIPT), *args], capture_output=True, text=True, env=env)


def test_budget_within_hard_and_planned_caps() -> None:
    b = sl.budget(PLAN)
    assert b["stage_caps_plus_endpoint"] <= 50
    assert b["total_with_misc"] <= b["planned_cap"] <= 50
    assert b["endpoint_cap"] == 2.5 and b["ok"]


@pytest.mark.parametrize("sid", GPU_STAGES)
def test_stage_dry_run_default_no_gcloud(sid: str, env: dict[str, str], tmp_path: Path) -> None:
    r = run(["--stage", sid], env)
    assert r.returncode == 0, r.stderr
    assert "DRY-RUN" in r.stdout and f"Stage:            {sid}" in r.stdout
    assert "--provisioning-model=SPOT" in r.stdout and "--instance-termination-action=DELETE" in r.stdout
    assert "--max-run-duration=" in r.stdout and "purpose=gwenlaya" in r.stdout
    assert "a100" not in r.stdout.lower() and "h100" not in r.stdout.lower()
    assert not (tmp_path / "calls.log").exists()


@pytest.mark.parametrize("sid", GPU_STAGES)
def test_worst_case_never_exceeds_stage_cap(sid: str) -> None:
    i = sl.stage_info(PLAN, sid)
    assert i["worst_usd"] <= i["cap_usd"]
    assert i["max_run_seconds"] / 3600 * i["rate_usd_per_h"] <= i["cap_usd"] + 1e-9
    assert i["zones"] and all(z.startswith(("us-east4", "us-central1")) for z in i["zones"])


@pytest.mark.parametrize("sid", ["C0", "C3", "S7", "nope"])
def test_non_vm_stages_refused(sid: str, env: dict[str, str]) -> None:
    r = run(["--stage", sid], env)
    assert r.returncode != 0 and "DRY-RUN" not in r.stdout


def test_t4_only_for_models_that_fit() -> None:
    assert sl.stage_info(PLAN, "S4a", "t4")["gpu"] == "t4"
    with pytest.raises(ValueError):
        sl.stage_info(PLAN, "S4b", "t4")
    with pytest.raises(ValueError):
        sl.stage_info(PLAN, "S2", "t4")
    with pytest.raises(ValueError):
        sl.stage_info(PLAN, "S1", "a100")


def test_t4_command_has_accelerator_flag(env: dict[str, str]) -> None:
    r = run(["--stage", "S4a", "--gpu", "t4"], env)
    assert "--accelerator=type=nvidia-tesla-t4,count=1" in r.stdout


def test_yes_spend_needs_env_even_for_stage(env: dict[str, str], tmp_path: Path) -> None:
    r = run(["--stage", "S1", "--yes-spend"], env)
    assert r.returncode != 0 and "GWAYA_CONFIRM_SPEND" in r.stdout
    assert not (tmp_path / "calls.log").exists()


def test_vm_names_and_labels_are_lowercase(env: dict[str, str]) -> None:
    r = run(["--stage", "S4a"], env)
    assert 'create "gwenlaya-s4a-' in r.stdout and "--labels=purpose=gwenlaya,run=s4a,stage=s4a" in r.stdout


def test_entry_commands_use_the_three_scripts() -> None:
    assert any("train_lora.py --recipe qwen3.5-9b-sft" in c for c in sl.entry_commands(PLAN, "S4b"))
    s4c = " ".join(sl.entry_commands(PLAN, "S4c"))
    assert "--max-steps 100" in s4c and "qwen3.8-27b-sft" in s4c
    assert any("run_study.py --stage S2" in c and "--lake-sync" in c for c in sl.entry_commands(PLAN, "S2"))
    assert any("quantize.py" in c for c in sl.entry_commands(PLAN, "S1") + sl.entry_commands(PLAN, "S4b"))
    for sid in GPU_STAGES:  # every sync target stays under gwenlaya_v4/
        for c in sl.entry_commands(PLAN, sid):
            assert "gs://" not in c or sl.LAKE in c


def test_recipes_and_stages_exist_in_scripts() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_lora  # noqa: E402
    for sid, recipes in sl.TRAIN_STAGES.items():
        for r in recipes:
            assert r in train_lora.RECIPES and train_lora.RECIPES[r]["stage"] in (sid, "S4d", "S4a", "S4b")
    for sid in sl.RUN_STUDY_STAGES:
        sl.get_stage(PLAN, sid)


def test_quota_and_accel_checks_fail_closed() -> None:
    assert sl.check_quota({"quotas": [{"metric": "GPUS_ALL_REGIONS", "limit": 1, "usage": 0}]})[0]
    assert not sl.check_quota({"quotas": [{"metric": "GPUS_ALL_REGIONS", "limit": 1, "usage": 1}]})[0]
    assert not sl.check_quota({"quotas": []})[0]
    assert not sl.check_accel([{"name": "x", "status": "RUNNING", "guestAccelerators": [{"acceleratorType": "t4"}]}])[0]
    assert sl.check_accel([{"name": "y", "status": "TERMINATED", "guestAccelerators": [{}]}, {"name": "z", "status": "RUNNING"}])[0]


def test_ledger_total_and_corrupt_line() -> None:
    assert sl.ledger_total('{"cost_usd_upper_bound": 1.5}\n\n{"cost_usd_upper_bound": 0.25}\n') == 1.75
    with pytest.raises(ValueError):
        sl.ledger_total("not json")


def test_packet_is_current_and_sums() -> None:
    fake = lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "DRY-RUN-STUB", "")  # noqa: E731
    text = sl.build_packet(PLAN, runner=fake)
    for sid in GPU_STAGES:
        assert f"## {sid}" in text
    assert "Human checklist" in text and "35.50" in text and "40.00" in text
    committed = ROOT / "experiments" / "launch_packet.md"
    if committed.exists():
        t = committed.read_text()
        assert all(f"## {sid}" in t for sid in GPU_STAGES) and "TBD" in t


def test_scripts_parse() -> None:
    for f in ("run_on_gcp.sh", "vm_startup.sh", "container_entry.sh", "stage_entry.sh"):
        assert subprocess.run(["bash", "-n", str(ROOT / "deploy" / f)]).returncode == 0
    assert "GWAYA_STAGE" in (ROOT / "deploy" / "container_entry.sh").read_text()
