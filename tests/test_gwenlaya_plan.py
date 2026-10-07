"""Consistency checks between experiments/plan.json and the GwenLaya pre-registration.

Offline: reads two repo files only (no GPU, network or model download).
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAN = json.loads((ROOT / "experiments" / "plan.json").read_text())
PREREG = (ROOT / "docs" / "GWENLAYA_PREREGISTRATION.md").read_text()

WHERE = {"cpu", "t4", "l4", "cloudrun-l4"}
STAGE_KEYS = {"id", "where", "models", "domains", "tasks", "arms", "quant", "n", "est_hours", "cap_usd"}
DATASET_KEYS = {"hf_id", "domain", "role", "license", "est_gb", "needs_decontam"}
EXCLUDED_IDS = ["KodCode", "Magpie", "Vezora", "Goedel-Pset", "PrimeIntellect", "gwaya_v2_verifier_curriculum"]


def test_budget_within_hard_cap():
    caps = sum(s["cap_usd"] for s in PLAN["stages"]) + PLAN["misc_cap_usd"]
    assert abs(sum(PLAN["misc_breakdown_usd"].values()) - PLAN["misc_cap_usd"]) < 1e-6
    assert PLAN["total_cap_usd"] <= 50.0
    assert abs(caps - PLAN["planned_cap_usd"]) < 1e-6
    assert caps + PLAN["reserve_usd"] <= PLAN["total_cap_usd"] + 1e-6
    assert all(s["cap_usd"] >= 0 for s in PLAN["stages"])


def test_stage_schema():
    ids = [s["id"] for s in PLAN["stages"]]
    assert len(ids) == len(set(ids))
    for s in PLAN["stages"]:
        assert STAGE_KEYS <= set(s), s["id"]
        assert s["where"] in WHERE, s["id"]
        if s["where"] == "cpu":
            assert s["cap_usd"] == 0.0, s["id"]
        if s["where"] == "l4":
            # max_hours is cap / planning rate (an upper bound); est_hours stays TBD until S1 measures
            rate = PLAN["planning_rates_usd_per_h"]["l4_spot_vm"]
            assert s["max_hours"] == round(s["cap_usd"] / rate, 1), s["id"]
            assert s["est_hours"] == "TBD", s["id"]


def test_stage_caps_match_prereg_table():
    for s in PLAN["stages"]:
        if s["where"] == "cpu":
            continue
        row = re.search(r"^\| %s \| %s \|.*\| ([0-9.]+) \| [^|]+ \|$" %(re.escape(s["id"]), re.escape(s["where"])),
                        PREREG, re.M)
        assert row, s["id"]
        assert float(row.group(1)) == s["cap_usd"], s["id"]


def test_datasets_schema_and_exclusions():
    for d in PLAN["datasets"]:
        assert DATASET_KEYS <= set(d), d["hf_id"]
        assert d["role"] in {"eval", "train", "dpo", "calib"}
        assert d["est_gb"] == "TBD" or isinstance(d["est_gb"], (int, float))
        for bad in EXCLUDED_IDS:
            assert bad not in d["hf_id"]


def test_hypotheses_are_preregistered():
    ids = [h["id"] for h in PLAN["hypotheses"]]
    assert ids == ["H%d" % i for i in range(1, len(ids) + 1)]
    for hid in ids:
        assert re.search(r"\*\*%s[ :(.]" % hid, PREREG), hid
    primary = [h["id"] for h in PLAN["hypotheses"] if h["family"] == "primary"]
    assert primary == ["H1", "H3"]
    assert "m = 2" in PREREG
    exploratory = {h["id"] for h in PLAN["hypotheses"] if h["family"] == "exploratory"}
    assert exploratory == {"H2", "H7", "H10", "H11", "H12"}


def test_stage_dependencies_exist_and_are_acyclic():
    stages = {s["id"]: s for s in PLAN["stages"]}
    position = {sid: i for i, sid in enumerate(stages)}
    for s in PLAN["stages"]:
        for dep in s.get("depends_on", []):
            assert dep in stages, (s["id"], dep)
            # listed order is a valid execution order
            assert position[dep] < position[s["id"]], (s["id"], dep)
    required = {"S2": {"C0", "C1", "S1"}, "C3": {"S2", "S3"}, "S5": {"S4a", "S4b", "C4", "C3"},
                "S6": {"S5", "C1", "C2", "C6"}, "S7": {"S6"}}
    for sid, deps in required.items():
        assert deps <= set(stages[sid]["depends_on"]), sid


def test_gpu_preflight_and_generation_rules():
    pre = PLAN["preflight"]
    assert any("GPUS_ALL_REGIONS usage == 0" in x for x in pre["before_every_vm_create"])
    assert "never stopped autonomously" in pre["blocker_2026_10_07"]
    gen = PLAN["generation_settings"]
    assert gen["thinking"].startswith("disabled")
    assert all(isinstance(v, int) for v in gen["max_new_tokens"].values())
    assert "--parallel" in gen["engine"]
    s7 = next(s for s in PLAN["stages"] if s["id"] == "S7")
    rate = PLAN["planning_rates_usd_per_h"]["cloudrun_l4_4cpu_16gib"]
    assert s7["n"] <= 6 and s7["max_instance_hours"] == round(s7["cap_usd"] / rate, 1)


def test_freeze_file_matches_current_files():
    freeze = (ROOT / "docs" / "PREREG_FREEZE.txt").read_text()
    for rel in ["docs/GWENLAYA_PREREGISTRATION.md", "experiments/plan.json"]:
        digest = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        # the latest frozen line for each file must match its current content
        lines = [ln.split() for ln in freeze.splitlines() if ln.strip().endswith(rel)]
        assert lines and lines[-1][0] == digest, rel
