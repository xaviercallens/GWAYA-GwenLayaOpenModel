"""
tests/test_webgwaya.py
======================
Unit and API integration tests for WebGWAYA console backend.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from webgwaya.app import app

client = TestClient(app)


def test_root_serves_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "WebGWAYA" in res.text


def test_system_status():
    res = client.get("/api/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["app_name"] == "WebGWAYA"
    assert "gpu" in data
    assert "chroma_available" in data


def test_scenarios_list():
    res = client.get("/api/scenarios")
    assert res.status_code == 200
    scenarios = res.json()
    assert len(scenarios) >= 5
    ids = [s["id"] for s in scenarios]
    assert "scenario_01_merge_intervals" in ids
    assert "scenario_02_trap_rain_water" in ids


def test_ast_audit_clean_code():
    res = client.post("/api/gwaya/ast-audit", json={"code": "def f(x):\n    return x + 1\n"})
    assert res.status_code == 200
    data = res.json()
    assert data["is_clean"] is True
    assert data["ast_shield"] == "PASS"
    assert data["energy"] < 1000.0


def test_ast_audit_rejects_pass_stub():
    res = client.post("/api/gwaya/ast-audit", json={"code": "def f(x):\n    pass\n"})
    assert res.status_code == 200
    data = res.json()
    assert data["is_clean"] is False
    assert data["ast_shield"] == "REJECT"
    assert data["energy"] == 1_000_000.0
    assert any("pass" in v.lower() for v in data["violations"])


def test_evaluate_valid_code_and_tests():
    code = "def add(a, b):\n    return a + b\n"
    spec = "assert add(2, 3) == 5\nassert add(0, 0) == 0\n"
    res = client.post("/api/gwaya/evaluate", json={"code": code, "test_spec": spec})
    assert res.status_code == 200
    data = res.json()
    assert data["verified"] is True
    assert data["tests_passed"] == 2
    assert data["tests_total"] == 2
    assert data["level"] == "verified_tests"


def test_evaluate_failing_tests():
    code = "def add(a, b):\n    return a - b\n"
    spec = "assert add(2, 3) == 5\n"
    res = client.post("/api/gwaya/evaluate", json={"code": code, "test_spec": spec})
    assert res.status_code == 200
    data = res.json()
    assert data["verified"] is False
    assert data["energy"] == 1_000_000.0


def test_benchmark_results_endpoint():
    res = client.get("/api/benchmark/results")
    assert res.status_code == 200
    data = res.json()
    assert "available" in data
    assert "models" in data


def test_lora_datasets_endpoint():
    res = client.get("/api/lora/datasets")
    assert res.status_code == 200
    datasets = res.json()
    assert len(datasets) >= 2
    ids = [d["id"] for d in datasets]
    assert "gwaya_verified_receipts" in ids


def test_lora_generate_config():
    payload = {
        "base_model": "qwen2.5-coder:1.5b",
        "r": 16,
        "lora_alpha": 32,
        "learning_rate": 0.0002,
        "epochs": 3,
        "quantization": "4bit",
        "dataset_name": "gwaya_verified_receipts",
    }
    res = client.post("/api/lora/generate-config", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "created"
    assert "config_path" in data
    assert "train_script_path" in data
