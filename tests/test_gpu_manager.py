"""
tests/test_gpu_manager.py
=========================
Unit tests for WebGWAYA GPU Lifecycle & VRAM Manager.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from webgwaya.gpu_manager import (
    DEFAULT_MODELS_DIR,
    find_ollama_bin,
    get_available_models,
    get_gpu_full_status,
    get_nvidia_telemetry,
    is_ollama_service_online,
    start_gpu_service,
    stop_gpu_service,
)


def test_find_ollama_bin():
    binary = find_ollama_bin()
    if binary is None:
        pytest.skip("Ollama binary not found on this system")
    assert binary.exists()
    assert binary.name.lower().startswith("ollama")


def test_get_nvidia_telemetry():
    telemetry = get_nvidia_telemetry()
    assert "available" in telemetry
    if not telemetry["available"]:
        pytest.skip("No NVIDIA GPU available on this system")
    assert "RTX" in telemetry["name"] or "NVIDIA" in telemetry["name"]
    assert telemetry["memory_total_mb"] > 0
    assert telemetry["memory_used_mb"] >= 0
    assert telemetry["memory_free_mb"] > 0
    assert telemetry["temp_c"] >= 0


def test_get_gpu_full_status_structure():
    status = get_gpu_full_status()
    assert "status" in status
    assert status["status"] in ("active", "stopped")
    assert "is_running" in status
    assert isinstance(status["is_running"], bool)
    assert "gpu" in status
    assert "models_count" in status
    assert "vram_freed" in status
    assert "message" in status


def test_mocked_gpu_lifecycle():
    with patch("webgwaya.gpu_manager.is_ollama_service_online", side_effect=[True, False]):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            res = stop_gpu_service()
            assert res["status"] == "stopped"
            assert res["is_running"] is False
            assert res["vram_freed"] is True

    with patch("webgwaya.gpu_manager.is_ollama_service_online", side_effect=[False, True]):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=9999)
            res = start_gpu_service(timeout_s=5.0)
            assert res["status"] == "active"
            assert res["is_running"] is True
