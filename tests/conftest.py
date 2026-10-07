"""Shared pytest fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _lora_output_to_tmp(tmp_path, monkeypatch):
    """Keep /api/lora/generate-config from writing into the repo's lora_configs/."""
    monkeypatch.setenv("WEBGWAYA_LORA_DIR", str(tmp_path / "lora_configs"))
