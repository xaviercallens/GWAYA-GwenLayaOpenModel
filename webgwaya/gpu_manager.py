"""
WebGWAYA GPU Lifecycle & VRAM Manager
====================================
Manages the lifecycle of GPU-accelerated services (Ollama CUDA daemon, models in VRAM)
allowing users to seamlessly STOP the GPU to release 100% of VRAM for external projects
(PyTorch training, 3D rendering, Stable Diffusion, etc.) and START it on demand.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("WebGWAYA.GPU")

OLLAMA_DEFAULT_HOST = "http://127.0.0.1:11434"

# Use environment variables for paths, with platform-specific defaults
if sys.platform == "win32":
    DEFAULT_MODELS_DIR = os.environ.get("OLLAMA_MODELS", r"C:\Users\%USERNAME%\AppData\Local\Ollama\models")
    CANDIDATE_OLLAMA_PATHS = [
        Path(os.environ.get("OLLAMA_BIN", r"C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe")),
    ]
else:
    # Linux/macOS defaults
    DEFAULT_MODELS_DIR = os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models"))
    CANDIDATE_OLLAMA_PATHS = []


def find_ollama_bin() -> Path | None:
    """Locates the Ollama executable on the system."""
    env_bin = os.environ.get("OLLAMA_BIN")
    if env_bin and Path(env_bin).exists():
        return Path(env_bin)

    for p in CANDIDATE_OLLAMA_PATHS:
        if p.exists():
            return p

    which_path = shutil.which("ollama")
    if which_path:
        return Path(which_path)

    return None


def get_nvidia_telemetry() -> dict[str, Any]:
    """Queries nvidia-smi for current GPU telemetry and VRAM utilization."""
    info: dict[str, Any] = {"available": False, "raw": "No GPU detected"}
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=4.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 8:
                info = {
                    "available": True,
                    "name": parts[0],
                    "driver": parts[1],
                    "memory_total_mb": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_free_mb": float(parts[4]),
                    "temp_c": float(parts[5]),
                    "power_w": float(parts[6]),
                    "util_pct": float(parts[7]),
                }
    except Exception as e:
        info["error"] = str(e)
    return info


def is_ollama_service_online(host: str = OLLAMA_DEFAULT_HOST) -> bool:
    """Checks whether the Ollama HTTP daemon is responsive."""
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_available_models(host: str = OLLAMA_DEFAULT_HOST) -> list[dict[str, Any]]:
    """Lists pulled models from Ollama."""
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = []
            for m in data.get("models", []):
                size_gb = round(m.get("size", 0) / (1024 ** 3), 2)
                models.append({
                    "name": m.get("name", "unknown"),
                    "size": f"{size_gb} GB" if size_gb > 0 else "unknown",
                    "modified_at": m.get("modified_at", ""),
                })
            return models
    except Exception:
        return []


def get_loaded_vram_models(host: str = OLLAMA_DEFAULT_HOST) -> list[dict[str, Any]]:
    """Lists models currently loaded in GPU VRAM via /api/ps."""
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            loaded = []
            for m in data.get("models", []):
                size_vram = round(m.get("size_vram", 0) / (1024 ** 2), 1)
                loaded.append({
                    "name": m.get("name", "unknown"),
                    "size_vram_mb": size_vram,
                    "expires_at": m.get("expires_at", ""),
                })
            return loaded
    except Exception:
        return []


def unload_all_vram_models(host: str = OLLAMA_DEFAULT_HOST) -> bool:
    """Signals Ollama to immediately offload any models currently active in VRAM."""
    loaded = get_loaded_vram_models(host)
    for m in loaded:
        m_name = m.get("name")
        if m_name:
            try:
                payload = json.dumps({"model": m_name, "keep_alive": 0}).encode("utf-8")
                req = urllib.request.Request(
                    f"{host.rstrip('/')}/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    pass
            except Exception as e:
                log.warning("Could not explicitly unload model %s: %s", m_name, e)
    return True


def stop_gpu_service(host: str = OLLAMA_DEFAULT_HOST) -> dict[str, Any]:
    """
    Completely stops the GPU service and background Ollama processes,
    releasing all GPU VRAM for other external projects.
    Sends SIGTERM first, then SIGKILL if needed.
    """
    log.info("Request received to STOP GPU service and release VRAM...")

    # 1. Unload models in VRAM if service is still responding
    if is_ollama_service_online(host):
        try:
            unload_all_vram_models(host)
        except Exception as e:
            log.warning("Error during VRAM model unload: %s", e)

    # 2. Terminate Ollama daemon and sub-processes to release all CUDA contexts
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/IM", "ollama.exe"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
        except Exception as e:
            log.warning("taskkill error: %s", e)
    else:
        # Linux/macOS: use pgrep/pkill -x (exact match on executable name)
        try:
            # First try SIGTERM (graceful shutdown)
            subprocess.run(["pkill", "-x", "ollama"], capture_output=True, text=True, check=False, timeout=5.0)
            time.sleep(1.0)

            # Check if still running, then force kill with SIGKILL
            pgrep_result = subprocess.run(
                ["pgrep", "-x", "ollama"],
                capture_output=True,
                text=True,
                check=False,
                timeout=3.0,
            )
            if pgrep_result.returncode == 0:  # Process still exists
                log.info("Ollama still running after SIGTERM, sending SIGKILL...")
                subprocess.run(["pkill", "-9", "-x", "ollama"], capture_output=True, text=True, check=False, timeout=3.0)
        except Exception as e:
            log.warning("pkill/pgrep error: %s", e)

    # Small pause to allow driver to reclaim memory
    time.sleep(3.0)

    # Verify status
    still_online = is_ollama_service_online(host)
    telemetry = get_nvidia_telemetry()

    return {
        "status": "stopped" if not still_online else "stopping",
        "is_running": still_online,
        "gpu": telemetry,
        "vram_freed": not still_online,
        "message": (
            "GPU service successfully STOPPED. All VRAM has been released for external workloads."
            if not still_online
            else "GPU service shutdown initiated."
        ),
    }


def start_gpu_service(host: str = OLLAMA_DEFAULT_HOST, timeout_s: float = 25.0) -> dict[str, Any]:
    """
    Starts the Ollama CUDA daemon, warming up GPU capabilities for WebGWAYA.
    Uses OLLAMA_BIN and OLLAMA_MODELS environment variables for configuration.
    """
    log.info("Request received to START GPU service...")

    if is_ollama_service_online(host):
        telemetry = get_nvidia_telemetry()
        models = get_available_models(host)
        return {
            "status": "active",
            "is_running": True,
            "gpu": telemetry,
            "models_count": len(models),
            "models": models,
            "message": "GPU service is already active and ready.",
        }

    ollama_bin = find_ollama_bin()
    if not ollama_bin or not ollama_bin.exists():
        raise RuntimeError(f"Ollama binary not found. Set OLLAMA_BIN env var or install Ollama.")

    # Use env var OLLAMA_MODELS if set, otherwise use default
    models_dir = os.environ.get("OLLAMA_MODELS", DEFAULT_MODELS_DIR)
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = str(models_dir)

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        [str(ollama_bin), "serve"],
        env=env,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    t0 = time.time()
    online = False
    while time.time() - t0 < timeout_s:
        if is_ollama_service_online(host):
            online = True
            break
        time.sleep(3.0)

    if not online:
        return {
            "status": "failed",
            "is_running": False,
            "gpu": get_nvidia_telemetry(),
            "message": f"Timed out waiting for GPU service to become healthy after {timeout_s}s.",
        }

    telemetry = get_nvidia_telemetry()
    models = get_available_models(host)
    return {
        "status": "active",
        "is_running": True,
        "gpu": telemetry,
        "models_count": len(models),
        "models": models,
        "message": "GPU engine successfully started. GPU is active and ready for inference.",
    }


def get_gpu_full_status(host: str = OLLAMA_DEFAULT_HOST) -> dict[str, Any]:
    """Retrieves unified GPU hardware and service lifecycle status."""
    online = is_ollama_service_online(host)
    telemetry = get_nvidia_telemetry()
    models = get_available_models(host) if online else []
    loaded_vram = get_loaded_vram_models(host) if online else []

    return {
        "status": "active" if online else "stopped",
        "is_running": online,
        "gpu": telemetry,
        "models_count": len(models),
        "models": models,
        "loaded_in_vram": loaded_vram,
        "vram_freed": not online,
        "message": (
            "GPU engine is active and ready for inference."
            if online
            else "GPU engine is STOPPED. VRAM is 100% released for external projects."
        ),
    }
