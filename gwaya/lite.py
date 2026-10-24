"""GWAYA-Lite: Q4 Qwen2.5-Coder tiers that fit 4-8 GB GPUs, a VRAM estimator and llama-server launch lines.

Sizes are ESTIMATES from the published Q4_K_M file sizes and the models' KV-cache geometry (layers x kv_heads x head_dim,
fp16 cache). They are not measurements: docs/PROPOSAL_HIVEMIND_GWAYA_CODER.md step P1 measures them on a real card, and
``estimate_vram_mib`` must be replaced by the measured numbers (or checked against them) before the table is relied on.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass

HIVEMIND_VRAM_CEILING_MIB = 6656  # HiveMind golden invariant: 8 GB card minus 1536 MiB WDDM reserve
CUDA_OVERHEAD_MIB = 600  # CUDA context + compute buffers, conservative


@dataclass(frozen=True)
class LiteTier:
    name: str  # model name sent to the server and used as the cascade ladder entry
    gguf_repo: str
    gguf_file: str
    weights_mib: int  # Q4_K_M file size
    kv_kib_per_token: int  # 2 * layers * kv_heads * head_dim * 2 bytes / 1024
    max_ctx: int = 32768


LITE_TIERS: dict[str, LiteTier] = {
    "1.5b": LiteTier("qwen2.5-coder:1.5b", "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
                     "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1070, 28),
    "3b": LiteTier("qwen2.5-coder:3b", "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
                   "qwen2.5-coder-3b-instruct-q4_k_m.gguf", 1840, 36),
    "7b": LiteTier("qwen2.5-coder:7b", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                   "qwen2.5-coder-7b-instruct-q4_k_m.gguf", 4466, 56),
}


def estimate_vram_mib(tier: LiteTier, ctx: int = 4096, kv_fp16: bool = True) -> int:
    """Estimated resident VRAM in MiB with every layer offloaded. ``kv_fp16=False`` models a q8_0 cache (half size)."""
    if not 256 <= ctx <= tier.max_ctx:
        raise ValueError(f"ctx must be in [256, {tier.max_ctx}] for {tier.name}")
    kv = tier.kv_kib_per_token * ctx / 1024 * (1.0 if kv_fp16 else 0.5)
    return int(round(tier.weights_mib + kv + CUDA_OVERHEAD_MIB))


def pick_tier(vram_mib: int = HIVEMIND_VRAM_CEILING_MIB, ctx: int = 4096, kv_fp16: bool = True) -> LiteTier | None:
    """Largest tier whose estimate fits ``vram_mib``; None when even the 1.5B tier does not."""
    for key in ("7b", "3b", "1.5b"):
        if estimate_vram_mib(LITE_TIERS[key], ctx, kv_fp16) <= vram_mib:
            return LITE_TIERS[key]
    return None


def server_command(tier: LiteTier, model_path: str, port: int = 8080, ctx: int = 4096, host: str = "127.0.0.1",
                   kv_fp16: bool = True) -> str:
    """llama-server command line for the tier (OpenAI-format /v1, all layers on GPU, thinking budget 0 as in gwaya.llama_tier)."""
    argv = ["llama-server", "-m", model_path, "--host", host, "--port", str(port), "-c", str(ctx), "-ngl", "99",
            "--alias", tier.name, "--reasoning-budget", "0"]
    if not kv_fp16:
        argv += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-fa"]
    return shlex.join(argv)
