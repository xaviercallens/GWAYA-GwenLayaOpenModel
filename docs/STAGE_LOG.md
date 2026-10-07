# Stage log

## L0 (local env), night 2026-10-07

- venv: numpy, scipy, datasets, safetensors, pytest, scikit-learn, transformers, peft, torch (CPU wheel) installed with uv; caches on /mnt/data/xdev-cache.
- llama.cpp release b11476 (ubuntu-x64 CPU) in `$NIGHT/bin`; Qwen3.5-2B and 4B Q4_K_M GGUFs from unsloth (revisions and sha256 in `$NIGHT/results/L0_env.json`).
- llama-server on 127.0.0.1:8091, `--parallel 2`, `-t 4` (D14). Logprobs are returned.
- Measured CPU speed is very low (about 3 tok/s for 2B, see L0_env.json), which constrains E-night size. This is the main finding of the step.
- Offline pytest (metrics_selective, run_study, oracles_fail_closed, rust_oracle, domains): 85 passed, 1 skipped (linker not reachable inside the sandbox). One failing test fixed (D16).
- GCP preflight (read-only): GPUS_ALL_REGIONS usage 1.0 / limit 1.0, held by socreateai-agora-hermes-node1 (not ours). No spend. Poller W running, log `$NIGHT/logs/gpu_slot.log`.
