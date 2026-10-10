# Laya at runtime (router + calibrator inside GwenLaya)

Status: wiring only. The frozen model (`docs/LAYA_FREEZE.txt`) is unchanged and its offline numbers are the only
evidence there is. **This wiring has not been evaluated end to end as a system.**

## What it does

- Pre-routing: `LayaRuntime.router(prompt, domain)` returns `{tier: P(correct)}` from the frozen ModernBERT-large +
  LoRA router. GwenLaya uses the registered A13b policy (`route_policy="start_at"`, `scripts/analyze_laya.py::run_policy`):
  start at the cheapest tier with `p >= tau_route` (else the last tier), then cascade upward on the gate. `tau_route = 0.5`
  is the value chosen on C for that registered result (`results/gwenlaya_v4/laya/laya_eprime.json`).
- Post-result analysis: `LayaRuntime.scorer(features)` returns the calibrator's raw sigmoid; GwenLaya maps it through the
  registered isotonic map (`docs/laya_isotonic_C.json`, fitted on C, n=849) via a `CalibrationArtifact`, then applies
  `tau_hi` / `tau_lo` to give VERIFIED / LIKELY_CORRECT / LIKELY_WRONG / ABSTAIN. VERIFIED always comes from the gate.
- Every Laya use is written to `result["evidence"]["laya"]`: `router`, `scorer`, `truncated_candidate`,
  `signals_incomplete`, `model_sha` (12 hex of the verified file hashes).

## Enable it

```
export GWAYA_DATA_ROOT=/mnt/data/home/xavkal/gwaya-data     # backbone is read from $GWAYA_DATA_ROOT/hf-home (no network)
export GWENLAYA_LAYA_DIR=$GWAYA_DATA_ROOT/laya_data/laya_v1  # contains router/ and calibrator/
export GWENLAYA_LAYA_DOMAINS=python,math                      # default; rust is gate-only, lean4 is refused
```

| variable | default | meaning |
|---|---|---|
| `GWENLAYA_LAYA_DIR` | unset = behaviour exactly as before | directory with `router/` and `calibrator/`; sha256 of every file is checked against `docs/LAYA_FREEZE.txt` (mismatch raises) |
| `GWENLAYA_LAYA_DEVICE` | `cpu` | torch device; fp32, batch size 1 |
| `GWENLAYA_LAYA_DOMAINS` | `python,math` | domains where Laya acts; only `python`, `rust`, `math` are accepted |
| `GWENLAYA_LAYA_TAU_ROUTE` | `0.5` | router threshold |
| `GWENLAYA_LAYA_THRESHOLDS` | unset | JSON `{domain: {tau_hi, tau_lo}}`; unset means +inf / -inf, so Laya reports p but never decides LIKELY_CORRECT/LIKELY_WRONG |
| `GWENLAYA_LAYA_TIER_MAP` | unset | `runtime_tier=trained_tier,...`; the heads know only `qwen3.5-2b-bf16`, `qwen3.5-4b-bf16`, `qwen3.5-9b-bf16` |
| `GWENLAYA_LAYA_THREADS` | torch default | `torch.set_num_threads` |

Requires `torch`, `transformers`, `peft`, `safetensors` and the cached `answerdotai/ModernBERT-large`.
`scripts/laya_runtime_thresholds.py` writes exploratory thresholds to `results/gwenlaya_v4/laya_runtime/thresholds_C.json`
(in-sample on C; no Laya `tau_hi`/`tau_lo` was ever registered), so using that file is a choice, not a registered setting.
A tier map says "treat this model as that trained tier"; it is an assumption the evaluation never tested.

## What is and is not covered

| domain | router / calibrator | why |
|---|---|---|
| python, math | on by default | in training; E' results in the paper |
| rust | off by default (`GWENLAYA_LAYA_DOMAINS=python,rust,math` to enable) | trained on, but at chance on fresh Rust R' (`docs/CLAIM_AUDIT_380.md`, `results/gwenlaya_v4/laya/laya_rfresh.json`); gate-only |
| lean4 | refused (cannot be enabled) | never seen by the frozen Laya; the gate-only policy in `gwenlaya.py` stands |

## Failure modes (all fail closed)

- Truncation: text is `[domain] [tier] [signals] [prompt] [candidate]`, cut at 512 tokens, prompt first. If any of the
  candidate is cut, `truncated_candidate=true` and LIKELY_CORRECT is vetoed. On fresh Rust the candidate was wholly outside
  the window for 103 of 207 rows; in a 20-row sample of C, 9 rows were truncated (`parity.json`). VERIFIED is unaffected.
- Missing signals: the calibrator was trained on rows that always had `mean_logprob` and `min_logprob`. A tier that cannot
  supply them (Ollama 0.1.44 returns none) sets `signals_incomplete=true` and LIKELY_CORRECT is vetoed. `Tier` has an
  optional `generate_with_signals(prompt, domain) -> (text, {"mean_logprob", "min_logprob"})` used when present. The
  llama.cpp tier (`gwaya/llama_tier.py`, below) supplies them; the default Ollama ladder never does. With the default Ollama ladder
  (`qwen2.5-coder:*`) tier names also do not match, so without `GWENLAYA_LAYA_TIER_MAP` Laya falls back to the static
  ladder and the scorer abstains (`router_tiers_unmatched`, `tier_not_in_laya_training`).
- A vetoed or abstaining scorer can still contribute LIKELY_WRONG when p <= `tau_lo` (veto is for LIKELY_CORRECT only).
  That path is not validated for out-of-distribution inputs either; leave `tau_lo` at -inf to switch it off.
- Log-probs from a different backend than the training generations (bf16 Qwen3.5) are a distribution shift that no
  evaluation covers.

## Local serving with log-probs

Ollama 0.1.44 returns no token log-probs, so with Ollama tiers `signals_incomplete` is always set and the calibrator can
never support LIKELY_CORRECT. The study's CPU runs used llama.cpp's `llama-server`, which does return them;
`gwaya/llama_tier.py` is a runtime tier for it.

```
scripts/gwenlaya_serve.sh start            # 2B on 127.0.0.1:8091, 4B on :8092 (or: start 2b); prints the exports below
export GWENLAYA_BACKEND=llamacpp
export GWENLAYA_LLAMA_URLS="qwen3.5-2b-bf16=http://127.0.0.1:8091,qwen3.5-4b-bf16=http://127.0.0.1:8092"  # cheapest first
export GWENLAYA_SERVED_QUANT=q4_k_m        # label only, recorded in the evidence
export GWENLAYA_LAYA_DIR=$GWAYA_DATA_ROOT/laya_data/laya_v1
scripts/gwenlaya_serve.sh status | stop
```

- **Protocol** (mirrors the registered study serving, `scripts/run_study.py::OpenAICompatBackend`): `/v1/completions`
  with the raw Qwen prompt of `OllamaGenerator.build_raw_prompt` (math prose with the empty think block, code fence
  prefill, `math_pot` domain), its stop sequences, `temperature 0`, `seed 0`, 1024 new tokens, `logprobs: 1`; output is
  wrapped by `OllamaGenerator.wrap_output`. The server runs with `--parallel 2 -c 8192 --jinja --reasoning-budget 0`
  (thinking off) and `-t 3` (the study used `-t 4`, D14). `mean_logprob` is the mean and `min_logprob` the minimum of the
  returned token log-probs, as in `scripts/data/build_laya_data.py`.
- **Failures raise**: connection refused, timeout (default 1800 s), HTTP errors and malformed bodies become a
  `generation_error` on the tier step. A server that returns no log-probs gives signals `{}`, so `signals_incomplete`
  is set (never zeros).
- **Names**: with `GWENLAYA_BACKEND=llamacpp` the names in `GWENLAYA_LLAMA_URLS` are used verbatim and `GWENLAYA_TIERS`
  is ignored. Use the trained names (`qwen3.5-2b-bf16`, `qwen3.5-4b-bf16`) so no tier map is needed. Without the
  switch nothing changes (Ollama, `GWENLAYA_TIERS`).
- **Quantisation shift**: the served weights are Q4_K_M GGUF, the Laya was trained on bf16 generations from vLLM on TPU.
  With the llama.cpp backend and Laya both on, every scorer evidence carries `quant_shift: true` and
  `served_quant` (from `GWENLAYA_SERVED_QUANT`, `unspecified` if unset), also at `evidence["laya"]`. The tier names say
  `-bf16` only because that is what the heads know. This is a label, not a correction: **the calibration is not claimed
  to hold under this shift.** The study's own CPU generations (2B/4B Q4_K_M) are the data to measure it on.
- **Ports**: `scripts/gwenlaya_serve.sh` refuses to start on a port that is already bound (another job's server is
  never touched) and only signals the pids in its own pidfiles (`$GWAYA_DATA_ROOT/serve/`). Override with
  `GWENLAYA_PORT_2B`, `GWENLAYA_PORT_4B`, `GWENLAYA_THREADS`. The smoke test used port 8093 because another job held 8091.
- **Smoke test** (`scripts/gwenlaya_live_smoke.py`, result `results/gwenlaya_v4/laya_runtime/live_smoke.json`): 3 tasks
  (python with 3 visible tests, python without tests, math without a gold answer) through the real frozen Laya and a 2B
  Q4_K_M llama-server (`-t 3`, machine shared with another job, load average about 4-6 on 8 logical cores), exploratory
  thresholds. Signals arrived (`signals_incomplete=false` in all 3), nothing was truncated, `quant_shift=true`.

| task | verdict | p_correct | route s | generate s | gate s | calibrate s | wall s |
|---|---|---|---|---|---|---|---|
| python, 3 visible tests | VERIFIED (gate) | 1.00 | 1.1 | 8.1 | 0.6 | 2.4 | 12.2 |
| python, no tests | LIKELY_WRONG | 0.43 | 1.5 | 9.6 | 0.0 | 2.9 | 14.0 |
| math, PoT gate, no gold | LIKELY_CORRECT | 0.97 | 1.1 | 71.1 | 17.0 (15.0 PoT generation) | 8.1 | 97.3 |

  **n = 3 is a smoke test, not an evaluation.** It shows the wiring runs end to end and gives order-of-magnitude
  latencies on this machine (one-off Laya load 39 s). The "LIKELY_WRONG" on the palindrome task is one sample of the
  exploratory in-sample python thresholds (`tau_lo` 0.69 > the p of 0.43), not evidence that the function is wrong, and
  not a measured error rate.

## Measured (scripts/laya_runtime_parity.py, `results/gwenlaya_v4/laya_runtime/parity.json`)

20 router rows (60 probabilities) and 20 calibrator rows of split C against the stored TPU v5e bf16 predictions:
router max / mean absolute difference 0.0038 / 0.0008; calibrator 0.0029 / 0.0006 (tolerance 0.02). Runtime is fp32
on CPU with the LoRA merged. Latency per call, i7-4930MX, 3 threads, machine shared with other jobs: router about
2.0-2.8 s mean (first call about 9-12 s), calibrator about 7-9 s mean (up to 512 tokens). One answer costs one router
call plus one calibrator call per tier tried, so a Laya-assisted answer adds roughly 10 s per tier on this CPU.

## Not validated

No end-to-end evaluation of the routed-and-scored system on this runtime exists: not accuracy, not the cost of the
cascade, not LIKELY_CORRECT precision under truncation or missing log-probs. The offline E' numbers describe the model
fed TPU-run Qwen3.5 generations with full signals. Specifically:

- **Quantisation shift**: the calibrator and router were trained on bf16 generations and signals; the llama.cpp tier
  serves Q4_K_M. Log-prob scale, text and correctness all shift with quantisation. Nothing measured says the
  calibration or the exploratory thresholds survive that; the evidence only labels it (`quant_shift`).
- **The n = 3 smoke test** (above) checks plumbing and cost only. It has no accuracy, calibration or precision content.
- **Calibrator latency**: about 2-3 s per call for short candidates and 8 s for a longer one in the smoke test, 7-9 s
  at 512 tokens in the parity run, on a shared 8-thread CPU with 2 torch threads; the 9B tier and a free machine were
  not timed. Router about 1-1.5 s per call.
- **Tier latency**: only the 2B Q4_K_M server was run live. The 4B path (port 8092) is covered by unit tests with a
  fake server, not by a live run here, and decodes at about 1.3 tok/s in the study, so a 1024-token answer can take
  about 13 minutes (the default timeout is 1800 s). The 9B tier has no server script.
- **Concurrency**: one request at a time per tier was used; no concurrent-load behaviour was checked.
- Rust stays gate-only and lean4 refused, as above.
