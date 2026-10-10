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
  optional `generate_with_signals(prompt, domain) -> (text, {"mean_logprob", "min_logprob"})` used when present; no
  llama.cpp client exists in the repo, so the default Ollama ladder never supplies them. With the default Ollama ladder
  (`qwen2.5-coder:*`) tier names also do not match, so without `GWENLAYA_LAYA_TIER_MAP` Laya falls back to the static
  ladder and the scorer abstains (`router_tiers_unmatched`, `tier_not_in_laya_training`).
- A vetoed or abstaining scorer can still contribute LIKELY_WRONG when p <= `tau_lo` (veto is for LIKELY_CORRECT only).
  That path is not validated for out-of-distribution inputs either; leave `tau_lo` at -inf to switch it off.
- Log-probs from a different backend than the training generations (bf16 Qwen3.5) are a distribution shift that no
  evaluation covers.

## Measured (scripts/laya_runtime_parity.py, `results/gwenlaya_v4/laya_runtime/parity.json`)

20 router rows (60 probabilities) and 20 calibrator rows of split C against the stored TPU v5e bf16 predictions:
router max / mean absolute difference 0.0038 / 0.0008; calibrator 0.0029 / 0.0006 (tolerance 0.02). Runtime is fp32
on CPU with the LoRA merged. Latency per call, i7-4930MX, 3 threads, machine shared with other jobs: router about
2.0-2.8 s mean (first call about 9-12 s), calibrator about 7-9 s mean (up to 512 tokens). One answer costs one router
call plus one calibrator call per tier tried, so a Laya-assisted answer adds roughly 10 s per tier on this CPU.

## Not validated

No end-to-end evaluation of the routed-and-scored system on this runtime exists: not accuracy, not the cost of the
cascade, not LIKELY_CORRECT precision under truncation or missing log-probs, not any local (Ollama/quantised) tier.
The offline E' numbers describe the model fed TPU-run Qwen3.5 generations with full signals.
