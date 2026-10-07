# Deviations from docs/GWENLAYA_PREREGISTRATION.md

Every departure from the pre-registration is recorded here with its reason, before the run it
affects. Nothing here changes a hypothesis's metric or MEI. Results affected by a deviation are
labelled in the paper.

## Night run 2026-10-07

Schedule: `experiments/night_schedule.json`. Branch: `night/2026-10-07`. Written before any
generation of the night.

### Context (read this session, 2026-10-07)

- `GPUS_ALL_REGIONS` limit 1, usage 1: held by `socreateai-agora-hermes-node1` (not a GwenLaya VM;
  never touched). It may stay busy all night. GPU work is therefore optional and must not be on the
  critical path.
- Prior spend: ~0.84 USD of TPU smoke tests (task text). The two lines in `spend_ledger.jsonl`
  carry seconds (240, 2466) but no `usd_estimate` field; the GCS `runs/LEDGER.jsonl` does not exist.
- TPU is not used: Qwen3.5/3.8 are unproven on TPU and have no TPU LoRA path.
- Local Ollama is version 0.1.44, returns no logprobs, and may not be restarted or upgraded.
- The scratchpad venv has no numpy/torch/transformers/peft/datasets; L0 installs them on /mnt/data.

### Deviations

| # | Prereg says | Night run does | Reason |
|---|---|---|---|
| D1 | E = full primary pool (1,552) on L4 | **E-night**: a seeded prefix (eval_order 20261007) of each primary domain, size fixed from measured CPU throughput in the freeze addendum before generation; full 1,552 only if G1 runs | No GPU slot guaranteed; CPU generation of 2B/4B costs tens of seconds per item. The prereg's truncation rule (section 9) already analyses truncated seeded n as is. |
| D2 | Tiers 2B/4B/9B/27B, B2 = largest tier | Local path: tiers **2B, 4B** only; B2/B3/B5 use 4B. With G1: 2B/4B/9B, 27B on a 300-item seeded subsample (exploratory, S6 reduction rule 2) | 9B/27B are impractical on 8 CPUs; 27B full E is in cut_order. |
| D3 | H3 cost metric = L4 GPU-seconds only | If G1 runs: L4 GPU-seconds (as registered). If not: **CPU-seconds** on this machine, labelled as a deviation; H3 then speaks to the CPU-deployment regime only | No L4 timings without the GPU slot. |
| D4 | Calibrator, thresholds and B3 threshold fit on a separate C; Laya LoRA on router-train | **5-fold cross-fitting on E-night grouped by source-problem cluster**, thresholds fit on training folds only, out-of-fold scores reported (the prereg's section 6 "secondary robustness" procedure, promoted to primary for this run). Procedure frozen in the addendum before generation | C generation would double CPU time; the review found MBPP+ overlap removes 154/170 Python C items, so C was not label-matched anyway (review F3). |
| D5 | Laya LoRA r=16 pre/post heads | LoRA if the 50-step CPU timing projects the 5-fold run <= 2 h; otherwise a **frozen-Laya-embedding logistic head**, and H5 is reported as "Laya-embedding probe vs LR" | CPU wall-clock limit (<= 4 h per local step). |
| D6 | Engine: llama.cpp server, same build for all quants | Same, but local runs use a prebuilt llama.cpp CPU release on 127.0.0.1:8091; G1 uses a CUDA build of the same release tag where available (tag recorded). Ollama not used for generation | Ollama 0.1.44 has no logprobs (needed for B3) and cannot be upgraded. |
| D7 | Lean 4: E (244), C, C2 elaboration, H11 | **Dropped tonight** | Lean is outside the pooled H1 analysis; CPU time is reserved for the primaries. |
| D8 | S3, S4a (SFT, DPO), S4b, S4c, S4d, S5 | **Deferred** (not run). H9 deferred; H10, H11, H12 dropped for this run | No verified traces without GPU; H9 prior 0.25 and ~0.5 power at MEI +3 pts (n=542); H10 0.05, H11 0.12, H12 0.12 < 0.15. Registered design stays in the paper. |
| D9 | B4 self-consistency k=5; second seed | **Dropped** | First S6 item in cut_order; 5x generation cost. Single seed is a stated limitation. |
| D10 | GSM8K, AIME25, LiveCodeBench, MBPP continuity anchor, 0.8B tier 0 | **Dropped** | Exploratory only, first in the section 9 cut order. |
| D11 | H8 quant factor on full E, 9B bf16/q8_0/q4 | Only if G1 sub-task (e) runs: 9B on a 300-item subsample; reported as an **exploratory estimate**, no TOST decision | A +-2 pt TOST is unattainable at n=300. |
| D12 | S7 endpoint 4-6 cold samples, cap 2.5 | E1: 4 cold samples, cap 2.0, only if G1 completed, the Laya head exists, L4 Cloud Run quota is present and the exact commands are authorized; else H14 NOT RUN | cut_order drops cold samples 5-6; no Cloud Build spend. |
| D13 | Primary E items verified on CPU only | Unchanged | (listed to make clear it is kept) |
| D14 | L0: llama-server `-t 8` | `-t 4` (still `--parallel 2`). Measured: `-t 8` gave 0.28-0.30 tok/s because 3 foreign CPU-bound python jobs (not ours, not touched) hold ~3 of 8 logical cores and the thread barrier stalls; `-t 4` gave 3.4 tok/s (llama-bench `-t 2/3/4` = 2.2/3.2/3.4 tg) | Throughput |
| D15 | L0 tok/s on 10 MBPP+ prompts at the default token budget | `max_tokens` = 64, not 256, to fit the 0.75 h wall clock at ~3 tok/s | Wall clock |
| D16 | Source-controlled bug fix: `gwaya/domains/math_check.py::_to_sympy_src` rejects multi-letter words | With sympy installed, `answers_equivalent('banana','apple')` returned False (words read as free symbols) instead of None (undecided); `tests/test_domains.py::test_unparseable_is_undecided` failed. Words now return None. Single-letter algebra (x+1 vs x+2) is unchanged | Fail-closed correctness |

### Hypothesis status for this run

| H | Prior (review) | Night status |
|---|---|---|
| H1 | 0.45 | Primary, run (local path always; L4 path if G1). Underpowered flag if paired n < 500 |
| H3 | 0.50 | Primary, run (CPU-seconds locally, D3; GPU-seconds if G1) |
| H4, H5, H6 | 0.15, 0.25, 0.40 | Secondary, run via cross-fitting (D4, D5) |
| H2, H7 | 0.08, 0.10 | Exploratory estimates (free from the same cache) |
| H8 | 0.30 | Exploratory estimate, only if G1(e) runs (D11) |
| H9 | 0.25 | Deferred (D8) |
| H10, H11, H12 | 0.05, 0.12, 0.12 | Dropped for this run (D8) |
| H13 | 0.75 | Run (ledger) |
| H14 | 0.50 | Only if E1 runs (D12) |
| H15 | 0.70 | Offline kill/resume test; real preemption timestamps if G1 is preempted |

### Budget for the night

GCP caps: G1 2.00 + E1 2.00 = 4.00 USD; with prior ~0.84 the planned total is 4.84 USD
(<= 45 rule, <= 40 planned, <= 50 hard). All other steps are local ($0).

### Outcomes

Filled in after the run from the results files only. Unknown = TBD.
