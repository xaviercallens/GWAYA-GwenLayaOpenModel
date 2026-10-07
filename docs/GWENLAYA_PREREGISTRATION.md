# Pre-registration: GwenLaya, a system that answers only when it can

Written 2026-10-07, **before any GwenLaya run**. No GwenLaya generation, training, quantization
or evaluation has been run. This document contains **no measured GwenLaya values**. Every unknown
is the literal token `TBD`. The machine-readable plan is `experiments/plan.json`. The plan and
this document must agree, and `tests/test_gwenlaya_plan.py` checks that they do.

This document merges three design-panel proposals: evaluation/statistics, data/training, and
systems/cost. Section 13 lists every conflict between them and how it was resolved.

**Revision 2 (2026-10-07, design freeze).** This revision resolves the blocking items from the
red-team review. The item-by-item response is in `docs/PROPOSAL_REVIEW.md` section 6. The sha256
of this file and of `experiments/plan.json` is in `docs/PREREG_FREEZE.txt`. Any later edit
requires a new dated line there. This design freeze is separate from the pre-E freeze addendum
(section 6), which is still `TBD`.

**History (disclosed).** This project previously published fabricated numbers and retracted them
(see `results/PREREGISTRATION.md`). The rules in this document are designed so that cannot happen
again:
- A number appears in a results table only if a results file in `gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/results/`
  contains it.
- No threshold is changed after the eval pool E has been generated.
- A FAIL is reported as a finding.

---

## 0. Evidence this design rests on (read 2026-10-07, this session)

| Fact | Value | Source |
|---|---|---|
| v3 MBPP-sanitized n=257, L4: A0 / A3 pass@1 | 64.2 / 71.98 | `results/results_n257_l4_gpu.json` |
| v3 A3-A0 | +7.78 pts, CI [4.67, 11.28], 20 gained / 0 lost, gate FAIL vs +8.0 | same |
| v3 A3 gate coverage / hidden-pass given VERIFIED | 80.93 % / 88.94 % | same |
| v3 `git_commit` field | empty | same |
| v3 n=60 CPU: A3-A0 | +6.67 pts, CI [1.67, 13.33], McNemar p 0.125, FAIL; hidden-pass given VERIFIED 86.54 % | `results/results_n60_cpu.json` |
| v3 multi-model | n_tasks = 20 per model (0.5b: hidden-pass given VERIFIED 52.94 %) | `results/gwaya_v3_multi_model/*/summary.json` |

Derived by arithmetic on the file values, not measured: the v3 A3 confident-wrong rate, i.e.
VERIFIED but failing the hidden tests, as a share of all problems, is 80.93 % x (1 - 0.8894) =
**8.95 %**.

What follows from this evidence:
1. **VERIFIED is not ground truth.** The v3 gate executed only the public test. In v4, the gate
   never sees the scoring checks, VERIFIED precision is reported as an endpoint in its own right,
   and the calibration label is the hidden/full-check outcome.
2. **Decisions are based on confidence intervals and checked for power.** A point-estimate miss
   like +7.78 vs +8.0 cannot decide a hypothesis again.
3. **Runs with n=20 are never confirmatory.** The multi-model n=20 numbers are cited as motivation
   only.

**Re-verified this session (read-only):**

| Item | Value | Source |
|---|---|---|
| `GPUS_ALL_REGIONS` | first read: limit 1, usage 0. **Re-read 2026-10-07 19:11 UTC: limit 1, usage 1** | `gcloud compute project-info describe` |
| us-central1 GPU quota | PREEMPTIBLE_NVIDIA_L4_GPUS = 3; PREEMPTIBLE_NVIDIA_T4_GPUS = 1 | `gcloud compute regions describe` |
| us-east4 GPU quota | PREEMPTIBLE_NVIDIA_L4_GPUS = 1; PREEMPTIBLE_NVIDIA_T4_GPUS = 1 | `gcloud compute regions describe` |
| Cloud Run `NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion` | 3 in us-central1 and europe-west1; **no value shown for us-east4** | `gcloud beta quotas info list --service=run.googleapis.com` |
| Data-lake bucket | location US-EAST4 | `gcloud storage buckets describe` |
| VM `socreateai-agora-hermes-node1` | us-east4-b, n1-standard-8, T4, SPOT. TERMINATED at the first read; **RUNNING at 19:11 UTC**, so it holds the only GPU quota unit. Not a GwenLaya VM. | `gcloud compute instances list` |
| Cloud Run L4 in us-central1 | the existing service `deepseek-prover-v2` has 1 L4, maxScale 1, so GwenLaya headroom is 2 of 3 | `gcloud run services list` / `describe` (read-only) |
| Cloud Run GPU idle | "Cloud Run might keep instances idle ... (up to 15 minutes, or 10 minutes for GPUs)"; "GPU is billed for the entire duration of the instance lifecycle"; instance-based billing is required for GPU | docs.cloud.google.com/run/docs/about-instance-autoscaling and /run/docs/configuring/services/gpu |
| minif2f-lean4 toolchain | the HF repo holds only `README.md`, `test.jsonl` and `valid.jsonl`, with **no** `lean-toolchain` or lakefile. The row headers import modules such as `Mathlib.Algebra.BigOperators.Basic` | HF API tree + row 0 of `test.jsonl` |
| `formal_verification/lean_oracle_v5.tar.gz` | 40 entries, `lake-manifest.json` has `"packages": []` and there is no `lean-toolchain`: **no Mathlib**, so it is not reusable for minif2f | `gcloud storage cat` piped to `tar t` (read-only) |
| HF repo createdAt, Qwen3.5 0.8B/2B/4B/9B | 2026-02-27 / 2026-02-28 | HF API (`/api/models/...`); all Apache-2.0, `image-text-to-text` |
| HF repo createdAt, Qwen3.8-27B | 2026-08-05 | HF API; Apache-2.0, `image-text-to-text` |
| HF repo createdAt, convaiinnovations/laya | 2026-09-18 | HF API; Apache-2.0, `text-classification` |
| livecodebench/code_generation_lite | lastModified **2025-06-05**; card license tag `cc` (variant unspecified) | HF API |
| MultiPL-E `humaneval-rs` test format | `tests` is a `fn main()` with one `assert_eq!` per line, so the asserts can be split into gate-visible and hidden. Checked on 1 row only. | datasets-server `/rows` |

Eval-set sizes, from datasets-server `/size`:

| Dataset | Sizes |
|---|---|
| mbppplus test | 378 |
| humanevalplus test | 164 |
| MultiPL-E humaneval-rs | 156 |
| MultiPL-E mbpp-rs | 354 |
| minif2f-lean4 | validation 244, test 244 |
| MATH-500 test | 500 |
| aime25 test | 30 |
| MBPP sanitized | train 120, validation 43, prompt 7, test 257 |

**Prices**, list prices from the Cloud Billing Catalog API (Compute `6F81-5844-456A`, Cloud Run
`152E-C115-5142`), read 2026-10-07 in USD, before any credits:

| SKU | Price |
|---|---|
| Spot L4 GPU, us-east4 | $0.2978/h |
| Spot L4 GPU, us-central1 | $0.3192/h |
| Spot T4 GPU, us-east4 | $0.1945/h |
| Spot T4 GPU, us-central1 | $0.209/h |
| Spot G2 core, us-east4 | $0.01329/h |
| Spot G2 RAM, us-east4 | $0.001557/GiB-h |
| Spot N1 core, us-east4 | $0.0152/h |
| Spot N1 RAM, us-east4 | $0.002039/GiB-h |
| pd-balanced, us-east4 | $0.11/GiB-month |
| Cloud Run L4, no zonal redundancy, us-central1 | $0.0001867/s |
| Cloud Run instance-billed CPU, us-central1 | $0.000018/vCPU-s |
| Cloud Run instance-billed memory, us-central1 | $0.000002/GiB-s |

Derived costs. The machine shapes are ASSUMED, not re-read:
- Spot g2-standard-8 (8 vCPU/32 GiB) + L4 in us-east4 = $0.4539/h, plus 150 GiB disk at about
  $0.0226/h.
- Spot n1-standard-8 (8 vCPU/30 GiB) + T4 in us-east4 = $0.3773/h.
- Cloud Run 4 vCPU/16 GiB + L4 = $0.0002907/s = $1.0465/h.
- **Planning rates: $0.55/h per L4 spot VM-hour and $0.45/h per T4 VM-hour**, all-in with margin.
- `deploy/gpu_prices.json` (T4 $0.29, L4 $0.35, self-labelled "unverified") understates the VM
  cost and must not be used for caps.

**Ollama tags**, from ollama.com/library, read this session:
- `qwen3.5:{2b,4b,9b}-{q4_K_M,q8_0,bf16}` exist.
- `qwen3.5:0.8b` lists `-bf16` and `-q8_0` but **no `-q4_K_M`**.
- `qwen3.8:27b-{q4_K_M,q8_0,bf16}` exist.
- `qwen2.5-coder:1.5b-instruct-*` exists.

**Not re-verified this session.** These come from the design panel or the task context, not from
my own reads:
- The design panel's note that the data-lake `laya_lora` adapter is untrained
  (samples_trained = 5). It is not used regardless (see section 10).
- The Laya architecture (ModernBERT-large encoder plus head).
- Whether the Lean-Workbook statements elaborate under a current Mathlib.
- The LiveCodeBench `contest_date` range. The download was not permitted this session.

---

## 1. System under test

For each prompt, GwenLaya runs these steps:
1. **Laya pre-generation head**: from the prompt and the domain tag, predict p_t(correct) for each
   tier t.
2. **Route** to the cheapest tier with p_t >= tau_route. If no tier qualifies, return ABSTAIN.
3. The **tier generates** at its deployed quant (q4_K_M).
4. The **GWAYA fail-closed gate** runs the *gate-visible* checks. The gate never sees the scoring
   checks.
5. **Laya post-generation head**: from the prompt, the candidate and the gate signals, compute a
   calibrated p(correct).
6. **Verdict** (precedence top-down):
   - **VERIFIED**: all gate-visible executed checks passed. p is still reported, and VERIFIED
     precision against the hidden checks is measured and published.
   - **LIKELY_CORRECT(p)**: p >= tau_hi.
   - **LIKELY_WRONG(p)**: p <= tau_lo. The answer is withheld.
   - **ESCALATE** to the next tier, when one exists and the budget rule allows it.
   - **ABSTAIN** otherwise.

**Answered** = verdict in {VERIFIED, LIKELY_CORRECT}.

**Generation settings (fixed; `generation_settings` in plan.json):**
- Thinking mode is **disabled** for every tier, domain and stage.
- `max_new_tokens` = 1024 in every domain.
- E decoding is greedy, except for B4's samples and S3.
- GL has **0 repair rounds**.
- **No verification runs on a GPU VM.** The gate and all scoring checks run on the local CPU over
  cached candidates (C3, C6, C7).

**Confident-wrong rate (CWR)** = P(answered and wrong) over all E items. "Wrong" is judged
against ground truth the gate never saw.

Gate-visible checks per domain (deployment-faithful, no gold answers):

| Domain | Gate-visible check | Ground truth (scoring only) |
|---|---|---|
| Python | first public test (MBPP+/HumanEval+ base test) in bwrap sandbox | full plus test suite |
| Rust | `rustc` + first `assert_eq!` | `rustc` + all asserts |
| Lean 4 | `lake env lean` kernel check + `#print axioms` audit; rejects sorryAx, `Lean.ofReduceBool`/native_decide | identical check (precision = 1 by construction) |
| Math | sandboxed program-of-thought re-execution that reproduces the boxed answer (sympy-equivalent) | sympy/numeric equivalence (rel tol 1e-6) with the gold answer |

Lean is therefore **excluded from the pooled confident-wrong analyses** (H1, exploratory H2). It contributes
to coverage, routing and the calibration of LIKELY_* verdicts.

## 2. Domains and evaluation pool E (frozen before generation)

| Domain | E sets (n read from datasets-server) | Role |
|---|---|---|
| Python | evalplus/mbppplus test (378) + evalplus/humanevalplus test (164) = 542 | primary |
| Rust | nuprl/MultiPL-E humaneval-rs (156) + mbpp-rs (354) = 510 | primary |
| Math | HuggingFaceH4/MATH-500 (500) | primary |
| Lean 4 | cat-searcher/minif2f-lean4 test (244) | secondary (routing/calibration) |
| Math | openai/gsm8k test, seeded random 500 of the test split | secondary |
| Math | math-ai/aime25 (30) | exploratory (no power) |
| Python | livecodebench/code_generation_lite | exploratory; see below |
| Python | MBPP-sanitized test (257), Qwen2.5-Coder-1.5B, A0/A3 | continuity anchor only |

- **Pooled primary n = 542 + 510 + 500 = 1,552 items.** Effective n is lower, because Rust tasks
  derive from HumanEval/MBPP. The design effect is `TBD` and is estimated on C before E is frozen.
- **LiveCodeBench is demoted to exploratory.** The lite dataset was last modified 2025-06-05,
  before every Qwen3.5 repository was created (2026-02-27), so the post-release subset is expected
  to be empty. Its exact count is `TBD`, because `contest_date` was not read. It therefore cannot
  serve as a contamination-free anchor. Its license tag is the unspecified `cc`, so it is used for
  evaluation only and never redistributed.
- **Manifest.** Before any generation, `gwenlaya_v4/datasets/manifest_eval.json` is written. It
  holds every E item id, the sha256 of the normalized statement and tests, the HF dataset revision
  SHA and the license string. Its hash is recorded in an addendum to this file.
- **Harness-sanity exclusion.** An item is excluded, and counted, if its reference solution fails
  its own checks in our sandbox, or if its tests cannot be split into gate-visible and hidden
  parts.

## 3. Models

| Role | HF repo (revision recorded at S1) | Ollama tag (q4 / q8 / bf16) |
|---|---|---|
| Tier 1 (smallest) | Qwen/Qwen3.5-2B | qwen3.5:2b-q4_K_M / -q8_0 / -bf16 |
| Tier 2 | Qwen/Qwen3.5-4B | qwen3.5:4b-q4_K_M / -q8_0 / -bf16 |
| Tier 3 | Qwen/Qwen3.5-9B | qwen3.5:9b-q4_K_M / -q8_0 / -bf16 |
| Tier 4 (largest) | Qwen/Qwen3.8-27B | qwen3.8:27b-q4_K_M (q8_0/bf16 not run: exceeds L4) |
| Continuity anchor | Qwen/Qwen2.5-Coder-1.5B-Instruct | qwen2.5-coder:1.5b-instruct-q4_K_M (and v3's `qwen2.5-coder:1.5b`) |
| Router + calibrator | convaiinnovations/laya (+ fresh LoRA r=16) | n/a (not a generator) |
| Exploratory tier 0 | Qwen/Qwen3.5-0.8B | qwen3.5:0.8b-q8_0 (no q4_K_M tag exists) |

- **Text backbone.** All Qwen3.5/3.8 repositories are tagged `image-text-to-text`. Only the text
  backbone is loaded, and the vision tower is never instantiated.
- **Support matrix.** At S1 a matrix is recorded for every model/quant/engine pair. Any
  unsupported pair is dropped and reported, not patched mid-budget.
- **Excluded models.** Qwen3.8 2.4T-A95B and Flash-Next (license "other"). There is no "Qwen3.8
  small". Qwen3-Coder-30B-A3B is an optional comparator, run only from the reserve and only by
  explicit user decision.
- **Local constraint.** Nothing is pulled into the local Ollama until the user has relocated its
  model store to /mnt/data. GGUF files for CPU work go under /mnt/data/home/xavkal/gwaya-data.

**Deployed variant per tier, chosen on C and frozen before E.** The QLoRA(+DPO) variant of a
tier is deployed if it is non-inferior to the base on C (answered accuracy margin -2 pts);
otherwise the base is deployed. Qwen3.8-27B is deployed fine-tuned only if the S4c probe was GO
and S4d completed. Otherwise it is base q4.

## 4. Arms and baselines

**One-cache design.** Every (model, quant, item, seed) generation is produced once and cached,
with per-call GPU-seconds taken from server timing. The gate, Laya and all routing arms are
deterministic post-processing of this cache, and escalation is replayed offline: the cost of a
routed answer is the sum of the GPU-seconds of every tier it invoked. Every comparison is exactly
paired. GPU-second comparisons use **L4 timings only**.

**Reuse and batching rules (binding for the S6 budget):**
- Each (tier, quant, item, seed) is generated **once**. GL, SH, LR, every B-arm and the 2x2
  ablation replay the cache, and no arm regenerates.
- The engine is the llama.cpp server with `--parallel 8`.
- Per-call GPU-seconds are the batch wall time split by each call's share of the batch's tokens.
- **S1 go/no-go.** S1 measures tokens/s at batch 8 per tier and projects the S6 hours. If the
  projection exceeds 0.8 x the S6 max hours, S6 is reduced in this order, and each reduction is
  recorded in the freeze addendum before E:
  1. B4 goes from k=5 to k=3 on a seeded subsample.
  2. The 27B tier runs on a seeded subsample. B2 becomes 9B, the GL tiers become 2B/4B/9B, and the
     27B results are exploratory.
  3. The q8_0 cell is dropped.
  4. GSM8K, AIME25 and LiveCodeBench are dropped.

  Until S1 has run, S6 `est_hours` is `TBD`. The cap/rate figure is only an upper bound.

| Arm | Description |
|---|---|
| **GL** | GwenLaya: full system, as in section 1 |
| B1 | always-smallest (Tier 1 q4), answers everything |
| B2 | always-largest (Tier 4 q4; Tier 3 if 27B is unsupported or OOM, reported) answers everything |
| B3 | B2 + length-normalized sequence logprob threshold (selective prediction), temperature-scaled on C |
| B4 | B2 + self-consistency confidence, k=5 (math: answer-majority share; code: output agreement) |
| B5 | v3-style gate only (VERIFIED or ABSTAIN), with the largest tier |
| B6 | Laya zero-shot (no LoRA) as router and calibrator |
| B7 | oracle router (cheapest correct tier), a ceiling only, never a comparator |
| LR | logistic regression on gate features (gate level, tests passed fraction, error class, repair rounds, mean/min logprob, agreement) without Laya |
| SH | Laya trained on shuffled labels, a negative control (expected AUROC about 0.5) |

**Ablation.** Laya {off = B3 score, on} x gate {off, on}, crossed with quant {bf16, q8_0, q4_K_M}.
The quant factor runs on **Qwen3.5-9B only**: 27B bf16 is about 55.6 GB of weights (an estimate
from 27.8B x 2 bytes) and does not fit an L4. All three quants are served by **the same engine**
(llama.cpp GGUF) so the serving stack is not a confound. If 9B bf16 does not fit at S1, the quant
factor moves to Qwen3.5-4B and this is reported.

## 5. Training (gate-verified traces)

**Sampling and verification.** The GPU only samples and the CPU only verifies.
- Rejection sampling: k=4 samples (code, math) or k=8 (Lean), temperature 0.8, top_p 0.95,
  fixed seeds.
- Teacher: Qwen3.5-9B. On-policy samples also come from each student.
- **S3 sampler, decided at S1.** At about 19.4 GB of bf16 weights (arithmetic), the 9B model
  leaves little KV headroom on a 24 GB L4. The options, in order:
  1. 9B bf16 on vLLM, if S1 shows the architecture is supported with batch >= 8.
  2. Otherwise, 9B with 8-bit/fp8 weights. This is stated as a change to the sampling
     distribution.
  3. Otherwise, a 4B bf16 teacher only.
- The prompt counts per domain are fixed from the S1 tokens/s and the S3 cap, and recorded
  before S3 starts.
- Keep rule: the candidate passes **all** available tests (not just the public one), the AST stub
  audit and grounding.
- Deduplicate by normalized hash and keep at most 2 accepted traces per prompt.
- A task is kept only if its tests reject a trivial baseline.

**Planning caps on accepted traces.** These are upper bounds, not targets that are promised.

| Domain | Cap | Sources |
|---|---|---|
| Python | ≤ 6,000 | TACO-verified, LeetCodeDataset, CodeRM-UnitTest, APPS intro/interview, laya-coding-curriculum-78k |
| Rust | ≤ 3,000 | (a) stdin/stdout tasks reuse their I/O cases via a new `RustCompilerOracle.verify_io`; (b) function tasks via a deterministic assert converter for supported signatures only |
| Lean 4 | ≤ 2,000 | Lean-Workbook statements that elaborate with `sorry` under a pinned Mathlib lake project on /mnt/data; 1 round (round 2 dropped in revision 2: no budget) |
| Math | ≤ 6,000 | GSM8K train, MATH train, NuminaMath-1.5 (closed-form answers only); new `gwaya/math_oracle.py` (fail-closed: unparseable output is UNVERIFIED) |

**Lean rule (pre-registered).** If fewer than 300 proofs are accepted in total, Lean is reported
as eval-only and no Lean SFT claim is made.

**Lean toolchain (C2, before S3).**
- Neither the minif2f-lean4 dataset nor `lean_oracle_v5.tar.gz` pins a Lean or Mathlib version
  (section 0).
- C2 picks a Mathlib commit under which the dataset headers elaborate, and records the
  `lean-toolchain` and Mathlib commit.
- It fetches the oleans with `lake exe cache get` (no source build).
- It elaborates all 488 statements with `sorry` and counts failures.
- It times one kernel check + `#print axioms` (`TBD`) and its peak RSS (`TBD`). The Lean hours for
  C3/C7 and the worker count (1 or 2 on the 31 GB local machine) are derived from that.
- Lean verification runs on the local CPU only. S3, S5 and S6 need C2 because their Lean prompts
  use its headers.

**DPO.**
- At most one pair per prompt.
- Chosen = an accepted trace. Rejected = preferably a *confident-wrong* candidate (passes the
  gate-visible check but fails the full check).
- Compile and stub failures make up at most 25 % of pairs.
- DPO targets correctness only. Abstention is Laya's job.

**QLoRA settings.**
- NF4 with double quantization.
- LoRA on all linear projections of the text backbone. Module names are read from
  `named_modules()` at S1.
- Loss on completion tokens only, using the official chat template.
- Cosine schedule with 3 % warmup, paged_adamw_8bit, gradient checkpointing.
- A COMMITTED checkpoint goes to GCS every 200 steps or 15 min.

| Model | Precision | r | lr | Epochs | Seq | Status |
|---|---|---|---|---|---|---|
| Qwen3.5-4B SFT | bf16 compute on L4 | 16 | 2e-4 | 2 | 2048 | core |
| Qwen3.5-4B DPO | bf16 | 16 (new LoRA) | 5e-6, beta 0.1 | 1 | 2048 | core |
| Qwen3.5-9B SFT | bf16 | 16 | 1e-4 | 2 | 2048 | core |
| Qwen3.8-27B SFT | bf16 compute, 4-bit base | 8 | 5e-5 | 1 | 1024 | conditional on S4c GO (peak VRAM ≤ 22 GiB in a 100-step probe) |

- `lora_configs/*` currently use Ollama tags as `base_model`. They are replaced by HF ids.
- **Laya LoRA (r=16).** It has a pre-generation head (per-tier p_t) and a post-generation head
  (p(correct)), trained on the router-train split described in section 6.
- A 50-step CPU timing run decides where the full Laya run happens: on CPU if the projected time
  is ≤ 12 h, otherwise inside S5 on L4.
- **Quantization of tuned weights.** merge_and_unload in bf16 on the local CPU (≤ 9B only; 31 GB
  RAM), then llama.cpp `convert_hf_to_gguf` and `llama-quantize` to q8_0 and q4_K_M, with an
  imatrix computed on train-split text only.
- **S4c gate (all of these must hold for GO):**
  - the training stack loads the 27B text backbone in 4-bit, with the transformers, peft and
    bitsandbytes versions recorded;
  - peak VRAM is ≤ 22 GiB over 100 steps;
  - the measured tokens/s extrapolates S4d to within its max hours.

  On NO-GO or OOM, the 9B result is used and the S4d cap returns to the reserve.
- **Serving a tuned 27B.** The adapter stays **unmerged**. It is converted to a GGUF LoRA and
  served with llama.cpp `--lora` over the same q4 base GGUF, so base and tuned share one quantizer.
  There is no 27B merge: a bf16 merge needs about 56 GB of RAM against 31 GB locally. If S1 shows
  that llama.cpp cannot load the adapter, the 27B tuned arm is dropped from S6 and this is
  reported.

## 6. Splits and calibration protocol

Three disjoint problem pools:

- **T (train).** Generator SFT/DPO and Laya LoRA. The router-train slice is sha256(source||id)
  mod 20 in {1, 2} of the training pools. The SFT/DPO pool is the remainder.
- **C (calibration/dev).**
  - Sources:
    - MBPP-sanitized train + validation + prompt (120 + 43 + 7 = 170 items, **minus any task_id
      also in MBPP+**; MBPP+ draws from across MBPP, and the overlap count is `TBD`).
    - minif2f-lean4 validation (244).
    - The sha256 mod 20 == 0 slice of the MATH train split and of the verified Rust translations.
  - Used to fit:
    - the temperature/isotonic map on the Laya score;
    - tau_route, tau_hi, tau_lo;
    - B3's threshold;
    - the deployed-variant choice.
  - **tau_hi** is chosen by split-conformal risk control so that the selective error is ≤ alpha =
    0.10.
  - **tau_lo** is the largest threshold whose LIKELY_WRONG precision is ≥ 0.90 on C.
- **E (eval).** Section 2.
- **Freeze rule.** The calibrator parameters, all thresholds and the E manifest hash go into a
  signed addendum **before E is generated**. Stage C6 writes it, and S6 depends on C6. E-based
  refitting is never allowed.
- **Secondary robustness.** 5-fold cross-fitting on E, grouped by source problem, with nested
  threshold selection. Labelled secondary.
- **Quantization transfer.** The calibrator is fit on the deployed quant (q4). Also reported:
  - (T) fit on bf16, applied to q4;
  - (R) 1-2 parameter temperature/isotonic refit on q4 C;
  - (F) full Laya retrain on q4 labels, only if budget allows.

## 7. Hypotheses (fixed before any run)

Every hypothesis states its primary metric, its decision rule and a minimal effect of interest
(MEI).
- **CIs** are 95 % cluster-stratified paired bootstrap intervals (section 8). CWR differences are
  GL minus comparator, so negative values are better.
- **The MEI is the effect the study is powered for.** "PASS" means the decision rule holds.
  "PASS (meaningful)" additionally requires the CI to exclude the MEI side named in the rule.

**Revision 2 relabels H2, H7, H10, H11 and H12 as exploratory.** The red-team priors that they
would be supported were 0.08, 0.10, 0.05, 0.12 and 0.12. Their IDs and metrics are kept, and each
one is reported as an estimate with a CI. None has a decision rule or enters Holm. The reasons
are given with each hypothesis below.

### Primary family (Holm, m = 2, FWER 0.05; pooled Python + Rust + Math, stratified by domain): H1, H3

- **H1: confident-wrong vs logprob selective prediction.**
  - Metric: ΔCWR = CWR(GL) − CWR(B3), at matched coverage. B3's threshold is set on C to hit GL's
    realized E coverage.
  - Decision: PASS if the Holm-adjusted one-sided exact McNemar p < 0.05 on the per-item
    confident-wrong indicators AND the CI upper bound < 0.
  - Matching: the coverage difference CI must lie within ±3 pts, otherwise the result is reported
    as "matching failed".
  - MEI: 3.0 pts absolute. HARM if the CI lower bound > 0.
- **H3: cost-aware routing.**
  - Metric: GPU-seconds per correct answered item, GL vs B2+gate (B5).
  - Decision: PASS if all three hold:
    - the CI of the ratio is < 1 (one-sided bootstrap, Holm-adjusted);
    - answered accuracy is non-inferior, with the CI lower bound of the difference > −2.0 pts;
    - the escalation rate is reported with its CI.
  - MEI: 25 % reduction in GPU-seconds per correct answered item.

### Secondary family A: calibration (Holm within the family)

- **H4.** The Laya post-generation head on E has ECE (15 equal-mass bins) ≤ 0.05, AND the ECE of
  GL is lower than that of temperature-scaled B3 (CI of the difference < 0). MEI: 0.02 ECE.
- **H5.** AUROC(Laya post-head) − AUROC(LR) ≥ 0 with CI excluding 0. MEI: +0.02. Negative
  control: the SH arm's AUROC CI must include 0.5, otherwise the pipeline is flagged as leaking.
- **H6.** Conformal guarantee: per domain, the selective error at tau_hi on E is ≤ 0.10. FAIL if
  the CI lower bound > 0.10.

### Secondary family B: quantization (Qwen3.5-9B, same engine)

- **H8.** q4_K_M vs bf16, by TOST with 90 % CIs. Equivalent if the answered-accuracy difference
  lies within ±2.0 pts, the ECE difference within ±0.02, and the AUROC difference within ±0.03.
  - If the TOST is not within the margins, write "q4 not shown equivalent".
  - If the CI excludes a margin, write "q4 degrades accuracy/calibration".
  - q8_0 vs bf16 is reported the same way.
  - Calibration transfer (T) vs (R) is descriptive.

### Secondary family C: training (H9 only after revision 2)

- **H9 (Python SFT).** Qwen3.5-4B SFT − base, pass@1 on MBPP+ ∪ HumanEval+ (542 items, greedy).
  PASS if the CI excludes 0. MEI +3.0 pts.

### Secondary family D: systems (engineering claims)

- **H13 (budget).** Ledger total ≤ $40.00 planned caps and ≤ $50.00 hard. Refuted if either is
  exceeded, or if any stage exceeds 1.25 x its cap twice.
- **H14 (endpoint).**
  - Cloud Run L4 cold start to first token for Qwen3.5-9B q4: median ≤ 60 s over 4–6 cold
    samples.
  - Warm time to first token at 512 input tokens: median ≤ 2 s.
  - These are design targets, not evidence-derived. With n ≥ 4, only the median and range are
    reported.
- **H15 (preemption).** Resume loses ≤ 15 min of work per event. Verified offline by a SIGKILL
  test and by checkpoint timestamps if a real preemption occurs.

### Exploratory (unadjusted, labelled as such)

Relabelled hypotheses. Each is reported as an estimate with a 95 % cluster bootstrap CI. None has
a PASS/FAIL rule.
- **H2 (Laya over the gate).** CWR(GL) − CWR(B5).
  - Why it is exploratory: VERIFIED takes precedence over Laya (section 1), so Laya cannot
    withhold VERIFIED-but-wrong answers. That was all of v3's confident-wrong mass (8.95 %,
    derived above). Routing to smaller tiers also lowers VERIFIED precision (the v3 n=20 trend,
    motivation only). In addition, "at B5's coverage" cannot be matched without using E.
  - Commitment kept: if the CI includes 0, publish "Laya adds no confident-wrong reduction beyond
    the executed gate".
- **H7 (VERIFIED precision).** VERIFIED precision against the hidden checks, per non-Lean domain.
  - It is reported with its CI and, for Rust, with the hidden-assert count. A point estimate
    below 0.95 is flagged as a gate weakness.
  - Why it is exploratory: v3 measured 88.94 % (A3) and 86.36 % (A4), and those were against
    weaker hidden checks.
- **H10 (Rust SFT).** Qwen3.5-4B SFT − base on MultiPL-E Rust (510), with the Python pass@1
  change reported alongside.
  - Why it is exploratory: by the reviewer's estimate the paired SE at n=542 is about 1.4–1.7
    pts, so a −1.0 pt non-regression margin is unattainable.
  - The Rust training traces are stdin/stdout programs, while MultiPL-E is function-style.
- **H11 (Lean SFT, conditional on ≥ 300 accepted proofs).** miniF2F test pass@8, SFT − base.
  - Why it is exploratory: it is underpowered at n=244, and round-2 expert iteration has no budget.
    Round 2 is dropped.
- **H12 (DPO).** CWR(SFT+DPO) − CWR(SFT) at the C threshold giving coverage 0.80 on the Laya
  post-head score.
  - Why it is exploratory: confident-wrong rejected samples are scarce, the update is small, and
    the calibrator was fit on pre-DPO outputs.

Other exploratory analyses:
- AIME25.
- The LiveCodeBench full-set result.
- GSM8K.
- Lean routing.
- Per-domain H1/H2.
- The continuity A3−A0 on MBPP-sanitized n=257 (Qwen2.5-Coder-1.5B) in the v4 harness.
- Qwen3.5-0.8B as tier 0.

## 8. Statistics

- **CIs.** Paired percentile bootstrap, 10,000 resamples, seed 0.
  - The resampling unit is the source-problem cluster: a HumanEval/MBPP Python task and its
    MultiPL-E Rust translation share a cluster.
  - Resampling is stratified by domain, and all arms are resampled jointly.
- **Tests.**
  - Exact McNemar on per-item indicators (H1; H2 and H12 as exploratory estimates).
  - Bootstrap for cost, calibration and the TOST.
  - Holm within each family.
- **Implementation.**
  - The existing `paired_stats` in `scripts/run_benchmark.py` (percentile bootstrap 10,000,
    Random(0), exact McNemar) is extended, not replaced.
  - New module `gwaya/selective_metrics.py` (pure numpy) with: ece, adaptive ECE, debiased ECE,
    brier (with decomposition), auroc, aurc and e-aurc, risk_at_coverage at {0.5, 0.7, 0.8,
    0.9}, coverage at risk ≤ 5 % / 10 %, cwr, cluster_stratified_bootstrap, holm, mcnemar_exact,
    tost.
  - Offline toy-case tests go in `tests/test_selective_metrics.py`.
- **Seeds.** Seed variance is not covered by the problem bootstrap. A second generation seed on
  the Python sets runs only if budget allows. Otherwise this is a stated limitation.

## 9. Power (computed this session, under ASSUMED discordance rates)

Method: exact one-sided McNemar at alpha = 0.05/3 (computed for the original m = 3; with m = 2 the
Holm level is 0.05/2, so these values are conservative and were not recomputed), computed by enumerating the binomial
distribution of discordant pairs. The script is in the session scratchpad (`power.py`). Clustering
is ignored, so these are upper bounds.

| Effect, in pts (P(B wrong-answered, GL not) / P(GL wrong-answered, B not)) | n=244 | n=378 | n=500 | n=750 | n=1000 | n=1552 |
|---|---|---|---|---|---|---|
| 6 (8 % / 2 %) | 0.77 | 0.94 | 0.99 | 1.00 | 1.00 | 1.00 |
| 4 (5 % / 1 %) | 0.59 | 0.85 | 0.94 | 0.99 | 1.00 | 1.00 |
| 3 (4 % / 1 %) | 0.37 | 0.63 | 0.79 | 0.94 | 0.99 | 1.00 |
| 2 (3 % / 1 %) | 0.17 | 0.33 | 0.47 | 0.69 | 0.83 | 0.97 |

- These power values are higher than the ones in the evaluation-design proposal (for example
  0.49 vs 0.59 for a 4-pt effect at n=244). The method there was not documented. The value
  that counts is the one re-computed on C with real discordance rates, before E is frozen.
- **Consequences:**
  - Per-domain confirmatory claims at n ≤ 500 are underpowered for effects ≤ 3 pts, which is
    why the primaries are pooled.
  - H11 (Lean, n=244) is expected to be underpowered, which is one reason it is exploratory.
  - If the C pilot gives power < 0.80 for H1's MEI at the effective n, this is stated in the
    freeze addendum. The MEI is **not** changed.
- **Budget truncation rule.**
  - E items are processed in a seeded random order (seed `eval_order` in plan.json).
  - If the cap stops a run, the truncated n is analysed as is. There is no optional stopping.
  - Cut order to protect the primaries: AIME25, LiveCodeBench, GSM8K, the q8 quant cell, the
    second seed, Lean E.

## 10. Exclusions and contamination control

**License exclusions:**
- KodCode (CC-BY-NC).
- Magpie-Qwen2.5-Coder-Pro.
- Vezora preference pairs.
- Goedel-LM/Goedel-Pset-v1 (no license).
- PrimeIntellect sets (license other or missing).
- `callensxavier/gwaya_v2_verifier_curriculum`: the HF API shows no license tag. Excluded until
  the author adds one.
- Qwen3.8 2.4T-A95B and Flash-Next.

**Licenses confirmed by HF API tag this session:**

| Dataset | License |
|---|---|
| proofnet | MIT |
| NuminaMath-LEAN | Apache-2.0 |
| aime25 | Apache-2.0 |
| MATH-500 | none (derived from MIT MATH; used for eval only, stated) |
| LiveCodeBench lite | `cc` (variant unspecified; eval only, not redistributed) |

**Artifact exclusion.** The data-lake `autoevolve_anse_datalake/models/laya_lora` adapter is
not used or cited as a calibrator.

**Decontamination.** New module `gwaya/data/decontam.py`, CPU only, with offline tests on planted
overlaps. It compares every train/C item against every E item, across domains:
- exact match on normalized hashes;
- any shared word 13-gram;
- MinHash Jaccard ≥ 0.8 on the normalized prompt+tests (code with AST-normalized identifiers);
- for Lean, an alpha-renamed statement hash.

The thresholds are conventional and not validated on this data.

**Mandatory cross-domain rules:**
- No HumanEval/MBPP test tasks, or their Rust translations, in T or C.
- No MBPP+ task_ids in C.
- No NuminaMath items whose source is MATH-test, AIME or AMC 2025.
- No MATH test split in T.
- Lean-Workbook is deduplicated against miniF2F and ProofNet.
- APPS, TACO, CodeContests and LeetCode are deduplicated against each other.

**Publication.** Removal counts per (training source, E set) are published in
`datasets/decontam_report.json`. Their hashes are frozen in the addendum.

**Residual risk, stated.** Qwen3.5 pretraining has likely seen HumanEval, MBPP and MATH. Absolute
scores on those sets are not clean. The claims are paired, within-model deltas.

## 11. Budget ($50.00 hard total; one GPU at a time; spot T4/L4 only; endpoint included)

`Max hours` is **cap / planning rate**, an upper bound enforced by `--max-run-duration`. It is
not a forecast. Every GPU-stage `est_hours` is `TBD` until S1 measures throughput.

| Stage | Where | What | Cap (USD) | Max hours |
|---|---|---|---|---|
| C0–C7 | cpu | preflight/manifest, decontam, oracles + tests, Mathlib pin, verification, Laya pilot, merge+quantize, calibrate+freeze, analysis | 0.00 | n/a |
| S1 | l4 | support matrix, batch-1/8 throughput, S6 go/no-go, S3 sampler choice, VRAM pilots | 1.50 | 2.7 |
| S2 | l4 | base tiers × {C, router-train}, continuity anchor | 3.50 | 6.4 |
| S3 | l4 | rejection sampling for SFT/DPO (sampler and prompt counts fixed at S1) | 6.00 | 10.9 |
| S4a | l4 | Qwen3.5-4B QLoRA SFT + DPO | 4.00 | 7.3 |
| S4b | l4 | Qwen3.5-9B QLoRA SFT | 5.50 | 10.0 |
| S4c | l4 | Qwen3.8-27B QLoRA go/no-go probe (100 steps) | 1.00 | 1.8 |
| S4d | l4 | Qwen3.8-27B QLoRA (only if S4c GO; otherwise the money returns to the reserve) | 3.00 | 5.5 |
| S5 | l4 | tuned tiers × {C, router-train}; Laya LoRA if CPU is infeasible | 2.50 | 4.5 |
| S6 | l4 | E eval matrix (one cache, arms replayed; 9B quant factor; continuity) | 6.00 | 10.9 |
| S7 | cloudrun-l4 | deploy, idle-to-zero once, 4–6 cold samples, warm session, teardown | 2.50 | ≤ 2.4 instance-h |
| misc: disk | n/a | pd-balanced boot/data disk, deleted with each VM | 2.00 | n/a |
| misc: GCS | n/a | storage, operations, egress | 1.00 | n/a |
| misc: build | n/a | Cloud Build + Artifact Registry for the S7 image | 1.50 | n/a |
| **Planned total** | | | **40.00** | |
| **Reserve** (never pre-allocated; drawn only by a logged user decision) | | | **10.00** | |

The sequential L4 upper bound is 60.0 VM-hours on the single GPU slot.

Disk line, as arithmetic on the section 0 price: 200 GiB × $0.11/GiB-month ≈ $0.030/h. $2.00 pays
for about 66 disk-hours, which covers 60 VM-hours only if each disk is deleted with its VM. The
GCS storage, Cloud Build and Artifact Registry prices were **not read** this session (`TBD`). Those
lines are caps checked against the ledger, not estimates.

**Spend controls.**
- **GPU preflight (C0, before every VM create).**
  - Read-only checks assert that `GPUS_ALL_REGIONS` usage == 0 and that no accelerator VM is
    RUNNING.
  - At 19:11 UTC on 2026-10-07 usage was 1/1, held by `socreateai-agora-hermes-node1`. That VM is
    not GwenLaya's. **The user stops it or schedules around it. It is never stopped
    autonomously.** No stage can start until then.
- **Between stages.** Every `purpose=gwenlaya` VM **and its disk** is deleted at the end of a
  stage. Weights and caches live in GCS. A TERMINATED VM frees the GPU quota, but its disk keeps
  billing.
- **Stage caps cover the whole VM lifecycle:** boot, downloads, re-downloads after a preemption,
  and idle time.
  - `--max-run-duration` = remaining stage cap / planning rate.
  - At the cap the stage stops and is reported **PARTIAL**. It is never extended silently.
  - Accrued cost is checked against the cap at every stage boundary and by an in-stage watchdog
    every 30 min.
  - Generation caches and checkpoints are committed to GCS every ≤ 15 min, and runs resume from
    the last COMMITTED object.
- **Ledger.** `gs://…/gwenlaya_v4/runs/LEDGER.jsonl` gets one line per VM session, plus disk, GCS
  and build lines. A stage is refused if ledger + stage cap > $40.00.
- **Launch.** Every launch goes through `deploy/run_on_gcp.sh` with its double spend
  confirmation, `--max-run-duration`, termination action DELETE and the self-delete trap.
- **Catalog price.** The price is rechecked before each stage.
- **Billing budget alert.** Creating one is a new cloud resource and requires the user's approval.
  Until then the ledger is the only guard, and the paper says so.
- **Every launch** of a spend-incurring command needs explicit user authorization of the exact
  command.

**Stage order and dependencies** (`depends_on` in plan.json; a test checks that it is acyclic):
- C0, C1, S1 → S2
- S1, C2 → S3
- S2, S3, C2 → C3
- C3, S1 → S4a, S4b
- S4b → S4c → S4d
- S4a, S4b, C3 → C4
- S4a, S4b, C4, C3 → S5
- S5, C2 → C6 (calibrate, then freeze the addendum)
- S5, C1, C2, C6 → S6
- S6, C2 → C7 (verify E)
- S6 → S7
- C7 → C5

The C0 addendum hash must exist before S2 runs.

**Cut order if over budget:** S4d, S4c, the 4B DPO, B4 k=5, the full-E 27B tier, the q8 cell, the
second seed, and S7 cold samples 5–6. The primaries (H1, H3) need only S2, S3, S4a/b, S5 and S6.

**Endpoint (S7).** These are commands to approve, not commands that were run.
- **Quota.**
  - Read-only check: Cloud Run `NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion` is 3 in
    us-central1.
  - `deepseek-prover-v2` already has 1 L4 with maxScale 1, so the headroom is 2.
  - The check is repeated before approval, and the first deploy must report the quota.
  - If the quota is absent, **S7 is dropped and the user is told**. Quota is never requested.
- **Topology.** Two Cloud Run services in **us-central1**:
  - `gwenlaya-gen`: L4, `--gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy --cpu 4
    --memory 16Gi --min-instances 0 --max-instances 1 --no-allow-unauthenticated`, llama.cpp
    server with the 9B q4 GGUF **baked into the image** in us-central1 Artifact Registry. This
    avoids cross-region reads from the US-EAST4 bucket, and the build cost is on the misc: build
    line.
  - `gwenlaya-gate`: CPU-only, Laya + gate.
- **Fail-closed gate.** If bwrap, rustc or Lean do not work inside Cloud Run, the gate service
  returns UNVERIFIED/ABSTAIN. All VERIFIED numbers in the paper come from the VM-based S6 run.
- The existing service `anse-serverless-laya` is never modified.
- **Idle billing (re-verified in the docs, section 0).**
  - GPU instances may stay idle for up to 10 minutes, and the GPU is billed for the whole
    instance lifecycle.
  - Arithmetic: each cold sample therefore costs up to 10 min × $1.0465/h ≈ $0.17 of idle, plus
    active time (`TBD`).
  - Idle-to-zero is measured once. Then **4 to 6 cold samples** are taken, each after scale-to-zero
    has been confirmed. Sampling stops early if the next sample would exceed the cap.
  - The warm measurements (30 per length bucket) run in one session.
- **Teardown.** Traffic is set to 0, or the service is deleted, at the end of S7, after the user
  approves.

## 12. Release checks

- **R1 Cost.** The ledger total is ≤ $50 and is compared with the billing-console total, which the
  user provides.
- **R2 Cleanup proof.** No `purpose=gwenlaya` VMs or disks remain, and all writes are under
  `gwenlaya_v4/`.
- **R3 Reproducibility.** Every results file carries:
  - a non-empty git commit (the v3 n=257 file's was empty);
  - the dataset and model revision SHAs;
  - tool versions and seeds;
  - the rows sha256;
  - hardware as read from the runtime;
  - the quant format.
- **R4 Integrity.** A script checks that every number in the paper or README tables traces to a
  results JSON path, or is `TBD`.
- **R5 Security.**
  - An unauthenticated request to the endpoint returns 401/403.
  - The gate is fail-closed when a toolchain is missing.
  - The sandbox security tests pass.
- **R6 Offline tests.** The pytest suite passes with no GPU, network or model downloads. The
  Playwright and gpu_manager tests are not run.

## 13. Conflicts between the designs and how they were resolved

1. **Math VERIFIED.**
   - Data design: VERIFIED may use the reference answer.
   - Eval design: the deployed gate must not use gold answers.
   - **Resolved for the eval design.** Gold answers are used for scoring only. The gate uses an
     executed program-of-thought re-check.
2. **Primary endpoint.**
   - The systems design's H-C4 targeted the v3 A3 CWR of 8.95 % (MBPP only).
   - **Resolved:** the matched-coverage, CI-based H1/H3. The 8.95 % is context only, because it
     comes from a different model and gate.
3. **Lean in the primaries.** Excluded, because the gate equals the ground truth. Adopted from the
   eval design.
4. **LiveCodeBench as the "cleanest anchor".**
   - Both the eval and the data designs proposed it as the cleanest anchor.
   - **Overruled by evidence read this session.** The lite dataset predates every Qwen3.5 release,
     so LiveCodeBench is exploratory only.
5. **Where Laya trains.**
   - Data design: on CPU. Systems design: CPU pilot only.
   - **Resolved:** a 50-step CPU timing run decides, with a threshold of 12 h, and S5 holds the L4
     fallback.
6. **T4 usage.**
   - The data design proposed fp16 QLoRA on T4, which has no bf16. The systems design proposed
     emulating a T4 on an L4.
   - **Resolved:** all training and every cost-metric generation run on L4, so GPU-seconds are
     comparable.
   - T4 (planning rate $0.45/h) is a **stock-out fallback** only, for ≤ 4B work after 3 failed L4
     zone attempts. Its GPU-seconds are reported separately and never pooled. A "T4 fit" claim is
     made only from a real T4 run.
7. **Quant engine.**
   - The data design served bf16 on vLLM.
   - **Resolved:** the same engine (llama.cpp GGUF) for bf16/q8/q4, to remove the serving-stack
     confound. vLLM is used only for S3 sampling throughput.
8. **Splits.**
   - The eval design used named C sets. The data design used hash slices.
   - **Resolved:** both. Named held-out sets plus the hash slice mod 20 == 0 form C, and
     mod 20 ∈ {1, 2} forms router-train. The MBPP+ id-overlap exclusion is added; it was found
     during the merge and is unverified.
9. **The data design's H1 (SFT +3) and H5 (quantization) as headline claims.**
   - **Demoted to secondary families C and B.** The publication's primary claims are about
     confident-wrong, calibration and cost, as the goal states.
10. **Budget.**
    - The data design planned about 53 GPU-hours at unverified prices.
    - **Resolved:** the systems design's staged caps, re-priced with catalog prices verified this
      session. $40.00 planned, $10.00 reserve.
11. **Power table.** Re-computed this session with a documented method. It disagrees with the eval
    design's numbers (section 9). The C pilot decides.
12. **Red-team blocking items (revision 2).**
    - Resolved by adding: the GPU preflight, the S7 time box, the reuse/batching/no-thinking rules
      with the S1 go/no-go, Lean placement, the S4c gate and the 27B serving path, ledger lines
      and the stage-stop rule, the S3 sampler choice, and the stage dependencies.
    - H2, H7, H10, H11 and H12 are relabelled exploratory.
    - The per-item response is in `docs/PROPOSAL_REVIEW.md` section 6. The review's non-blocking
      items F2–F13 are **not** all resolved by this revision. Section 6 there lists them as open.

## 14. What would falsify us

- **H1 fails** (CI includes 0) → GwenLaya does not reduce confident-wrong answers relative to a
  simple logprob threshold at matched coverage. This is the central claim, and a FAIL is
  published as the main result.
- **H2 (exploratory) CI includes 0** → Laya is unnecessary; the executed gate does the work.
- **H5 fails, or SH shows AUROC ≠ 0.5** → Laya is no better than a logistic regression on gate
  features, or the pipeline leaks labels.
- **H6 fails** → the stated probabilities are not trustworthy, and the LIKELY_CORRECT(p) verdicts
  must not be presented as calibrated.
- **H7 (exploratory): VERIFIED precision point estimate < 0.95** → "VERIFIED" overclaims, as it did in v3 (88.94 %).
- **H8 not equivalent** → q4 deployment costs accuracy or calibration. It is reported, not
  hidden.
- **Other contamination signals** → claims are restated on decontaminated data only, beyond the
  LiveCodeBench issue already disclosed in section 2.
- **Any threshold changed after E is generated** → the affected result is void.

## Addenda

- Freeze addendum (before E is generated): `TBD`.
- Power re-check on C: `TBD`.
- Design effect: `TBD`.
- Support matrix from S1: `TBD`.
