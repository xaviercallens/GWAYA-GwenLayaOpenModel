# Red-team review: GwenLaya v4 pre-registration

Reviewed 2026-10-07 against `docs/GWENLAYA_PREREGISTRATION.md` and `experiments/plan.json` as they
stand in the working tree. The reviewer's stance is a skeptical senior reviewer at a top ML venue.
No GwenLaya run exists, so every probability below is a **prior**, not a result. Numbers are tagged
with how they were obtained:

- **[file]**: read this session from a results file in this repo.
- **[HF]**: read this session from the HF datasets-server API, using the script described in section 1.
- **[lit]**: a published result, cited qualitatively. arXiv IDs marked (v) were confirmed this
  session by an alphaXiv search. The others are from memory and must be checked before they are cited.
- **[est]**: my own back-of-envelope arithmetic, not a measurement.

Nothing in this review was measured on a GwenLaya model.

---

## 1. New evidence read this session (it changes the design)

| Fact | Value | Source |
|---|---|---|
| MBPP-sanitized train / validation / prompt task_ids that are also MBPP+ task_ids | 108/120, 39/43, 7/7 → **154 of the 170 planned C items** | [HF] `evalplus/mbppplus` test (378 ids) vs `google-research-datasets/mbpp` sanitized |
| MBPP-sanitized test (continuity anchor, n=257) ids also in MBPP+ | **224 / 257** | [HF] same |
| MultiPL-E `mbpp-rs` ids also in MBPP+ (E) | 316 / 354 | [HF] `nuprl/MultiPL-E` names `mbpp_<id>_*` |
| `mbpp-rs` count of `assert` per task | median 3, min 3; 331/354 tasks have ≤ 3 | [HF] substring count of "assert" in `tests` |
| `humaneval-rs` count of `assert` per task | median 7; 1 task has 1; 20/156 have ≤ 3 | [HF] same |
| MBPP+ row fields | `test_list` (the 3 original asserts) and `test` (plus harness) | [HF] row 0 |
| v3 n=257 A3: coverage / hidden pass given VERIFIED | 80.93 % / 88.94 % | [file] `results/results_n257_l4_gpu.json` |
| v3 n=257 A4 (more candidates): coverage / hidden pass given VERIFIED | 68.48 % / 86.36 % | [file] same |
| v3 multi-model (n=20 each) hidden pass given VERIFIED, 0.5b / 1.5b / 3b / 7b | 52.94 / 84.21 / 89.47 / 100.0 % | [file] `results/gwaya_v3_multi_model/*/summary.json` |
| v3 n=257 file: per-item rows | **not present**; only `rows_sha256` | [file] |

The script that produced the [HF] rows was run from the session scratchpad and was not committed.
The assert counts are substring counts and were not parsed.

Consequences:
- **C is almost empty for Python.** Once the mandated MBPP+ exclusion is applied, the MBPP part of C
  shrinks from 170 to 16 items. The Python thresholds and the Python calibrator would then be fit
  on roughly 16 named items plus whatever the hash slice contributes. That slice comes from
  TACO/LeetCode/APPS-style problems, which have a different format and different test strictness
  from E.
- **The continuity anchor is not independent of E.** 224 of its 257 tasks are E items, so it must
  not be presented as a separate replication.
- **Rust ground truth is weak.** For about 94 % of `mbpp-rs` (331/354), "hidden" means at most 2
  asserts once the first one is given to the gate. A low Rust confident-wrong rate would partly
  reflect a lenient ground truth.

## 2. Design flaws that would make a positive result wrong or uninterpretable

Ranked by severity.

**F1. VERIFIED takes precedence over Laya, so Laya cannot remove the dominant error type (affects
H2, H1, H4).**
- In section 1, step 6, VERIFIED wins whenever the gate-visible checks pass, and Laya's p is only
  reported.
- v3's confident-wrong mass was entirely VERIFIED-but-wrong: derived 8.95 % of problems, from
  [file] 80.93 % × (1 − 0.8894).
- In GL, Laya can only *add* answers (LIKELY_CORRECT) or route. It cannot withhold a VERIFIED
  wrong answer.
- Routing to smaller tiers also lowers VERIFIED precision. In v3 it rose with size, from 52.94 %
  (0.5b) to 100 % (7b), though at n=20 this is motivation only.
- So CWR(GL) ≥ VERIFIED-wrong(routed tiers) ≳ VERIFIED-wrong(largest tier) = CWR(B5) at B5's
  natural coverage.
- "At B5's realized coverage" (H2) is undefined for GL. Matching it requires dropping GL answers
  by some score after E is seen, which conflicts with the freeze rule.
- Fix: let a frozen Laya threshold demote VERIFIED to LIKELY_WRONG when p is very low, or state
  that H2 tests only the LIKELY_CORRECT increment. Define the matching operation before E.

**F2. H1's comparator is a straw man, and the comparison is confounded (affects H1).**
- B3 is the largest tier with a logprob threshold and *no execution*. GL has executed tests.
- It is long established that executing tests filters wrong code. Examples: CodeT (arXiv 2207.10397),
  AlphaCode's filtering, and v3 itself. Token-probability confidence for code is known to be poorly
  calibrated (Spiess et al., arXiv 2402.02047 (v)).
- A PASS on H1 would therefore show "execution beats logprob". It would not show anything about
  Laya or routing.
- GL and B3 also use different generators: GL routes over 4 tiers, B3 uses only the largest. CWR
  at matched coverage therefore mixes generator capability with confidence quality.
- Fix: make the primary comparator B5 + logprob, or B5 + LR (gate features), on the *same routed
  cache*. Report full risk-coverage curves (AURC / E-AURC, already in `selective_metrics`) at
  pre-registered coverage points with thresholds frozen on C.
- Missing baselines: a white-box probe on the generator's hidden states, and P(True)/verbalized
  confidence. Probes are a strong recent baseline for code correctness (arXiv 2512.07404 (v),
  2501.12934 (v)).

**F3. The calibration labels mean different things in C and in E (affects H4, H6, H5).**
- In C, "correct" for MBPP-sanitized means passing the 3 original asserts, and for the hash-slice
  sources it means passing their own tests.
- In E, "correct" means passing the EvalPlus *plus* suites, which are much stricter (EvalPlus,
  arXiv 2305.01210, reports large pass-rate drops under the plus tests).
- A calibrator and a conformal threshold fit on lenient labels will be overconfident on strict
  labels. Split-conformal guarantees need C and E to be exchangeable, which they are not in source,
  format or label strictness.
- Fix: build C from items that have plus-style hidden suites (for example, hold out a cluster-level
  slice *of E's own source distribution*). Or pre-register that H6 is a test of shift robustness,
  not of the conformal guarantee.

**F4. Thinking mode, decoding and token budget are unspecified (affects H1, H3, H8, H9, H14).**
- Qwen3.5/3.8 support thinking and non-thinking chat modes, and the documents do not say which is
  used.
- The mode changes accuracy, the length-normalized logprob (B3), GPU-seconds (H3) by possibly an
  order of magnitude, quantization sensitivity on long chains (H8), and what "time to first token"
  means (H14).
- Temperature/greedy for E, max_new_tokens and the prompt templates are also not fixed. The
  section 1 pipeline does not say whether GL includes v3-style repair rounds, yet LR uses "repair
  rounds" as a feature.

**F5. Clustered data analysed with an unclustered decision test (affects H1, H2, H12).**
- The decision rule uses exact McNemar, which assumes independent items.
- 316/354 `mbpp-rs` items share a source problem with MBPP+ items [HF], and `humaneval-rs` mirrors
  HumanEval+. The pooled 1,552 items therefore have far fewer independent clusters, so McNemar's p
  is anti-conservative.
- Use a clustered McNemar test (Obuchowski 1998; Durkalski et al. 2003) or the cluster bootstrap for
  the decision itself. Holm adjusts the p but not the 95 % CI that is also part of the rule. Use
  Holm-consistent (for example 1 − 0.05/3) intervals.

**F6. The budget cannot cover the E matrix as specified (affects all primaries via truncation).**
- [est] On an L4 (≈300 GB/s memory bandwidth, NVIDIA spec from memory, not re-read), a 27.8B model
  at q4_K_M (≈ 4.8 bits/weight → ≈ 17 GB) decodes at most ≈ 18 tok/s at batch 1.
- [est] One pass over 1,552 items at 1,000 output tokens each would take about 24 h at batch 1.
- S6 has 10.9 L4-hours. In that time it must cover 4 tiers, B4 with k=5 on the largest tier (5×
  the largest-tier cost), the 9B bf16/q8 cells, the SFT/DPO arms, Lean, GSM8K and the continuity
  anchor.
- Batching helps, but the GPU-second accounting for H3 is then ill-defined (see F8). **B4 k=5 is
  not in the cut order.**
- The truncation rule is honest. Even so, primaries cut to a few hundred items lose most of their
  power (prereg section 9: at n=500 power is 0.79 for 3 pts and 0.47 for 2 pts, before clustering).
- Fix: run S1 throughput first, then size E and decide whether B2 is 27B or 9B before the freeze.

**F7. H9 and H10 confound quantization pipelines.**
- Base tiers are Ollama's q4_K_M GGUFs. Tuned tiers are re-quantized locally with an imatrix built
  on train text.
- A tuned-vs-base delta therefore includes a quantizer difference of unknown sign.
- H9 says "greedy" but does not say at which precision it is run. Compare base and tuned through
  the identical convert + imatrix + quantize pipeline, or in bf16.

**F8. The cost metric (H3) leaves out real costs.**
- It omits CPU gate time (rustc compiles, Lean, sandbox), Laya inference, and the latency of
  sequential escalation.
- Per-call GPU-seconds under concurrent batching are not attributable to individual calls.
- Pre-register concurrency = 1 for the cost cells, or report throughput-normalized cost.

**F9. Laya is an unvalidated component, and there is no control for its pretraining.**
- `convaiinnovations/laya` was created 2026-09-18 and is a custom "LayaTypedDecisions" class. Its
  architecture is not verified (prereg section 0, "Not re-verified").
- PEFT may not support the custom class.
- H5 compares Laya against LR, but no baseline separates "Laya's weights" from "any 400M encoder
  fine-tuned identically". Add a plain ModernBERT-large (or same-size encoder) LoRA arm with the
  same inputs.
- The router-train slice (hash mod 20 ∈ {1,2}, i.e. 10 % of T) yields `TBD` labels per tier and
  per domain. If it holds a few thousand mostly-correct labels, the encoder is unlikely to beat LR
  on gate features. LR already sees the tests-passed fraction, which is near-decisive for code.

**F10. Conformal procedure and per-domain claim do not match (affects H6).**
- Selective risk is not monotone in the threshold, so "split-conformal risk control" should be
  Learn-then-Test (Angelopoulos et al., arXiv 2110.01052) with fixed-sequence testing.
- The claim is per domain, so the thresholds must be per domain (Mondrian). A pooled tau_hi gives
  no per-domain guarantee.
- The documents also leave open whether the 0.10 bound applies to LIKELY_CORRECT only or to all
  answered items. If VERIFIED precision is near v3's 88.94 % [file], all-answered error already
  exceeds 0.10.

**F11. Some decision rules are ambiguous or one-sided in the wrong place.**
- H9, H10 and H11 say "PASS if the CI excludes 0" without a direction, so a significant *drop*
  satisfies the text.
- "PASS (meaningful)… exclude the MEI side named in the rule" is not operational.
- H6 PASS means only "not FAIL", so with a wide CI an underpowered study passes by default.
- H13 is largely self-fulfilling, because the ledger refuses stages.

**F12. Contamination caveat is understated.**
- Qwen3.8-27B (Aug 2026) and the Qwen3.5 models (Feb 2026) postdate all of E, and per section 2
  LiveCodeBench lite also predates them.
- Within-model deltas are robust to contamination only if contamination affects both arms equally.
  For selective prediction it does not: memorized items have high logprob and pass tests, which
  inflates both precision and AUROC in ways that differ by score type.
- Report results on a contamination-stratified split, for example items with versus without
  13-gram hits in a public pretraining-corpus sample, or on post-cutoff items. If none are
  available, say so.

**F13. Smaller issues.**
- B3's threshold is targeted at "GL's realized E coverage", which uses E information. This is
  conservative for GL but breaks the stated freeze rule. Pre-register the coverage target instead.
- The "first public test" of HumanEval+ is undefined, because its base test is a single `check`
  function. The split rule must be specified before E.
- Lean "round 2" expert iteration has no separate GPU budget.
- DPO "confident-wrong" rejected samples are rare (count `TBD`), so the 25 % cap on
  compile/stub-failure pairs may bind and shrink the DPO set.
- Results are single-seed (acknowledged).

## 3. Per-hypothesis priors

The prior is P(the hypothesis is *supported* by the pre-registered decision rule when run as
written). This includes the chance that design issues, matching failure or truncation prevent
support. It is not P(the underlying idea is true).

| H | Prior | Main reasoning | Flaws |
|---|---|---|---|
| H1 GL < B3 CWR at matched coverage | **0.45** | Execution of gate-visible tests is a strong filter for Python/Rust (CodeT; v3 coverage 80.93 % at 88.94 % precision [file]). Logprob is poorly calibrated for code (2402.02047 (v)). That pushes the prior up. Pulling it down: about 0.25 chance that coverage matching fails (±3 pts) under C→E shift, the weaker math gate, possible floor effects if the largest tier is very accurate, and F5/F6. | F2, F3, F5, F6, F13 |
| H2 GL < B5 CWR | **0.08** | Structurally blocked by F1: Laya cannot veto VERIFIED, and routing lowers VERIFIED precision (v3 trend [file], n=20). The MEI of 2 pts has power 0.97 only before clustering and truncation (prereg section 9). | F1, F5, F6 |
| H3 GPU-s per correct answer lower, accuracy non-inferior | **0.50** | Cascades routinely save compute (FrugalGPT arXiv 2305.05176; AutoMix). If 2B/4B solve a large share, a ratio < 1 is likely (about 0.85). The −2 pt non-inferiority on answered accuracy is the binding term (about 0.6), because small-tier VERIFIED precision is lower. If 27B does not fit, B2 becomes 9B and the savings shrink. | F4, F6, F8 |
| H4 ECE ≤ 0.05 and below B3 | **0.15** | Calibrators fit on one distribution degrade under shift (Guo et al. 2017; Ovadia et al. 2019). F3 makes Python labels systematically stricter in E. With 15 equal-mass bins and n ≈ 1.5k, estimator noise alone is a sizeable fraction of 0.05. "ECE of GL" is not defined (which score?). | F3, F11 |
| H5 Laya AUROC > LR, CI excludes 0 | **0.25** | LR already sees tests-passed fraction and logprob features. Few router-train labels; a text encoder is unproven on code correctness; generator-internal probes beat external readers in recent work (2512.07404 (v)). The SH control only detects pipeline label leakage, not decontamination failures. | F9, F3 |
| H6 per-domain selective error ≤ 0.10 (not FAIL) | **0.40** | Lenient FAIL rule (CI lower bound > 0.10) helps. Pulling it down: no exchangeability (F3), a pooled tau (F10), weak math gate, and the 0.10 bound may include VERIFIED items whose v3 precision was 88.94 % [file]. | F3, F10, F11 |
| H7 VERIFIED precision ≥ 0.95 in every non-Lean domain | **0.10** | v3 was 88.94 % (n=257) and 86.36 % for A4 [file], against *weaker* hidden checks than MBPP+. EvalPlus plus tests are stricter. Stronger generators help (v3 7b 100 % at n=20, motivation only). Rust may *look* good only because hidden = 2 asserts on most of `mbpp-rs` [HF]. Math PoT-vs-CoT agreement is correlated, not independent. Needs all three domains. | F3, F12 |
| H8 q4_K_M ≡ bf16 (TOST, 3 margins) | **0.30** | Weight-only 4-bit typically recovers most accuracy (Kurtic et al., arXiv 2411.02355 (v)). But [est] paired SE at n ≈ 1.5k with 10 % discordance is about 0.8 pts, so a ±2 pt TOST needs a true |Δ| ≲ 0.7. The ECE ±0.02 margin is near estimator noise, and all three margins must hold. Long reasoning chains amplify quantization divergence. | F4, F6 |
| H9 4B SFT > base on Python | **0.25** | Small LoRA SFT on traces from a slightly larger, same-family teacher, applied to a heavily post-trained instruct model, usually gives small or negative gains (LoRA learns less: arXiv 2405.09673 (v)). Training mixes 4 domains. Mismatch between thinking and non-thinking modes can regress the model. Quantization confound F7. | F4, F7, F11 |
| H10 Rust SFT gain AND Python drop LB > −1 | **0.05** | [est] With n=542 and 10–15 % discordance, the paired SE is about 1.4–1.7 pts. The 95 % lower bound is then about −3 pts even when the true change is zero, so the −1.0 margin is essentially unattainable. Rust traces come mostly from stdin/stdout tasks, while MultiPL-E is function-style. | F7, F11 |
| H11 Lean SFT gain (conditional) | **0.12** | Requires ≥ 300 accepted proofs, then a significant gain at n=244, which the prereg itself calls underpowered. Published prover gains rely on orders of magnitude more sampling (DeepSeek-Prover, arXiv 2405.14333; Lean Workbook, arXiv 2406.03847). Round 2 has no budget. | F6, F13 |
| H12 DPO reduces CWR | **0.12** | Rejected confident-wrong pairs are scarce. The update is small: LoRA, 1 epoch, lr 5e-6. "Coverage matched to 0.80" does not name the confidence score. The calibrator was trained on pre-DPO outputs. | F11, F13 |
| H13 budget ≤ $40 / $50 | **0.75** | The ledger enforces it by refusing stages, so support is close to tautological. The residual risk is a ledger/billing mismatch: boot time, disks, Cloud Run idle billing, egress. Support comes at the price of truncating the science (F6). | F6, F11 |
| H14 cold start ≤ 60 s, warm TTFT ≤ 2 s | **0.50** | Warm prefill of 512 tokens on a 9B q4 model on L4 should be well under 2 s (about 0.85). The cold start includes a cross-region read of a ≈ 6 GB GGUF or a large image pull (about 0.6). n ≥ 4 is descriptive only. | F4 |
| H15 resume loses ≤ 15 min | **0.70** | Checkpoints every ≤ 15 min bound the loss, plus the upload time. A preemption during an upload falls back to the previous COMMITTED checkpoint, giving up to about 30 min. VM re-provisioning time is not counted. The offline SIGKILL test is plausible to pass. | none |

The joint probability that **all three primaries** pass is about 0.03. The leading **publishable
outcome** is H1 PASS with H2 FAIL: "the executed gate does the work; Laya adds nothing measurable
beyond it". The prereg already commits to publishing that (section 7, H2), and it should be framed
as the expected result, not a surprise.

## 4. Required changes before the freeze addendum (ordered)

1. Resolve F1: give Laya a veto over VERIFIED, or redefine H2. Define coverage matching as a
   pre-registered operation.
2. Rebuild C. Measure the MBPP+ overlap (154/170 removed, [HF]) and source a C that matches E's
   label strictness (F3).
3. Fix thinking mode, decoding, max tokens, prompts, repair rounds and concurrency (F4, F8).
4. Run S1 throughput, then size E and the arm list to the S6 cap. Add B4 to the cut order (F6).
5. Replace the H1 comparator with B5 + LR or B5 + logprob on the same cache. Add a hidden-state
   probe arm and a plain-encoder arm (F2, F9).
6. Use clustered tests and Holm-consistent CIs. Give every decision rule a direction. Use
   per-domain LTT thresholds (F5, F10, F11).
7. Use the same quantization pipeline for base and tuned models (F7). Widen or drop the −1.0 pt
   Python margin in H10, with a power check.
8. Define the HumanEval+ visible/hidden split and report the Rust hidden-assert counts with every
   Rust precision number (F13, section 1).

## 5. What was and was not checked

- **Checked:**
  - Both proposal files, read in full.
  - The v3 results files (`results_n257_l4_gpu.json`, `results_n60_cpu.json`, the four
    multi-model `summary.json`).
  - The HF datasets-server rows for `evalplus/mbppplus`, MBPP sanitized (all splits) and MultiPL-E
    `mbpp-rs`/`humaneval-rs`.
  - alphaXiv existence of arXiv 2402.02047, 2411.02355, 2405.09673, 2512.07404 and 2501.12934.
- **Not checked:**
  - Any Qwen3.5/3.8/Laya behavior. No model was run.
  - The contents of the cited papers beyond their abstracts and titles.
  - The arXiv IDs not marked (v): 2207.10397, 2305.01210, 2305.05176, 2110.01052, 2405.14333 and
    2406.03847.
  - L4 bandwidth.
  - The HumanEval+ test format.
  - Lean-Workbook elaboration.
  - Cloud Run cold-start behavior.
- **The priors are subjective.** They come from the reasoning above, not from a bootstrap or a
  posterior.

## 6. Response to the blocking items (revision 2, 2026-10-07)

Every item below was resolved by a change to `docs/GWENLAYA_PREREGISTRATION.md` and
`experiments/plan.json`. None was rebutted. The checks re-run this session were read-only:
`gcloud` describe/list, `gcloud storage cat`, the HF API and Google's Cloud Run docs. Nothing was
created, started or stopped.

| # | Item | Resolution | Evidence re-read this session |
|---|---|---|---|
| 1 | GPU slot occupied | C0 preflight (`plan.preflight`): before every VM create, assert `GPUS_ALL_REGIONS` usage == 0 and no RUNNING accelerator VM. The Hermes VM is the user's to stop or schedule around, never stopped autonomously. Every GwenLaya VM and its disk is deleted at the end of each stage. | 19:11 UTC: `GPUS_ALL_REGIONS` limit 1.0, usage 1.0; `socreateai-agora-hermes-node1` us-east4-b, n1-standard-8, T4, SPOT, RUNNING |
| 2 | S7 infeasible / idle unmodelled | S7 cap 3.50 → 2.50 (≤ 2.4 instance-h). Idle-to-zero is measured once, then 4–6 cold samples are taken, and samples 5–6 are in the cut order. The image is built into us-central1 Artifact Registry, billed to a new misc: build line (1.50). There is a read-only quota check, the first deploy must report the quota, and S7 is dropped if no quota exists. | Docs: "up to 15 minutes, or 10 minutes for GPUs" idle; "GPU is billed for the entire duration of the instance lifecycle". Cloud Run L4 quota 3 in us-central1. `deepseek-prover-v2` holds 1 L4 (maxScale 1), so the headroom is 2. |
| 3 | S6 too small | Reuse rule: each (tier, quant, item, seed) is generated once, and every arm replays the cache. Thinking is disabled, `max_new_tokens` = 1024, and the engine is llama.cpp with `--parallel 8`. The S1 go/no-go projects S6 hours from batch-8 tokens/s. The reduction order is B4 k=3 subsample, then a 27B subsample (B2 → 9B, 27B exploratory), then q8, then secondary sets. GPU `est_hours` = `TBD` and `max_hours` = cap/rate. | Planning arithmetic only; no throughput was measured. |
| 4 | Lean placement | C2 picks and records the Mathlib commit, uses `lake exe cache get`, elaborates all 488 statements, and records a timed single check (`TBD`). Lean workers are ≤ 2. All verification runs on the local CPU over cached candidates (C3, C6, C7), never on the GPU VM. C2 is a dependency of S3, C3, C6, S6 and C7. | The minif2f-lean4 HF tree has only `README.md`, `test.jsonl` and `valid.jsonl` (no toolchain pin). `lean_oracle_v5.tar.gz` has `"packages": []` and no `lean-toolchain`, so no Mathlib and it is not reused. |
| 5 | S4d unproven / unservable | The S4c gate adds three conditions: the stack loads the model (library versions recorded), peak VRAM ≤ 22 GiB, and the measured tokens/s extrapolates within the S4d max hours. NO-GO falls back to 9B. The tuned 27B adapter stays unmerged and is served with llama.cpp `--lora` over the same q4 base. If that fails at S1, the arm is dropped and this is reported. S4d cap 4.00 → 3.00. | Arithmetic only; library support is unchecked (now an S1/S4c task). |
| 6 | Unbudgeted costs | The misc cap is 1.50 → 4.50, as disk 2.00 + GCS 1.00 + build 1.50. A stage cap covers the whole VM lifecycle, including re-downloads after a preemption. `--max-run-duration` = remaining cap / rate. At the cap a stage stops and is reported PARTIAL. A 30-min watchdog checks cost, and the boundary check is kept. Stages sum to 35.50, +4.50 misc = 40.00 planned, +10.00 reserve = 50.00. | Disk arithmetic uses the pd-balanced price recorded in prereg section 0; GCS/Build/AR prices were not read (`TBD`). |
| 7 | S3 under-specified | The S3 sampler is chosen at S1: 9B bf16 vLLM if supported with batch ≥ 8, else 9B 8-bit/fp8 (stated as a change to the distribution), else a 4B teacher. Prompt counts per domain are fixed from S1 tokens/s and recorded before S3. S3 cap 7.00 → 6.00. | none |
| 8 | Missing dependencies | All the required `depends_on` lists are added (S2, C3, S5, S6, S7), plus C4, C5, S4a–d and the new C6 (calibrate + freeze before S6) and C7 (verify E). The plan's stage order is now a valid execution order, and a test enforces it. | `tests/test_gwenlaya_plan.py` |
| 9–13 | H2, H7, H10, H11, H12 | Relabelled **exploratory**. Each is reported as an estimate with a CI and has no decision rule. The primary family is now H1 and H3 (m = 2), and calibration (H4–H6), quantization (H8) and training (H9) remain secondary. Lean round 2 is dropped. | Priors from section 3 of this review |

**Still open (not blocking, not resolved by this revision).** F2 (H1's comparator), F3 (C/E label
shift and the 154/170 MBPP C shrinkage), F5 (clustered McNemar), F7 (base Ollama GGUF vs local
re-quantization for H9), F9 (plain-encoder arm), F10 (LTT / Mondrian thresholds), F11
(directionless rules for H9) and F12 must be resolved before the pre-E freeze addendum (C6). H14's
targets are unchanged.

**Freeze.** The sha256 of both revised files is recorded in `docs/PREREG_FREEZE.txt`.
