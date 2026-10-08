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
| D17 | Harness-sanity: reference solution passes its own checks | Python and Math as registered (reference solution run through the repo checkers in bwrap; Math: gold must self-verify and equal the last boxed answer of the reference solution). **Rust has no reference solution in MultiPL-E**, so Rust items are only required to (a) split into >= 2 asserts (a gate-visible first assert plus a different hidden remainder) and (b) typecheck with `rustc --emit=metadata` against an `unimplemented!()` body for both the full and the gate tests | Weaker than registered for Rust: a bad Rust test that typechecks is not caught. Single-assert items are excluded because gate and hidden would be identical |
| D18 | Decontaminate T and C against E | Run on the only local candidate calib source (MBPP sanitized train/validation/prompt, 170 items): all 170 are flagged by the id rule (they are MBPP+ or mbpp-rs items) and removed, so there is no MBPP-derived C. No T set exists tonight (SFT/DPO deferred, D8) and calibration uses cross-fitting on E (D4/D5) | Nothing to decontaminate beyond that; the report is `experiments/night/decontam_report.json` |
| D19 | E-night size from measured throughput | n_d = 28/26/26 (sum 80) from the L0 sequential wall tok/s and an **assumed** mean completion length (no generation has run); 0.85 margin; L3 (4B, 4 h) binds | The CPU is 8 logical cores shared with 3 foreign jobs; the paired n will be far below 500 and H1 carries the underpowered flag |
| D20 | Prompt text and gate-visible test definitions (not fixed in the prereg) | HumanEval+ gate = first input/expected pair of the combined plus list (not literally the docstring example); MBPP+ gate = `test_list[0]`, shown in the prompt as "Your code should pass this test"; prompts add a one-line instruction (full function, boxed answer for Math) | Needed to make the items answerable; recorded so the frozen prompt is the one in `tasks_E_night.jsonl` |
| D21 | L2 run_study stage and instrumentation | Stage `night_L2` is defined in `experiments/night/plan_night.json` (plan.json copy plus one stage; plan.json untouched). `run_study.py` OpenAI backend now stores `token_logprobs` (llama.cpp `logprobs.content` format) and per-call `cpu_seconds` (utime+stime delta of the llama-server pid, valid because calls are sequential; includes idle-spinning worker threads, so it is server CPU time, not pure model compute). A first 2-item attempt stored no logprobs (parser bug) and was discarded (`cache/L2_discarded_nologprobs`). The 4B llama-server on 8091 was stopped by pid and replaced by the 2B one; L3 must restart 4B. | Required by the L2 spec; bug found by the first attempt |
| D22 | Bootstrap RNG `random.Random(0)` (`paired_stats`) | `scripts/analyze_study.py` uses `numpy.random.default_rng(0)`, 10,000 cluster-stratified resamples (clusters within domain, all arms jointly); percentile CIs. Point estimates and tests are unaffected; the CI endpoints differ from a `Random(0)` run only by Monte-Carlo noise. Holm is taken over the kept family (hypotheses that ran), so m can be 1 or 2. Analysis inputs are `rows.jsonl`/`gens.jsonl` plus an optional cross-fitted `scores_crossfit.jsonl`; hypotheses without inputs are written `not_run` | Implementation choice; vectorised and cluster-aware |

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

## D23 Analysis verifier fixes (2026-10-08)
- Scored the 80 L2 generations with `run_study.py --mode score` into `night/cache/L2score` (no new generation, no GCP spend). Arm `base` only: 14/80 VERIFIED, 66/80 FAILED or UNVERIFIED (as read from rows.jsonl).
- Only one unregistered arm exists, so no paired comparison is possible: McNemar, paired-bootstrap deltas and Holm for H1/H2/H3/H7 stay not_run. Verifier items on McNemar and delta CIs are therefore not fixable with current data; they need a second arm.
- `analyze_study.py` now emits a descriptive single-arm summary (accuracy, confident-wrong rate, ECE, Brier, AUROC, AURC, cost per correct, cluster bootstrap CIs) using raw confidence exp(mean_logprob). Not cross-fitted, not a pre-registered test; ECE here is for an uncalibrated score. H4-H6 remain not_run (no cross-fitted scores).
- Renamed misleading `client_seconds_total` to `server_reported_seconds_total` (sum of gpu_s) and added `client_wall_seconds_total` (sum of wall_s).
- Confident-wrong rate here counts every UNVERIFIED answer as wrong (answered=true for all rows), so it is an upper bound.

## Post-L2 review fixes (2026-10-08, after L2 generation, before any registered test)

Timing, stated plainly: D1-D20 were written before any generation; D21 was written during L2
(committed with the L2 generations, 64441eb); D22 and D23 were written **after** the L2 generations
existed (d4aecfd, 6e0a559) but before any registered test could run; D24-D26 below are written after
the external review of the interim paper. No registered test has run, so no p-value or verdict was
seen before any of these choices.

| # | Prereg says | What was actually done / is now done | Reason |
|---|---|---|---|
| D24 | Math: the boxed answer is scored by sympy/numeric equivalence with the gold answer (prereg section 1, gate/ground-truth table); unparseable math output is UNVERIFIED, fail-closed (section 5, math oracle row) | **Not registered, found in review:** `gwaya/domains/math_check.py::extract_final_answer` (used by `run_study --mode score`) falls back from the last `\boxed{}` to the last `####` line and then to **the last number in the text**. None of the 26 L2 math generations contains `\boxed` (`numbers_v4.json` `single_arm.math_extraction.boxed.n` = 0), so all 26 were scored by the last-number fallback, including all 3 passes (`single_arm.math_extraction.last_number.n_pass_hidden` = 3; the texts end in Python comments, `print(...)` or `return ...`). `analyze_study.py` now also reports the **registered boxed-only rescoring** (`single_arm.boxed_only.*`: math 0/26 pass). Both versions are reported; the math generations are invalid in either case (prompt-path defect). **Frozen for the planned math regeneration:** the registered rule, boxed-only (last `\boxed{}`; no `\boxed` = UNVERIFIED); the fallback may appear only as a labelled sensitivity analysis. `math_check.py` itself is not changed tonight, so `run_study` scoring must pass a boxed-only flag (or be patched under a new deviation) before the regeneration is scored | Unregistered scoring rule; fail-closed requirement |
| D25 | Gate-visible check = the first public test only; the gate never sees the scoring checks (prereg sections 0-1; freeze addendum section 2) | **Defect found in review:** for HumanEval+ the v1 builder (`python_gate_tests_plus`) only rewrote the loop header to `zip(inputs[:1], results[:1])` and kept the complete `inputs = [...]` / `results = [...]` literals, so every HumanEval+ `gate_payload` contained the whole hidden suite (11/11 rows in `tasks_E_night.jsonl`, 161/161 in `tasks_E_primary.jsonl`; MBPP+, Rust and Math gate payloads were correctly restricted). The builder now rewrites the literals themselves with `ast` to their first element and fails closed; `tests/test_eval_manifest.py` asserts that no hidden element appears in the gate text. `scripts/data/regen_gate_payloads_d25.py` wrote `tasks_E_night.d25.jsonl` (80 rows, 11 gates rewritten, prompts and checker payloads byte-identical as JSON) and `tasks_E_primary.d25.jsonl` (1,536 rows: 156 rewritten, 5 HumanEval+ items excluded fail-closed: HumanEval/14, /83, /100, /130 define a `ref_func` reference implementation that the gate would contain, and HumanEval/149 failed the leak check because a hidden input literal also occurs inside the first expected output; none of the 5 is in E-night). New sha256s are in `docs/FREEZE_ADDENDUM_2026-10-08_D25.md`. **No reported number changes:** the L2 run called no gate (`gate` is null in all 80 rows), so the leaked payload was never executed or read by any recorded component | Leak of hidden checks into the gate payload |
| D26 | Primary family: Holm, m = 2, FWER 0.05 (prereg section 7) | D22's "Holm over the kept family, so m can be 1 or 2" was a decision-rule change made after the L2 generations existed; it is **reverted**: `analyze_study.py` now always uses m = 2, and a primary that did not run enters Holm with p = 1 (so the other primary is tested at 0.025). Also after L2 (analysis presentation, no registered metric changed): the single-arm descriptives are reported per domain with Python as the headline and the pooled numbers as a labelled sensitivity row, and the per-row hidden-check outcome is reported as PASS_HIDDEN / FAIL_HIDDEN / UNDECIDED (rows.jsonl `score` VERIFIED / FAILED / UNVERIFIED), because the registered VERIFIED is the gate verdict, which was not recorded in any row | Restore the registered decision rule; stop pooling invalid and valid domains |

D22 note (appended, not rewritten): the Holm sentence in D22 is superseded by D26. The RNG part of
D22 stands.

D24 note (appended 2026-10-08, v3.3.1, not rewritten): the regenerated `numbers_v4.json`
(from `night/cache/L2fix/rows.jsonl`) now has `single_arm.math_extraction.last_number.n_pass_hidden`
= 0, because the re-scored rows apply the registered boxed-only rule by default (D28); the value 3
above refers to the v3.3.0 rows (`night/cache/L2score`).

## Post-v3.3.0 fixes (2026-10-08, after release v3.3.0, before any registered test)

Written after v3.3.0 was released. No registered test has run. The night generations
(`night/cache/L2score/gens.jsonl`, unchanged) were re-scored into `night/cache/L2fix/rows.jsonl`
(copy: `results/gwenlaya_v4/night_L2fix/rows.jsonl`), and `papers/numbers_v4.json`,
`papers/tables_v4.tex` and `papers/figures_v4/` were regenerated from it.

| # | Prereg says | What was actually done / is now done | Reason |
|---|---|---|---|
| D27 | Rust hidden check: compile the candidate with the MultiPL-E test and run all asserts (prereg section 1); harness sanity on a passing program (D17 weakened this to typecheck only) | **Defect, v3.3.0 Rust result withdrawn.** Inside bwrap, rustc's linker `cc` resolves through `/etc/alternatives`, which the sandbox did not mount, so every Rust program failed to link; the v3.3.0 Rust result (0/26) was a harness artefact. Fixed in PR #4 (merged, 8cb37ad). Also fixed: the pass nonce was written into `main.rs`, so a candidate could read its own source or binary and forge a pass (reproduced, then closed: nonce via stdin, separate compile and run sandboxes, run sandbox without `/proc`, `unsafe` / `link_section` / `asm` rejected); `assert_ne!` now counted with `assert_eq!`. Validation (`results/gwenlaya_v4/rust_harness_validation.json`): 25/25 hand-written reference solutions VERIFIED and 9/9 wrong solutions FAILED, on 25 of the 26 night Rust tasks (`rs/HumanEval_132_is_nested` has no reference in the file). Night Rust re-scored: 6/26 PASS_HIDDEN, 20 FAIL_HIDDEN, 0 UNDECIDED (`numbers_v4.json` `single_arm.by_domain.rust.count.*`) | Harness defect (link failure) and forgeable pass signal |
| D28 | Math: boxed answer scored by equivalence; unparseable = UNVERIFIED; a math prompt asking for `\boxed{}` | Night math generations went through the code path (python system prompt, ```` ```python ```` prefill, ```` ``` ```` stop), so they stay invalid (unchanged conclusion). Code fixed (2c1f6b1): prose math prompt ending in `\boxed{}`, Qwen3.x non-thinking via an empty think block; **boxed-only scoring is now the default** (the registered rule); the last-number fallback (D24) is opt-in only. Night re-score: math 0 PASS_HIDDEN, 0 FAIL_HIDDEN, 26 UNDECIDED. Also fixed: the self-consistency vote key for math used the extraction method instead of the answer; B4 did not run (D9), so no reported number changes | Prompt-path defect; restore the registered scoring rule as default |
| D29 | Python hidden checks run in bwrap | With a uv-managed interpreter the venv `python` symlink chain went through a directory not bound into the sandbox, so Python could not start inside bwrap on the re-scoring machine. Fixed by recreating the interpreter symlinks (2c1f6b1). Night Python unchanged: 11/28 PASS_HIDDEN | Environment defect |
| D30 | Not registered (TPU not in the plan; arms B1-B6/GL only) | **Unregistered descriptive TPU run #3** (one on-demand v5e chip, vllm-tpu 0.31.0, bf16, greedy, max_tokens 1024, non-thinking, fixed prompts): all 80 night tasks for Qwen/Qwen3.5-2B, Qwen/Qwen3.5-4B, Qwen/Qwen2.5-Coder-1.5B-Instruct (`results/gwenlaya_v4/tpu_run3/`). Python scoring on the TPU host was invalid (sandboxed checks launched from inside the vLLM process all failed), so all TPU generations were re-scored locally with the same checkers (`*.rescored.jsonl`, `summary.json`); Rust and math re-scores equal the TPU-host status on every item, Python does not. Math truncations at 1024 tokens count as UNVERIFIED. Reported as descriptive only: single sample, no logprobs, no gate, no calibration, not a registered arm. Qwen3.5 runs through the PyTorch fallback of tpu-inference with `SKIP_JAX_PRECOMPILE=1`; this supersedes the v3.3.0 statement that Qwen3.5-4B did not finish XLA compilation (that was run #1). No TPU LoRA path for dense Qwen3.5/Qwen3.8 (unchanged) | GPU slot unavailable (G); scoring defect on the TPU host (I) |
