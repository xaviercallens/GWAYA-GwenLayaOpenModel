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

D27 note (appended 2026-10-08, revision 3.4.0, not rewritten): the Rust harness counts `assert!`,
`assert_eq!` and `assert_ne!` calls (`gwaya/oracles.py` `_RUST_ASSERT_MACRO`), not only
`assert_eq!`/`assert_ne!` (claim audit B3).

D30 note (appended 2026-10-08, revision 3.4.0, not rewritten): "re-scored ... with the same checkers"
stands, but run #3 is **not** a comparison "across hardware and quantisation" only (claim audit B12):
engine (vLLM `llm.chat` vs llama.cpp raw completions), code system prompt and code prefill/stop also
differed, so it is a loose consistency check. The cause of the TPU-host Python failure is not recorded
(the gate self-test on that host passed; audit B9). Run #2 scores are unused because its own gate
self-test reported `python_ok False` (not because of D29, which concerns the re-scoring machine).
The same-protocol comparison is D38/A7 below.

## Full-E baselines on TPU v5e (2026-10-08, revision 3.4.0; exploratory, no registered test)

Branch `bench/tpu-e-eval`. Analysis plan `docs/ANALYSIS_PLAN_E_TPU.md` (committed 8ee9da3, 22:22:55;
original sha256 1f2fa2c1...0d46), with four dated post-hoc amendments. Results:
`results/gwenlaya_v4/e_tpu/numbers_e_tpu.json` (sha256 c76bc506...226b). None of the rows below changes a
hypothesis's metric or MEI; H1/H3 remain not run.

Timing, stated plainly: D31, D32, D34-D37 were fixed in the plan or the generation scripts. The plan was
committed (22:22:55) after all generation and as scoring started: the 2B/4B scoring run started at
22:21:55, 60 s before the commit, and finished at 22:36:38; the Coder rows were complete at 22:22:45,
10 s before it. The scoring logs print no outcomes and no scored E outcome is known to have been inspected
before the commit, but the descriptive 80-task runs had already shown approximate pass rates.
D33 and D40 arose during generation. D38 and D39 were decided after the first analysis run and are post hoc.
D41 describes how scoring was run.

| # | Prereg says | What was actually done | Reason / when |
|---|---|---|---|
| D31 | Generation with llama.cpp (D6), GGUF q4 on L4 (registered) | One on-demand TPU v5e chip (v5litepod-1, us-west4-a), vLLM-TPU 0.31.0, **bf16**, greedy, `max_new_tokens` 1024, thinking off, raw Qwen protocol prompts and stop sequences from `OllamaGenerator` (the code-domain prompt is unchanged since the L2 commit 64441eb; only the math path changed, D28). HF weights loaded at revision `main` (not pinned) | No GPU slot (G); before generation |
| D32 | Tiers 2B/4B/9B/27B; B2 = largest tier | Tiers **2B and 4B** only ("largest" = 4B); Qwen2.5-Coder-1.5B-Instruct added as a continuity anchor (A1, A2, A9 only) | G, B; before generation |
| D33 | One engine configuration per tier | Configured `max_num_seqs`: 96 for Coder and 2B, **32** for the 4B, because at 96 the 4B failed with `ValueError: Cannot fit both KV pools under gpu_memory_utilization=0.92: the mamba pool needs 193 blocks (2 per request x 96 requests) ...` (`gen_qwen3.5-4b-bf16_s96.log`). Effective concurrency was lower: the engine re-split the KV cache for at most **54** concurrent requests (2B) and **11** (4B at 32). The "Mamba and attention pools together exceed the HBM budget" line is an INFO message that also appears in the successful runs; vLLM calls the recurrent-state pool "mamba" although Qwen3.5 is hybrid full/linear attention. The 4B therefore ran at lower effective concurrency than the 2B, which makes it dearer per token and understates the cascade's cost penalty | HBM limit; during generation (corrected after re-audit E14) |
| D34 | H3 cost in L4 GPU-seconds (D3: CPU-seconds locally) | **Accelerator chip-seconds**: chunk wall time split by token share (registered attribution rule), throughput not latency, not comparable to GPU-seconds. The first chunk of each model (62 s / 144 s / 192 s vs 9-10 / 21-23 / 50-55 s later, consistent with lazy compilation) is included. CPU time of the gate is not counted | Plan |
| D35 | B3 threshold set on split C to match GL coverage | A4 baseline answers the **k most confident tasks** (exp(mean log-prob)), with k = the gate's answered count **in the same domain**; no outcome labels are used to choose k; ties broken pessimistically. k is matched on E rather than taken from a threshold frozen on a separate split. No direction of bias is claimed (the earlier "favours the baseline" was unsupported, re-audit AB11/E45). Pooled number: coverage matched within each domain (math k = 0), -0.063 (-0.074 to -0.054); the earlier global-threshold pooled -0.042 is withdrawn (re-audit E47). Not B3, not H1 | No split C; plan |
| D36 | GL = gate + Laya router/calibrator; H3 = GL vs B5 | A5 "gate cascade" = 2B -> executed gate -> 4B (run_study arm `gwenlaya`, static ladder, no Laya), compared with B1/B2 (not B5), Holm over two exploratory contrasts. Not GL, not H3 | No Laya head; plan |
| D37 | Math gate = sandboxed program-of-thought re-execution that reproduces the boxed answer (prereg section 1 table; needs no task payload) | **Not implemented.** Our gate executes only the task's `gate_payload`, which is empty for all 500 math rows, so every math gate verdict is UNVERIFIED: coverage 0, the cascade always escalates, matched-coverage math comparisons are empty. This is an implementation gap in the checker, not a property of math tasks | Implementation gap; stated in plan (A3); reworded after re-audit E42. **Superseded for the gate by D43 (revision 3.5.0)**; the E-set numbers of 3.4.0 stay as published |
| D38 | (A7 in plan: CPU vs TPU on 80 night tasks) | A7 **restricted to Python/Rust** (54 tasks): CPU night math came from the defective code prompt (D24/D28). Decided after seeing a pooled p driven by math | Post hoc (plan amendment 1) |
| D39 | (A8 in plan: timeout audit) | A8 became a correction: all 1,536 non-passing Python/Rust rows re-checked serially (3 flips); contention flips applied as an **overlay** (`scripts/tpu/serial_overlay.py`, originals untouched; py/MBPP/271 for 2B and 4B); `rs/mbpp_130_max_occurrences` (4B) is nondeterministic (Rust HashMap order) and stays FAILED; all 1,572 passing rows re-run once (0 changes); sensitivity S1 counts every item with two different outcomes as wrong. Overlay importer bug (re-attributed seconds inflated cost) fixed and tested before the final numbers | Post hoc (plan amendments 2 and 4) |
| D40 | Log-probabilities captured (D21) | A first TPU pass (VM gwenlaya-tpu-e-gen-210955; ledger 963 s, 0.321 USD) returned **no log-probabilities** (`mean_logprob` null in all 1,536 Coder and 1,536 2B rows; 4B failed to start). Per the maintainer it used `logprobs=0`. It was set aside and not used; all three models were regenerated with `logprobs=1` (VM gwenlaya-tpu-e-gen2-214445); `gen_batch.py` now fails fast without log-probabilities | Implementation defect; during generation |
| D41 | -- (scoring parallelism not specified) | 2B/4B scored with `--check-workers 5` (parallel prefetch of a memoised check cache); Coder with one check process. Contention timeouts are handled by D39. Both scoring runs ran from working trees with uncommitted changes. `results.json` records HEAD at the **end** of each run (66e2313 Coder, 5e10ad2 tiers, both dirty); the Coder run started 22:05:17 (before 66e2313, 22:08:27) and the 2B/4B run 22:21:55 (before 5e10ad2, 22:25:43), so the code that ran was 39a5a0a and 66e2313 respectively plus unidentified uncommitted changes | Wall clock; scoring |

Also recorded: an orphan TPU VM (gwenlaya-tpu-e-gen2-213823, us-central1-a) was created at 21:38:23 by a
driver the maintainer killed by mistake and deleted at 21:44:35; ledger 360 s, 0.12 USD (upper bound).

D42 (appended 2026-10-09 after the re-audit `docs/CLAIM_AUDIT_340.md`): **E is spent for a registered H1/H3.**
Prereg section 6 (freeze rule) requires the calibrator, all thresholds and the E manifest hash in a signed
addendum before E is generated, and forbids E-based refitting. The full E set has been generated and its
outcomes analysed and published (revision 3.4.0) before split C or the Laya head exist. A registered H1/H3
therefore needs fresh generations on a new evaluation set E' after the freeze (for example problems released
after the models' training cutoff), or a recorded deviation stating that the E generations predate the freeze
and their outcomes were seen, in which case the test is a deviated test.

Further notes from the re-audit (2026-10-09): the cascade arm (`gwenlaya`) abstains on 757 of 1,536 tasks
(all 500 math, 77 Python, 180 Rust); "cascade accuracy" counts the final candidate whether or not it was
answered (selective coverage 0.507, CWR 0.103 pooled, `A5.cascade_selective_*`). Within each domain every
source-problem cluster has one item, so the cluster bootstrap is effectively item-level stratified by domain.
A warm-up-excluded cost sensitivity S2 (post hoc) is reported next to the planned cost: pooled cascade/4B
ratio 1.243 (1.224-1.260) without the first chunk vs 1.357 with it.

## Revision 3.5.0 (2026-10-09): D43-D47. All are post hoc relative to the E baselines of 3.4.0 and to the registered H1/H3

All of the following were decided after the E baselines (A1-A9) had been seen. They are exploratory and change
the status of no registered hypothesis. The plan addenda A10, A11, A12 in `docs/ANALYSIS_PLAN_E_TPU.md` were
written before the corresponding results were computed (A12: before the 9B run completed).

D43. **Program-of-thought math gate added after seeing the E baselines.** Prereg says the math gate is a sandboxed
program-of-thought re-execution (D37 recorded that it was not implemented). It was implemented
(`gwaya/domains/math_gate.py`) and run on E math for the 2B and 4B after the zero math coverage had been seen
(addendum A10). The gold answer is never an input. One greedy program (max 1,024 new tokens) per tier and math task;
programs are a second accelerator generation and are charged to every gated arm for each tier consulted; ungated arms
are not charged. The math rows of the overlay were replaced by the new replay (originals untouched). The 3.4.0
statement "math gate coverage 0" is superseded; the cascade cost and selective numbers change for math and pooled.

D44. **Qwen3.5-9B added post hoc.** The registered ladder lists 2B/4B/9B/27B; D32 had cut it to 2B/4B. After the
E results, to restore the registered ladder's largest feasible tier (addendum A12), the 9B was run on all 1,536 E tasks plus math programs
(addendum A12, written before the 9B run completed; a smoke test of 6 tasks per domain was not analysed). The 27B
was not attempted (budget). The reading rule for "cascade pays off" was fixed in A12 before results.

D45. **Multi-chip chip-seconds.** The 9B ran on a `v5litepod-4` slice (tensor parallel 4). Attributed seconds are the
slice wall time split by token share times 4 chips (`--chips 4` in `scripts/import_remote_gens.py`), so they are
comparable in unit with the single-chip 2B/4B but not a measure of equal engine concurrency. The 9B effective
concurrency was not measured, so its cost is less certain. Dollar cost at the 1.20 USD/chip-hour list price.

D46. **A11 cross-fitted calibrated abstention added post hoc** (`scripts/analyze_abstention.py`, addendum A11;
`results/gwenlaya_v4/e_tpu_mathgate/abstention_a11.json`). It replaces a calibrator fitted on split C (which does not
exist) with a 5-fold cross-fitted logistic regression, run once on E. Model M3 uses the gate verdict as a feature, so
its gain over a log-prob-only score is calibration gain, not independent evidence. CIs resample clusters of fixed
out-of-fold predictions and do not include model-fit variability. ECE percentile-bootstrap CIs sometimes exclude their
own point estimate (known upward bias of ECE under resampling). A11 was run for the 4B and 2B only, not the 9B.

D47. **Spend.** The ledger has 11 lines totalling 4.9136 USD (list-price estimate, not the bill): the PoT run
(`tpu_pot-082416`, 0.33), the 9B smoke (`tpu_nineb-smoke-084942`, 0.7827) and the 9B full run
(`tpu_nineb-full-091403`, 1.272) were added since 3.4.0 (2.5289). The 9B lines are v5litepod-4 at 4 x 1.20 USD/hour.


## Revision 3.6.0 (2026-10-09): D48-D52. Laya training, the fresh set E', and the shuffled-label control.
D48. **Split of the new pool.** A13 splits the new pool P by cluster hash mod 10 (0-5 train, 6 calibration C, 7-9 evaluation E')
instead of the reserved 20-bucket rule (1-2 router-train, 0 calibration), which would have left about 10 percent of a small pool for training.
D49. **E is used for training.** All 1,536 E tasks are in the Laya training set (E is burned for the registered tests anyway). E' and C share no
cluster, task id or normalised prompt hash with the training data (checked: E' 0/0/0; C 0/0/1, the single prompt-hash match affects only C).
D50. **The registered shuffled-label criterion was withdrawn before E' was generated.** A13b required the shuffled-label (SH) controls to give
AUROC 0.4-0.6 on C. They did not: SH calibrator 0.693 pooled (0.55-0.69 within domain x tier), SH router 0.572 / 0.637 / 0.389. We then scored the
UNTRAINED, same-seed twin of each head on C: calibrator 0.48 pooled but 0.24-0.51 within strata (Python 0.24-0.38), router 0.653 / 0.482 / 0.575.
SH and twin rankings are almost uncorrelated (0.14), so the SH value is not an initialisation effect; both are far from 0.5 in either direction. Our
reading (not proven): a head fed the strongly predictive gate/log-prob features, or a router scored pooled over domains with different base rates,
ranks by those inputs in an arbitrary direction when it has learned no labels, so the registered criterion cannot be met reliably by a correct pipeline.
The control that was meant to detect leakage is replaced by the direct leakage check above. The SH numbers, the twin numbers and this reading are in the
paper. If the criterion had been applied as written, E' would not have been generated; it was generated after this deviation was committed, and the
trained Laya's C AUROC (0.952 calibrator; router 0.78 / 0.81 / 0.82) is far from both controls.
D51. **27B abandoned; generator LoRA not trained.** See A13b: the 27B did not finish loading in 33 minutes on a 4-chip slice and an 8-chip slice is blocked by the
per-zone quota (4 chips); the Qwen3.5-0.8B torch_xla LoRA probe finished no step in 25 minutes (pure-PyTorch linear-attention fallback).
D52. **A first Laya run was lost and repeated.** The first launch (about 0.9 USD) trained the calibrator but could not save it (XLA-device tensors cannot be read by
safetensors) and the router found no examples (tier names not passed); both were fixed and the run repeated, with the save moved to before any prediction.

## Revision 3.7.0 (2026-10-09): D53-D58. A fresh Rust set R' for the frozen Laya (addendum A14)
D53. **R' was built post hoc, after the E' results were known.** Rust had no fresh evaluation in 3.6.0 (MultiPL-E Rust is all in E). R' was decided after seeing that
Laya's E' advantage was on math and Python only. The plan (A14, commits 0f47b15/ecdd44e) was committed before any model generated for R'; no Laya score and no model output
on R' had been seen. The results are exploratory/confirmatory in the A13b sense only (same definitions, Rust only); no registered test is involved.
D54. **Tasks were filtered by reference and stub checks, using the harness.** Of 106 Exercism practice exercises, 69 were kept: an exercise is kept only if its reference solution
is VERIFIED by the real checker with latency at most 4 s and the unmodified starter stub is not VERIFIED. 37 were dropped, with reasons in
`results/gwenlaya_v4/laya/data/rust_fresh_report.json`. This selects tasks that are solvable by the checker and may differ in difficulty and style from a random Rust workload. No model answer was used in the filter.
D55. **Harness property found (edition 2015); measured in A15.1.** The Rust oracle invoked `rustc` without `--edition`, i.e. edition 2015, in every published run. Exercises whose reference needs the 2021 prelude or let-chains
were dropped from R'. A15.1 (below, D58) re-checked every stored Rust answer under edition 2021: confirmed differences are small (see D58); all published numbers stay labelled edition 2015.
D56. **tau reused.** The router threshold tau = 0.5 was chosen on C (3.6.0) and is reused on R' without re-choosing; C contains no Rust. The isotonic map is the frozen one (identical to the frozen file).
D57. **Spend and bookkeeping.** R' generation and Laya prediction added three ledger lines (`tpu_rf-9b-182548`, `tpu_rf-small-182548`, `tpu_laya-pred-r-184359`), after the E' prediction line. The `meta.n_Eprime_rows` field and the `router.E_prime` key in `laya_rfresh.json` are names inherited from `scripts/analyze_laya.py`; the data are R' (207 rows, 69 tasks).
D58. **Rust oracle default edition changed to 2021; A15 results.** Plan addendum A15 (commit f0f514b, before any edition or depth result) fixed a decision rule: published numbers stay labelled edition 2015; the default for FUTURE runs becomes 2021; if confirmed flips exceed 2 percent of any set's tasks for any tier, that set is re-analysed under 2021. Result (`results/gwenlaya_v4/edition/rust_edition_sensitivity.json`, commit 8d3323f): confirmed hidden-test fail-to-pass flips E 4B 1, E 9B 1 (of 507), E 2B 0; R' 4B 1 (of 69), R' 2B and 9B 0; no pass-to-fail flip; the same item also moves the gate from fail to verified. All are below 2 percent (R' 4B is 1/69), so no re-analysis was done. `gwaya/oracles.py` now has `DEFAULT_RUST_EDITION = "2021"` (env `GWAYA_RUST_EDITION` overrides). This supersedes the A15 sentence "the default is still 2015" and the D55 statement that the sensitivity was not run. A15.2 (visible-test depth dial, `rust_gate_depth.json`) is descriptive, not a test: A15 planned k = 1, 2, 4 on Python and Rust of E and R'; what was run is k = 1, 2, 3 on the Rust tasks of E only (505 tasks with at least 3 assertions), with edition 2015 pinned. Python and R' are not covered; the change from k = 4 to 3 and the scope are deviations from A15.
