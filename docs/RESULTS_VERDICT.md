# Results verdict, night run 2026-10-07/08

Written 2026-10-08 on branch `night/2026-10-07`. Sources: `papers/numbers_v4.json` (schema
`gwenlaya_v4.numbers/1`, n_boot 10000, seed 0), the raw rows
`/mnt/data/home/xavkal/gwaya-data/night/cache/L2score/{rows,gens}.jsonl` (hash chains OK per
numbers_v4 meta), `/mnt/data/home/xavkal/gwaya-data/spend_ledger.jsonl`, and `docs/DEVIATIONS.md`.
No number below comes from anywhere else. Unknown = TBD.

**Revised 2026-10-08 after external review** (see `docs/REVIEW_RESPONSE.md`): H13 now uses the
machine wording and its 0.97 is withdrawn; the descriptive result is per domain with Python as the
headline (numbers from `single_arm.by_domain.*` in numbers_v4, not hand counts); the hidden-check
outcome is written PASS_HIDDEN / FAIL_HIDDEN / UNDECIDED, because the registered VERIFIED is the
gate verdict and no gate verdict was recorded; math scoring used an unregistered last-number
fallback (D24); the HumanEval+ gate payloads leaked the hidden tests (D25, no number affected);
primary Holm is back to the registered m = 2 (D26).

**Revised 2026-10-08 for v3.3.1** (branch `fix/math-and-rescore`; D27-D30 in `docs/DEVIATIONS.md`).
numbers_v4 is now built from the re-scored rows `night/cache/L2fix/rows.jsonl` (copy in
`results/gwenlaya_v4/night_L2fix/rows.jsonl`; same generations). Additional sources:
`results/gwenlaya_v4/rust_harness_validation.json` and `results/gwenlaya_v4/tpu_run3/`
(`summary.json`, `*.rescored.jsonl`, `status.jsonl`). Changes: the v3.3.0 Rust result (0/26) was a
sandbox link failure and is withdrawn; the fixed harness gives Rust 6/26 (D27). Math is scored
boxed-only by default and all 26 night math items are UNDECIDED (D28). Python unchanged (D29). A
descriptive, unregistered TPU run was added (D30). Ledger now 5 lines, 1.5399 USD.

## Bottom line

**No pre-registered test ran.** The only scored data are 80 items from one unregistered arm
(`base`, Qwen3.5-2B q4_K_M, CPU). H1, H3 (primaries) and H2, H7 need paired registered arms
(GL, B1, B2, B3, B5). H4-H6 need a cross-fitted scores file, which does not exist. Only H13 has a decision rule that
can be evaluated; it is not refuted on the known lines, but it holds largely by construction (the
ledger refuses over-cap stages), so it is not evidence about the system. Every probability in the
table below, including H13's, is the **red-team prior** from `docs/DEVIATIONS.md` ("Hypothesis
status for this run"), written before any data, not evidence. The Rust correction and the TPU run
(v3.3.1) are descriptive and change no verdict.

## Verdicts

| H | Role | Verdict | probability_true | Derivation |
|---|---|---|---|---|
| H1 | primary | not_run | 0.45 (prior) | No GL or B3 arm; McNemar and paired bootstrap impossible. Even if run, power at the 3 pt MEI with n=80 was 0.05 (STAGE_LOG L1 review), so it would very likely be inconclusive. |
| H3 | primary | not_run | 0.50 (prior) | No GL or B5 arm. Cost unit would be CPU-seconds (D3), not GPU-seconds. |
| H4 | secondary A | not_run | 0.15 (prior) | No cross-fitted scores; no Laya head trained. |
| H5 | secondary A | not_run | 0.25 (prior) | Same. |
| H6 | secondary A | not_run | 0.40 (prior) | Same. |
| H8 | secondary B | not_run | 0.30 (prior) | Needs G1 (GPU slot held by a foreign VM, D11). |
| H9 | secondary C | not_run (deferred) | 0.25 (prior) | Deferred, D8. |
| H13 | systems | not refuted (interim; ledger $1.54, catalog list-price estimate) | 0.75 (prior) | Ledger check, holds by construction; see below. |
| H14 | systems | not_run | 0.50 (prior) | Needs E1, which needs G1 (D12). |
| H15 | systems | not_run | 0.70 (prior) | See below. |
| H2 | exploratory | not_run | 0.08 (prior) | No GL or B5 arm. |
| H7 | exploratory | not_run | 0.10 (prior) | `gate` is null in every row, so VERIFIED precision of the gate cannot be computed. |
| H10 | exploratory | dropped | 0.05 (prior) | D8. |
| H11 | exploratory | dropped | 0.12 (prior) | D7, D8. |
| H12 | exploratory | dropped | 0.12 (prior) | D8. |

### H13 derivation

Decision rule: ledger total <= 40 USD (planned) and <= 50 USD (hard); refuted if either is exceeded
or a stage exceeds 1.25x its cap twice. numbers_v4 H13 verdict: "not refuted": 5 ledger lines,
0 without `usd_estimate`, `usd_known_total` 1.5399. Lines: TPU smoke attempt 1 (spot, 240 s,
0.0329), TPU smoke (on demand, 2466 s, 0.822), L1 upload (13 s, 0.0), TPU run #2 (on demand, 1073 s,
0.3577), TPU run #3 (on demand, 982 s, 0.3273). Prices are Cloud Billing Catalog list prices (v5e
on-demand 1.20 USD/chip-h; v5e spot 0.494237 USD/chip-h). The billed amount is TBD until the
billing export is read. G1/E1 never ran. The red-team review
(PROPOSAL_REVIEW.md, prior 0.75) notes that the ledger enforces H13 by refusing stages, so support
is close to tautological. An earlier version of this file gave probability_true = 0.97; that was
author judgement with no statistical basis and is withdrawn. The probability column shows the prior.
H13 is a whole-study claim, so this is interim.

### H15 note

numbers_v4 marks it not_run. `tests/test_run_study.py::test_resume_after_kill_mid_run` passes
(rerun 2026-10-08: 2 resume tests pass), but it simulates a death by calling `run_task` once and
appending a torn line. It sends no SIGKILL and measures no lost wall time, so it is not the
registered verification. It is weak supporting evidence that resume is item-granular; the
probability stays at the prior.

## Descriptive single-arm results (not a pre-registered test)

Arm `base`, n = 80 (python 28, rust 26, math 26). Confidence is raw exp(mean_logprob), uncalibrated,
not cross-fitted. 95 % cluster-stratified bootstrap CIs, all from numbers_v4
(`single_arm.by_domain.*`, `single_arm.boxed_only.*`, `arm.base.*`). "Pass" = PASS_HIDDEN (rows.jsonl
`score` = VERIFIED, i.e. passed the hidden checks); this is not the registered VERIFIED (gate),
which was not recorded in any row (`gate_recorded` = 0 in every domain).

Headline, Python only (n = 28):

| Metric | Point | 95 % CI |
|---|---|---|
| pass rate (PASS_HIDDEN) | 0.393 | 0.214-0.571 |
| mean raw confidence | 0.896 | 0.875-0.917 |
| confident-wrong rate (upper bound, UNDECIDED counted wrong) | 0.607 | 0.429-0.786 |
| ECE | 0.519 | 0.392-0.689 |
| Brier | 0.476 | 0.334-0.617 |
| AUROC | 0.717 | 0.483-0.917 |
| AURC | 0.438 | 0.221-0.712 |
| cost per pass (server CPU-seconds) | 330 | 174-726 |

Rust, descriptive (n = 26; v3.3.1, harness fixed and validated after scoring, D27):

| Metric | Point | 95 % CI |
|---|---|---|
| pass rate (PASS_HIDDEN) | 0.231 | 0.077-0.385 |
| mean raw confidence | 0.894 | 0.882-0.906 |
| confident-wrong rate (upper bound) | 0.769 | 0.615-0.923 |
| ECE | 0.675 | 0.549-0.813 |
| Brier | 0.605 | 0.477-0.725 |
| AUROC | 0.875 | 0.709-0.990 |
| AURC | 0.557 | 0.321-0.851 |
| cost per pass (server CPU-seconds) | 494 | 261-1552 |

With 6 passes, one seed, an uncalibrated score and probable benchmark exposure, the Rust AUROC is
not read as evidence that the score ranks Rust answers well.

Outcome counts (script output, v3.3.1): python 11 PASS_HIDDEN / 17 FAIL_HIDDEN / 0 UNDECIDED; rust
6 / 20 / 0; math 0 / 0 / 26. Math is scored boxed-only by default (D28) and no night math generation
contains `\boxed` (`math_extraction.boxed.n` = 0). The v3.3.0 counts (rust 0/13/13, the 3 math passes
from the last-number fallback) are superseded.

Sensitivity only (not valid estimates): pooled n = 80 (boxed-only math): pass rate 0.212
(0.125-0.300), mean confidence 0.875 (0.863-0.886), ECE 0.662 (0.585-0.741), Brier 0.589
(0.522-0.654), AUROC 0.806 (0.675-0.917), AURC 0.601 (0.458-0.758), cost per pass 970 (629-1648).
Math: AUROC undefined (no pass). In `tables_v4.tex` the "last-number fallback" math row and the
"sensitivity (mixed)" pooled row now equal the boxed-only rows, because the rows were re-scored
boxed-only; their status labels are hard-coded in the script and predate D27/D28.

## Descriptive TPU run #3 (not a registered arm, not a test; D30)

One on-demand TPU v5e chip, vllm-tpu 0.31.0, bf16, greedy (temperature 0), max_tokens 1024,
non-thinking, fixed prompts. All 80 night tasks per model. Scoring on the TPU host was invalid for
Python (every Python item FAILED there), so all rows were re-scored locally with the same checkers;
Rust and math re-scores equal the TPU-host status on every item. Counts from
`results/gwenlaya_v4/tpu_run3/summary.json`; truncations from `*.rescored.jsonl`
(`finish_reason` = length); throughput from the `SUMMARY` lines of `status.jsonl`.

| Model | Python /28 | Rust /26 | Math /26 | math truncated (UNVERIFIED) | load_s | gen_s | tok_per_s |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-2B | 11 | 6 | 16 | 9 | 106.2 | 99.7 | 256.3 |
| Qwen/Qwen3.5-4B | 20 | 13 | 16 | 9 | 124.3 | 117.4 | 220.2 |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 18 | 15 | 8 | 4 | 186.3 | 8.7 | 2094.2 |

Qwen3.5-2B bf16 on TPU has the same Python (11/28) and Rust (6/26) counts as the q4_K_M CPU arm; the
item-level outcome agrees on 22/28 Python and 24/26 Rust items (computed from
`night_L2fix/rows.jsonl` vs `res_qwen35_2b.rescored.jsonl`). This is a consistency check across
hardware and quantisation, not a test. Single sample, no logprobs, no gate, no calibration; public
benchmarks the models have probably seen. Qwen3.5 runs through the tpu-inference PyTorch fallback
(with `SKIP_JAX_PRECOMPILE=1`); Qwen2.5-Coder is JAX-native.

Absolute scores are on public benchmarks probably seen in Qwen3.5 pretraining (prereg section 10);
they are not clean capability estimates.

## Results that look wrong or underpowered

1. **Math is invalid as a capability measure.** The L2 review found all 26 math completions were
   produced with the python system prompt and a ```python prefill (0/26 contain `\boxed`). The
   scored rows are those same generations; no CPU regeneration has happened. Under boxed-only
   scoring (now the default, D28) all 26 are UNDECIDED. The prompt is fixed in code (D28); only the
   TPU run used it.
2. **The v3.3.0 Rust result (0/26) was wrong, caused by our harness (D27).** No Rust program could
   link inside the sandbox. Fixed and validated on hand-written solutions (25/25 correct pass, 9/9
   wrong fail) after the generations existed; Rust is now 6/26. The validation is not an independent
   reference set, and one of the 26 tasks has no reference solution in it.
3. **Pooled numbers mix domains and defects**, so Python (n = 28) stays the headline (chosen in D26,
   before the Rust fix); Rust is a second descriptive domain; pooled numbers are a sensitivity
   analysis.
4. **The confidence signal is badly calibrated, and its ranking is unresolved.** On Python the
   AUROC CI (0.483-0.917) includes 0.5; ECE 0.519, with mean confidence 0.896 against a pass rate of
   0.393.
5. **Underpowered by design.** n = 80 paired at most; the L1 review computed power 0.05 for H1 at
   the 3 pt MEI. Any H1 run at this n is expected to be inconclusive, not a refutation.
6. **Cost unit is CPU-seconds on a shared 8-core machine** with 3 foreign CPU-bound jobs (D14), and
   `cpu_seconds` includes idle-spinning threads (D21). Not comparable to the registered L4
   GPU-seconds.

## What would change these verdicts

Regenerate the 26 night math items on the CPU arm with the fixed prompt (D28); generate at least the
B3/B5/GL arms on the same 80 items with the fixed harnesses; build `scores_crossfit.jsonl`.
Only then can H1, H3, H2, H7 and H4-H6 move off not_run.
