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
descriptive, unregistered TPU run was added (D30). Ledger now 5 lines, 1.5399 USD (superseded in revision 3.4.0: 8 lines, 2.5289 USD).

**Revised 2026-10-08 for revision 3.4.0** (branch `bench/tpu-e-eval`; D31-D41 in `docs/DEVIATIONS.md`).
Additional sources: `docs/CLAIM_AUDIT.md`, `docs/ANALYSIS_PLAN_E_TPU.md`,
`results/gwenlaya_v4/e_tpu/numbers_e_tpu.json` (sha256 c76bc506...226b), `papers/tables_e_tpu.tex`,
`results/gwenlaya_v4/e_tpu/{data,audit,overlay}/`, and the ledger read at 2026-10-08 23:30 CEST.
Changes: (1) the claim-audit corrections (two wrong numbers_v4.json hashes, stale spend, bib title;
H13 "by construction" and the run #3 CPU-vs-TPU reading reworded); (2) exploratory full-E baselines on
one TPU v5e chip, reported below. **No verdict changes**: every registered hypothesis that was not run
is still not run, and the probability column is still the prior.

## Bottom line

**No pre-registered test ran.** The scored data are the 80 night items (one unregistered CPU arm,
`base`, Qwen3.5-2B q4_K_M) and the exploratory full-E TPU baselines (1,536 tasks x 3 models); none is a
registered arm. H1, H3 (primaries) and H2, H7 need paired registered arms
(GL, B1, B2, B3, B5). H4-H6 need a cross-fitted scores file, which does not exist. Only H13 has a decision rule that
can be evaluated; it is not refuted on the known lines (8 ledger lines, 2.5289 USD). The GPU stage
launcher is designed to refuse a stage that would push the ledger over its cap, so H13 was expected to
hold largely by construction (red-team prior 0.75); the TPU runs up to v3.3.1 were launched by separate
scripts with a run-time limit but no ledger check, so the low total reflects low usage. In either case it
is not evidence about the GwenLaya system. The full-E TPU baselines (revision 3.4.0) are exploratory and
are not a test of H1, H3 or any other registered hypothesis. Every probability in the
table below, including H13's, is the **red-team prior** from `docs/DEVIATIONS.md` ("Hypothesis
status for this run"), written before any data, not evidence. The Rust correction and the TPU run
(v3.3.1) are descriptive and change no verdict.

## Verdicts

| H | Role | Verdict | probability_true | Derivation |
|---|---|---|---|---|
| H1 | primary | not_run | 0.45 (prior) | No GL or B3 arm (no Laya head, no split C); the exploratory A4 matched-coverage comparison (gate vs the k most confident log-prob answers, k matched on E) is not H1. H1/H3 can no longer be run as registered on the E generations (freeze rule, D42): a registered test needs fresh generations E' or a recorded deviation. McNemar and paired bootstrap impossible. Even if run, power at the 3 pt MEI with n=80 was 0.05 (STAGE_LOG L1 review), so it would very likely be inconclusive. |
| H3 | primary | not_run | 0.50 (prior) | No GL or B5 arm. The exploratory A5 gate cascade (no Laya) vs always-4B in TPU chip-seconds is not H3. |
| H4 | secondary A | not_run | 0.15 (prior) | No cross-fitted scores; no Laya head trained. |
| H5 | secondary A | not_run | 0.25 (prior) | Same. |
| H6 | secondary A | not_run | 0.40 (prior) | Same. |
| H8 | secondary B | not_run | 0.30 (prior) | Needs G1 (GPU slot held by a foreign VM, D11). |
| H9 | secondary C | not_run (deferred) | 0.25 (prior) | Deferred, D8. |
| H13 | systems | not refuted (interim; ledger $2.53, catalog list-price estimate) | 0.75 (prior) | Ledger total; expected to hold largely by construction for the GPU launcher; low usage on TPU; not evidence about the system. See below. |
| H14 | systems | not_run | 0.50 (prior) | Needs E1, which needs G1 (D12). |
| H15 | systems | not_run | 0.70 (prior) | See below. |
| H2 | exploratory | not_run | 0.08 (prior) | No GL or B5 arm. |
| H7 | exploratory | not_run | 0.10 (prior) | No registered arm with a gate. Descriptive analogue (A3, 4B gate on TPU, not the registered setting): precision 0.830 Python, 0.887 Rust, both below the registration's 0.95 flag. |
| H10 | exploratory | dropped | 0.05 (prior) | D8. |
| H11 | exploratory | dropped | 0.12 (prior) | D7, D8. |
| H12 | exploratory | dropped | 0.12 (prior) | D8. |

### H13 derivation

**Update for revision 3.4.0.** The ledger read at 2026-10-08 23:30 CEST has 8 lines; the sum of
`usd_estimate` (recomputed) is 2.5289 USD. The three added lines are the first E-generation pass
(963 s, 0.321, set aside: no log-probabilities), an orphan VM created by a driver the maintainer killed by
mistake (360 s, 0.12, upper bound) and the E-generation pass used (1644 s, 0.548). The v3.3.1 text below
("holds by construction ... the ledger enforces H13 by refusing stages") overstated it (claim audit B15):
the ledger file enforces nothing; the GPU stage launcher is designed to refuse over-cap stages; the TPU
runs up to v3.3.1 had a run-time limit but no ledger check. `deploy/tpu/drive.sh` (committed 39a5a0a)
checks ledger + worst case against a cap and is a dry run by default; its log for the second E pass shows
the check (ledger 1.9809, worst case 2.5, cap 45). The text below is kept as history.

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
`night_L2fix/rows.jsonl` vs `res_qwen35_2b.rescored.jsonl`). Corrected wording (claim audit B12):
Qwen3.5-2B in bf16 under vLLM on TPU, with a different code system prompt and no code prefill, gives the
same Python and Rust pass counts as the q4 CPU arm; hardware, quantisation, engine and code prompt all
changed, so this is a loose consistency check, not an isolation of any one factor. Superseded by the
same-protocol comparison A7 below. Single sample, no logprobs, no gate, no calibration; public
benchmarks the models have probably seen. Qwen3.5 runs through the tpu-inference PyTorch fallback
(with `SKIP_JAX_PRECOMPILE=1`); Qwen2.5-Coder is JAX-native.

Absolute scores are on public benchmarks probably seen in Qwen3.5 pretraining (prereg section 10);
they are not clean capability estimates.

## Exploratory full-E baselines on TPU v5e (revision 3.4.0; not registered arms, not a test)

Plan: `docs/ANALYSIS_PLAN_E_TPU.md` (A1-A9; amendments 1-4 are post hoc). 1,536 tasks of
`tasks_E_primary.d25.jsonl` (Python 529, Rust 507, math 500); one TPU v5e chip, vLLM-TPU 0.31.0, bf16,
greedy, one sample, `max_new_tokens` 1024, raw protocol prompts identical to the llama.cpp path;
scored locally in bwrap with the unchanged study code. All numbers from `numbers_e_tpu.json`
(`metrics.*`, `paired_tests`, `A7_night_cpu_vs_tpu`, `stability_audit`); 95 % source-problem cluster
bootstrap CIs, 10,000 resamples, seed 0 (within a domain every cluster has one item, so this is in effect
an item-level bootstrap stratified by domain). Tables: `papers/tables_e_tpu.tex`. Revised 2026-10-09 after
the re-audit `docs/CLAIM_AUDIT_340.md` (numbers file regenerated, sha256 9197f367...47bc).

| Pass rate (PASS_HIDDEN) | Python | Rust | Math | Pooled |
|---|---|---|---|---|
| Qwen3.5-2B | 0.452 | 0.304 | 0.514 | 0.423 (0.398-0.447) |
| Qwen3.5-4B | 0.675 | 0.529 | 0.612 | 0.606 (0.582-0.631) |
| Qwen2.5-Coder-1.5B | 0.647 | 0.422 | 0.368 | 0.482 (0.457-0.507) |

Negative and null results, stated first:
1. **Raw log-prob confidence is badly calibrated.** 4B mean confidence 0.927 / 0.930 / 0.938 (Python /
   Rust / math) vs pass rates 0.675 / 0.529 / 0.612; ECE 0.252-0.598 across all model x domain cells.
2. **The gate cascade (2B -> executed gate -> 4B; GwenLaya without Laya, not GL) gains no accuracy and is
   not cheaper than always-4B.** Pooled accuracy difference cascade - 4B = -0.003 (-0.014 to 0.009);
   McNemar 40 vs 44 discordant, p = 0.744 (Holm 0.744). Python: -0.026 (-0.053 to -0.002; upper bound
   within Monte-Carlo noise of 0, borderline). Pooled cost ratio cascade / 4B in chip-seconds = 1.357
   (1.329-1.385) with the first (compilation) chunk, as planned, and 1.243 (1.224-1.260) without it (S2,
   post hoc); Python 0.997 vs 0.900, Rust 1.221 vs 1.090, math 1.514 vs 1.398. Pooled the cascade is more
   expensive either way; on Python it is break-even with warm-up and ~10 % cheaper without, at a 2.6-point
   accuracy cost. Cost is configuration dependent (configured max_num_seqs 96 / 32, effective KV capacity at
   most 54 (2B) / 11 (4B) concurrent requests per the engine logs; not latency; not comparable to
   GPU-seconds) and not robust. Cost per correct: 0.821 vs 0.603 chip-s. Escalated: 0.675 pooled. The
   cascade arm abstains on 757 tasks (all 500 math); "cascade accuracy" counts the final candidate whether
   answered or not; selective coverage 0.507, CWR 0.103 pooled. Oracle-router ceiling (A6): 0.644
   (0.620-0.668) vs 0.606 for the 4B: little accuracy headroom for routing between these two tiers.
3. **The gate has zero coverage on math**, because the registered math gate (program-of-thought
   re-execution) is not implemented in our checker (it executes only `gate_payload`, empty for the 500 math
   rows); the cascade escalates every math task. An implementation gap, not a property of math tasks.
4. **Truncation:** 0.400 / 0.320 / 0.152 of math generations (2B / 4B / Coder) hit 1,024 tokens; no
   truncated generation passed (truncated math has no complete `\boxed{}`). Math pass rates are lower
   bounds under this budget.

Most favourable (exploratory, with caveats): gate-only on the 4B answers 0.813 (Python) and 0.596 (Rust)
of tasks with precision 0.830 and 0.887. A baseline that answers the k most confident tasks, k = the
gate's answered count in the same domain (no labels used to choose k; pessimistic ties), has a
confident-wrong rate 0.076 (0.057-0.098) higher on Python and 0.112 (0.093-0.134) higher on Rust. Pooled
with coverage matched within each domain (math k = 0): gate 0.070 vs baseline 0.133, difference -0.063
(-0.074 to -0.054); the earlier global-threshold -0.042 is withdrawn. Caveats: the gate executes visible
tests the log-prob does not see; k is matched on E rather than frozen on a separate split; public
benchmarks probably seen in pretraining; one greedy sample; this is not H1 (GL vs B3 with the threshold set
on split C). No direction of bias from the matching is claimed.

A7 (same raw protocol, night Python + Rust, 54 tasks; math excluded post hoc): CPU q4 llama.cpp vs TPU bf16
vLLM, Qwen3.5-2B: Python 11 vs 10 (27/28 agree), Rust 6 vs 8 (22/26 agree, p = 0.625), pooled p = 1.0.
Consistency check, not a test.

Verification: serial re-check of all 1,536 non-passing Python/Rust rows: 3 flips, of which 2 changed the
headline (py/MBPP/271 for 2B and 4B, contention, applied as an overlay) and 1 did not
(rs/mbpp_130_max_occurrences for 4B, nondeterministic, stays FAILED);
5 checks per model time out even alone (stay FAILED); all 1,572 passing rows re-run once, 0 changes ("none
observed", not "none exist"). Sensitivity S1 (unstable items counted wrong) differs from the headline by at
most 0.0019 (4B Python).

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
