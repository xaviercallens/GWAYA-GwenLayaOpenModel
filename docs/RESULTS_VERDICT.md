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

## Bottom line

**No pre-registered test ran.** The only scored data are 80 items from one unregistered arm
(`base`, Qwen3.5-2B q4_K_M, CPU). H1, H3 (primaries) and H2, H7 need paired registered arms
(GL, B1, B2, B3, B5). H4-H6 need a cross-fitted scores file, which does not exist. Only H13 has a decision rule that
can be evaluated; it is not refuted on the known lines, but it holds largely by construction (the
ledger refuses over-cap stages), so it is not evidence about the system. Every probability for a hypothesis that did
not run is the **red-team prior** from `docs/DEVIATIONS.md` ("Hypothesis status for this run"), not
evidence.

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
| H13 | systems | not refuted (interim; known spend $0.00, 2,706 TPU-seconds unpriced) | 0.75 (prior) | Ledger check, holds by construction; see below. |
| H14 | systems | not_run | 0.50 (prior) | Needs E1, which needs G1 (D12). |
| H15 | systems | not_run | 0.70 (prior) | See below. |
| H2 | exploratory | not_run | 0.08 (prior) | No GL or B5 arm. |
| H7 | exploratory | not_run | 0.10 (prior) | `gate` is null in every row, so VERIFIED precision of the gate cannot be computed. |
| H10 | exploratory | dropped | 0.05 (prior) | D8. |
| H11 | exploratory | dropped | 0.12 (prior) | D7, D8. |
| H12 | exploratory | dropped | 0.12 (prior) | D8. |

### H13 derivation

Decision rule: ledger total <= 40 USD (planned) and <= 50 USD (hard); refuted if either is exceeded
or a stage exceeds 1.25x its cap twice. numbers_v4 H13 verdict: "not refuted on the known lines
(lower bound; unpriced lines exist)": 3 ledger lines, `usd_known_total` 0.0, 2 lines without
`usd_estimate` (`seconds_without_estimate` 2706.0, the two TPU smoke tests; they also lack the
`resource` field required by the ledger schema). They are unpriced (TBD) until the billing export is
read; the ~0.84 USD planning figure comes from task text, not from the ledger, a price list or a
bill, and is not used. No stage has a spend over its cap (G1/E1 never ran). The red-team review
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

Outcome counts (script output): python 11 PASS_HIDDEN / 17 FAIL_HIDDEN / 0 UNDECIDED; rust 0 / 13 /
13; math 3 / 18 / 5. Math answer extraction: all 26 by the unregistered last-number fallback
(`math_extraction.boxed.n` = 0), including all 3 passes; registered boxed-only rescoring: math 0/26.

Sensitivity only (not valid estimates): pooled n = 80 as scored: pass rate 0.175 (0.10-0.25),
mean confidence 0.875 (0.863-0.886), ECE 0.700 (0.626-0.771), Brier 0.630 (0.571-0.686), AUROC 0.618
(0.422-0.811), AURC 0.716 (0.586-0.859), cost per pass 1178 (732-2114). Pooled with boxed-only math:
pass rate 0.138 (0.075-0.200), ECE 0.737 (0.676-0.798), AUROC 0.758 (0.568-0.922). Math as scored:
AUROC 0.348 (0.133-0.583). Rust: AUROC undefined (no pass).

Absolute scores are on public benchmarks probably seen in Qwen3.5 pretraining (prereg section 10);
they are not clean capability estimates.

## Results that look wrong or underpowered

1. **Math is invalid as a capability measure.** The L2 review found all 26 math completions were
   produced with the python system prompt and a ```python prefill (0/26 contain `\boxed`). The
   scored rows are those same generations; no regeneration has happened. The 3/26 math passes came
   from the unregistered last-number fallback (D24) and say nothing about the model's math ability.
2. **Rust 0/26 PASS_HIDDEN with 13/26 UNDECIDED is suspicious.** The Rust harness has never been
   validated on a passing program (D17). A zero rate with half the items undecided is as consistent
   with a harness or toolchain problem as with model weakness. Treat it as unverified until one
   known-good Rust solution passes end to end.
3. **Pooled numbers mix domains and defects**, so Python (n = 28) is the only headline; pooled
   numbers are a sensitivity analysis.
4. **The confidence signal is badly calibrated, and its ranking is unresolved.** On Python the
   AUROC CI (0.483-0.917) includes 0.5; ECE 0.519, with mean confidence 0.896 against a pass rate of
   0.393.
5. **Underpowered by design.** n = 80 paired at most; the L1 review computed power 0.05 for H1 at
   the 3 pt MEI. Any H1 run at this n is expected to be inconclusive, not a refutation.
6. **Cost unit is CPU-seconds on a shared 8-core machine** with 3 foreign CPU-bound jobs (D14), and
   `cpu_seconds` includes idle-spinning threads (D21). Not comparable to the registered L4
   GPU-seconds.

## What would change these verdicts

Fix the math prompt path and regenerate the 26 math items; validate the Rust harness on one passing
program; generate at least the B3/B5/GL arms on the same 80 items; build `scores_crossfit.jsonl`.
Only then can H1, H3, H2, H7 and H4-H6 move off not_run.
