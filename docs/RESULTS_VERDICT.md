# Results verdict, night run 2026-10-07/08

Written 2026-10-08 on branch `night/2026-10-07`. Sources: `papers/numbers_v4.json` (schema
`gwenlaya_v4.numbers/1`, n_boot 10000, seed 0), the raw rows
`/mnt/data/home/xavkal/gwaya-data/night/cache/L2score/{rows,gens}.jsonl` (hash chains OK per
numbers_v4 meta), `/mnt/data/home/xavkal/gwaya-data/spend_ledger.jsonl`, and `docs/DEVIATIONS.md`.
No number below comes from anywhere else. Unknown = TBD.

## Bottom line

**No pre-registered test ran.** The only scored data are 80 items from one unregistered arm
(`base`, Qwen3.5-2B q4_K_M, CPU). H1, H3 (primaries) and H2, H7 need paired registered arms
(GL, B1, B2, B3, B5). H4-H6 need a cross-fitted scores file, which does not exist. Only H13 has a
decision that can be evaluated, and only provisionally. Every probability for a hypothesis that did
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
| H13 | systems | supported (interim) | 0.97 | Deterministic check, see below. |
| H14 | systems | not_run | 0.50 (prior) | Needs E1, which needs G1 (D12). |
| H15 | systems | not_run | 0.70 (prior) | See below. |
| H2 | exploratory | not_run | 0.08 (prior) | No GL or B5 arm. |
| H7 | exploratory | not_run | 0.10 (prior) | `gate` is null in every row, so VERIFIED precision of the gate cannot be computed. |
| H10 | exploratory | dropped | 0.05 (prior) | D8. |
| H11 | exploratory | dropped | 0.12 (prior) | D7, D8. |
| H12 | exploratory | dropped | 0.12 (prior) | D8. |

### H13 derivation

Decision rule: ledger total <= 40 USD (planned) and <= 50 USD (hard); refuted if either is exceeded
or a stage exceeds 1.25x its cap twice. From numbers_v4: 3 ledger lines, `usd_known_total` 0.0,
2 lines without `usd_estimate` (`seconds_without_estimate` 2706.0, the two TPU smoke tests).
DEVIATIONS.md puts those at about 0.84 USD (task text, not read from the ledger). No stage has a
spend over its cap (G1/E1 never ran). So the rule holds with a margin of roughly 39 USD.
probability_true = 0.97 is not a bootstrap: the check is deterministic, and the residual 0.03 is
for ledger incompleteness (two unpriced lines; `runs/LEDGER.jsonl` on GCS does not exist) and for
future spend in the remaining nights. It is **interim**: H13 is a whole-study claim.

### H15 note

numbers_v4 marks it not_run. `tests/test_run_study.py::test_resume_after_kill_mid_run` passes
(rerun 2026-10-08: 2 resume tests pass), but it simulates a death by calling `run_task` once and
appending a torn line. It sends no SIGKILL and measures no lost wall time, so it is not the
registered verification. It is weak supporting evidence that resume is item-granular; the
probability stays at the prior.

## Descriptive single-arm results (not a pre-registered test)

Arm `base`, n = 80 (python 28, rust 26, math 26). Confidence is raw exp(mean_logprob), uncalibrated,
not cross-fitted. 95 % cluster-stratified bootstrap CIs from numbers_v4.

| Metric | Point | 95 % CI |
|---|---|---|
| accuracy | 0.175 | 0.10-0.25 |
| coverage | 1.0 | 1.0-1.0 |
| confident-wrong rate (upper bound, UNVERIFIED counted wrong) | 0.825 | 0.75-0.90 |
| ECE | 0.700 | 0.626-0.771 |
| Brier | 0.630 | 0.571-0.686 |
| AUROC | 0.618 | 0.422-0.811 |
| AURC | 0.716 | 0.586-0.859 |
| cost per correct (server CPU-seconds) | 1178 | 732-2114 |

Per domain, counted from rows.jsonl: python 11 VERIFIED / 17 FAILED; rust 0 VERIFIED /
13 FAILED / 13 UNVERIFIED; math 3 VERIFIED / 18 FAILED / 5 UNVERIFIED.

## Results that look wrong or underpowered

1. **Math is invalid as a capability measure.** The L2 review found all 26 math completions were
   produced with the python system prompt and a ```python prefill (0/26 contain `\boxed`). The
   scored rows are those same generations; no regeneration has happened. The 3/26 math VERIFIED
   says nothing about the model's math ability.
2. **Rust 0/26 VERIFIED with 13/26 UNVERIFIED is suspicious.** The Rust harness has never been
   validated on a passing program (D17). A zero rate with half the items undecided is as consistent
   with a harness or toolchain problem as with model weakness. Treat it as unverified until one
   known-good Rust solution passes end to end.
3. **Pooled accuracy 0.175 is therefore driven by python** (11 of 14 VERIFIED) and is a lower bound
   on what the 2B model can do.
4. **The confidence signal is barely informative and badly calibrated.** The AUROC CI (0.42-0.81)
   includes 0.5; ECE about 0.70 means raw exp(mean_logprob) sits near 1 while accuracy is near 0.2.
5. **Underpowered by design.** n = 80 paired at most; the L1 review computed power 0.05 for H1 at
   the 3 pt MEI. Any H1 run at this n is expected to be inconclusive, not a refutation.
6. **Cost unit is CPU-seconds on a shared 8-core machine** with 3 foreign CPU-bound jobs (D14), and
   `cpu_seconds` includes idle-spinning threads (D21). Not comparable to the registered L4
   GPU-seconds.

## What would change these verdicts

Fix the math prompt path and regenerate the 26 math items; validate the Rust harness on one passing
program; generate at least the B3/B5/GL arms on the same 80 items; build `scores_crossfit.jsonl`.
Only then can H1, H3, H2, H7 and H4-H6 move off not_run.
