---
name: gwaya-sealed-read
description: Use when registering a goal, freezing a policy, or reading a sealed evaluation set (E, G, any new one) in GWAYA - registration order, certificate arithmetic, one-read discipline, duplicate-labelling and flaky-label checks. Written from phase 3 (A33 to A33.4).
---

# Registration, freeze and the sealed read

Details and numbers: `LL.md` items 12 to 18, `docs/NEXT_STEPS_PHASE4.md` Section 2, `docs/ANALYSIS_PLAN_E_TPU.md` A33 to A33.4.

## 1. Before registering a certified goal, compute what it can certify
Fixed-sequence exact binomial at risk 0.10, delta 0.10, one threshold: accepted n -> max errors: 50:1, 88:4, 150:9, 250:18, 400:31, 600:50. Probability to certify with true error r (n=88/150/250/400): r=0.05: .55/.78/.95/.99; r=0.08: .16/.23/.37/.47; r=0.10: ~.06. So "precision 0.90" does not certify; about 0.95 at the target coverage, or a larger set, is needed. Rerun the 5-line scipy computation for any new risk/delta/n and put the table in the addendum.

## 2. Registration order (each step is a commit before the next)
1. Addendum: goal, data, stages, decision rules, stop rules, "not claimed". Implement EVERY rule in code now (phase 3 had to add rule (b) and `sealed` later).
2. Amendments before any number they govern (A33.1 slice change, A33.2 withhold V2 numbers until the test pool is fixed, A33.3 drop a model, A33.4 freeze).
3. Dev readout with the registered rule, unchanged. Nulls stay nulls ("not met" even if one interval is above zero).
4. Freeze the policy: named primary and what is reported beside it, fit on all dev data, calibrator on dev out-of-fold scores, threshold ladder in declared order. Commit code + freeze text + dev output BEFORE any label of the sealed set is produced.

## 3. The read
- Keep the sealed pool file read-only (`chmod 400`), score hidden labels only in the final stage, detached with `setsid nohup`, one writer, pid file. Never start a second run while `pgrep` shows the first.
- After the run: row counts must equal unique keys (`scored` rows vs candidates). If not, diff the copies (labels can be non-deterministic, e.g. one BigCodeBench sample flipped), rerun the frozen code on first-copy and last-copy dedupes, and DISCLOSE; do not re-fit anything.
- Commit the result file right after, then write the paper. No revision after the read; met or not met.

## 4. What is already spent (do not re-score or retune)
E (read once, A31), G for the 27B only (phase 3). G is unread for the 2B/4B/9B (they have no G generations).
