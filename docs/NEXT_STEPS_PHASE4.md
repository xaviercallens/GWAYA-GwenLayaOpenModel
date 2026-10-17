# Next steps and research directions after phase 3 (written 2026-10-17)

Status at writing: branch `phase3/exec`, paper revision 3.13.0 draft (commit `4ee58be`), ledger 41.75 USD of the 50 USD cap, the user
ordered no further spend. Everything below is a proposal; each item that runs needs a rule written before it runs (as A23-A33 did)
and, if it spends, the user's authorisation. Nothing here is a result.

## 1. What phase 3 established, in one paragraph

On fresh Python the best policies we have accept the top 40% of tasks at precision about 0.88 (27B, sealed G, 88 tasks, Wilson
[0.79, 0.94]) against 0.64 for accepting everything; the visible gate alone accepts 78% at 0.81. Verifier V2 (model-written tests
executed on every candidate) did not beat the greedy D3 baseline for any tier, a 27B did not beat the 9B on the certified readout,
and nothing certifies at risk 0.10, delta 0.10. The phase-2 sealed E result (9B, precision 0.886 at coverage 0.73, previously seen
set) remains the strongest positive finding and is not strengthened or contradicted by phase 3.

## 2. The certificate is stricter than the goal sounds (computed 2026-10-17, exact binomial)

To certify selective risk <= 0.10 at delta 0.10 with one threshold, the accepted set must contain at most this many errors:

| accepted n | 50 | 88 | 150 | 250 | 400 | 600 |
|---|---|---|---|---|---|---|
| max errors | 1 | 4 | 9 | 18 | 31 | 50 |
| observed error needed | 0.020 | 0.045 | 0.060 | 0.072 | 0.077 | 0.083 |

Probability of certifying when the true error rate among accepted tasks is r (n = 88, 150, 250, 400): r = 0.05: 0.55, 0.78, 0.95, 0.99;
r = 0.08: 0.16, 0.23, 0.37, 0.47; r = 0.10: about 0.06; r = 0.12: about 0.01. So "precision 0.92" would still fail most of the time at
these sizes, and our observed 0.875 at 88 accepted could never certify. The goal needs a true precision of about 0.95 at coverage 0.4, or
a much larger sealed set. The gap is therefore real, not only a small-sample artefact. Decide, before the next sealed set is opened,
whether the goal stays "risk <= 0.10 certified" (hard, needs a different kind of evidence) or becomes a different, declared statement
(for example a one-sided Wilson bound at a stated coverage, or risk 0.15), and say so in an addendum first. Changing it after a read
would be post hoc, as A30 was.

## 3. Directions, ordered by expected value per unit of risk

**A. Better evidence per task, not more selectors (highest value).** Every phase-2 and phase-3 lever that added learned scoring over the same signals has now been null
(A26, A28 to A30, A32, A33). What is untested is evidence that is independent of the generator:
1. Tests written by a *different* model than the one that generated the candidates, for every tier including the largest (phase 3
   could only decouple the 2B/4B/9B; the 27B wrote its own tests). Cheap on F: 27B-generated candidates, 9B-written tests.
2. Behavioural agreement by execution on shared probe inputs across candidates (listed untested in A32; V2 only used model-written
   asserts' inputs). Probe inputs from a property-based generator need no model.
3. A stronger visible gate (more assertions per task) before any further best-of-k (3.12.0 follow-up; the gate drives most of the signal).
4. Measure the contradiction rate of synthesized tests against the hidden check on F (planned in A33, not run; local, free).

**B. Make the evaluation powerful enough to see a real effect (highest value, cheapest).** The dev certificate had about 81 calibration
tasks per fold and certified nothing for any baseline, so rule (a) could hardly be met. Options, each declared before use: a larger
development set (F has 407 Python tasks; G or a new set could double it), a single pre-declared threshold instead of a five-step ladder,
or a calibration/test split that keeps more tasks in calibration. Section 2 above gives the sizes.

**C. Fill the gaps phase 3 left (needs budget; about 3-4 USD estimated, scale from the F sampling runs: 9B 8 samples on 407 tasks
took 1,192 s on a v5litepod-4).** G generations (greedy + 8 samples) for the 2B, 4B and 9B so the sealed readout covers every tier, at
equal candidate counts with the 27B (4 samples; the equal-count reporting was not run). Only G is unread for these tiers, but the
sealed policy is already frozen for the 27B, so a new sealed read for the other tiers needs its own freeze addendum first. The ledger
has 8.25 USD under the cap and 5.25 USD under the A33 ceiling, so this fits only after the ceiling is deliberately kept or raised.

**D. The 35B-A3B tier.** Not measured. The AWQ community checkpoint fails at TP=4 (weight shape not aligned); untested options: TP=2 with
two data-parallel replicas, a GPTQ-Int4 checkpoint, an 8-chip slice if a quota can ever be obtained (the user ruled out quota requests,
so this is parked), or a GPU path. Results from a quantized community checkpoint would not speak to the bf16 model.

**E. Laya v3 (T7 of A33, A23 rule).** Not run; pool-O greedy outputs exist for 2B/4B/9B (6,000 tasks each; not yet scored, scoring is local and free).
Training needs one v5litepod-1 run (the v2 run cost 1.46 USD). Rule (d) of A33 is the A23 rule, unchanged.

**F. Phase-2 positive finding: replicate on a fresh set.** The E result was read once on a previously seen set. The cleanest next
step with the strongest prior is to score the frozen D3 `lr|core` and the hand rule gate-then-log-prob on a *fresh* Python set with a
paired test, tier by tier, with the goal stated in Section 2's terms. G is spent for the 27B only; for the 2B/4B/9B it is unread.

## 4. Process fixes (cheap, do first)

- Find why two identical T5 VMs ran in parallel on 2026-10-16 (about 5 USD wasted): add a lock file or a check for a live driver of
  the same stage name in `scripts/tpu/launch_*.sh` before it creates a slice.
- `drive.sh` now has SSH keepalives (`805bfc5`). Also poll the remote status file for the end marker instead of waiting on the SSH
  session, and run the ledger booking from the real creation time, not the timestamp in the VM name (the 200546 line booked the
  full 150 minutes).
- Detach every long local job with `setsid nohup` (a plain background job died after one row on 2026-10-17).
- Independent audit of the phase-3 numbers (as for 3.10.0 and 3.12.0): recompute V2 dev, the sealed table and the spend from the raw
  files, then tag 3.13.0. The phase-3 section is a draft until then. Also update the paper abstract, hypothesis status, README
  status paragraph, and fix the stale `pyproject` version (3.4.0).
- Read the billed TPU amount (the ledger is a list-price estimate); check whether `gwenlaya-tpu-1` (not ours) is billing.

## 5. What not to do

- Do not re-score or retune on G for the 27B, or on E; both are read.
- Do not add another learned selector over the same signals without new evidence (Section 3A): every such lever so far was null.
- Do not state a goal of "precision 0.90" for a certified claim without Section 2's arithmetic.
- Do not claim anything about the 35B-A3B or about the 2B/4B/9B on G.
