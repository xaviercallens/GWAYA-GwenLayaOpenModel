# Phase 3 plan (v2): a certified 0.90, a stronger verifier, a larger generator, and a staged RL program

Dated 2026-10-15. Proposal, not registered: each stage becomes an addendum (A33+) in `docs/ANALYSIS_PLAN_E_TPU.md`
with its decision rule committed before the run. No result exists for anything below. v1 of this file (commit 6820933)
is superseded; what changed: a literature review, the goal is restated as a certified selective-risk statement, a larger
Qwen3.5 tier is costed, ten hypotheses are scored and three selected, and reinforcement learning is laid out in phases.

## 1. Where we stand

| quantity (Qwen3.5-9B, Python) | value | source |
|---|---|---|
| greedy pass rate on fresh F (407 tasks) | 0.396 | A29/A30 |
| some candidate correct among greedy + 8 samples (T=0.8) | 0.617 | A30 |
| gate-verified samples that fail the hidden check | 29% (494 / 1,698) | A30 |
| frozen D3 policy on sealed E (seen, easy), frozen 0.85 threshold | 0.886 [0.850, 0.914] at coverage 0.73 | A31 |
| D3 on F python alone, precision >= 0.84 with Wilson LB >= 0.80 | met only at coverage 0.17 (0.915) | audit 3.12.0 |
| learned selector over the 9-candidate pool, F python | 0.884 at coverage 0.32 vs 0.846 at 0.30 | A32 |
| gate with 8 hidden tests instead of 1 on E (dial, not deployable) | 0.832 -> 0.941 precision, coverage 0.87 -> 0.77 | A17 |

The one-assertion visible gate is the weak link; the candidate pool already contains a correct answer for 62% of tasks;
every phase-2 lever that added learned scoring over the same signals was a null. New evidence per task is needed, and the
measurement of "0.90" must become a statement that survives a sealed read.

## 2. Literature review (what transfers to our setting, what does not)

Selection with generated tests. CodeT (Chen et al. 2022, arXiv 2207.10397) generates tests from the prompt with the
example cases removed, executes every candidate on every test, groups candidates into consensus sets by their pass
pattern and ranks a set by (number of candidates) x (number of tests passed); pass@1 rises by about 10 points on
HumanEval and MBPP for Codex models and 4 to 13 points for 6B-16B open models, and still by 4 to 9 points with only
10 tests per problem. Neither count alone works: ranking by candidate count only (self-consistency) or by tests only is
worse than the baseline on HumanEval (their Table 13). Test quality tracks generator quality: weak generators write
"toxic" tests that a wrong candidate passes and the reference fails. Transfer: our 9-candidate pools and a 9B test
writer are in the regime where CodeT's gain was smallest but still positive; the dual score, not pass count alone, is
what to implement.

Weighting unreliable tests. ACES (Sun et al. 2026, arXiv 2604.03922) weights each generated test by its leave-one-out
AUC against the ranking induced by the other tests, needs only the binary pass matrix, and beats majority voting and
CodeT on HumanEval+ (74.4 vs 69.5 vs 67.9 pass@1 with ~200 candidates and ~500 tests); with 100 tests it already
exceeds majority voting at 500. Transfer: cheap and label-free; our pools are small (9 candidates), so the closed-form
variant (ACES-C) is the realistic one and gains will be smaller than reported.

Reliable expected outputs. CoTT (Lee et al. 2026, arXiv 2609.12489) builds tests as (input, expected output) where the
expected output is the majority of 20 general solutions executed on the input, and only when that majority is weak
(< 0.8) asks the model for an input-specific program; with Llama-3.1-8B this improves reranking over always-inductive
and costs 15% less than always-transductive. Transfer: this is exactly how to turn our sample pool into expected outputs
without an oracle; the gating threshold 0.8 is a free parameter we fix in advance.

Oracle-free agreement. DiffCodeGen (He et al. 2026, arXiv 2605.20473) fuzzes inputs from one candidate under coverage
guidance, runs all candidates, clusters by behaviour and returns the medoid of the largest cluster, with no extra model
calls; it matches or beats majority voting with generated tests at about 4% of the tokens. Transfer: our
"behavioural agreement" arm can be built from the visible assert's inputs by type-aware mutation plus fuzzing, at zero
TPU cost, on the pools we already have.

When selection fails. Verification Trap (He et al. 2026, arXiv 2610.05170): when generator and test writer share a
mistaken premise, generated tests inherit the blind spot and the selector picks a verifier-compatible wrong candidate
although a correct one is in the pool; adding more tests or samples from the same loop recovers little (+1 point),
whereas a premise-agnostic auditor recovers +5 points. Transfer: our test writer must be decoupled from the generator
where possible (different tier, or a non-LLM channel such as fuzzing), and the final accept decision should not rest on
one evidence channel.

Abstention with guarantees. Task abstention for code (Zhou et al. 2026, arXiv 2605.17029) uses Learn-then-Test over
execution clusters of 64 samples on generated tests, filters tests that cause widespread failures or high output
entropy, and raises abstention precision by 26 points over the best baseline; rules calibrated on MBPP transfer to
HumanEval. PriceCheck (Lee et al. 2026, arXiv 2609.37493) prices each label-free check (agreement rate on correct and
on wrong answers, cost), composes them into cascades with early stopping, and certifies one schedule at a stated
selective-risk target; a scorer's ranking quality alone does not determine the error rate among served answers, and
generator confidence (mean log-prob) never certified at any target in their math setting. Certified by Abstention
(Balaji 2026, arXiv 2610.09541) gives the hard constraint for our budgets: a Clopper-Pearson selective certificate at
alpha = 0.10, delta = 0.10 and a 20-point grid needs at least 51 error-free accepted calibration points; with
calibration sets of 67 to 158 problems the standard certificate mostly abstains; a fixed-sequence procedure started at
that floor covers 2 to 8x more; and selecting best-of-n against the same verifier pushes the realised error above the
target while the empirical failure rate stays below delta. Transfer: (i) our Wilson-on-rank-curve readings are not
certificates; (ii) with ~400 calibration tasks a certificate at 0.90 is feasible only with a small, pre-declared family
of thresholds; (iii) the verifier that selects a candidate and the verifier that certifies acceptance must be
different evidence, or the certificate is calibrated on the wrong distribution.

Search and repair. ReMCTS (Wang et al. 2026, arXiv 2609.34717): program candidates as tree states, 8 iterations x
branch 5 with visible-test feedback, structured failure memory; improves over direct generation in 8 of 10
model-dataset pairs on open Qwen3 models, and budget-matched Best-of-32 or Reflexion-16 do not explain the gain; search
without any test ("proxy only") is positive on average but unstable. Transfer: a shallow tree with execution feedback
is worth one stage; a tree without executed evidence is not.

RL with verifiable rewards for small code models. GRPO training dynamics for 1.5B-7B (Ghosh et al. 2026, arXiv
2609.39321): gains are reliable on math and heterogeneous on code (17 of 36 configurations improve; a 4B regresses on
APPS until LoRA is restricted to a few layers and the correctness reward is up-weighted); reward and accuracy are only
partly aligned. RobustTests (Zhang et al. 2026, arXiv 2608.24135): synthesising tests from near-correct faulty
programs, validating inputs, clustering tests by their failure profile and using a stepwise dense reward lifts a 32B by
3 points on LiveCodeBench; about 10% of accepted synthetic tests remain invalid, which is why dense rewards beat sparse
ones. Also relevant: CATCH (arXiv 2609.39533) and hacktrace (arXiv 2610.03055) on reward hacking in coding RL;
PreciseCoder (arXiv 2610.18755) on what unit-test-only rewards teach. Transfer: RL is a phase after the verifier, not
before it, because the reward is the verifier; it needs dense rewards, a held-out test generator, and an explicit
abstention action; and on TPU the only RL stack for our family is Tunix for Qwen3 (not Qwen3.5) and MaxText for the
Qwen3.5-35B-A3B MoE, so the generator for an RL phase is chosen by tooling, not by preference.

Autonomous experiment loops. Karpathy's autoresearch (github.com/karpathy/autoresearch, March 2026): a fixed
evaluation file the agent never edits, one file it may edit, a human-edited program.md, a fixed 5-minute budget per
experiment, one scalar metric (val_bpb), keep or discard, a log of about 100 experiments overnight. The pattern, not the
code, is what we adopt in Section 5: a frozen scorer, one editable selector file, a fixed per-experiment budget, one
scalar, append-only log, and the human steering only through the plan file.

## 3. The goal, restated

Old (A30): precision >= 0.84 at the largest coverage whose Wilson lower bound is >= 0.80. The coverage is chosen with
the labels, there is no coverage floor, and the number is not a certificate.

New. For the 9B tier on fresh Python, a frozen selective policy S is "at 0.90" when all of:
1. Certified risk: on the sealed set G, a fixed-sequence Learn-then-Test over a family of at most five thresholds
   declared and committed before G is read certifies selective risk <= 0.10 at delta = 0.10 (exact binomial, in the
   declared order, stop at the first failure). The certified threshold's coverage on G is the headline number.
2. Coverage floor: that coverage is >= 0.40. (Today's equivalent on F python is 0.17.)
3. Separation of channels: the evidence used to pick a candidate (synthesised tests, agreement) and the evidence used
   to accept it (the scorer threshold) are computed from different generations or different tools, and the
   calibration rows are produced under the same selection rule as deployment (selection-conditional calibration).
4. Point estimates alongside: precision with Wilson 95% at the certified coverage, and the whole rank curve, so that
   a certificate that abstains is visible as such.
5. Against baselines, paired by task: gate-only, the frozen D3 policy of A31, and accept-everything; and the paired
   difference in correct accepted tasks at matched coverage with a cluster bootstrap CI.
Secondary readouts: the 4B tier under the same rule; coverage at the same certificate on an E-type (seen, easy) set,
target >= 0.60; cost in generated tokens per task relative to always-9B greedy. Each tier separately, no pooling.
Risk is 0.10 and not tighter because the certification floor at alpha = 0.10, delta = 0.10 with a 5-member family is
about 44 error-free accepted calibration points (log(5/0.1)/-log(0.9)), which 400 tasks at coverage 0.40 can supply;
at alpha = 0.05 the floor nearly doubles and the certificate would mostly abstain.

## 4. Ten hypotheses, scored, and the three selected

Scoring, each 0 to 3: G = expected gain toward the Section 3 goal (from the literature and our own A17/A29/A32
numbers); F = feasibility with our tooling and the TPU quota (16 v5e chips per region, no GPU); I = evidence
independence from the generator (Verification Trap); C = cost (3 = free, 0 = > $8). Score = G + F + I + C; ties by G.

| # | hypothesis | G | F | I | C | sum | note |
|---|---|---|---|---|---|---|---|
| H1 | Prompt-only synthesised tests (16 per task) with expected outputs from pool majority and transductive fallback (CoTT), tests weighted by leave-one-out consistency (ACES-C), dual CodeT score as scorer features | 3 | 3 | 2 | 2 | 10 | A17 dial says test count is the lever; one TPU stage ~$3 |
| H2 | Oracle-free behavioural clustering on fuzzed and mutated inputs (DiffCodeGen), medoid of the largest cluster, cluster share as a feature | 2 | 3 | 3 | 3 | 11 | no model call, runs on existing pools |
| H3 | Thinking mode at generation (2k reasoning budget) | 1 | 2 | 1 | 1 | 5 | 9B already truncates 21%; cost x2-3; kept only under a paired rule |
| H4 | Depth-2 repair tree with execution feedback and failure memory (ReMCTS-style, <= 12 generations) | 2 | 2 | 1 | 1 | 6 | gain documented on Qwen3 dense; coupled to the same generator |
| H5 | Larger generator: Qwen3.5-27B dense or Qwen3.5-35B-A3B on v5litepod-8, greedy + 8 samples on F | 3 | 1 | 2 | 1 | 7 | raises the base rate, the most direct lever on precision at fixed coverage; never loaded; see 4.1 |
| H6 | Certified selective risk (fixed-sequence LTT, declared family) replaces Wilson-on-rank-curve | 2 | 3 | 3 | 3 | 11 | changes what "0.90" means; required by Section 3 |
| H7 | Escalation routing on disagreement: accept at 4B when its pool agrees and passes the suite, else 9B | 1 | 3 | 2 | 3 | 9 | both pools exist on F; simulated offline; cuts cost, not precision |
| H8 | Offline accept policy by policy gradient (A27.1) | 0 | 3 | 2 | 3 | 8 | A32 says surface features saturate; low gain |
| H9 | RL on the generator with a verifiable dense reward and an abstention action (GRPO), staged | 2 | 1 | 1 | 0 | 4 | needs the verifier first and a new spend authorisation; Section 6 |
| H10 | Upstream task abstention from execution clusters (cluster ratio / semantic entropy, LTT) instead of prompt features | 1 | 3 | 2 | 3 | 9 | our U lever redone with executed evidence; needs > 9 samples to be sharp |

Selected (top three by score, with H6 as the measurement frame for all of them):
1. **Verifier V2 = H1 + H2** (one TPU stage for test synthesis, the rest offline). This is the only item with a
   documented effect of the needed size (A17; CodeT) and the one that attacks the 29% verified-but-wrong rate directly.
2. **H6, certified goal**, free, applied to every arm; without it the number is not comparable to the literature.
3. **H5, the larger generator** as the one generation-side experiment, because it moves the base rate, which bounds
   coverage at any precision (Chow's bound: coverage <= base rate / (1 - alpha)). H4 (repair tree) is the alternate if
   H5 fails its feasibility smoke; H3 rides inside H5 as an arm only if the 27B run budget allows a second pass.
H7 and H10 run inside the free autoresearch loop of Section 5 since both pools and all features already exist.

### 4.1 Feasibility and cost of a larger Qwen3.5 tier on TPU
- Candidates: Qwen3.5-27B (dense, Apache-2.0 per the model card) and Qwen3.5-35B-A3B (MoE, 3B active, 256 experts,
  hybrid Gated DeltaNet + attention), both on Hugging Face under Qwen/.
- Memory: 27B bf16 weights ~54 GB, 35B ~70 GB; a v5litepod-4 has 64 GB HBM, a v5litepod-8 128 GB. Both need
  v5litepod-8 (TP=8) with KV cache; the project quota is 16 v5e chips per region, so v5litepod-8 is allowed.
- Cost: 8 chips x 1.20 USD/h = 9.60 USD/h list. The A30 9B run (407 tasks x 8 samples) took 20 min on 4 chips; a 27B
  dense pass is roughly 3x the compute, so 30 to 45 min on 8 chips, about 5 to 7 USD; the MoE at 3B active should be
  faster per token but is memory-bound and untested on vllm-tpu.
- Risk: no Qwen3.5 > 9B has ever been loaded here; Qwen3.5 runs through the PyTorch fallback on tpu-inference with
  SKIP_JAX_PRECOMPILE=1 and had a 22-minute precompile once; TP=8 and the MoE path are untested. A 15-minute smoke
  (load + 8 prompts, hard cap, ~2.5 USD) precedes any run; failure of the smoke ends H5 for this phase.
- What a larger tier would and would not show: a higher base rate on F and therefore a higher coverage ceiling at
  0.90; it does not by itself fix the gate, and the D3 scorer would need refitting per tier (one logistic regression,
  grouped CV, as today).

## 5. The autoresearch loop for the offline part (free)
- Frozen: `scripts/phase3/eval_selector.py` (the metric), the pools (F python 2B/4B/9B x 9 candidates, synthesised
  tests once H1 has run, fuzzed inputs), the folds (5, grouped by cluster, hash as A26), and the sealed G.
- Editable: one file, `scripts/phase3/selector.py` (features, model class, threshold family, escalation rule).
- Metric, one scalar: certified coverage of the 9B tier on dev F under the Section 3 rule with the fold-internal
  calibration (fixed-sequence LTT at alpha 0.10, delta 0.10, family <= 5), averaged over folds; ties broken by the
  paired count of correct accepted tasks at coverage 0.40. Lower-is-better variants are not allowed: one number.
- Budget: 5 minutes CPU per experiment; keep if the metric rises by >= 0.01 coverage, else discard; every run appends
  one line (hash of selector.py, metric, coverage at 0.90 point estimate, 4B metric) to `results/.../phase3_loop.jsonl`.
- Human steering only through `docs/PHASE3_PROGRAM.md` (the program.md of this loop): allowed features, forbidden
  inputs (hidden tests, G, anything from E), and the order of ideas: H7, H10, H2 variants, ACES weighting variants.
- Stop: 60 experiments or no kept change in 15 consecutive runs. The loop never touches generation or spend.

## 6. Reinforcement learning, in phases (each gated; needs a new spend authorisation beyond the current ceiling)
R0 (free, inside the loop): the offline accept policy as a REINFORCE/linear policy with lambda = 4 (A27.1), reported
   against the logistic scorer at equal coverage; expected null, kept for completeness.
R1 (rejection-sampling fine-tune, ~3 USD): from the F pools, keep candidates that pass Verifier V2 (never the hidden
   check), fine-tune the generator on them; generator and tooling: Qwen3-4B through Tunix on one v5e (Tunix has no
   Qwen3.5 path; the A28 T2 LoRA probe on Qwen3.5 timed out), or Qwen3.5-4B QLoRA on one L4 if the GPU slot is free.
   Rule: pass rate on F python rises with paired CI > 0 AND verified-but-wrong rate does not rise by > 3 points.
R2 (GRPO with dense verifiable reward and abstention, ~6 to 10 USD): reward = stepwise pass fraction on a
   RobustTests-style suite (tests synthesised against near-correct faulty candidates from our own pools, validated,
   clustered by failure profile) plus agreement; an explicit CANNOT_SOLVE action with reward 0 and wrong answers at
   -lambda; group size 8, LoRA restricted to a few layers as the small-model study advises; test generator held out
   from the policy (no self-play) to avoid the co-training pathology; reward hacking monitored with a held-out
   hidden-test audit that never enters the reward. Rule: as R1 plus abstention precision >= 0.80.
R3 (only if R2 passes): a second GRPO round with the test generator refreshed from R2's failures, and the Section 3
   certificate re-run on G with selection-conditional calibration.
Why RL after the verifier and not before: every cited RL result for code is bounded by test quality (RobustTests,
CATCH, PreciseCoder); with a one-assertion gate the reward would teach the gate's blind spots.

## 7. Stages, budget and decision rules (current ceiling 47.76 USD, ledger 26.33, so 21.43 available, >= 3 reserve)

| stage | what | accel, est. cost | decision rule, fixed before the run |
|---|---|---|---|
| S0 | Sealed set G (300 to 400 fresh Python tasks: LiveCodeBench-style recent problems, FullStackBench items not in F, BigCodeBench-Hard remainder; built with the F builder, decontaminated against E, E', F, P); fuzzing and mutation tooling; H2 and H7 offline on existing pools; the loop of Section 5 opened | CPU, 0 | G sealed at commit; loop metric baseline recorded |
| S1 | Verifier V2 test synthesis: 9B thinking on, prompt only, 16 asserts per task on F and G, plus expected outputs by pool majority with transductive fallback (CoTT, gate 0.8) | v5litepod-4, ~3 USD | descriptive: tests per task, parse rate, share contradicting the visible assert, "toxic" rate measured on F only against the hidden check; stop if < 6 valid tests per task |
| S2 | H5 smoke: Qwen3.5-27B on v5litepod-8, load + 8 prompts, 15-minute cap | v5litepod-8, ~2.5 USD | pass = engine ready and 8 completions; else H5 ends and S3 switches to H4 |
| S3a | H5 run: 27B greedy + 8 samples on F python (and G) | v5litepod-8, ~6 USD | primary: certified coverage of 27B under Section 3 on dev F vs 9B; "lever" only if coverage rises by >= 0.10 with the paired CI above zero |
| S3b | (alternate) H4 repair tree on 9B, depth 2, <= 12 generations per task | v5litepod-4, ~7 USD | same rule against the 9B pool |
| S4 | Freeze selector + threshold family; read G once; certificate per tier | CPU, 0 | Section 3, all five conditions |

Total about 12 to 18 USD against 21.43 available. RL phases (Section 6) are outside this budget and wait for a new
authorisation. Every launcher dry-runs first; the user launches; scoring is local; each stage has a go/no-go against
the ledger.

## 8. Expected outcome, stated now
- The verifier stage is the one most likely to move the number; a plausible result is certified coverage 0.30 to 0.40
  at risk 0.10 on F for the 9B, i.e. the floor met at the edge or just missed; on E-type sets, met.
- The larger tier raises the base rate and the coverage ceiling, if it loads; whether the gate and scorer transfer to it
  is unknown, and one run cannot separate "bigger model" from "different error profile".
- Certified numbers will be lower than the Wilson-on-rank-curve numbers of A30/A31 by construction; that is the point.
- Nulls are reported as nulls; the goal above is fixed before S1 launches and is not revised after a read.

## 9. Risks and what is not claimed
- Synthesised tests are wrong in both directions; their toxic rate is measured on F against the hidden check (dev
  only) and reported as the verifier's own error rate. Tests the whole pool fails are kept as evidence, not dropped.
- Coupling: generator and test writer share a model family; H2's fuzzing channel and a different tier for test writing
  are the decoupling; a residual trap rate is expected and reported (recoverable mis-selection: tasks with a correct
  candidate in the pool where the selector picks a wrong one).
- Selection pressure: calibration rows are produced under the deployed selection rule; the certificate is about the
  deployed pipeline, not about the scorer in isolation.
- Budget and quota: a v5litepod-8 run that hangs in precompile costs 9.60 USD/h; the smoke's hard cap and the driver's
  watchdog bound that; no quota request will be made.
- Not claimed: anything about math, Rust or Lean; human preference; transfer beyond function-level Python; that G is
  free of pre-training exposure (decontamination covers our own sets only).
