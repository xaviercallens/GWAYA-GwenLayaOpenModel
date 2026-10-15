# Phase 3 plan (v3): a certified 0.90, a stronger verifier, Qwen3.5-35B-A3B and 27B at spot cost, a Laya track, and a staged RL program

Dated 2026-10-15. Proposal, not registered: each stage becomes an addendum (A33+) in `docs/ANALYSIS_PLAN_E_TPU.md`
with its decision rule committed before the run. No result exists for anything below. v1 (commit 6820933) and v2
(f1c73c5) are superseded. v3 adds, at the user's request: both Qwen3.5-35B-A3B and Qwen3.5-27B as target tiers with the
real tooling status (Section 4.1); a quantization and FinOps section (TurboQuant and what is actually available on TPU,
int8 weights, fp8 KV, spot and Flex-start pricing, zone ladders, compile caches; Section 4.2); and a Laya track
(Section 4.3): the 20x data scale-up that was planned as A23 and never run, hidden-state probes, and multicalibration.
"Reach the next level" is read as: the Laya calibrator must become measurably better than the hand logistic regression
it has tied with since 3.6.0, under the Section 3 certificate.

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

Learned calibrators and the Laya question. Three results bear on why Laya v1 ties with a logistic regression on
gate and log-prob features. (i) Hidden states beat text. On LiveCodeBench, a linear probe on Qwen3-4B's hidden state at
the final prompt token, read before any output token, predicts first-attempt correctness with AUC 0.881 (0.842 after
regressing out prompt length; prompt length alone 0.657) from 444 tasks (Di Cicco 2026, arXiv 2606.14530). After
generation, a contrastive correctness direction in the hidden states (representation engineering, "LAT") beats
log-likelihood and verbalised confidence by 13 to 29 points of selection accuracy on BigCodeBench for four 7-8B models
and lifts pass@1 by 21% on HumanEval and 51% on BigCodeBench when used to rank 10 samples, fitted from only 10% of the
tasks (Ribeiro et al. 2025, arXiv 2512.07404). Certified by Abstention (above) found the same: an unreadable residual
probe bought 2 to 3x the certified coverage of log-prob and surface features. Laya reads the prompt and candidate as
text with a separate encoder; none of these signals are available to it. (ii) Group-wise calibration. Multicalibration
over groups defined by language, length and complexity raises Brier skill score over Platt or histogram binning on
LiveCodeBench, MultiPL-E and McEval for Qwen3-Coder-30B-A3B and GPT-OSS-20B (Campos et al. 2025, arXiv 2512.08810);
whole-output mean log-prob, including the reasoning trace, was the best raw score. Our dev table carries source
benchmark, tier and length for free. (iii) Data scale and label quality. Laya v1 saw 322 Python training rows; its
Python AUROC CI includes zero against the regression. The verification-ceiling study (Gureja et al. 2025, arXiv
2509.20837) shows that for training on synthetic code the richness of the tests matters more than their number (two
tests beat one by 3 points, more than two hurts by filtering out hard problems), that relaxed pass thresholds of 0.6 to
0.8 help only with rich tests, and that human review found substantial blind spots in synthetic tests. Transfer: the
A23 pool O (OpenCodeInstruct, re-verified in our sandbox) is the right source for scaling Laya's labels, its asserts
are the weak kind, and labels must come from execution on our checks, not from the source's pass flags.

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
| H11 | Laya data scale x20 on pool O (A23), same recipe, with the regression as competitor | 1 | 3 | 2 | 2 | 8 | cheapest untested factor for Laya; one TPU stage at about 1 to 2 USD |
| H12 | Hidden-state probe (prompt-final and candidate-mean) as scorer input | 2 | 2 | 3 | 3 | 10 | 0.88 AUC pre-generation on LCB for Qwen3-4B; needs Transformers passes, not vLLM |
| H13 | Multicalibration over source, tier, length, gate groups | 1 | 3 | 3 | 3 | 10 | free, inside the loop; measured by certified coverage |
| H14 | Int8 weights + spot + compile cache for every generation stage | 0 | 2 | 3 | 3 | 8 | no precision gain; halves the slice and cuts the bill 3 to 4x; gated by the A25 accuracy rule |

Selected (top three by score, with H6 as the measurement frame for all of them):
1. **Verifier V2 = H1 + H2** (one TPU stage for test synthesis, the rest offline). This is the only item with a
   documented effect of the needed size (A17; CodeT) and the one that attacks the 29% verified-but-wrong rate directly.
2. **H6, certified goal**, free, applied to every arm; without it the number is not comparable to the literature.
3. **H5, the larger generators (35B-A3B first, then 27B), in int8 on 4 chips at spot (H14)**, because the base rate
   bounds coverage at any precision (coverage <= base rate / (1 - alpha)). H4 (repair tree) is the alternate if both
   smokes fail; H3 rides inside H5 as an arm only if the run budget allows a second pass.
The Laya track (H11, H12, H13) runs in parallel because H11 needs one cheap TPU stage and H12/H13 are offline; it is
judged by the same certificate. H7 and H10 run inside the free autoresearch loop of Section 5 since both pools and all
features already exist.

### 4.1 Target tiers: Qwen3.5-35B-A3B and Qwen3.5-27B, and what the TPU stack supports today
- Models: Qwen3.5-35B-A3B (MoE, 35B total, about 3B active per token, 8 of 256 routed experts plus 1 shared, 40-layer
  hybrid Gated DeltaNet + attention) and Qwen3.5-27B (dense, with a vision encoder we do not use); both Qwen/ on Hugging
  Face. **Excluded by the user (2026-10-15): Qwen3.5-397B-A17B and Qwen3.5-122B-A10B.** The 397B needs about 800 GB
  of bf16 weights (a v5litepod-64 or larger, above the 16-chip quota and at 76.80 USD/h on-demand), and a 397B
  generator defeats the point of a GwenLaya combination, which is to make small tiers trustworthy. It is mentioned
  below only because the TPU stack lists it as a tested model, which is evidence about the Qwen3.5 MoE code path, not a
  candidate. The 122B (about 244 GB bf16, v5litepod-16 at the quota edge) is excluded for the same reason.
- Tooling status (tpu-inference README, read 2026-10-15): tested models include Qwen3-4B, Qwen3-30B-A3B, Qwen3-32B,
  Qwen3-Coder-480B-A35B and Qwen3.5-397B-A17B; Qwen3.5-9B is listed as untested; Qwen3.5-35B-A3B and Qwen3.5-27B are
  not listed at all. Our own runs show Qwen3.5-2B/4B/9B work through the PyTorch (torchax) path with
  SKIP_JAX_PRECOMPILE=1 and max_num_batched_tokens 1024. The fused-MoE kernel row in the package's support table is
  unconfirmed for every quantization format. So: the 35B-A3B's architecture family (Qwen3.5 MoE) has a tested 397B
  sibling (excluded as a tier, see above), which is encouraging for the model code path, and nothing is known about its performance on v5e; the 27B is
  the 9B's architecture scaled, which we know runs.
- Memory and slice. bf16 weights: 27B about 54 GB, 35B about 70 GB. A v5litepod-4 has 64 GB HBM, a v5litepod-8 128 GB
  (quota: 16 chips per region, so both are allowed). In bf16 both need v5litepod-8. With int8 weights (W8A8
  compressed-tensors, listed for v5/v6 in the support table, loading-compatibility only), 27B is about 27 GB and 35B
  about 35 GB, and both fit a v5litepod-4 with room for the KV cache at our 1.5k-token contexts. That halves the
  hourly rate (4 x 1.20 = 4.80 USD/h against 9.60) and is the main cost lever (Section 4.2).
- Decode cost. The MoE activates about 3B parameters per token, so its per-token compute is 9B-class or lower while
  its memory traffic is 35B-class; on a memory-bound v5e the MoE should land between the 9B and the 27B in tokens per
  chip-second, and is the better first try. The 27B is about 3x the 9B per token: the A30 9B pass (407 x 8 samples)
  took 20 minutes on 4 chips, so a 27B pass is 45 to 60 minutes on 4 chips in int8, or 30 to 45 on 8 in bf16.
- Order of trials: (1) 35B-A3B int8 on v5litepod-4; (2) 27B int8 on v5litepod-4; (3) only if int8 loading fails,
  bf16 on v5litepod-8. Each trial is a 15-minute smoke (load, compile, 8 prompts, hard cap); the first that passes
  gets the S3 run. A quantized arm must pass the A25 accuracy rule (paired pass-rate lower bound above -2 points
  against bf16 on a 120-task subset) before any number from it enters a table.
- What a larger tier shows: a higher base rate on F and so a higher coverage ceiling at risk 0.10 (coverage <=
  base rate / 0.9); it does not fix the gate; the scorer is refit per tier (one logistic regression, grouped CV).

### 4.2 Quantization and FinOps: paying less for the same evidence
Quantization ladder, from what exists to what does not:
- Weights int8 (W8A8 compressed-tensors) is the one that changes the bill: it moves both target tiers from 8 chips to
  4. Int4 AWQ (W4A16) is also listed for v5; it would put the 27B on a single v5e chip (about 14 GB) if the kernels
  are real, which the table does not promise. Both are "checkpoint loads" claims only; throughput and accuracy are
  ours to measure, under the A25 rule.
- KV cache fp8 via `kv_cache_dtype` is exercised by tpu-inference's own e2e tests (A25 established this on
  2026-10-13; the A25 fp8-vs-bf16 arm was planned and never run). At our contexts (prompt about 400 tokens, 1,024 new)
  the KV cache is small next to the weights, so fp8 KV buys batch size, not a smaller slice; it matters for the 8-sample
  and tree stages where max_num_seqs is the throughput limit (the 4B failed to start at 96 sequences for HBM reasons).
- TurboQuant (Zandieh et al. 2025, arXiv 2504.19874): data-oblivious vector quantization of the KV cache by random
  rotation plus per-coordinate optimal scalar quantization, with a 1-bit QJL residual for unbiased inner products;
  quality-neutral at 3.5 bits per channel and marginally degraded at 2.5 bits on LongBench with Llama-3.1-8B, over
  4.5x KV compression, no calibration data. It is not in tpu-inference 0.31.0, not in mainline llama.cpp, and is
  reported only in GPU vLLM forks (A25 checked the wheel). Its benefit is long-context decoding, which is not our
  regime. Decision: not pursued on TPU in phase 3; revisited only if a stage needs > 8k-token contexts (the repair tree
  with long feedback could), and then as a GPU-side experiment. Related 2026 work (RoPE-aware bit allocation, WUSH-KV,
  query-adaptive KV quantization) is noted for the same reason and not planned.
- Mixed-precision MoE weight quantization (per-expert bit allocation, e.g. arXiv 2610.20353, PagedWeight 2607.16184)
  is the relevant line for the 35B-A3B if int8 does not fit a future smaller slice; out of scope until the model runs.

Pricing and provisioning (v5e, per chip-hour, list): on-demand 1.20 USD in us-central1/us-east5/us-south1 and 1.56
in europe-west4; spot 0.4942 as read from the billing catalogue on 2026-10-08 (`deploy/tpu/drive.sh`, about 59%
off; a third-party aggregator quotes lower spot prices in other regions, unverified); DWS Flex-start 0.60 (50% off,
queued start, not preempted once running). Our spot experience: no capacity in us-central1-a and us-east5-b, a
preemption after 4 minutes in us-west4-a; on-demand in us-south1-a worked first time. The project can create
v5litepod-4 and v5litepod-8 in more than twenty zones (asia-east1, asia-northeast1, asia-southeast1, europe-north1,
europe-west1/3/4, us-central1, us-east1/5, us-south1, us-west1/4), so capacity, not quota, is the constraint.
Rules for phase 3 runs:
1. Spot first with a zone ladder of at least six US zones, then Flex-start, then on-demand; the launcher's
   `--zones` list and a `--provisioning` switch (spot | flex-start | on-demand) with the price table per mode.
2. Resumable by construction: generation writes one task at a time (already the case), downloads partial results on
   preemption, and resumes from the last complete task in the next zone; the stage is worth its spot discount only if
   a preemption costs < 10 minutes of lost work, which the chunking gives.
3. Compile cache on the bucket: persist the vLLM/XLA compilation cache under gs://.../gwenlaya_v4/xla_cache/<model>/
   and restore it at start; the 9B spent 262 s of a 961 s run in "engine ready", the 4B once 22 minutes. Saves 15 to
   30% on short runs.
4. Prefix sharing: the 8 samples of a task are one request with n=8, not eight requests, so the prompt is prefilled
   once; keep max_num_seqs on the ladder (32, 16, 8) and measure tokens per chip-second per arm as A25 asked.
5. Hard caps and the watchdog stay (`--max-minutes`, delete on exit, a 15-minute cap on every smoke); a run that
   hangs in precompile on 8 chips costs 9.60 USD/h and the cap bounds it.
6. Ledger per mode: the ledger records the provisioning mode and the price used, so the paper's spend table can
   state list on-demand, spot, and the mix.
Expected effect on the stage table (Section 7): int8 on 4 chips plus spot turns the 27B pass from about 6 USD to about
1.5 to 2 USD, and the whole non-RL plan from 12 to 18 USD on-demand to 6 to 9 USD at spot; the risk is time, not
money, when capacity is scarce.

### 4.3 The Laya track: data scale, hidden-state probes, multicalibration
Why a separate track. Since revision 3.6.0 the Laya calibrator has been indistinguishable from a logistic regression
on gate and log-prob features (AUROC difference -0.004 [-0.015, +0.006] on F), and on Python alone below it on every
tier; A28 showed feedback retrofitting does not help; A22 showed a different text layout does not help. Three
untried levers remain, two of them nearly free.
L1. Data scale (A23, planned 2026-10-13, never run). Pool O was probed (399 rows scanned, 294 kept, 74% keep rate,
    drops: too few passing tests 66, prompt too long 31) and never built. Build the 6,000-task deterministic subset
    (about 8,200 rows scanned), generate greedy answers for 2B/4B/9B on TPU (18,000 generations; E's 1,536 x 3 took
    27 minutes on one chip, so about 1.7 hours on one chip, 2 USD on-demand or about 0.9 at spot), score locally
    (about 15 minutes with 8 processes), and train v3 exactly as A23 says: same recipe, same layout, Python head on
    P_train + O_train, isotonic on P_calib + O_calib, plus the learning curve at 500 / 1,000 / 2,000 / 4,000 / all.
    The honest competitor is the same logistic regression trained on the same larger pool. Labels are our executed
    checks; the source's pass flags are never used. Decision rule (A23, unchanged): v3 beats v1 and the regression on F
    python AUROC with the paired cluster-bootstrap CI above zero; otherwise "data scale is not the lever".
L2. Hidden-state probe as a Laya input. Two readouts: (a) the prompt-final hidden state of the generator before any
    output token (the routing signal, "will this tier solve it"), and (b) the mean hidden state over the candidate's
    tokens after generation (the acceptance signal), each at a layer chosen by nested CV in the upper third of the
    stack, through standardise, PCA 95%, L2 logistic regression, as in the cited work. Feasibility: vLLM-TPU does not
    expose hidden states, so the forward passes run with Transformers: prompt-only passes for 407 F tasks on the 2B
    and 4B are cheap on CPU; the 9B and the candidate-conditioned pass are the TPU VM's job in the same session as a
    generation stage (torch_xla forward only, no training) or an L4 if the slot is free. The probe is first compared on
    F (grouped CV) against the regression and against Laya v1; if it wins, it becomes a feature of the stacked scorer
    D4, and "Laya v4" is the stack, not a bigger text encoder. Decision rule: probe AUROC on F python exceeds the
    regression's by >= 0.03 with paired CI above zero, and the certified coverage of Section 3 rises on dev.
L3. Multicalibration. Replace the single isotonic map by group-conditional calibration (iterative grouped linear
    binning or group-conditional unbiased regression) over groups available at test time: source benchmark, tier,
    prompt-length tercile, candidate-length tercile, gate verdict. Free; runs in the Section 5 loop; judged by the
    group-average squared calibration error and by certified coverage, never by ECE alone (a base-rate predictor has
    low ECE).
L4. Features that the literature says help and Laya lacks: whole-output mean log-prob including any reasoning trace
    (when thinking is on), token-level log-prob dispersion, and the executed-evidence features of Verifier V2. These
    go into D4 regardless of L1 to L3.
Not planned: a larger text encoder, more epochs, or another layout; the evidence says the text channel is not where
the information is. The Laya name is kept for the stacked router-and-calibrator; its inputs change.
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

Costs are list on-demand / spot (0.4942 per chip-hour) and assume int8 on v5litepod-4 where a tier allows it.

| stage | what | accel, est. cost on-demand / spot | decision rule, fixed before the run |
|---|---|---|---|
| S0 | Sealed set G (300 to 400 fresh Python tasks: LiveCodeBench-style recent problems, FullStackBench items not in F, BigCodeBench-Hard remainder; F builder, decontaminated against E, E', F, P); fuzzing and mutation tooling; H2 and H7 offline on existing pools; loop opened; pool O built to 6,000 tasks (CPU) | CPU, 0 | G sealed at commit; loop baseline recorded; pool O report committed |
| S1 | Verifier V2 test synthesis: 9B thinking on, prompt only, 16 asserts per task on F and G, expected outputs by pool majority with transductive fallback (gate 0.8) | v5litepod-4, 3 / 1.3 USD | descriptive: tests per task, parse rate, contradiction and toxic rates on F (dev only); stop if < 6 valid tests per task |
| S2a | Smoke: Qwen3.5-35B-A3B int8, v5litepod-4, 15-minute cap | 1.2 / 0.5 USD | pass = engine ready and 8 completions; A25 accuracy subset follows in S3 |
| S2b | Smoke: Qwen3.5-27B int8, v5litepod-4, 15-minute cap (bf16 on v5litepod-8 only if int8 fails to load) | 1.2 / 0.5 USD (2.4 / 1 on 8 chips) | same |
| S3a | Run: first passing tier, greedy + 8 samples on F python and G, plus the 120-task bf16 accuracy subset | v5litepod-4, 4 to 5 / 1.7 to 2 USD (27B dense is the upper end) | primary: certified coverage under Section 3 on dev F vs 9B; lever only if coverage rises by >= 0.10 with paired CI above zero; A25 rule for the quantized arm |
| S3b | (alternate) H4 repair tree on 9B, depth 2, <= 12 generations per task | v5litepod-4, 7 / 3 USD | same rule against the 9B pool |
| S4 | Laya track: greedy 2B/4B/9B on the 6,000 pool O tasks (18,000 generations), scored locally; Laya v3 and the regression trained on P + O; learning curve; hidden-state probe passes for the 9B in the same session | one v5e chip, 2 + 1 / 0.9 + 0.5 USD | A23 rule for v3; L2 rule for the probe |
| S5 | Freeze D4, threshold family, escalation rule; read G once; certificate per tier | CPU, 0 | Section 3, all five conditions |

Total about 13 to 16 USD on-demand, 7 to 9 USD at spot, against 21.43 available; the RL phases (Section 6) wait for a
new authorisation. Every launcher dry-runs first; the user launches; scoring is local; each stage has a go/no-go
against the ledger, and a spot stage that finds no capacity in its zone ladder within 30 minutes falls back to
Flex-start, then on-demand, with the user told before the mode changes.

## 8. Expected outcome, stated now
- The verifier stage is the one most likely to move the number; a plausible result is certified coverage 0.30 to 0.40
  at risk 0.10 on F for the 9B, i.e. the floor met at the edge or just missed; on E-type sets, met.
- The larger tier raises the base rate and the coverage ceiling, if it loads; whether the gate and scorer transfer to it
  is unknown, and one run cannot separate "bigger model" from "different error profile". The 35B-A3B is the more
  likely to run (tested MoE siblings) and the 27B the more likely to be strong per token; neither has run anywhere here.
- On the Laya track, data scale alone is the expected null (the text channel ties the regression at 322 rows and the
  reason is probably the channel, not the count); the hidden-state probe is the item with a documented effect of the
  needed size, and multicalibration should move certified coverage by a few points at most.
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
