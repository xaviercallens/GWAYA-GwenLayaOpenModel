# Phase 3 plan: from 0.85 to 0.90 precision with a stronger verifier, deep-thinking generation and disagreement routing

Dated 2026-10-15. Proposal, not yet registered: each stage becomes an addendum (A33+) in `docs/ANALYSIS_PLAN_E_TPU.md`
with its decision rule committed before the run. No result exists for anything below.

## 1. Where we stand and why the ceiling is where it is

| quantity (Qwen3.5-9B, Python) | value | source |
|---|---|---|
| greedy pass rate on fresh F (407) | 0.396 | A29/A30 |
| some candidate correct in 9 (greedy + 8 samples, T=0.8) | 0.617 | A30 |
| gate-verified samples that fail the hidden check | 29% (494 / 1,698) | A30 |
| frozen D3 policy on sealed E, frozen 0.85 threshold | precision 0.886 [0.850, 0.914] at coverage 0.73 | A31 |
| D3 on F python alone, A30 criterion | 0.915 at coverage 0.17 only | audit 3.12.0 |
| learned selector over the pool, F python | 0.884 at coverage 0.32 (B0 0.846 at 0.30) | A32 |
| gate with 8 hidden tests instead of 1, E (dial, not deployable) | precision 0.832 -> 0.941, coverage 0.87 -> 0.77 | A17 |

Three facts drive the plan. (i) The one-assertion visible gate is the weak link: 29% of VERIFIED samples are wrong, and the
A17 dial shows that eight tests remove most of that (the only effect of this size in the study). (ii) The candidate pool
already contains a correct answer for 62% of tasks; the selector, not the generator, is the bottleneck, and the A32
scorer with surface features does not find it. (iii) Every phase-2 lever that added learned scoring over the same signals
was a null. The next lever must add new *evidence per task*: more executed tests and more candidates that disagree on
them, plus a generator that can use that evidence.

"0.85 to 0.90" is made precise as: **precision >= 0.90 (point) with Wilson 95% lower bound >= 0.85, at coverage >= 0.40
on fresh Python (9B)**, against today's 0.17 coverage at that precision on F python. The E-type number (easy, seen set)
is a secondary readout: coverage >= 0.60 at the same precision. A coverage floor is part of the goal so that a tiny
accepted set cannot satisfy it.

## 2. Three workstreams

### W1. Post-evaluation: a test-time verifier that is stronger than one assertion
1. **Test synthesis from the prompt only (TS).** For each task the 9B (thinking on) writes 8 to 16 candidate asserts from
   the prompt and the signature, *before seeing any candidate*. Tests are kept only if they parse and reference the
   target function. This avoids the model confirming its own candidate's bugs. Candidates are then executed on the
   synthesized suite in the sandbox; the per-candidate signal is the pass vector, not a single verdict.
2. **Behavioural agreement (BA).** Candidates are clustered by their output vectors on the synthesized inputs (and on
   generated probe inputs where no expected value exists: agreement needs no oracle). Cluster size and cross-tier
   agreement become features; the AlphaCode/CodeT idea, which we never ran (A29 "S2" never executed).
3. **Scorer D4.** Stacked logistic regression as D3, plus: pass fraction on TS, size of the candidate's BA cluster,
   number of distinct clusters, whether the visible assert and the TS suite agree, repair depth (W2). Threshold by the
   same Wilson-LB rule on grouped CV, and, as a new addition, a **risk-controlling threshold** (RCPS / learn-then-test)
   that gives a distribution-free guarantee on precision at the chosen coverage instead of a point estimate.
4. **Guard.** Hidden tests never select, never train, never appear in any prompt. TS tests that contradict the visible
   assert are dropped; TS tests that no candidate passes are kept (they are evidence of failure, not of noise) but a
   suite in which zero tests are passed by anyone is flagged for the audit.

### W2. Generation: deep thinking and a repair tree
1. **Thinking mode (TM).** Qwen3.5 thinking on, reasoning budget 2k tokens, answer cap 1,024 (today 21% of 9B
   generations are cut at 1,024 with thinking off; the budget must be measured, not assumed). Greedy TM vs greedy
   non-thinking, paired on F python, McNemar and paired bootstrap: TM is kept only if pass rate rises with CI above zero
   AND the confident-wrong rate (VERIFIED-but-wrong) does not rise.
2. **Tree of repair (TR).** Depth-2 search per task: root = 4 TM samples; each sample is run on the visible assert and
   the TS suite; a failing node is expanded once by a repair prompt that shows the code, the failing test and the actual
   value (the loop already in `gwaya/assistant.py`); nodes are selected by TS pass fraction then log-prob. Budget: at most
   12 generations per task. Hidden tests are never in the loop. Output: the best leaf plus the whole tree for the scorer.
3. **Why a tree and not longer thinking alone.** Thinking improves the first draft; the tree lets execution evidence
   steer the second. A29 showed blind sampling adds wrong-but-verified candidates; expansion only from executed
   failures adds targeted candidates instead.

### W3. Routing: escalate on disagreement, not on prompt features
1. Current routers (Laya, prompt-only U) are about equal to a hand logistic regression and the 2B->4B->9B cascade never
   paid off (cost ratio 1.74 vs always-9B). The new policy is **evidence-based escalation**: run the 4B tree first; if its
   candidates form one BA cluster that passes the TS suite, accept at 4B; otherwise escalate to the 9B tree. Routing
   decisions use only executed evidence and cost, no learned prompt features.
2. Metrics: accuracy and precision of the escalation policy at its coverage vs always-9B TR, and cost in generated tokens
   per task (the paper's cost ratio). Success = same precision at >= 0.95 x the always-9B coverage and <= 0.70 x its cost.
3. The Laya router is kept as a baseline arm only; no further Laya training (A28 said feedback retrofitting is a null).

## 3. Stages, budget and decision rules

| stage | what | accelerator, est. cost (list) | decision rule (fixed before the run) |
|---|---|---|---|
| S0 | Fresh sealed set **G**: 300 to 400 Python tasks from sources not yet used or post-cutoff (LiveCodeBench-style recent problems, FullStackBench items not in F, BigCodeBench-Hard remainder), built with the F builder and decontaminated against E, E', F, P. Also the TS/BA sandbox tooling and a CPU dry run on 20 F tasks. | CPU, $0 | G is sealed from the moment its task file is committed; read once by the final stage. |
| S1 | TS suites for F and G (9B thinking on, prompt only). | v5litepod-4, ~$3 | Descriptive: tests per task, parse rate, share that contradict the visible assert. Stop if < 4 valid tests per task on average. |
| S2 | TM greedy vs non-thinking greedy on F python (9B, 4B). | v5litepod-4, ~$3 | TM kept only if paired pass-rate gain CI > 0 and confident-wrong rate not higher. Else TR uses non-thinking samples. |
| S3 | TR (4 roots x 1 repair, <= 12 gens/task) on F python, 4B and 9B; execute TS/BA locally. | v5litepod-4, ~$7 | Primary dev readout: D4 at grouped-CV threshold reaches 0.90 / LB 0.85 at coverage >= 0.40 on F python 9B. Secondary: W3 escalation cost rule. |
| S4 | Freeze D4 + escalation policy; read G once. | CPU, $0 | Same rule on G. "Met" needs both F (dev, CV) and G (sealed). |

Total about $13 at list price against $21.43 available under the $47.76 ceiling, leaving the >= $3 reserve. Every launcher
dry-runs first; the user launches; scoring is local. Each stage gets a go/no-go against the ledger.

## 4. Expected outcome, stated now
- S1 and S3 are the stages most likely to move the number: A17's dial says test count is the lever, and BA is the
  cheapest new evidence. A plausible outcome is coverage 0.30 to 0.40 at 0.90 on F for the 9B, i.e. the goal met at the
  edge or just missed; on E-type sets, met.
- S2 (thinking mode) may be a null for pass rate on BigCodeBench-style tasks and will raise cost; it is kept only under its
  rule.
- W3 is expected to cut cost, not to raise precision.
- Nulls will be reported as nulls; the post hoc history of the 0.84 goal (A30) is not repeated: the goal here is fixed
  before S1 launches.

## 5. Risks and what is not claimed
- Synthesized tests can be wrong; a wrong test that every candidate fails hurts coverage, a wrong test that a wrong
  candidate passes hurts precision. The share of TS tests contradicted by the hidden check is measured on F (dev only) and
  reported as the verifier's own error rate.
- Reward hacking: candidates may special-case visible inputs. BA on generated probe inputs is the guard; the rate of
  VERIFIED-on-TS-but-hidden-fail is the headline cost, as in A29.
- Thinking traces are not used as features (they can leak test text and are not calibrated evidence).
- 407 + ~350 tasks: intervals of about +/- 0.05 at coverage 0.4; single seed; Python only; no claim for math, Rust, Lean.
- G may still overlap pre-training data of the generators; decontamination covers our own sets only.
- Budget: a 9B tree at 12 generations per task is ~1.5x the A30 sampling run per stage; if S1 shows thinking mode doubles
  token counts, S3 drops to 8 generations per task before it drops the tree.
