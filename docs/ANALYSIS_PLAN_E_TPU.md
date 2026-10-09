# Analysis plan: full E-set baselines on TPU v5e (written before any E-set result was seen)

Date: 2026-10-08. Status: **exploratory / descriptive.** None of this is a pre-registered hypothesis
test: H1 and H3 compare against GL, which needs the Laya router and calibrator (not trained yet).
This plan fixes the analysis *before* the scored rows were inspected; anything not listed here is a
post-hoc analysis and will be labelled as such in the paper.

## Data
- Task set: `tasks_E_primary.d25.jsonl` (1,536 tasks: Python 529, Rust 507, math 500; the D25 rebuild
  without hidden tests in gate payloads). Clusters = source problems (`cluster` field).
- Generations: vLLM-TPU, one v5e chip, bf16, greedy, thinking off, `max_new_tokens` 1024, raw Qwen
  prompts and stop sequences identical to the llama.cpp protocol (`OllamaGenerator`), log-probabilities
  of the sampled tokens. Models: Qwen2.5-Coder-1.5B-Instruct (continuity anchor), Qwen3.5-2B, Qwen3.5-4B.
  Raw files and the importer (`scripts/import_remote_gens.py`) are committed or reproducible; nothing is
  scored on the TPU host.
- Scoring: `scripts/run_study.py --mode score` in the bwrap sandbox; `correct` = hidden-check outcome
  VERIFIED (PASS_HIDDEN); FAILED and UNVERIFIED both count as not correct. Math is scored boxed-only.
- Cost unit: accelerator chip-seconds = chunk wall time split by token share (the registered rule), at
  the batch size each model was run with (96 sequences for Coder and 2B, 32 for 4B because 96 did not fit
  in HBM). Cost is therefore throughput-at-that-batch, not latency, and is not comparable to L4 GPU-seconds.

## Analyses (all with source-problem cluster bootstrap, 10,000 resamples, numpy `default_rng(0)`, percentile 95% CI)
- **A1** Pass rate (PASS_HIDDEN) per model x domain and pooled.
- **A2** Raw confidence = exp(mean log-prob of generated tokens): ECE (15 equal-mass bins), Brier, AUROC,
  AURC, per model x domain. Python/Rust/math are never pooled for a headline.
- **A3** Gate-only on the largest tier (4B): coverage (share of tasks the gate VERIFIES), precision
  P(pass hidden | gate VERIFIED), confident-wrong rate CWR = P(answered and not correct) over all tasks;
  per domain. Math has no gate-visible check in the task files; if the gate answers nothing there it is
  reported as coverage 0, not dropped.
- **A4** (exploratory) Matched-coverage comparison: CWR of "4B answers iff mean-log-prob >= t" with t chosen
  on E so coverage equals the gate's coverage in the same domain, versus the gate. Choosing t on E favours
  the log-prob baseline, so a gate advantage here is conservative. The registered B3 sets t on a separate
  split C (not available), so this is not B3.
- **A5** (exploratory, **not a registered arm**) Gate cascade = GL without Laya: answer with 2B, run the
  executed gate, escalate to 4B only if the gate does not verify; final answer is the last tier's. Compare
  with always-smallest (B1) and always-largest (B2): accuracy, accelerator cost per task, share escalated,
  cost per correct answer. Paired cluster-bootstrap CIs for accuracy and cost differences. Exact McNemar on
  the pooled paired accuracy contrasts cascade-vs-B2 and cascade-vs-B1, Holm over these two (m = 2).
- **A6** Oracle router (cheapest correct tier of {2B, 4B}) as a ceiling only, never as a comparator.
- **A7** Same-protocol consistency check on the 80 night tasks: Qwen3.5-2B q4_K_M llama.cpp CPU (night arm,
  re-scored) versus Qwen3.5-2B bf16 vLLM TPU: pass counts, item agreement, exact McNemar on discordant pairs.
  Hardware, quantisation and engine all differ; prompts and stop sequences do not.
- **A8** Timeout audit: every row whose check timed out is re-checked serially; changes are reported.
- **A9** Truncation: share of generations with `finish_reason = length` per model x domain, and their
  pass rate (a truncated math answer has no `\boxed{}` and is UNVERIFIED).

## Decision rules and what would count against us
There are no pass/fail gates. Reported honestly regardless of direction: the cascade may be no cheaper
than always-largest; the gate may have low coverage; raw confidence may be near-useless on code (the
night arm's Python AUROC CI included 0.5). A cascade that saves cost *and* keeps accuracy would be
described as an exploratory observation on public benchmarks the models have probably seen, not as
evidence for any registered hypothesis.

## Known limits (stated now)
Single greedy sample; one hardware/engine; contamination of public benchmarks by pretraining is possible;
tau for raw-confidence selection is not set on a held-out split; the 4B tier is the largest available, so
B2/B5 are "largest of two", not the registered 9B/27B; CPU contention can cause spurious timeouts (A8).

## Amendments (made after the first analysis run; the original text above is unchanged, sha256 1f2fa2c1...)
Dated 2026-10-08. These were decided after seeing the first results and are labelled as post-hoc.

1. **A7 excludes math.** The first run reported a pooled CPU-vs-TPU McNemar p of 0.0004 driven entirely by
   math (CPU 0/26 versus TPU 15/26). The CPU night math generations came from the defective code prompt
   (D24/D28), so that difference measures the defect, not the engine. A7 is restricted to Python and Rust.
   Seeing the pooled number before deciding is a researcher degree of freedom; the reason (a documented
   defect found before this run) is independent of the outcome.
2. **A8 became a correction procedure.** The serial re-check found tasks that fail in the parallel run but
   pass alone (contention timeouts). Those tasks are re-scored serially for all arms
   (`scripts/tpu/serial_overlay.py`); the originals stay untouched, the overlay takes precedence in
   `analyze_e_tpu.py`, and every corrected row is listed in the numbers file under
   `serial_recheck_corrections`. Only failures can be spurious, so only non-passing rows were re-checked.
   Checks that time out even when run alone remain FAILED and are counted and reported.
3. **Cascade cost reading.** The first run showed the cascade is not cheaper than always-largest for this
   2B/4B pair. This is reported as the result; no tuning of the gate or the pair was done afterwards.
4. **Stability audit and policy (decided before the passing-row audit finished).** The serial overlay showed
   that `rs/mbpp_130_max_occurrences` (Qwen3.5-4B) is not a contention artefact: the candidate iterates a Rust
   `HashMap` and takes the first maximum, so with tied counts its result depends on the per-run random hash
   order and it passes about half the time. A re-check of failures alone cannot see the opposite error (an
   unstable candidate that happened to pass), so every passing Python/Rust row was re-run once more
   (`recheck_failures.py --verified`). Policy: **headline numbers** = original scores with the serial overlay
   applied to contention flips; **sensitivity S1** = additionally count every item that produced two different
   outcomes in any run as not correct (fail-closed). Both are reported; if they differ by more than the
   confidence intervals, S1 is described as the conservative reading.
5. **Changes made after the independent re-audit of revision 3.4.0 (post-hoc, 2026-10-09).**
   - A4 pooled: the earlier "pooled" matched-coverage figure applied ONE threshold across all domains, which is not
     the per-domain matching the plan describes. It is removed and replaced by a per-domain-matched pooled value
     (sum over domains of gate-wrong minus coverage-matched-baseline-wrong, divided by N).
   - Wording: the coverage-matched baseline picks the k most confident tasks with k = the gate's answered count in
     the same domain. No outcome labels are used to choose k, so "threshold tuned on E, which favours the baseline"
     was unsupported and is withdrawn; ties are broken pessimistically (wrong first).
   - S2 (new, post-hoc): cost ratios excluding the first generation chunk of each model, which includes XLA
     compilation. The audit showed the Python cascade-vs-largest cost ratio depends on this; both are reported.
   - Math gate: the frozen sentence above ("Math has no gate-visible check in the task files") is accurate about the
     task files but misleading. The registered math gate (program-of-thought re-execution that reproduces the boxed
     answer) needs no task payload; the zero coverage is an implementation gap in our checker.
   - Concurrency: engine logs show the KV cache re-split for at most 54 concurrent requests (2B) and 11 (4B), not
     the requested 96 and 32. Chip-second costs are therefore throughput at those effective concurrencies; the 4B
     ran at lower concurrency than the 2B, which understates the cascade's cost penalty.

## Addendum A10: the registered math gate, implemented after the E analysis (written before any program was generated)
Dated 2026-10-09. Post hoc relative to A1-A9 and labelled so: the math-gate gap was found in the re-audit, after E
had been analysed.

- **Gate.** Registered design (GWENLAYA_PREREGISTRATION.md l.166): the tier that answered writes a short Python
  program that computes the answer and prints it; the program runs in the bwrap sandbox (no network, no host files,
  time and memory limits); the gate VERIFIES iff the program's last output line is equivalent to the answer's final
  `\boxed{}` value (same equivalence as the scorer). The gold answer is never an input (tested). No boxed answer, no
  program, crash, timeout, unparseable output, or a MISMATCH are all UNVERIFIED (a wrong program is not a refutation).
- **Generation.** One greedy program per (tier, math task), prompt = the problem without its boxed-answer
  instruction, system prompt in `gwaya/generators.py`, `max_new_tokens` 1024, same TPU, engine and settings as the
  answers. Cache keys carry a `|pot` suffix and call index 1; records are marked `kind: pot` and are never scored as
  answers.
- **Cost.** The program is a second accelerator generation. Every arm that uses the gate (gate-only and the cascade) is
  charged the program's attributed chip-seconds for each tier it consults; ungated arms are not. Cost is therefore
  comparable to the code domains only if the same charge is applied; it is stated.
- **Analyses (math only, plus the pooled cascade recomputed with the new math rows).** Gate coverage, precision
  P(correct | VERIFIED), confident-wrong rate vs answer-everything, and the per-domain coverage-matched log-prob
  baseline (same construction as A4). Cascade 2B -> gate -> 4B on math: accuracy, cost with program cost, escalation
  share, cost per correct answer, and the oracle ceiling (unchanged). Math rows from the new replay replace the old
  math rows in the overlay (originals untouched).
- **What would count against the gate.** Low coverage (programs often fail), low precision (programs reproduce the same
  mistake as the reasoning), or a cost that exceeds the benefit. Any of these is reported as found.
- **Not a registered test.** This changes the status of no hypothesis.

## Addendum A11: cross-fitted calibrated abstention (written before any such model was fitted)
Dated 2026-10-09. Exploratory; **not** the registered GL arm or H1/H3 (no Laya, no separate split C). Because E has
already been analysed, the freeze rule cannot hold for E; this analysis replaces a fitted-on-C calibrator with a fully
pre-specified cross-fitted procedure that is run once.

- **Unit and tiers.** Each of the 1,536 E tasks x tier, for the 4B (headline) and the 2B (secondary).
- **Features** (stored outputs only; the gold answer and hidden tests are never inputs): gate verdict one-hot
  {VERIFIED, FAILED, UNVERIFIED} (code: visible-test gate; math: the A10 program-of-thought gate); mean token
  log-prob; minimum token log-prob; mean of the lowest 10% of token log-probs (at least one token); log(1 + completion
  tokens); truncated flag (finish_reason == length); domain one-hot (3). Output = P(correct), correct = hidden check.
- **Models** (all evaluated only on out-of-fold predictions): M0 gate verdict (answer iff VERIFIED, also used as a
  binary score); M1 mean log-prob only; M2 logistic regression on every non-gate feature; M3 logistic regression on all
  features. Logistic regression = scikit-learn `LogisticRegression(penalty="l2", C=1.0, max_iter=1000)` after
  `StandardScaler` fitted on the training portion; no interactions; no tuning of C.
- **Folds.** 5 folds by source problem: fold = int(sha256(cluster)[:8], 16) mod 5 (deterministic, no RNG).
- **Discrimination and calibration** (per domain; a pooled value is only a convenience): AUROC, Brier, ECE (15
  equal-mass bins), AURC of the out-of-fold P(correct); 95% CIs by the same cluster bootstrap as A1-A9
  (10,000 resamples, numpy `default_rng(0)`). Paired differences M3-M0, M3-M1, M2-M1.
- **Target-risk operating points.** Selective risk = P(wrong | answered). For alpha in {0.05, 0.10}: inside each
  outer training portion, inner 4-fold cross-fitted predictions choose the threshold with the largest coverage whose
  inner selective risk is <= alpha (require >= 30 answered; otherwise answer nothing); the model is then fitted on the
  whole training portion and the threshold applied to the held-out fold. Report realised coverage and realised
  selective risk (pooled over folds, per domain) with cluster-bootstrap CIs, for M1, M3 and for the gate alone (which
  has no threshold). Compare realised risk with alpha: a method "meets the target" only if the upper CI bound <= alpha
  is NOT required; we report whether the point estimate and the CI cover alpha, without a pass/fail claim.
- **What would count against it.** M3 not beating M0 or M1 on AURC; realised risk above alpha; coverage collapsing on
  Rust or math. All are reported as found. No hyperparameter, feature or fold choice will be changed after seeing results.

## Addendum A12: a wider tier gap with Qwen3.5-9B (written before the 9B run completed; no 9B result seen)
Dated 2026-10-09. Exploratory and post hoc relative to A1-A9; it restores the registered ladder's largest feasible tier
(9B; the 27B did not fit the budget and is not attempted here).

- **Generation.** Qwen3.5-9B, bf16, vLLM-TPU, one `v5litepod-4` slice (tensor parallel 4), greedy, same raw protocol,
  prompts, stop sequences and log-probabilities as the 2B/4B; answers for all 1,536 E tasks and, for math, a
  program-of-thought program per task (A10). A 6-task-per-domain smoke test (18 tasks) showed the model starts and
  generates; its outputs are not analysed. Effective concurrency will be read from the engine log and reported.
- **Cost unit.** Attributed seconds are the slice wall time split by token share **times the 4 chips in use**
  (`--chips 4` in the importer), so chip-seconds are comparable with the single-chip 2B/4B. Dollar cost = chip-seconds x
  $1.20 / 3600 (on-demand list price estimate). Different effective concurrency across tiers is stated, not corrected.
- **Analyses.** (a) A1, A2 for the 9B. (b) A3/A4: gate-only on the 9B as the largest tier, per domain, with the A10 math
  gate, and the per-domain-matched log-prob baseline. (c) Cascades with the executed gate and no router, replayed offline
  from the cached generations: 2B -> 9B, 4B -> 9B, and the three-tier ladder 2B -> 4B -> 9B. For each: accuracy
  (final candidate, answered or not), selective coverage and confident-wrong rate, escalation share, cost per task and
  cost per correct answer (math charged for the programs of every tier consulted), cost ratio versus always-9B with and
  without the first generation chunk (S2), and paired cluster-bootstrap CIs; exact McNemar for each cascade versus
  always-9B, Holm over the three cascades. (d) The oracle-router ceiling over {2B, 4B, 9B} (cheapest correct tier),
  as a ceiling only. (e) A11 repeated with the 9B as the tier.
- **Reading rule fixed in advance.** A cascade "pays off" only if its cost ratio versus always-9B is below 1 in BOTH the
  with-warm-up and without-warm-up readings AND the accuracy difference's CI does not exclude zero on the harmful side.
  Anything else is reported as not paying off. We expect the answer to depend on the tier gap; no tuning follows.
- **What would count against us / limits.** A gate-driven cascade that is still not cheaper than always-9B; a 9B that is
  not much better than the 4B (little headroom); cost depending on effective concurrency (2B <= 54, 4B <= 11, 9B to be
  read from the log); public benchmarks possibly seen in pretraining; single greedy sample.

## Addendum A13: Laya trained on tier labels, a fresh evaluation set E', and a 27B tier (written before any of it was generated; no result seen)
Dated 2026-10-09. TPU budget for this addendum: USD 25 (hard), inside the project cap of USD 50 (spent so far 4.91).

**Why.** E is burned for the registered H1/H3 (the calibrator must be frozen before the evaluation set is generated; it was not, because no Laya existed). A fair test needs a fresh set E' that no model, router or calibrator has seen, and a Laya trained on labels from other tasks.

**Data (new pool P, disjoint from E).** Math: MATH train (EleutherAI/hendrycks_math, train split only; MATH-500 is drawn from the test split) and GSM8K train, deterministic seeded sample. Python: MBPP tasks whose id is not in MBPP+ (so not in E); original asserts as hidden tests, first assert visible to the gate. Decontamination: drop any P task whose normalised prompt hash or cluster equals an E task. Rust: no new pool exists (MultiPL-E Rust is all in E); Rust is used for training from E and is NOT evaluated on fresh data. Size is whatever these sources give, stated when built.
**Split (deviation D48 from the reserved 20-bucket rule, which would leave about 10 percent of a small pool for training):** by cluster, sha256(cluster) mod 10: 0-5 train, 6 calibration C, 7-9 evaluation E'. E (all 1,536 tasks) is added to TRAIN only. Nothing in E' or C is used for training, threshold choice or model selection, and E' is generated and scored only after Laya and its calibration are frozen (artifact sha256 recorded before E' generation).

**Tiers and labels.** Qwen3.5-2B, 4B, 9B (done on E) and 27B (smoke-tested first on a v5litepod-8; abandoned and reported if it does not run). Each tier answers every P task with the raw protocol (greedy, log-probs), math tasks also get the program-of-thought program for the math gate. Labels: correct (hidden tests / boxed answer) and gate verdict per tier.

**Laya.** ModernBERT-large with LoRA r=16 (scripts/train_laya.py), router head (per-tier P(correct) from domain + prompt) and calibrator head (P(correct) from prompt + candidate + gate/log-prob signals). Trained on the TPU if torch_xla works (feasibility probe first, then train); otherwise the run is reported as not done rather than moved to an unplanned machine. Fixed in advance: epochs 3, lr 2e-4, seed 1234, no tuning on C beyond fitting the monotone (isotonic) calibration map; the shuffled-label control (SH) is trained identically and must give AUROC near 0.5.
**Analyses on E' only (pre-specified):** (a) calibrator P(correct) versus the gate alone (B5), the gate with the A11 cross-fitted logistic model on hand-built features (M3), and raw mean log-prob (B3): AUROC, AURC, ECE, Brier, selective risk at coverage matched to the gate; paired cluster bootstrap (10,000, seed 0), Holm over the three comparisons. (b) Router: cascade order and escalation chosen by the router vs the static ladder vs always-largest, as accuracy, confident-wrong rate and chip-second cost per correct answer, tiers {2B,4B,9B} and {2B,4B,9B,27B}; the oracle ceiling reported as a ceiling. (c) Registered hypothesis status: H1 and H3 are reported as tested on E' with the deviation D48 stated, supported only if the paired CI excludes zero in the favourable direction. Everything else stays exploratory.
**LoRA of the generator.** There is no TPU training path for Qwen3.5 dense in MaxText or Tunix. We run one bounded probe (Qwen3.5-0.8B, peft on torch_xla, 50 steps, at most USD 1) to turn this into a measured statement; no generator adapter is claimed unless it trains and passes the harness.
**Limits stated in advance.** Small P (about 2-3k tasks); math and Python only for fresh evaluation; 27B effective concurrency and cost less certain; single greedy sample; TPU cost is a list-price estimate.

## Addendum A13b: what was run, and the exact E' analysis (written after the Laya training job was launched, before E' was generated; no Laya score and no E' result seen)
Dated 2026-10-09.

**Changes to A13 forced by measurement (reported, not hidden).** (1) The 27B tier did not run: Qwen3.5-27B on a v5litepod-4 never finished loading in 33 minutes (no token generated), and an 8-chip slice is blocked by the per-zone v5e quota of 4 chips (no quota requests, by instruction). The tier set is therefore {2B, 4B, 9B}; "bigger Gwen" is the 9B already reported in A12. (2) The generator-LoRA probe (Qwen3.5-0.8B, peft on torch_xla, bf16) loaded the model, fell back to the pure-PyTorch linear-attention path and finished no training step in 25 minutes; no generator adapter is trained or claimed. (3) Laya trains on the TPU: torch_xla 2.8 + peft, ModernBERT-large, LoRA r=16, measured 2.35 s/step including compile in the timing probe.
**Training data actually used.** Train = E (1,536 tasks) + P_train (1,545 tasks) = 3,081 tasks, 9,243 (task, tier) calibrator rows; labels from the 2B, 4B, 9B tiers; hyper-parameters as in A13 (3 epochs, lr 2e-4, batch 16, max length 512, seed 1234, bf16 autocast); validation = 10 percent task-grouped hash inside train; shuffled-label controls trained identically. Calibration split C = P_calib (283 tasks, 849 rows). E' = P eval split (828 tasks: 659 math, 169 Python), generated only after the freeze file below is committed.
**Freeze.** After training, sha256 of adapter, heads, and the isotonic maps fitted on C are written to `docs/LAYA_FREEZE.txt` and committed BEFORE any E' generation. If the shuffled-label control gives AUROC outside 0.4-0.6 on C, the pipeline is declared broken and E' is not generated.
**Primary tier and family.** The tier whose answers are scored is Qwen3.5-9B (the largest tier), math and Python pooled on E'; per-domain and per-tier are secondary. Confirmatory family (Holm over 2): **H1** = the Laya calibrator's P(correct) versus raw mean log-prob (B3) on the area under the risk-coverage curve (AURC; lower is better); **H3** = Laya versus the executed gate alone (B5) on selective risk at the gate's own coverage (Laya ranks the answers and keeps the same number the gate verifies). Supported only if the paired cluster bootstrap CI (10,000, seed 0) of the difference excludes zero in Laya's favour after Holm; reported with the deviation D48 (split) and D49 (training on E) stated. **Exploratory:** Laya versus a logistic regression on the hand-built features (gate, mean log-prob, min log-prob, domain, tier) trained on the same training rows (does text help beyond the features?); AUROC, ECE (15 equal-mass bins), Brier for all scores.
**Router policy (exploratory).** For each task, start at the cheapest tier whose router P(correct) is at least tau, else start at the 9B; then run the gate cascade upward and answer at the first VERIFIED tier (else the last consulted tier's candidate, unanswered). tau is chosen on C from {0.5,0.6,0.7,0.8,0.9} as the value with the lowest mean chip-second cost among those whose C accuracy is at least that of the static 2B->4B->9B ladder on C (if none, 0.9). Compared on E' with the static ladder and always-9B: accuracy, confident-wrong rate, cost per task and per correct answer (programs charged for every consulted tier), paired bootstrap CIs. Rust has no fresh evaluation data, so nothing about Rust is claimed from E'.
**Limits.** E' has only 169 Python tasks (low power) and its math is mostly GSM8K and easy MATH, so absolute accuracies are higher and not comparable with E; training on E is documented; single greedy sample; TPU costs are list-price estimates.
