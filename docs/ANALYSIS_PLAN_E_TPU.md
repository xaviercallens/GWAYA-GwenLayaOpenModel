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
