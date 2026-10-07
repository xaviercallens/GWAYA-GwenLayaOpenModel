# Freeze addendum, night run 2026-10-07 (C0-lite)

Written 2026-10-08 by stage L1, **before any candidate generation**. At the time of writing
`$NIGHT/cache/` does not exist and no run_study generation has been started. This addendum is
registered in `docs/GWENLAYA_PREREGISTRATION.md` (Addenda) with its sha256. It does not change the
registered hypotheses, MEIs or decision rules. Every departure from the registered design is in
`docs/DEVIATIONS.md` (D1 to D20).

## 1. Primary-E manifest

- File: `experiments/night/manifest_eval.json` (same bytes as `$NIGHT/results/manifest_eval.json`).
- **manifest sha256: `bd1684dadcf38328fb6fab7619e5ea2f4fbc5ccc0e31022e059360b2d8c615e0`**
- Per item: task_id, native id, source-problem cluster, sha256 of the whitespace-normalised
  statement plus tests, sha256 of the gate-visible tests, HF dataset revision, license, exclusion reason.
- HF revisions (read from the Hub on 2026-10-08):
  - `HuggingFaceH4/MATH-500`: `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, license `none on card (derived from MIT MATH); eval only`, rows 500
  - `evalplus/humanevalplus`: `d32357cf319e50e9c8d8dab5ea876c72b0fd321b`, license `apache-2.0`, rows 164
  - `evalplus/mbppplus`: `b2d74c91837c3f2a20c1299ae98133cbe7cfa077`, license `apache-2.0`, rows 378
  - `nuprl/MultiPL-E`: `28441b6024e71d4a1c1c0f6bf171c935cd5a43f2`, license `mit`, rows 510

| Domain | Rows | Excluded (harness sanity) | Included | n_d (E-night) |
|---|---|---|---|---|
| math | 500 | 0 | 500 | 26 |
| python | 542 | 8 | 534 | 28 |
| rust | 510 | 3 | 507 | 26 |
| total | 1552 | 11 | 1541 | 80 |

Exclusion reasons (counted, not hidden): python: reference_fails_hidden_checks 8; rust: harness_typecheck_fails_hidden 1, single_assert_no_gate_hidden_split 1, tests_not_splittable 1.

## 2. Gate-visible versus hidden checks (as implemented)

- Python, HumanEval+: gate = the first input/expected pair of the combined evalplus input list
  (`zip(inputs, results)` sliced to `[:1]`); hidden = the full plus suite.
- Python, MBPP+: gate = `test_list[0]` (the public base test, also shown in the prompt) with
  `test_imports`; hidden = the full plus `test` field.
- Rust: gate = `let candidate = ...;` plus the first `assert_eq!`; hidden = every assert (`tests` as released).
- Math: no gold answer reaches the gate (empty gate payload); hidden = sympy/numeric equivalence with the gold answer.
- Source clusters: a HumanEval or MBPP Python task and its MultiPL-E Rust translation share
  cluster `HumanEval/N` or `MBPP/N`; MATH-500 items are singleton clusters.

## 3. E-night size n_d and order

- Order: per domain, `random.Random(f"20261007:{domain}").shuffle(sorted(included task_ids))`; E-night
  is the first n_d of each domain, written to `tasks_E_night.jsonl` interleaved by rank/n_d so every
  prefix keeps the domain proportions. Truncation at a time limit therefore keeps the seeded order, and
  the truncated n is analysed as is.
- **n_d = python 28, rust 26, math 26, sum 80**
  (largest-remainder apportionment of 80 in proportion 542:510:500).
- Sizing rule: n_total = min over stages of floor(0.85 x hours x 3600 x tok_s / mean_tokens_per_item), with
  L2 = 3 h at 2.734 tok/s (2B) and L3 = 4 h at 1.292 tok/s (4B),
  both the sequential wall rates measured in L0 (`L0_env.json`). Result per stage: {'L2': 127, 'L3': 80}; binding stage **L3**.
- **mean_tokens_per_item = 196.6 is ASSUMED, not measured.** ASSUMED: 1.25 x Qwen-token length of the reference solution + 20 (python); 1.6 x its Python sibling (rust); reference-solution length + 20 (math); capped at 1024. Not measured; see addendum.
  If real completions are longer, L2 and L3 hit their time limit and report a truncated n; if shorter,
  they finish early and n_d is not enlarged after the fact.
- Files: `tasks_E_night.jsonl` sha256 `e842edd13951b4082c4495f7bcc845b032989f4a457e0c5d6e9c44f2b72cd679` (80 lines);
  `tasks_E_primary.jsonl` (all 1541 included items in the same interleaved seeded order, used only if G1 runs)
  sha256 `971f44c61bad5044f8487383f8a3ea27bc2c292c565a6ff8ef1254fce1cc7097`; `E_night_sizing.json` sha256 `3353241c468bd5f68b2b7e242f0294b453fea24a6dd119f1786455e14e303611`.
- Generation settings are unchanged from the registration: greedy, thinking off, max_new_tokens 1024, logprobs on, 0 repair rounds in GL.

## 4. Decontamination

- Tool: `scripts/data/decontaminate.py` with `experiments/night/decontam_report.json` (sha256 `d128e06c5949c86c29537f71a1d547ef953be6aa22f7bc814a280ec4d48f1907`). Eval index: the five primary-E sets; train/calib sources: MBPP sanitized train, validation, prompt.
- Result: status `done`, complete True.
  - `google-research-datasets/mbpp:sanitized:prompt`: n_in 7, flagged 7, kept 0; removals by eval set: evalplus/mbppplus: 7; nuprl/MultiPL-E:mbpp-rs: 5
  - `google-research-datasets/mbpp:sanitized:train`: n_in 120, flagged 120, kept 0; removals by eval set: evalplus/mbppplus: 120; nuprl/MultiPL-E:mbpp-rs: 107
  - `google-research-datasets/mbpp:sanitized:validation`: n_in 43, flagged 43, kept 0; removals by eval set: evalplus/mbppplus: 43; nuprl/MultiPL-E:mbpp-rs: 34
- Every MBPP-sanitized item is removed, because it is an MBPP+/MultiPL-E mbpp-rs item. There is therefore **no MBPP-derived calibration set**, and no T/C source is generated or used tonight (D18).

## 5. Cross-fitting procedure (replaces the separate C set; D4/D5)

- Unit: source-problem cluster. Fold of an item = `int(sha256(f"1234|{cluster}").hexdigest(), 16) % 5`, so a Python task and its Rust translation are always in the same fold.
- Per fold k, per tier and per domain: fit all calibrators (LR, SH, B3 temperature scaling, the Laya head or the frozen-embedding probe) on the other 4 folds only; predict fold k. Thresholds tau_route, tau_hi (split-conformal, alpha 0.10) and tau_lo (LIKELY_WRONG precision >= 0.90) are chosen on the training folds only, per domain, then applied to the held-out fold. All reported E scores are out-of-fold.
- B3's threshold is set on the training folds to hit GL's out-of-fold coverage on the same training folds; matching on the held-out fold is reported with its CI (registered rule: coverage difference within +-3 pts, otherwise "matching failed").
- SH uses labels shuffled within the training folds with seed 1234.
- If a domain has fewer than 20 training-fold items or a single outcome class in the training folds (expected at n_d about 26 per domain, about 21 training items), that domain's thresholds are not fitted separately: calibrators and thresholds are fitted on the pooled training folds of all domains with the domain as a feature, and the result is labelled pooled. Per-domain calibration claims (H6) are then TBD.

## 6. Arms run tonight

GL, B1 (always 2B), B2 (always the largest local tier, 4B; deviation from 9B/27B unless G1 runs), B3, B5 (gate only, 4B), B6 (Laya zero-shot features), LR, SH (negative control). B7 is a descriptive ceiling only. B4 is dropped (5x generation cost). Tiers: 2B and 4B Q4_K_M on CPU. Lean 4, GSM8K, AIME25, LiveCodeBench, the MBPP continuity anchor, Qwen3.5-0.8B and the second seed are not run (D8, D10).

## 7. Analysis plan

Unchanged from sections 7 and 8 of the preregistration: paired percentile bootstrap, 10,000 resamples, seed 0, resampling cluster-stratified with all arms jointly; exact McNemar; Holm within the primary family H1, H3 (m = 2). Cost is CPU-seconds per call (D3). H4 to H6 secondary, H2 and H7 exploratory; H8, H9 to H12, H14 not run unless stated in DEVIATIONS. Paired n is the number of items present in both the 2B and the 4B caches. A paired n below 500 carries the underpowered flag. Negative and null results are reported.

## 8. Power statement

Method: the registered one (exact one-sided McNemar, enumerated, clustering ignored, so an upper bound); the code reproduces the registered table values (tests/test_eval_manifest.py). Evaluated at the sized n = 80 (the most that can be paired; fewer if truncated):

| Effect (P(B confident-wrong, GL not) / P(GL confident-wrong, B not)) | Power at alpha 0.025 (Holm, m = 2) | Power at alpha 0.05/3 (registered table) |
|---|---|---|
| 6 pts (8%/2%) | 0.261 | 0.206 |
| 4 pts (5%/1%) | 0.115 | 0.104 |
| 3 pts (4%/1%) | 0.052 | 0.049 |
| 2 pts (3%/1%) | 0.017 | 0.016 |

**H1 is expected to be badly underpowered at this n.** At the registered MEI (3 pts) the power is far below 0.80. The MEI is not changed. A non-significant H1 on E-night is
inconclusive, not evidence of no effect, and will be reported as such. The design effect is `TBD` (estimated from the realised clusters of E-night after generation; 80 items contain 77 distinct clusters in the file at the time of writing).
