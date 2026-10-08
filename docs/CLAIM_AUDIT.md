# Claim audit: GwenLaya v4 interim report (revision v3.3.1)

Audited 2026-10-08 on branch `bench/tpu-e-eval` (base `main` = release v3.3.1, `4914b5c`). This was an
independent check. The auditor changed no paper, number, code or data file. The only file written is
this one; scratch scripts are in the session scratchpad.

Scope: `papers/gwenlaya_v4.tex` (line numbers below refer to it), `papers/gwenlaya_v4.bib`,
`docs/DEVIATIONS.md`, `docs/RESULTS_VERDICT.md`, `docs/GWENLAYA_PREREGISTRATION.md` and `README.md`.
`papers/numbers_v4.json` was **not** used as a source. Every number was recomputed from the raw rows,
generations, ledger and logs, or the row says it was not.

## Summary

| Status | Count |
|---|---|
| VERIFIED | 61 |
| VERIFIED_WITH_NOTE | 29 |
| NOT_VERIFIABLE | 2 |
| WRONG | 4 |
| OVERSTATED | 3 |
| **Total claims checked** | **99** |

By class:
- **A (own quantitative results), 45 rows.** All 14 pass/fail counts are correct. These are Python, Rust and math on CPU, the nine TPU cells, and the 22/28 and 24/26 agreement counts. Re-running the repo checkers on the stored generations reproduces every stored status: 0 mismatches over 80 CPU and 240 TPU items. All CIs agree with an independent bootstrap within 0.02, except two cost-per-pass upper bounds. Those differ by Monte-Carlo noise on a ratio statistic. Three rows are wrong: two quoted sha256 values and the spend total, which is now stale.
- **B (code/system), 20 rows.** The harness claims hold in the code and in tests that were re-run. The 25/25 and 9/9 Rust validation reproduces exactly. Two claims are overstated: H13 holding "by construction", and the CPU-vs-TPU comparison being described as a check "across hardware and quantisation".
- **C (external/citations), 19 rows.** Every cited work exists and its arXiv id, authors and year match. One bib entry is wrong: the gwayav3 title does not match the Zenodo record. One README badge is overstated.
- **D (logic/statistics), 15 rows.** The reasoning and arithmetic hold, including the power values and the n=60 bootstrap argument. Two rows have minor notes.

## Must-fix before publication (WRONG / OVERSTATED)

1. **A32. WRONG. The sha256 of `papers/numbers_v4.json` (v3.3.0), l.585.** The paper quotes `e5488f6c…b32b07c`. That is the file as committed in `b5be32a`, before the spend repricing. The file at tag `v3.3.0` (`ebd032f`) hashes to `fccb1695538e76eba93669d1a232d24c7373e8aa2ad37d900f0e19c153275995`.
   **Fix:** replace the hash with `fccb1695538e76eba93669d1a232d24c7373e8aa2ad37d900f0e19c153275995`.
2. **A33. WRONG. The sha256 of `papers/numbers_v4.json` (v3.3.1), l.588.** The paper quotes `952bfc77…6eb74`. That is an uncommitted scratch regeneration (`scratchpad/reanalysis2/numbers_v4.json`) that differs from the committed file only in the `single_arm_note` string. The committed file at tag `v3.3.1` and on HEAD hashes to `5f3c2a6e1e069663af306f67f4e2ba0e32155044908cba35fd5826f092d8a1c0`.
   **Fix:** replace the hash with `5f3c2a6e1e069663af306f67f4e2ba0e32155044908cba35fd5826f092d8a1c0`. Re-hash again if `analyze_study.py` is re-run.
3. **A28. WRONG (stale). Spend: "5 lines", "\$1.54" and "1.5399" at l.70, 274, 286, 538 and 550, and in RESULTS_VERDICT.** The claim was true at the v3.3.1 release (08:04). The ledger now has **7 lines totalling \$1.9809**. The two added lines are `tpu_e-gen-210955` (963 s, \$0.321, 21:36) and `tpu_e-gen2-orphan` (360 s, \$0.12, 21:44, an orphan VM created by a killed driver).
   **Fix:** either update to "7 lines, \$1.98 (sum of `usd_estimate` = 1.9809)" and add the two rows to the spend table, or pin the snapshot: "Read from `spend_ledger.jsonl` as of 2026-10-08 08:04 (its first 5 lines, \$1.54); two later lines (\$0.441, a post-v3.3.1 TPU run including a 6-minute orphan VM) are outside this report." Either way, H13 is still not refuted.
4. **C12. WRONG. Bib entry `gwayav3`.** The title "GWAYA v3: A Fail-Closed Verification Gate for LLM-Generated Code" does not match the record. Zenodo record 10.5281/zenodo.23121788 is titled "GWAYA v3.1: Fail-Closed Gate & Real Toolchain Verification for LLM Code Generation", version 3.1.0, published 2026-10-03.
   **Fix:** `title = {{GWAYA} v3.1: Fail-Closed Gate \& Real Toolchain Verification for {LLM} Code Generation}`.
5. **B15. OVERSTATED. "the ledger refuses any stage that would exceed its cap, so H13 holds largely by construction" (l.292–294; abstract l.70–71; RESULTS_VERDICT).**
   - The ledger file `spend_ledger.jsonl` enforces nothing.
   - The only ledger-checking launcher in the released code is `deploy/run_on_gcp.sh`. It reads the GCS `runs/LEDGER.jsonl`, which the paper itself says does not exist.
   - All recorded spend came from TPU drivers that are not in the repo (`scratchpad/tpu_smoke/drive*.sh`). These have a run-time limit but no ledger-total check.
   - `deploy/tpu/drive.sh`, which does check the ledger, is untracked and dated 21:09, after the v3.3.1 release.

   **Fix:** "The GPU stage launcher is designed to refuse a stage that would push the ledger over its cap, so H13 was expected to hold largely by construction (red-team prior 0.75). The TPU runs recorded here were launched by separate scripts with a run-time limit but no ledger check, so the low total reflects low usage. In either case it is not evidence about the GwenLaya system."
6. **B12. OVERSTATED. TPU-vs-CPU reading: "a consistency check of the re-scored pipeline across hardware and quantisation" (l.521–522; abstract l.67–69); and "with the fixed prompts of D28" (l.490).** The TPU run differs from the CPU arm in more than hardware and quantisation:
   - **Engine:** vLLM `llm.chat` versus a llama.cpp `/v1/completions` raw prompt.
   - **Code system prompt:** TPU uses "Reply with one complete Python code block only. No explanations." and, for Rust, "…no main function". CPU uses the `_SYSTEM_PROMPT` "runnable code block only. No placeholders…".
   - **Prefill and stop:** the CPU arm used a ```` ```lang ```` prefill and a ```` ``` ```` stop; the TPU run used neither.
   - **Scope of D28:** D28 fixed only the math prompt.

   **Fix:** "Qwen3.5-2B in bf16 under vLLM on TPU, with a different code system prompt and no code prefill, gives the same Python and Rust pass counts as the q4 CPU arm. Hardware, quantisation, engine and code prompt all changed, so this is a loose consistency check, not an isolation of any one factor." Also change l.490 to "with the D28 math prompt and TPU-specific code system prompts".
7. **C17. OVERSTATED. README "CI passing" badge.** The badge is a static shields.io image. The `.github/workflows/ci.yml` it links to is untracked, so it is not in the repository, and `gh run list` shows no runs.
   **Fix:** remove the badge, or commit the workflow and use the live workflow-status badge.

Recommended, not a WRONG: commit the scripts that produced the published TPU and validation numbers. They exist only in the session scratchpad: `tpu_smoke/gen_tasks.py`, `run3.sh`, `drive3.sh` and `rust_reference.py`. Without them, D30 and the 25/25 and 9/9 validation cannot be reproduced from the repo (see B4 and B10).

## Full table

Legend: V = VERIFIED, VN = VERIFIED_WITH_NOTE, NV = NOT_VERIFIABLE, W = WRONG, O = OVERSTATED. "boot" =
my bootstrap: 10,000 resamples with Python `random.Random(12345)` (deliberately not the paper's numpy
`default_rng(0)`), stratified by domain. Each E-night item is its own cluster in `tasks_E_night.jsonl`
(28/26/26 distinct clusters), so this equals the paper's cluster bootstrap. Percentile intervals.
Confidence = exp(mean_logprob) from `L2fix/gens.jsonl`. I checked that it equals the mean of
`token_logprobs` for all 80 items. ECE uses 15 equal-mass bins (the registered definition) and,
for comparison, 10 equal-width bins.

### A. Quantitative claims about our own results

| ID | Claim (tex line) | Evidence | Status / fix |
|---|---|---|---|
| A1 | 80 scored items, one unregistered arm (python 28, rust 26, math 26) (l.54–55, 201) | `L2fix/rows.jsonl`: 80 rows, all `arm=base`, domains 28/26/26 | V |
| A2 | Python 11 PASS / 17 FAIL / 0 UNDECIDED (l.59, 364) | rows: VERIFIED 11, FAILED 17. Re-ran `gwaya.domains.checkers.check` on the 80 stored generations: identical status for all 80 | V |
| A3 | Python pass rate 0.393, CI 0.214–0.571 (l.60, 341, 365) | 11/28 = 0.3929; boot [0.214, 0.571]. Wilson for reference: [0.236, 0.576] | V |
| A4 | Python mean conf 0.896 [0.875, 0.917] | 0.8965, boot [0.875, 0.917] | V |
| A5 | Python ECE 0.519 [0.392, 0.689] (l.61, 341, 366) | 15 equal-mass bins: 0.519, boot [0.390, 0.686]. 10 equal-width bins: 0.504 [0.328, 0.674] | VN: the result depends on binning; state "ECE, 15 equal-mass bins" |
| A6 | Python Brier 0.476 [0.334, 0.617] (l.366) | 0.4756, boot [0.332, 0.613] | V |
| A7 | Python AUROC 0.717 [0.483, 0.917] (l.62, 341, 367) | pairwise Mann-Whitney with ties = 1/2: 0.7166, boot [0.489, 0.914]; includes 0.5 | V |
| A8 | Python AURC 0.438 [0.221, 0.712] (l.367) | 0.4378, boot [0.222, 0.703] | V |
| A9 | Python CWR 0.607 [0.429, 0.786] (l.368) | 17/28 = 0.6071, boot [0.429, 0.786] | V |
| A10 | Python cost/pass 330 [174, 726] CPU-s (l.341, 369) | sum of `cpu_seconds` 3624.84 / 11 = 329.5; boot [175, 703] | VN: upper bound differs by 23 s (ratio statistic, Monte-Carlo noise) |
| A11 | Rust 6 / 20 / 0 (l.59, 126–127, 371–372) | rows and an independent re-run of the checker: 6 VERIFIED, 20 FAILED | V |
| A12 | Rust pass rate 0.231 [0.077, 0.385] | 0.2308, boot [0.077, 0.385] | V |
| A13 | Rust mean conf 0.894 [0.882, 0.906]; ECE 0.675 [0.549, 0.813]; Brier 0.605 [0.477, 0.725]; AUROC 0.875 [0.709, 0.990]; AURC 0.557 [0.321, 0.851] (l.342, 374–376) | 0.8939 [0.882, 0.906]; 0.6754 [0.546, 0.815]; 0.6048 [0.475, 0.726]; 0.8750 [0.714, 0.991]; 0.5568 [0.318, 0.849] | V |
| A14 | Rust cost/pass 494 [261, 1552] | 2962.70 / 6 = 493.8; boot [258, 1587] | VN: upper bound differs by 35 s (ratio noise) |
| A15 | Math boxed-only: 0/0/26, none contains `\boxed`; mean conf = ECE = 0.832 [0.807, 0.857] (l.63–64, 135, 344, 382–384) | 26 UNVERIFIED; 0 of 26 texts contain `\boxed`; mean conf 0.8320 [0.807, 0.857]; ECE identical, as it must be with all y = 0 | V |
| A16 | Pooled 0.212 [0.125, 0.300]; conf 0.875 [0.863, 0.886]; ECE 0.662 [0.585, 0.741]; AUROC 0.806 [0.675, 0.917]; cost 970 [629, 1648] (l.345, 386) | 0.2125 [0.1375, 0.300]; 0.8747 [0.863, 0.886]; 0.6622 [0.583, 0.740]; 0.8058 [0.675, 0.914]; 970.5 [627, 1650] | VN: pass-rate lower bound differs by 0.0125 (< 0.02, Monte-Carlo) |
| A17 | v3.3.0 Rust 0/26 (0/13/13); D23 "14/80 VERIFIED"; D24 3 math passes via last-number (l.57, 118, 416) | `L2score/rows.jsonl`: rust 13 FAILED + 13 UNVERIFIED; 14 VERIFIED in total; math 3 VERIFIED, 18 FAILED, 5 UNVERIFIED | V |
| A18 | v3 n=60 "CPU": A3−A0 +6.67, CI [1.67, 13.33], 4/0, McNemar p = 0.125 (l.91) | `results_n60_cpu.json` matches; 2·0.5^4 = 0.125. The file has `hardware: "unspecified"` and `ollama_version: "unspecified"` | VN: "CPU" comes from the file name and README, not the file contents |
| A19 | n=257: A0 64.2, A3 71.98, +7.78, CI [4.67, 11.28], 20/0, p rounds to 0.0, `git_commit` empty (l.93–95, 100) | file matches; 20/257 = 7.78 pts; exact two-sided p = 2·0.5^20 = 1.9e-6 | V |
| A20 | Coverage 80.93 %, precision 88.94 %, so ≈ 8.95 % confident-wrong (l.103–105) | 0.8093 × (1 − 0.8894) = 0.0895 | V |
| A21 | Power 0.052 at 3-pt MEI, 0.261 at 6 pts, α = 0.025, n = 80 (l.237–239, 267, 661) | exact one-sided McNemar power by enumeration: 0.052 (4 %/1 % discordance), 0.261 (8 %/2 %); also 0.049/0.206 at α = 0.05/3. Matches `E_night_sizing.json` | VN: depends on the assumed discordance rates (prereg §9 says so) |
| A22 | TPU table counts: 2B 11/6/16, 4B 20/13/16, Coder 18/15/8 (l.512–514) | `*.rescored.jsonl` counts match, and so does `summary.json`. Re-running the checkers locally on the raw TPU texts reproduces every one of the 240 statuses | V |
| A23 | Math truncated 9 / 9 / 4, all UNVERIFIED (l.506, 512–514, 524) | `finish_reason=length`: 9, 9, 4, all `no_boxed_answer`. Other UNVERIFIED: 1, 1, 3 `unparseable_answer` | V |
| A24 | Load s / gen s / tokens / tok/s: 106.2 / 99.7 / 25,568 / 256.3; 124.3 / 117.4 / 25,852 / 220.2; 186.3 / 8.7 / 18,219 / 2,094.2 (l.512–514) | SUMMARY lines in `status.jsonl` match. Summed per-row `completion_tokens` = 25,568 / 25,852 / 18,219. tok/s is tokens over unrounded `gen_s` (`gen_tasks.py`), so 25568/99.7 = 256.4 versus 256.3 is rounding | VN: the load and gen timers are self-reported and cannot be recomputed |
| A25 | CPU-vs-TPU (Qwen3.5-2B) item agreement 22/28 Python, 24/26 Rust (l.520) | recomputed. Python disagreements: MBPP/731, HE/153, MBPP/809 (CPU fail, TPU pass); MBPP/238, /171, /233 (CPU pass, TPU fail). Rust: HE_39 (fail→pass), mbpp_266 (pass→fail) | V |
| A26 | TPU-host Python 28/28 FAILED for each model; Rust and math rescore = host status on every item (l.495–498) | SUMMARY lines: python FAILED 28 ×3; `status_tpu_host == status` for 26/26 Rust and 26/26 math for every model | V |
| A27 | Smoke run #1: Qwen2.5 load 208.9 s, 442 tokens in 0.61 s, 724.6 tok/s; gate UNVERIFIED (no bwrap); Qwen3.5-4B exit 124 at 1,500 s; run #2 Qwen3.5-0.8B 16 tasks (l.466–476) | read-only `gcloud storage cat` of `res_qwen25.json` and `status.jsonl`: values match; 442/0.61 = 724.6. Run #2 summary n = 16 | V |
| A28 | Ledger 5 lines, total \$1.54 (1.5399) (l.70, 274, 286, 538, 550) | per-line USD = seconds × price/3600 is correct for all 5 lines and they sum to 1.5399. **The ledger now has 7 lines, \$1.9809** | **W (stale)**: see must-fix 3 |
| A29 | v5e list prices: on-demand \$1.20/chip-h; spot us-west4 \$0.494237 (l.288–289, 551) | cached Cloud Billing Catalog SKU dump (scratchpad `sk_6F81-5844-456A.json`, read 2026-10-07): "TpuV5e running in Americas" 1.2, "…Las Vegas" 1.2, "…Spot…Las Vegas" 0.494237 | V |
| A30 | L2 took 72 minutes wall (l.243) | sum of per-call `wall_s` = 71.9 min; STAGE_LOG "about 72 min" | VN: sum of sequential call times, not a start/end timestamp |
| A31 | 13 of the 15 sha256 values in the frozen-inputs table (l.578–592): manifest, tasks_E_night, tasks_E_primary, FREEZE_ADDENDUM, decontam_report, L2score rows/gens, L2fix rows/gens, rust_harness_validation, tpu_run3/summary, both .d25 files | `sha256sum` on each file: all 13 match; L2, L2score and L2fix `gens.jsonl` are byte-identical | V |
| A32 | numbers_v4.json (v3.3.0) sha `e5488f6c…` (l.585) | tag v3.3.0 file = `fccb1695…`; `e5488f6c` = commit b5be32a | **W**: see must-fix 1 |
| A33 | numbers_v4.json (v3.3.1) sha `952bfc77…` (l.588) | committed file = `5f3c2a6e…`; `952bfc77` = scratch file | **W**: see must-fix 2 |
| A34 | llama.cpp b11476 asset sha `2cda5ff9…4b5e`; GGUF 2B rev f6d5376b sha `aaf42c8b…9223`; 4B rev e87f1764 sha `00fe7986…11a4` (l.569–572) | GitHub release API digest matches; HF API LFS sha256 at those revisions matches both | V |
| A35 | HF dataset revisions MATH-500 6e4ed1a2, humanevalplus d32357cf, mbppplus b2d74c91, MultiPL-E 28441b60 (l.594–595) | HF API `/revision/<sha>` resolves all four | V |
| A36 | rows/gens hash chains verify (l.595–596) | independent re-implementation of `chain_hash` (sha256 of prev + canonical JSON without `hash`): L2fix rows, L2fix gens and L2score rows all verify | V |
| A37 | D25: 11/11 E-night and 161/161 primary HumanEval+ gates leaked; 4 carried a reference impl; d25 = 80 / 1,536 rows, 5 excluded; prompts and hidden checks unchanged (l.435–443) | my literal check: hidden `inputs` list present in 11/11 night gates (0/11 after d25); primary 157/161 by my heuristic plus 4 with `ref_func` = 161. d25 prompts and `checker_payload` identical for all 80; 11 gates changed. `gate_regen_d25_report.json`: 1,536 rows, 5 excluded | VN: 4 of the 161 primary items were confirmed via `ref_func`, not by my literal match |
| A38 | D18: all 170 MBPP-sanitized calibration items removed (l.216) | decontam_report `train_sets`: 7 + 120 + 43 = 170 flagged, 0 kept | V |
| A39 | D14/D15/D19: `-t 8` 0.28–0.30 tok/s, `-t 4` 3.4; llama-bench 2.2/3.2/3.4; 64 tokens; assumed 196.6 tokens/item (l.213, 217) | `L0_env.json` (2.22/3.19/3.42; max_tokens 64), `E_night_sizing.json` (196.6) | VN: these are recorded values, not re-measured. The 0.28–0.30 figure is not in `L0_env.json`, which says "~0.3" (D14 text). The actual mean was 194.1 tokens/item |
| A40 | E pool 542 (164 + 378) / 510 / 500 = 1,552; 1,541 included; primary file 1,541 (l.162–163, 201, 232) | freeze addendum rows 164/378/510/500; `tasks_E_primary.jsonl` 1,541 rows (534 / 507 / 500) | V |
| A41 | Commits: v3.2.0 = 825d1ba, first draft 9b78e18, PR #4 = 8cb37ad, fixes 2c1f6b1 (l.561–566) | `git tag --points-at 825d1ba` gives v3.2.0; the other commits exist with matching subjects | V |
| A42 | Data lake `night/2026-10-07/` holds 6 objects; TPU logs under `tpu_smoke/`; scored rows not in the lake (l.600–607) | read-only `gcloud storage ls`: 6 objects; `tpu_smoke/` has smoke, run2 and run3 folders; no `gwenlaya_v4/results/` | V |
| A43 | `gate` null in all 80 rows (l.320, 429) | 0 non-null | V |
| A44 | D1–D20 before generation; D21 during L2; D22–D23 after generations; D27–D30 after v3.3.0 (l.191–193) | git timestamps: 9b41bda 23:42 and 521835e 00:35 < L2 64441eb 01:56; d4aecfd 02:04; 6e0a559 02:13; v3.3.0 03:18 < 97526d3 06:39 and 2c1f6b1 07:18 | V |
| A45 | The only GPU quota unit was held all night by a foreign VM (l.188–189) | `night/logs/gpu_slot.log`: 131/131 polls (23:50 to 21:54) show usage 1/1 by `socreateai-agora-hermes-node1` | V |

### B. Code and system claims

| ID | Claim (tex line) | Evidence | Status / fix |
|---|---|---|---|
| B1 | Rust link failure: `cc` resolves through `/etc/alternatives`, which was not mounted (l.116–118) | `/usr/bin/cc -> /etc/alternatives/cc`; pre-fix `gwaya/sandbox.py` (97526d3^) binds no `/etc/alternatives`; the fixed sandbox `--ro-bind-try`s it | V |
| B2 | Forgery closed: nonce on stdin, separate compile and run sandboxes, no `/proc` in the run sandbox, `unsafe` / `link_section` / `asm` rejected (l.119–123) | `gwaya/oracles.py` l.249–291 (`stdin_data=nonce`, two `run_in_sandbox` calls, `mount_proc=False`); regex l.76–78 also rejects `no_mangle`, `export_name`, `used`, `link`, `global_asm`, `naked_asm`. Pre-fix code embedded the nonce in the program. `tests/test_rust_harness_hardening.py`: 15 passed | V |
| B3 | `assert_ne!` counted; Rust PASS = "all `assert_eq!`/`assert_ne!` lines" (l.123, 449) | `_RUST_ASSERT_MACRO = assert(_eq\|_ne)?!` | VN: plain `assert!` is also counted. Suggested: "all `assert!`/`assert_eq!`/`assert_ne!` calls" |
| B4 | Validation: 25/25 hand-written references pass, 9/9 wrong fail, on 25 of 26 tasks (`HumanEval_132` has none) (l.124–126, 424) | re-ran `scratchpad/rust_reference.py` on `tasks_E_night.jsonl`: 25 reference VERIFIED, 9 wrong FAILED; status map identical to the committed JSON; the missing task is `rs/HumanEval_132_is_nested` | VN: the JSON contains no solution source and the generator script is not in the repo, so it cannot be reproduced from the repo. Commit `rust_reference.py` |
| B5 | Examined compile failures are model errors (String/str indexing by `usize`, `.unique()`) (l.127–129) | spot check: gen 1 (`mbpp_748`) indexes a `String` with `str1[j]`; 1 Rust gen calls `.unique()` | VN: spot-checked, not all 20 failures |
| B6 | Math went through the code path because the prompt builder falls back to `python` (l.405–406) | 64441eb `gwaya/generators.py` l.90 `_FENCE_TAG.get(domain…, "python")`; all 26 math texts start with a code fence | V |
| B7 | Boxed-only is the scoring default; fallback opt-in; self-consistency key bug fixed (l.132–137) | `check_math(..., boxed_only=True)` default; `extract_final_answer(boxed_only=False)` used only on request; key-bug fix is described in the 2c1f6b1 message | VN: I did not inspect the consensus-key diff line by line |
| B8 | D29: a uv-managed interpreter could not start in bwrap; fixed (l.138–141); run #2 Python invalid "(D29)" (l.477) | 2c1f6b1 message; run #2 TPU `gate_selftest`: `python_ok False bwrap`; run #3: `python_ok True` | VN: I cannot reproduce the pre-fix environment. D29's text says "on the re-scoring machine" only, while l.477 uses it for the TPU host. Suggested: "(D29; run #2 self-test `python_ok False`)" |
| B9 | Run #3 TPU-host Python failed "because sandboxed checks were launched from inside the vLLM process" (l.143–144, 494–496) | outcome verified (A26); the cause is not logged; the run #3 self-test passed Python in bwrap | NV. Suggested: "failed for every item for a reason not recorded (the gate self-test on the same host passed)" |
| B10 | Qwen3.5 runs through the PyTorch fallback with `SKIP_JAX_PRECOMPILE=1`; Qwen2.5 is JAX-native; arch `Qwen3_5ForConditionalGeneration` (l.470–474, 525–527) | GCS smoke log: "not registered in tpu-inference. Falling back to vLLM-native Pytorch"; the JAX-native list includes `Qwen2ForCausalLM`; `scratchpad/tpu_smoke/run3.sh` sets `SKIP_JAX_PRECOMPILE=1` for both Qwen3.5 steps | VN: the run #3 scripts are not in the repo |
| B11 | Run #3: one on-demand v5e, us-west4-a, vLLM 0.31.0, bf16, greedy, max_tokens 1024, non-thinking, all 80 prompts submitted together (l.488–490, 508) | run3 logs `dtype=bfloat16`; `gen_tasks.py`: `temperature=0.0, max_tokens=1024`, `enable_thinking=False`, one `llm.chat` call over all prompts; ledger zone us-west4-a; venv check shows 0.31.0 | V |
| B12 | TPU vs CPU is a check "across hardware and quantisation"; run #3 used "the fixed prompts of D28" | the engine, code system prompt and prefill/stop differ (see must-fix 6) | **O** |
| B13 | Run #3 used the fixed (2c1f6b1) code | `scratchpad/tpu_smoke/repo.tgz` (07:13): generators, math_check, sandbox, oracles and checkers are byte-identical to 2c1f6b1 | V |
| B14 | No TPU LoRA path: Qwen2.5 `enable_lora` failed at engine init; Tunix 0.1.7 lists Qwen2/Qwen3; MaxText has Qwen3.5 MoE only (l.478–482) | GCS `lora_serve_qwen25` log: "Engine core initialization failed"; `tunix_install`: `0.1.7 [... 'qwen2', 'qwen3' ...]`; `maxtext_qwen_support`: `qwen3.5_35b_a3b`, `qwen3.5_397b_a17b` (MoE) and qwen2.5 dense | VN: an absence claim limited to two stacks; vLLM-TPU LoRA was tested only on Qwen2.5 |
| B15 | H13 holds "by construction" because the ledger refuses over-cap stages (l.70–71, 292–294) | see must-fix 5 | **O** |
| B16 | H15 test passes but sends no SIGKILL; it appends a torn line (l.299–302) | ran `test_resume_after_kill_mid_run`: 1 passed; code calls `run_task` once and writes a torn line, with no `os.kill` | V |
| B17 | CWR counts UNDECIDED as wrong, so it is an upper bound (l.690) | `answered=True` in 80/80 rows; `cwr = mean(ans & ~cor)` | V |
| B18 | Reliability diagram uses equal-mass bins; pooled curve dashed (l.357–359) | `make_figures`: 15 equal-mass groups; pooled curve `"--"` | V |
| B19 | The D25 builder fails closed and a test asserts no hidden element in the gate (l.440–442) | `tests/test_eval_manifest.py`: 12 passed (l.58–60 assert hidden literals absent); d25 night file has 0 leaks (A37) | V |
| B20 | Python PASS = "the full EvalPlus test suite run in bwrap" (l.448–449) | `checker_payload.tests` is 20–93 kB for HE+ and 3–11 kB for MBPP+; checker runs in bwrap (`GWAYA_ALLOW_UNISOLATED` unset; re-scoring worked) | VN: I did not diff payloads against upstream EvalPlus |

### C. External facts and citations

| ID | Claim | Evidence | Status / fix |
|---|---|---|---|
| C1 | `evalplus`: Liu, Xia, Wang, Zhang, NeurIPS 2023 | arXiv 2305.01210: title and 4 authors match | VN: venue from auditor knowledge, not fetched |
| C2 | `multiple`: MultiPL-E, "A Scalable and Polyglot Approach…", IEEE TSE 2023, 13 authors | Crossref 10.1109/TSE.2023.3267446: same title, TSE vol 49(7), pp. 3675–3691, 2023; arXiv 2208.08227 authors match (the arXiv title says "Extensible") | VN: add `volume={49}, number={7}, pages={3675--3691}, doi={10.1109/TSE.2023.3267446}` |
| C3 | `math`: Hendrycks et al., NeurIPS D&B 2021 | arXiv 2103.03874: title and 8 authors match | VN: venue not fetched |
| C4 | `letsverify`: Lightman et al., ICLR 2024 | arXiv 2305.20050: 10 authors match; web search confirms ICLR 2024 | V |
| C5 | `selective`: Geifman & El-Yaniv, NeurIPS 2017 | arXiv 1705.08500 matches | VN: venue not fetched |
| C6 | `guo`: Guo, Pleiss, Sun, Weinberger, ICML 2017 | arXiv 1706.04599 matches | VN: venue not fetched |
| C7 | `conformal`: Angelopoulos & Bates, arXiv:2107.07511, 2021 | arXiv record matches | V |
| C8 | `holm`: Scand. J. Stat. 6(2):65–70, 1979 | web search: matches | V |
| C9 | `mcnemar`: Psychometrika 12(2):153–157, 1947 | Crossref 10.1007/BF02295996: matches | V |
| C10 | `vllm`: Kwon et al., SOSP 2023 (29th) | arXiv 2309.06180: title and 9 authors match | VN: venue not fetched (SOSP '23 is the 29th SOSP) |
| C11 | `llamacpp` release b11476, 2026 | GitHub release b11476 published 2026-10-07 | V |
| C12 | `gwayav3` title "GWAYA v3: A Fail-Closed Verification Gate for LLM-Generated Code", v3.1.0, DOI 10.5281/zenodo.23121788 | Zenodo API: DOI and version 3.1.0 correct; the title differs | **W**: see must-fix 4 |
| C13 | Qwen3.5 0.8B/2B/4B exist; Qwen3.8 exists (l.48, 474–478) | HF API: Qwen/Qwen3.5-{0.8B,2B,4B}, apache-2.0, `Qwen3_5ForConditionalGeneration`; Qwen/Qwen3.8-27B and others listed | V |
| C14 | Laya head on "the `convaiinnovations/laya` encoder" (l.159) | HF model exists, apache-2.0, architecture `LayaTypedDecisions` | VN: I did not verify that it is an encoder |
| C15 | Pool sizes; miniF2F Lean 4 has 244 items (l.162–164) | HE+ 164, MBPP+ 378, MultiPL-E Rust 510, MATH-500 500 (addendum and decontam `n_items`) | VN: the miniF2F count (244, the standard test split) was not fetched |
| C16 | Benchmarks "very probably seen in Qwen3.5 pretraining" (l.389–391) | the claim is hedged as a probability | NV (no fix needed) |
| C17 | README badge "CI passing" | `.github/` untracked; no workflow runs | **O**: see must-fix 7 |
| C18 | README: n=60 run was "Local CPU, Ollama 0.1.44" | `results_n60_cpu.json`: `hardware` and `ollama_version` "unspecified"; other README numbers (68.3/71.7/75.0, p 0.50/0.125/0.289; n=257 rows) match the files | VN: the provenance of "CPU / 0.1.44" is not recorded in the result file |
| C19 | README links: Zenodo v3.1.1 DOI 10.5281/zenodo.23123923; HF dataset; 7-page v3 PDF; MIT license | Zenodo API: v3.1.1 record; HF dataset `callensxavier/gwaya-v3-verified-report` exists; `pdfinfo` reports 7 pages; LICENSE is MIT | V |

### D. Logical and statistical claims

| ID | Claim | Check | Status / fix |
|---|---|---|---|
| D1 | No pre-registered hypothesis test was completed (l.54) | only one unregistered arm exists; no `scores_crossfit.jsonl`; H1/H3/H2/H7 need paired arms | V |
| D2 | Holm m = 2, a primary that did not run enters with p = 1, so the other is tested at 0.025 (l.168–169) | Holm: the smallest p is compared to α/m = 0.05/2 | V |
| D3 | Non-significant H1 at n = 80 would be inconclusive (l.239–240, 661–662) | power 0.052 (A21) | V |
| D4 | n=60 bootstrap lower bound is positive "almost by construction" (l.97–100) | with 4/0 discordance, Δ ≤ 0 needs zero discordant draws: (56/60)^60 = 0.0159 < 0.025, so the 2.5th percentile is > 0 | V |
| D5 | Python AUROC CI includes 0.5 (l.62, 367) | the paper's [0.483, 0.917] and my [0.489, 0.914] both include 0.5 | V |
| D6 | "Badly calibrated": conf ≈ 0.9 versus pass < 0.5 (l.61, 366, 396–398) | 0.896 vs 0.393 (Python), 0.894 vs 0.231 (Rust) | V |
| D7 | Math ECE equals mean confidence | with all labels 0, ECE = mean p under any binning | V |
| D8 | The machine table's math "as scored" and boxed-only rows, and its two pooled rows, coincide (l.349–351, 693–694) | `tables_v4.tex` l.15 = l.18 and l.19 = l.20 | V |
| D9 | Same counts, not the same items (l.68–69, 518–520) | A25 | V |
| D10 | Truncated math counts are lower bounds under this budget (l.524–525) | truncated items are UNVERIFIED, never PASS, so pass counts can only rise with a larger budget | V |
| D11 | Hypotheses were pre-registered "before any GwenLaya run" (l.50–53) | prereg first committed 2026-10-07 23:33 (0b4f104), after the TPU smoke tests (21:54–22:38, which produced no study data) and before L2 (about 00:44 to 01:56) | VN: suggested "before any GwenLaya evaluation data were generated" |
| D12 | "Run #2 … outputs are not used in this report" (l.553–554) | l.475–476 cites run #2 (Qwen3.5-0.8B generated 16 tasks) as feasibility evidence | VN: suggested "run #2 scores are not used; only its generation count is cited (Section 5)" |
| D13 | v3.3.1 is a re-score only; generations unchanged (l.246–247, 568) | sha of L2, L2score and L2fix `gens.jsonl` all `03f6e2b6…` | V |
| D14 | The validation covers 25 of 26 Rust tasks and is not an independent reference set (l.642–645) | B4; the references were written for this purpose | V |
| D15 | The scored arm is not a registered arm (l.55, 307) | arm `base` is not in the registered arm set (B1–B6, GL, LR, SH) | V |

## Method and limits

- **Recomputation.**
  - `scratchpad/audit_recompute.py` is stdlib plus numpy and does not import repo code. It reads `night/cache/L2fix/{rows,gens}.jsonl` and `tasks_E_night.jsonl`. It computes counts, Wilson intervals, ECE (15 equal-mass and 10 equal-width bins), Brier, pairwise AUROC, pessimistic-tie AURC, CWR, and cost per pass (total `cpu_seconds` over passes), with my own bootstrap (`random.Random(12345)`, 10,000 resamples, stratified by domain).
  - `scratchpad/audit_rescore.py` re-ran `gwaya.domains.checkers.check` in bwrap (`GWAYA_ALLOW_UNISOLATED` unset) on all stored CPU and TPU texts, then compared the results with the stored statuses.
- **Not independently re-checkable:**
  - TPU load and generation timers (self-reported by the run).
  - The billed GCP amount (the paper already marks it TBD).
  - The cause of the TPU-host Python failure (B9).
  - The pre-fix uv environment (D29).
  - Venue metadata for C1, C3, C5, C6 and C10, which comes from auditor knowledge. The arXiv id, title, authors and year were fetched live.
  - The miniF2F size.
  - Whether `convaiinnovations/laya` is an encoder.
  - The v5e prices were read from a Cloud Billing Catalog dump cached in the scratchpad on 2026-10-07. I did not query the live catalog (the public pricing page did not render).
- **Not done:**
  - The figures were not compared pixel-by-pixel; only their generating code was read.
  - The red-team priors were checked against `DEVIATIONS.md` (all 15 match) but not against `PROPOSAL_REVIEW.md` row by row; H13's 0.75 and the "close to tautological" quote were confirmed there.
  - Prereg wording was checked only for H1/H3/H13, the MEIs, Holm, the budget and §10.
- **Live state.** A TPU E-generation job ran during the audit and added two ledger lines (A28). The ledger was only read, and nothing under `night/cache` was modified. Any later run will change the ledger total and, if `analyze_study.py` is re-run, the `numbers_v4.json` hash again.
