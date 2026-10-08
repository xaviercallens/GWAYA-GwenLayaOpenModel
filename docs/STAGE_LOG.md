# Stage log

## L0 (local env), night 2026-10-07

- venv: numpy, scipy, datasets, safetensors, pytest, scikit-learn, transformers, peft, torch (CPU wheel) installed with uv; caches on /mnt/data/xdev-cache.
- llama.cpp release b11476 (ubuntu-x64 CPU) in `$NIGHT/bin`; Qwen3.5-2B and 4B Q4_K_M GGUFs from unsloth (revisions and sha256 in `$NIGHT/results/L0_env.json`).
- llama-server on 127.0.0.1:8091, `--parallel 2`, `-t 4` (D14). Logprobs are returned.
- Measured CPU speed is very low (about 3 tok/s for 2B, see L0_env.json), which constrains E-night size. This is the main finding of the step.
- Offline pytest (metrics_selective, run_study, oracles_fail_closed, rust_oracle, domains): 85 passed, 1 skipped (linker not reachable inside the sandbox). One failing test fixed (D16).
- GCP preflight (read-only): GPUS_ALL_REGIONS usage 1.0 / limit 1.0, held by socreateai-agora-hermes-node1 (not ours). No spend. Poller W running, log `$NIGHT/logs/gpu_slot.log`.

### L0 stage review (2026-10-08 00:20)

Verdict: **trustworthy, continue** (L1 next; G1 likely SKIPPED).

- Checked: `L0_env.json` has the build tag, both GGUF sha256 and revisions, tok/s per tier, logprobs true, pytest counts. `toks_2B.json` / `toks_4B.json` have 10 rows each with `lp: true`. llama-server `/health` = ok, both slots idle. D16 diff is limited to multi-letter words in `_to_sympy_src` (sqrt/pi kept), which is fail-closed. No generation has happened yet, so leakage and UNVERIFIED-rate checks do not apply at this stage.
- Main risk: throughput. 2B ≈ 2.7 tok/s wall, 4B ≈ 1.3 tok/s wall (3 foreign CPU jobs still hold about 3 of 8 cores). At max_new_tokens 1024, L2 (3 h) and L3 (4 h) will produce only tens to low hundreds of items. Paired n will be far below 500, so H1 must carry the underpowered flag. L1 must compute n_d from these measured rates and write it to the freeze addendum **before** generation. Any lower max_new_tokens or thinking budget is a new deviation that has to be logged.
- GPU: the poller shows GPUS_ALL_REGIONS 1.0/1.0 held by socreateai-agora-hermes-node1 on every poll (23:50, 00:00, 00:10). G1 runs only if the slot frees up. E1 depends on G1, so expect both to be SKIPPED.
- Ledger: the two existing lines have no `usd_estimate`, so the prior ~$0.84 is still counted by hand under the schedule's hard_rule. Spend is ~$0.84 of $45; L0 spent $0.

## L1 (C0-lite manifest and freeze), night 2026-10-07

- Loaded humanevalplus (164), mbppplus (378), MultiPL-E humaneval-rs (156) and mbpp-rs (354), MATH-500 (500); HF revisions and licenses are in `experiments/night/manifest_eval.json`.
- Manifest: 1552 rows, 11 excluded by harness sanity (python 8, rust 3, math 0), 1541 included. Python reference solutions failed on timeouts (3), a MemoryError, one wrong test, and 3 specs with no assert statement; Rust uses a typecheck-only check because MultiPL-E has no reference solution (D17).
- Gate-visible versus hidden split, Python/Rust clusters (HumanEval/N, MBPP/N), decontamination against the MBPP sanitized calib source (all 170 removed, D18).
- E-night n_d = python 28, rust 26, math 26 (sum 80), bound by L3 (4B, 4 h) at 1.29 tok/s with an **assumed** 197 tokens per item (D19). Power at n = 80 is low for every registered effect size (6 pts: 0.26 at alpha 0.025); H1 will be inconclusive unless n grows.
- Freeze addendum `docs/FREEZE_ADDENDUM_NIGHT_2026-10-07.md`, its sha256 is in the prereg Addenda and a new block in `docs/PREREG_FREEZE.txt`. No generation had happened when it was written.
- Code: `scripts/data/build_eval_manifest.py`, `run_c0_lite.py`, `run_c0_finalize.py`; 11 offline tests in `tests/test_eval_manifest.py`.
- Uploaded manifest, both task files, sizing, addendum and decontam report to `gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/night/2026-10-07/`; ledger line with usd_estimate 0.0. No GCP spend.

### L1 stage review (2026-10-08 00:37)

Verdict: **trustworthy, continue** (G1 expected SKIPPED; L2 next).

- Row counts match the plan and the addendum. Manifest has 1552 rows (python 542 = 164 + 378, rust 510 = 156 + 354, math 500) and 1541 included. No duplicate task_ids. `tasks_E_primary.jsonl` has 1541 unique lines, `tasks_E_night.jsonl` has 80 (python 28, rust 26, math 26). E-night equals the `in_e_night` items, is a subset of primary, and contains no excluded item. Every row has a prompt and a checker payload. Gate payloads are empty only for math (500/500), which matches the design.
- Hashes recomputed: the manifest, both task files, the sizing file and the decontam report all match the sha256 values in the addendum. The addendum sha256 (`ac2970…376c`) matches `addendum.sha` and the prereg Addenda entry. The repo manifest and the `$NIGHT/results` manifest are byte-identical. The GCS prefix has the 6 objects (29.2 MiB), and their sizes match the local files.
- Sandbox: the Python harness sanity ran through `check_python` (bwrap is present at /usr/bin/bwrap). 534 of 542 references came back VERIFIED, so the sandbox works and is not failing everything closed. UNVERIFIED appears only 3 times (MBPP/737, 787, 794). Those are per-item timeouts or memory errors, not an infrastructure failure. Rust sanity is only a `rustc --emit=metadata` typecheck, because MultiPL-E has no reference solution (D17). The Rust harness is therefore not validated end to end on a passing program, so watch the Rust VERIFIED rate in L2. Math: 500/500 gold answers self-verify and match the reference solutions. That is plausible for MATH-500 but has not been audited by hand.
- Leakage: none of the eval items come from the eval side's own sources. The 243 mbppplus and 147 mbpp-rs matches are against MBPP-sanitized train/val/prompt. All 170 of those train items were removed, and no T/C set is used tonight (D18). Verification rechecks are ok and `missing_eval_sets` is empty.
- `pytest tests/test_eval_manifest.py`: 11 passed (rerun in this review).
- Main risk: **mean tokens per item (196.6) is assumed, not measured (D19).** At 1.29 tok/s the 4B tier makes about 18.6k tokens in 4 h. If real completions run close to max_new_tokens 1024 (typical for math without thinking), L3 will finish only about 20 to 40 items and paired n will be far below 80. L2 should log the mean completion tokens of its first about 10 items to the stage log. n_d must not change after the freeze, and truncation is reported as is. H1 is already underpowered at n = 80 (power 0.05 at the 3 pt MEI) and will be reported as inconclusive if it is not significant.
- GPU slot: GPUS_ALL_REGIONS is still 1.0/1.0, held by socreateai-agora-hermes-node1 (00:10, 00:20, 00:31). Expect G1 and E1 to be SKIPPED.
- Spend: ~$0.84 of $45 (L1 spent $0.0, storage is negligible). The two TPU ledger lines still have no `usd_estimate`.

## L2 (2B generation on E-night), night 2026-10-07

- Ran `run_study.py --stage night_L2` (plan `experiments/night/plan_night.json`, D21) against llama-server 2B Q4_K_M on 127.0.0.1:8091, greedy, thinking off, max_new_tokens 1024, `--cpu-pid` for per-call CPU-seconds. Status COMPLETE (log ends DONE), 80 of 80 items (python 28, rust 26, math 26), about 72 min wall (limit 3 h).
- `$NIGHT/cache/candidates_2b.jsonl`: 80 lines, every line has non-empty `token_logprobs` and `cpu_seconds` (checked). Server CPU total 16498 s, client wall total 4313 s.
- Finish reasons: 75 stop, 5 length (truncated at 1024 tokens). Mean completion tokens 194.1 (python 118.8, rust 98.5, math 371.0), versus the 196.6 assumed in D19: the assumption held for the 2B tier. Mode was generate, so nothing is scored yet.
- Code: logprob parser for llama.cpp format, `proc_cpu_seconds`, SIGINT handled as PARTIAL; 3 new tests in `tests/test_run_study.py` (20 pass).
- Note: the 4B server was stopped; L3 must start it again (same flags, model Qwen3.5-4B-Q4_K_M.gguf).

### L2 stage review (2026-10-08)

Verdict: **not trustworthy for the math domain. Do not start L3 as planned.** Python and Rust generations look sound. Fix the math prompt path first and regenerate the 26 math items on 2B (see below).

- Row counts match the plan. `candidates_2b.jsonl` and `L2/gens.jsonl` each have 80 lines and 80 unique task_ids (python 28, rust 26, math 26). Their task_id set equals `results/tasks_E_night.jsonl`. No completion is empty or duplicated, and none has `<think>` tags. `token_logprobs` lengths are 15 to 1024 and `cpu_seconds` is 29.4 to 1348.5 s. The log ends `COMPLETE ... DONE rc=0`. The stage only generated, so the sandbox was not exercised and the UNVERIFIED rate and scoring checks do not apply yet. Leakage: no new data source, so nothing new to flag.
- **Blocking defect (math prompt).** All 26 of 26 math completions are a fenced ```python program. 0 of 26 contain `\boxed`, and 21 just `print` a numeric approximation. Cause: `OpenAICompatBackend.generate` (scripts/run_study.py:244) builds the prompt with `OllamaGenerator.build_raw_prompt`. For any domain not listed in `_FENCE_TAG`, including `math`, that function falls back to tag `python`. It uses the system prompt "You are a careful python programmer. Reply with one complete, runnable code block only ... Do not explain." It also prefills the assistant turn with "```python", and "```" is a stop sequence. That overrides the frozen user prompt "Put the final answer in \boxed{}". Every math item would therefore score FAIL or UNVERIFIED for a reason that has nothing to do with model capability. L3 (4B) uses the same code path and would waste about 4 h of CPU to produce the same artifact.
- Remediation (has to be logged as a new deviation in DEVIATIONS.md before any rerun): for `domain == "math"`, use a math system prompt (for example `data_build.py`'s "Reason briefly and put the final answer in \boxed{}.") with no code prefill and no "```" stop. Then regenerate only the 26 math items for 2B, keeping the old ones in a discarded cache, and run L3 with the fix. The python and rust prompt path (code-only system prompt plus fence prefill) is unchanged. It is not in the freeze addendum, so record it as a clarification in the same deviation.
- Consequence for D19: the math mean of 371.0 tokens was measured on code output, not on reasoning. Real \boxed reasoning will probably be longer, and 4 of the 5 `length` truncations are already math. The L2 confirmation of the 196.6 tok/item assumption does **not** carry over to math or to L3 sizing. Recheck the mean math tokens after the rerun. n_d stays frozen, and truncation is reported as is.
- Python and Rust look plausible: 28 of 28 contain `def`, 26 of 26 contain `fn`, and only 1 python item hit length. Rust quality looks weak (the sampled item indexes a `String` by integer, which will not compile). That is a model result, not an infra problem, but the Rust harness still has never been validated on a passing program (D17).
- Spend: ~$0.84 of $45. L2 spent $0, local CPU only.

## Analysis script (night 2026-10-07/08)

- `scripts/analyze_study.py` implements the pre-registered analysis (cluster-stratified paired bootstrap, exact McNemar, Holm over the kept family, ECE/Brier/AUROC/AURC, CWR, cost per correct answered item; H1, H3 primary, H4-H6 from a cross-fitted scores file, H2/H7 exploratory, H13 from the ledger). Outputs `papers/numbers_v4.json` (each number with its source), `papers/tables_v4.tex`, `papers/figures_v4/`. 14 synthetic known-answer tests in `tests/test_analyze_study.py` pass (D22).
- Run on the current data it has only generation descriptives (no scored rows exist yet; the math prompt defect from the L2 review is still open), so H1/H2/H3/H7 are `not_run` and every arm cell is TBD. H13 reports only the 0.0 USD known in the ledger (2 lines carry no usd_estimate).

## v3.3.1 corrections: Rust harness, math scoring, re-score, TPU run #3 (2026-10-08)

Branch `fix/math-and-rescore`. No new CPU generation; no registered test ran. Deviations D27-D30.

- **Rust harness (D27, PR #4, 8cb37ad).** In bwrap, rustc's linker `cc` resolves through
  `/etc/alternatives`, which was not mounted, so no Rust program could link; the v3.3.0 Rust 0/26
  was a harness artefact. The nonce-in-source forgery was reproduced and closed (nonce via stdin,
  separate compile/run sandboxes, no `/proc` in the run sandbox, `unsafe`/`link_section`/`asm`
  rejected); `assert_ne!` now counted. Validation: 25/25 reference solutions VERIFIED, 9/9 wrong
  solutions FAILED (`results/gwenlaya_v4/rust_harness_validation.json`).
- **Math and Python fixes (D28, D29, 2c1f6b1).** Prose math prompt ending in `\boxed{}`, Qwen3.x
  non-thinking via an empty think block, boxed-only scoring by default (last-number fallback
  opt-in), self-consistency math key fixed; uv-managed interpreter symlinks recreated so Python
  starts inside bwrap.
- **Re-score of the night arm.** The unchanged L2 generations were re-scored into
  `night/cache/L2fix/rows.jsonl` (sha256 `a5e72614...70a8`; copy in `results/gwenlaya_v4/night_L2fix/`)
  and `analyze_study.py` was rerun (numbers_v4.json sha256 `952bfc77...eb74`). Result: python 11/28
  PASS_HIDDEN (unchanged), rust 6/26, math 0 pass / 26 UNDECIDED (boxed-only; generations still
  invalid). Hash chains OK per numbers_v4 meta.
- **TPU run #3 (D30).** One on-demand v5e chip (us-west4-a, 982 s, ledger 0.3273 USD catalog
  estimate), vllm-tpu 0.31.0, bf16, greedy, max_tokens 1024, non-thinking. All 80 tasks for
  Qwen3.5-2B, Qwen3.5-4B, Qwen2.5-Coder-1.5B-Instruct. Qwen3.5 ran via the PyTorch fallback with
  `SKIP_JAX_PRECOMPILE=1`. TPU-host Python scoring invalid (all FAILED); all rows re-scored locally
  (`results/gwenlaya_v4/tpu_run3/summary.json`). Python/Rust/Math passes: 2B 11/6/16, 4B 20/13/16,
  Coder 18/15/8; math truncations at 1024 tokens (UNVERIFIED): 9, 9, 4. Descriptive only.
- **Spend.** Ledger 5 lines, sum of usd_estimate 1.5399 USD (catalog list-price estimates; billed
  amount TBD). TPU run #2 (1073 s, 0.3577) is in the ledger; its outputs are not used in the paper.
- **Paper.** `papers/gwenlaya_v4.tex` revised to v3.3.1 (corrections section, Rust and math text,
  descriptive TPU section, spend, limitations).

### v3.3.1 review note

Verdict: **descriptive results only; still no registered test.** Python and Rust night numbers are
usable as descriptive results; math remains invalid until regenerated on CPU with the fixed prompt.
The Rust validation was done after the generations existed and uses hand-written solutions, not an
independent reference set. The TPU 2B counts match the CPU arm in aggregate but not item by item
(22/28 Python and 24/26 Rust items agree).

## Revision 3.4.0: claim audit fixes and full-E baselines on TPU v5e (2026-10-08)

Branch `bench/tpu-e-eval` (base `main` = v3.3.1, 4914b5c). No registered test ran. Deviations D31-D41.

- **Claim audit** (`docs/CLAIM_AUDIT.md`, 99 claims: 61 V, 29 VN, 2 NV, 4 W, 3 O). Fixed in the paper:
  numbers_v4.json hashes (tag v3.3.0 `fccb1695...5995`, tag v3.3.1/HEAD `5f3c2a6e...a1c0`; recomputed
  with `git show <tag>:papers/numbers_v4.json | sha256sum`), stale spend, bib title (already fixed in
  39a5a0a, verified), H13 wording, run #3 CPU-vs-TPU wording, run #2 / D29 cross-reference, Rust assert
  counting, registration-timing wording. **Correction to the v3.3.1 entry above:** the numbers_v4.json
  hash `952bfc77...eb74` quoted there is an uncommitted scratch file; the committed file is `5f3c2a6e...a1c0`.
- **TPU E generation.** First pass (VM gwenlaya-tpu-e-gen-210955, 963 s, 0.321 USD): Coder and 2B
  generated but returned no log-probabilities, 4B failed to start; set aside. An orphan VM
  (gwenlaya-tpu-e-gen2-213823, 360 s, 0.12 USD) was created by a driver killed by mistake and deleted.
  Second pass (gwenlaya-tpu-e-gen2-214445, us-west4-a, 1644 s, 0.548 USD, via `deploy/tpu/drive.sh` with
  ledger + worst-case cap check): 1,536 tasks x {Coder-1.5B, Qwen3.5-2B at 96 seqs, Qwen3.5-4B at 32 seqs
  (96 exceeded HBM)}, bf16, greedy, logprobs=1. Raw files and logs in
  `gs://.../gwenlaya_v4/tpu_runs/gwenlaya-tpu-e-gen2-214445/`; hashes in
  `results/gwenlaya_v4/e_tpu/SHA256SUMS.txt` (verified with `sha256sum -c`; later split into
  `SHA256SUMS_repo.txt` and `SHA256SUMS_dataroot.txt`, each checked from its own root).
- **Local scoring** with `scripts/import_remote_gens.py` + `run_study.py --mode score` in bwrap (Coder one
  check process; tiers 5 workers, six arms). Serial re-check of all 1,536 non-passing Python/Rust rows:
  3 flips (py/MBPP/271 for 2B and 4B, applied as an overlay; rs/mbpp_130_max_occurrences for 4B,
  nondeterministic, stays FAILED). All 1,572 passing rows re-run: 0 changes. 5 checks per model time out
  even when run alone. Overlay importer bug (inflated seconds) fixed and tested.
- **Analysis** `scripts/analyze_e_tpu.py` (A1-A9) -> `numbers_e_tpu.json` (sha256 c76bc506...226b),
  `papers/tables_e_tpu.tex`, `papers/figures_v4/e_*.png`. Headline (pooled pass rate): 2B 0.423, 4B 0.606,
  Coder 0.482. Negative: cascade vs always-4B accuracy -0.003 (-0.014 to 0.009), cost ratio 1.357; gate
  coverage 0 on math; raw confidence ~0.93 vs 0.53-0.68 pass on the 4B. Most favourable (exploratory):
  4B gate vs the k most confident log-prob answers at matched coverage, CWR -0.076 (Python), -0.112 (Rust)
  (wording corrected 2026-10-09, see below).
- **Spend.** Ledger 8 lines, sum of usd_estimate 2.5289 USD (read 2026-10-08 23:30 CEST; list-price
  estimates; billed amount TBD).
- **Paper.** `papers/gwenlaya_v4.tex` revised to 3.4.0 (corrections since v3.3.1, full-E section with
  methods/verification/results, hypothesis status, spend, reproducibility, claim-audit appendix);
  compiles cleanly.

### 3.4.0 review note

Verdict unchanged: **no registered test.** The full-E results are exploratory baselines on public
benchmarks with one greedy sample. They show that, for this 2B/4B pair, a gate-only cascade neither
saves accelerator time nor gains accuracy over always using the 4B, and that raw log-prob confidence is
badly calibrated. The executed gate gives fewer confident-wrong answers than a log-prob threshold at
matched coverage on code. That comparison is not H1, and part of the difference comes from running a
visible test.

## Revision 3.4.0 re-audit corrections (2026-10-09)

Independent re-audit `docs/CLAIM_AUDIT_340.md` (144 claims: 108 V, 22 VN, 2 NV, 4 W, 8 O; all E-set numbers
reproduce; defects in definitions, explanations and timing). `numbers_e_tpu.json` (sha256 9197f367...47bc),
`tables_e_tpu.tex` (bb7f5786...fea6) and the figures were regenerated by the coordinator with new keys
(`A4.*.domain_matched`, `S2.*_nowarm`). Paper and docs corrected:

- Pooled A4: the global-threshold -0.042 is withdrawn; per-domain-matched pooled -0.063 (-0.074 to -0.054).
- "Threshold tuned on E favours the baseline" removed everywhere; the baseline is top-k with k = the gate's
  count in the same domain, no labels used, pessimistic ties.
- Scoring commit attribution: `results.json` records HEAD at the end of a run; runs started 22:05:17 (Coder)
  and 22:21:55 (2B/4B), before the recorded commits.
- Plan timing: committed after generation and as the 2B/4B scoring started (60 s earlier), not "before the
  results were scored".
- Engine concurrency: configured 96/32, effective at most 54 (2B) / 11 (4B); the 4B failure at 96 is a
  `ValueError: Cannot fit both KV pools`.
- Math gate: zero coverage is an implementation gap (registered program-of-thought re-execution not built).
- H1/H3 cannot be run as registered on these E generations (freeze rule; D42).
- Cost reported with and without the compilation chunk (S2): pooled 1.357 / 1.243; Python 0.997 / 0.900.
- Cascade abstentions (757 tasks), item-level bootstrap note, borderline Python interval, "no complete
  \boxed{}", truncation not post hoc, throughput typo (4,865), stale RESULTS_VERDICT bottom line.
- Paper recompiled cleanly.
