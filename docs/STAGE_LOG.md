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
