# Claim audit: GwenLaya v4 interim report, revision 3.4.0

Audited 2026-10-08/09 on branch `bench/tpu-e-eval` (HEAD `f80ac16` plus the uncommitted 3.4.0 edits to
`papers/gwenlaya_v4.tex`, `.bib`, `docs/DEVIATIONS.md`, `docs/RESULTS_VERDICT.md`, `docs/STAGE_LOG.md`).
This is an independent, adversarial check of a paper the auditor did not write. The auditor changed no paper,
doc, result, script, test or data file; this file is the only output. Line numbers (`l.`) refer to
`papers/gwenlaya_v4.tex` as it stands in the working tree.

`papers/numbers_v4.json` and `results/gwenlaya_v4/e_tpu/numbers_e_tpu.json` were **not** used as sources.
Every E-set number below was recomputed with the auditor's own code (no import of `scripts/analyze_*.py`) from
the scored rows, the overlay, `gens_summary.jsonl`, the raw generations in the data lake, the task file, the
audit JSONs, the ledger and the night rows. The derived JSON was opened only afterwards, to locate where the
paper's pooled A4 number comes from (claim E47).

## Summary

| Status | Count |
|---|---|
| VERIFIED | 108 |
| VERIFIED_WITH_NOTE | 22 |
| NOT_VERIFIABLE | 2 |
| WRONG | 4 |
| OVERSTATED | 8 |
| **Total claims checked** | **144** |

By class: A (own numbers) 52, B (code/system) 37, C (external/citations) 18, D (logic/statistics) 13,
E (wording/consistency) 24. The 8 OVERSTATED rows cover 5 distinct issues, because AB4/E3/DOC2 and
AB11/E45 are the same issue in different places.

**Every quantitative E-set result reproduces.** That covers all pass counts, A2 calibration point estimates,
A3/A4/A5/A6, McNemar counts and p, Holm, S1, truncation, token and chip-second totals, the stability-audit
counts and A7. Every bootstrap CI endpoint the auditor recomputed is within 0.005 of the paper (threshold 0.02,
or 0.05 for ratios). All 24 hashes in the paper's table match. The defects are in **definitions, explanations
and timing statements**, not in the arithmetic.

## Must-fix before publication

1. **E47 WRONG: the explanation of the pooled A4 number (l.662–663).** The paper says the pooled −0.042 "mixes
   in a domain where the comparison is empty". It does not. The pooled number uses **one global log-prob
   threshold** at the gate's pooled coverage (k = 732 of 1,536). At that threshold the baseline answers **280
   math tasks** (and 229 Python, 223 Rust). The gate answers none. So the pooled comparison is not "0 on math
   by construction". A per-domain-matched pooled difference, which is what the plan's A4 ("coverage equals the
   gate's coverage in the same domain") implies, is **−0.063 [−0.074, −0.054]**.
   *Fix:* "The pooled −0.042 uses a single threshold over all domains at the gate's pooled coverage, so the
   log-prob baseline answers 280 math tasks that the gate cannot answer. Matching coverage within each domain,
   as in the plan, gives −0.063 (CI −0.074 to −0.054). Neither pooled number is a headline." Alternatively,
   drop the pooled number.
2. **E25 WRONG: "Both scoring runs started from working trees with uncommitted changes (`dirty: true` at
   `66e2313` and `5e10ad2`)" (l.547–548).** `run_study.py` reads `git_info()` in `build_results()`, after
   `study.run()`, so the recorded commit is HEAD **when the run ended**:
   - The Coder scoring started at 22:05:17 (`results.json` `started_utc` 20:05:17Z), **before** `66e2313`
     existed (22:08:27). Its args have no `check_workers` key, which matches the pre-`66e2313` CLI.
   - The 2B/4B scoring started at 22:21:55, **before** `5e10ad2` (22:25:43) and before the plan commit
     (22:22:55).

   *Fix:* "Both scoring runs ran from working trees with uncommitted changes. `results.json` records the HEAD
   at the end of each run (`66e2313`, `5e10ad2`). The runs started at 22:05:17 and 22:21:55, so the code that
   ran was `39a5a0a` and `66e2313` respectively, plus uncommitted changes I cannot identify."
3. **NS1 WRONG and a missing limitation: "run GL, B3 (threshold on C) and B5 on the same E generations, to
   test H1 and H3 as registered" (l.1070–1071).** Prereg §6 ("Freeze rule") requires the calibrator, all
   thresholds and the E manifest hash in a signed addendum **before E is generated**, and forbids any E-based
   refitting. The full E set has now been generated, and its outcomes analysed and published, before C or Laya
   exist. A later H1/H3 run on these generations cannot be "as registered".
   *Fix:* say so in Limitations and Next steps. Either (a) regenerate E after the C6 freeze (a new seed or new
   generations), or (b) record a deviation stating that the E generations predate the freeze and that their
   outcomes were seen, so any H1/H3 run on them is a deviated test.
4. **AB4/E3/DOC2 OVERSTATED: timing of the plan against 2B/4B scoring** (abstract l.67–68 "committed before the
   2B/4B results were scored"; l.459; DEVIATIONS l.136–137). The 2B/4B scoring process started at **22:21:55,
   60 s before the plan commit**, and ended at 22:36:38, which is the file time the paper quotes. The scoring
   log prints only progress counts, not outcomes, so nothing suggests the results were seen. Still, "before
   ... were scored" is not literally true.
   *Fix:* "The plan was committed at 22:22:55, while the 2B/4B scoring was in progress (started 22:21:55,
   rows complete 22:36:38; the scoring log prints no outcomes)."
5. **AB11/E45 OVERSTATED: "the threshold was tuned on E, which favours the baseline"** (abstract l.84–85;
   l.655–656; table/figure captions; D35; limitations l.1045). The matched baseline answers the top-k tasks by
   exp(mean log-prob), with k set to the gate's count. It never uses E's labels. The only label use is
   pessimistic tie-breaking, which works *against* the baseline, and the 4B confidences contain no ties in any
   domain. Fitting k on E removes coverage mismatch, but nothing in it can lower the baseline's error. So
   "favours the baseline" and "the gate advantage is conservative" are unsupported.
   *Fix:* "The cut-off is set on E only to match coverage exactly. It uses no labels, so it neither helps nor
   hurts the baseline's error, apart from pessimistic tie-breaking, which never triggered here." Drop "This
   holds even though the threshold was tuned on E".
6. **E14 OVERSTATED: the batch sizes that define the cost regime** (l.502–505, 706–707, 719, D33 l.753, §8
   l.838–839, limitations l.1048: "96 for the 2B, 32 for the 4B").
   - The `max_num_seqs` settings were 96 and 32. But the engine logs in the data lake report that the hybrid
     KV cache was re-split for **54 concurrent requests** (2B: `_mamba_num_blocks=433`, 8/request) and **11
     concurrent requests** (4B at 32: `_mamba_num_blocks=89`). The effective concurrency was therefore at most
     54 and 11.
   - The quoted message "Mamba and attention pools together exceed the HBM budget" is an INFO line that also
     appears in the successful 2B and 4B-at-32 logs. The fatal error at 96 was `ValueError: Cannot fit both KV
     pools under gpu_memory_utilization=0.92 ...`.

   *Fix:* report "configured `max_num_seqs` 96/32; the engine log reports KV capacity for 54 (2B) and 11 (4B)
   concurrent requests", and quote the ValueError. The cost caveat's direction (a smaller 4B batch favours the
   cascade) still holds, and it is stronger than stated.
7. **E42 OVERSTATED: "The registration's 'gate' has nothing to execute there" (l.638).** The registered math
   gate (prereg §1 table) is a sandboxed program-of-thought re-execution that reproduces the boxed answer. It
   needs no task payload. What is missing is an implementation, not a registered check.
   *Fix:* "The gate implementation executes only the task's `gate_payload`, which is empty for all 500 math
   rows. The registered math gate (program-of-thought re-execution, prereg §1) is not implemented."
8. **DOC1 OVERSTATED (stale): `docs/RESULTS_VERDICT.md` l.37–38** still says "The only scored data are 80 items
   from one unregistered arm". That contradicts the 3.4.0 section of the same file.
   *Fix:* "The scored data are the 80 night items (one unregistered CPU arm) and the exploratory full-E TPU
   baselines (1,536 tasks × 3 models); none is a registered arm."
9. **E18 WRONG (cosmetic):** the Coder chunk 2–8 throughput is 4,507–**4,865** tok/s (log and raw chunk walls:
   45,395 / 9.331 s = 4,865.0), not 4,866.

**Should fix (VERIFIED_WITH_NOTE rows with a concrete wording fix):**
- **E49, cascade definition.** The `gwenlaya` arm itself **abstains** when the 4B's gate does not verify: 757
  of 1,536 tasks, including all 500 math tasks (77 Python and 180 Rust abstentions, none hidden-passing). The
  reported "cascade accuracy" is the hidden pass rate of the final candidate, abstained or not. On math, 306
  "correct" answers are ones the arm abstained on. Its selective coverage is 0.507.
- **E35, the bootstrap.** Within each domain every source-problem cluster has exactly one item (the 457
  shared problems span Python+Rust), so the "cluster bootstrap" is in effect an item-level bootstrap
  stratified by domain. A cross-domain cluster bootstrap widens pooled CIs only slightly (≤0.004).
- **E51, the Python result.** The upper bound of the Python cascade−4B CI (−0.002) is within Monte-Carlo noise
  of 0. The auditor gets 0.000 within domain and −0.002 cross-domain.
- **E34/DOC3, `sha256sum -c`.** `SHA256SUMS.txt` mixes data-root-relative and repo-relative paths. One
  `sha256sum -c` from the data root fails the `numbers_e_tpu.json` entry. Every entry matches when resolved
  against its own root.
- **E38, truncation wording.** Five truncated math answers contain "\boxed" (four unclosed; one complete, the
  2B FAILED item). Say "no complete \boxed{}".
- **T3, architecture wording.** Qwen3.5 is hybrid full/linear attention (4B: 24 `linear_attention` + 8
  `full_attention` layers, HF `config.json`). "Mamba" is vLLM's name for the state pool.

## Full table

Status: V = VERIFIED, VN = VERIFIED_WITH_NOTE, NV = NOT_VERIFIABLE, W = WRONG, O = OVERSTATED.

### Abstract and Sections 1–5

| ID | Claim (line) | Cl. | Evidence | Status | Fix |
|---|---|---|---|---|---|
| AB1 | 15 hypotheses; H1 = fewer CW than log-prob selective prediction at matched coverage; H3 = lower cost per correct answered item than largest tier with the gate; registered before any evaluation data (l.57–60) | E | prereg §7 (H1 GL vs B3, H3 GL vs B5 = B2+gate); `0b4f104` 2026-10-07 23:33 | V | – |
| AB2 | Both primaries need Laya and split C (l.61–62) | D | prereg §6: C fits the Laya calibration map, tau_route/hi/lo and B3's threshold | V | – |
| AB3 | 1,536 tasks (529/507/500); 3 models; greedy, 1 sample, bf16, vLLM, one v5e chip; scored locally (l.64–67) | A | task file counts; raw rows `temperature` 0; logs `v0.31.0`; `nothing scored on host` per pipeline | V | – |
| AB4 | Plan committed before 2B/4B were scored (l.67–68) | E | `tiers/results.json` started 20:21:55Z = 22:21:55 CEST; plan commit 22:22:55 | O | Must-fix 4 |
| AB5 | Pooled pass 0.423 / 0.606 / 0.482 (l.70) | A | own: 650, 931, 740 of 1,536 (with overlay) | V | – |
| AB6 | 4B mean conf 0.927–0.938; pass 0.529–0.675 (l.72–73) | A | own: 0.9272/0.9303/0.9378; 0.6749/0.5286/0.6120 | V | – |
| AB7 | Cascade−4B −0.003 (−0.014, 0.009); cost ratio 1.357 (1.329–1.385) (l.76–77) | A | own −0.0026 [−0.0143, 0.0091]; 1.3569 [1.3289, 1.3837] | V | – |
| AB8 | Oracle 0.644 vs 0.606 (l.77–78) | A | own 0.6439 | V | – |
| AB9 | Gate verifies no math answer; cascade escalates every math task (l.79–80) | A | 4B and 2B gate VERIFIED on math = 0/500; escalated math = 500/500 | V | – |
| AB10 | 4B gate − matched log-prob CWR: −0.076 Python, −0.112 Rust (l.82–84) | A | own −0.0756, −0.1124 | V | – |
| AB11 | "This holds even though the threshold was tuned on E" (l.84–85) | D | top-k uses no labels; no ties in 4B conf (0/0/0) | O | Must-fix 5 |
| AB12 | 1,536 non-passing code rows re-checked → 3 changes; 1,572 passing → none (l.88–89) | A | audit JSONs: 644+412+480; flips 1+2+0; 392+624+556, 0 flips | V | – |
| AB13 | Corrects 4 W + 3 O; spend $2.53 from 8 ledger lines (l.91–93) | A | CLAIM_AUDIT.md summary; ledger 8 lines sum 2.5289 | V | – |
| R1 | n=60: +6.67, CI [1.67, 13.33], 4/0, p=0.125, hardware "unspecified" (l.109–111) | A | `results_n60_cpu.json` `paired_vs_A0.A3` | V | – |
| R2 | n=257: 64.2, 71.98, +7.78, [4.67, 11.28], 20/0, empty `git_commit` (l.112–114) | A | `results_n257_l4_gpu.json` | V | – |
| R3 | n=257 p printed 0.0, i.e. p<0.001; n=60 bootstrap lower bound positive by construction (l.116–121) | D | 2·0.5^20 = 1.9e-6; argument correct | V | – |
| R4 | Coverage 80.93 %, precision 88.94 %, ≈8.95 % CW by arithmetic (l.124–125) | A | file values; 0.8093·0.1106 = 0.0895 | V | – |
| C1 | A32: `e5488f6c` = b5be32a; tag v3.3.0 = `fccb1695…5995` (l.136–140) | A | `git show b5be32a:` / `v3.3.0:papers/numbers_v4.json \| sha256sum` | V | – |
| C2 | A33: tag v3.3.1 and HEAD = `5f3c2a6e…a1c0`; `952bfc77` was an uncommitted scratch file (l.141–143) | A | hash recomputed (V); the scratch-file origin cannot be checked (file gone) | VN | – |
| C3 | A28: ledger 8 lines, $2.5289 (l.144–146) | A | ledger read | V | – |
| C4 | C12: bib title matches Zenodo (l.147–148) | C | Zenodo API, record 23121788: title, v3.1.0, Callens | V | – |
| C5 | B15/B12 rewording (l.149–160) | E | wording matches §7 H13 paragraph and §9 | V | – |
| C6 | Rust harness counts `assert!`/`assert_eq!`/`assert_ne!` (l.168–169, 209–210) | B | `gwaya/oracles.py` `_RUST_ASSERT_MACRO = \bassert(?:_eq\|_ne)?!` | V | – |
| C7 | Registration first committed `0b4f104` 2026-10-07 23:33 (l.176–178) | B | `git show -s 0b4f104` | V | – |
| C8 | Scripts committed in `39a5a0a` (l.187–190) | B | `git show --stat 39a5a0a` lists all four paths | V | – |
| C9 | MultiPL-E bib metadata (l.186) | C | Crossref 10.1109/TSE.2023.3267446: 49(7) 3675–3691, 13 authors match | V | – |
| C10 | D27: nonce on stdin, separate sandboxes, no /proc, unsafe/link_section/asm rejected; 25/25 and 9/9 (l.200–214) | B | code read (`oracles.py` l.75–77, 249–286); `test_rust_harness_hardening.py` passes (39 with test_run_study); validation JSON not re-run this round (prior audit reproduced it) | VN | – |
| D1 | E pool: Python 542, Rust 510, MATH-500 500, miniF2F 244 (l.248–250) | E | prereg §2 table | V | – |
| D2 | Arms B1–B6, LR, SH (l.251–254) | E | prereg §4 | V | – |
| D3 | Holm m=2; non-run primary enters with p=1 → other tested at 0.025 (l.255–256) | D | m=2 is registered; the p=1 rule is D26's implementation (consistent with Holm), not prereg text | VN | Add "(D26)" |
| D4 | H1/H3 metrics, tests, MEIs (l.258–261) | E | prereg §7 | V | – |
| D5 | Secondary vs exploratory partition (l.263–265) | E | prereg §7 rev. 2 | V | – |
| N1 | Table `tab:dev` D1–D30 rows | E | spot-checked against DEVIATIONS.md (D1, D4, D14/15, D22, D25–D30) | V | – |
| N2 | 80 of 1,541 included items (l.330) | A | `tasks_E_primary.jsonl` 1,541 rows | V | – |
| N3 | Power 0.052 / 0.261 at α=0.025 (l.335–337) | A | FREEZE_ADDENDUM_NIGHT §8 table | V | – |
| N4 | Night: Python 11/28, Rust 6/26, math 26 UNDECIDED; Table `tab:calib` CIs | A | counts recomputed from `L2fix/rows.jsonl`; CIs not re-bootstrapped this round (prior audit A3, A10–A16 matched) | VN | – |

### Section 6: full-E TPU baselines

| ID | Claim (line) | Cl. | Evidence | Status | Fix |
|---|---|---|---|---|---|
| E1 | None registered; H1 GL vs B3, H3 GL vs B5; GL needs Laya, B3 needs C (l.447–449) | E | prereg §7. The hook-introduced slip (H3 needing B3) is fixed: every H3 mention (l.59, 260, 448, 779, 1047) says B5 | V | – |
| E2 | Plan `8ee9da3` 22:22:55; original sha `1f2fa2c1…0d46` (l.453–455) | B | `git log`; `git show 8ee9da3:… \| sha256sum` | V | – |
| E3 | 2B/4B rows written 22:36:38, "after the plan" (l.459) | E | mtime true; process started 22:21:55 | O | Must-fix 4 |
| E4 | Coder rows 22:22:45, 10 s before the plan (l.460–462) | B | mtime 22:22:45.646 vs commit 22:22:55 | V | – |
| E5 | Analysis script `5e10ad2` 22:25 (l.463) | B | git log | V | – |
| E6 | Four dated amendments (l.466–476) | E | ANALYSIS_PLAN amendments 1–4 | V | – |
| E7 | Truncation pass counts are post hoc (l.477–478) | E | plan A9 already lists "and their pass rate", so this is over-cautious | VN | Drop "the truncation pass counts" from the post-hoc list |
| E8 | v5litepod-1, us-west4-a; vLLM-TPU 0.31.0 from logs (l.484–485) | B | ledger line 8; logs "V1 LLM engine (v0.31.0)"; PyPI vllm-tpu 0.31.0 exists (uploaded 2026-10-06) | V | – |
| E9 | bf16, greedy, 1024 tokens, thinking off, log-prob per token (l.486–487) | B | `gen_batch.py` SamplingParams(temperature 0, max 1024, logprobs=1); Qwen3 empty think block in math prefill | V | – |
| E10 | Prompts via `build_raw_prompt` + `stop_sequences`, same path as the CPU arm; code prompt unchanged since `64441eb` (l.488–493) | B | `git diff 64441eb HEAD -- gwaya/generators.py`: code system prompt, ```lang prefill and ``` stop unchanged; `64441eb:run_study.py` uses `build_raw_prompt` | V | – |
| E11 | `run_e_gen.sh` sets `SKIP_JAX_PRECOMPILE=1` (l.494–495) | B | script read | V | – |
| E12 | 1,536 tasks, 1,079 clusters (l.496–497) | A | task file | V | – |
| E13 | Weights at revision `main`, not pinned (l.501–502) | B | logs `revision=main`. HF API: main last modified 2026-03-02 (4B `851bf6e8…`, 2B `15852e8c…`), so those are very likely the weights used | V | Optional: cite those shas as "probable" |
| E14 | 2B at 96, 4B at 32 "so the 4B cost is throughput at a smaller batch"; quoted HBM message (l.502–505; also 706, 719, 753, 838–839, 1048) | C | lake logs: re-split at 54 (2B) and 11 (4B@32) concurrent requests; fatal error at 96 is `ValueError: Cannot fit both KV pools…` | O | Must-fix 6 |
| E15 | First pass: Coder and 2B `mean_logprob` null in all rows; 4B failed at start (l.507–510) | A | lake `gwenlaya-tpu-e-gen-210955`: 1,536+1,536 rows, 0 non-null; status rc=1 | V | – |
| E16 | That pass requested `logprobs=0` (maintainer report) (l.510–511) | B | not checkable (code of that pass not committed); the paper says so | VN | – |
| E17 | Second pass `logprobs=1`; fail-fast; 3×1,536 non-null; sha256 match locally and in the lake (l.511–516) | B/A | `gen_batch.py` `require_logprobs`, exit 3; raw rows 0 null; lake objects downloaded and hashed: all 3 match | V | – |
| E18 | Chunk timings 62/144/192 s; 9–10, 21–23, 50–55 s; 4,507–4,866, 2,864–3,120, 1,085–1,128 tok/s (l.519–522) | A | logs and raw `chunk_wall_s`; Coder max is 4,865 | W | Must-fix 9 |
| E19 | Slow first chunk matches lazy compilation, not instrumented (l.522–523) | D | hedged | V | – |
| E20 | Completion tokens 353,862 / 509,089 / 464,253 (l.527) | A | gens_summary sums | V | – |
| E21 | Chip-seconds 129.4 / 296.7 / 561.0 = sums of chunk walls incl. first (l.528–529) | A | 129.45 / 296.74 / 561.03; raw chunk-wall sums identical | V | – |
| E22 | Importer → hash-chained gens; `run_study --mode score` local, env unset; chip-s by token share (l.536–541) | B | `run_tiers.sh` and `score_local.sh` `unset GWAYA_ALLOW_UNISOLATED`; share is **prompt+completion** tokens (`import_remote_gens.py` l.56, 76) | VN | Say "prompt+completion token share" |
| E23 | Six arms, `--check-workers 5` recorded (l.543–546) | B | `tiers/results.json` args | V | – |
| E24 | Coder one check process, no option recorded (l.546–547) | B | Coder args have no `check_workers` key | V | – |
| E25 | "Both scoring runs started from … dirty: true at 66e2313 and 5e10ad2" (l.547–548) | B | `git_info()` called at end of run; start times precede both commits | W | Must-fix 2 |
| E26 | Re-ran 2 test files: 22 tests passed (l.549–550) | B | re-run: 22 passed | V | – |
| E27 | 644 / 412 / 480 = 1,536 non-passing re-checked (l.553–554) | A | audit JSONs; consistent with 1,036 − passes per model | V | – |
| E28 | Flips: MBPP/271 (2B, 4B) contention; mbpp_130 nondeterministic, passes ~half the time, not re-measured (l.555–561) | A | flips match the JSONs. "About half" is the amendment's assertion; nobody measured it | VN | – |
| E29 | Overlay 16 rows; MBPP/271 VERIFIED in 8 rows; mbpp_130 stays FAILED (l.565–571) | A | overlay read: 8+8 rows, all six arms | V | – |
| E30 | Overlay bug fixed; importer keeps original seconds; test guards it (l.572–575) | B | overlay `gpu_s` equals the original rows' for all 16; test exists and passes | V | – |
| E31 | 5 timeouts alone per model (l.579–580) | A | audit JSONs 5/5/5 | V | – |
| E32 | 1,572 passing re-run (392/624/556), none changed; "none observed" reading (l.581–585) | A | `recheck_pass_*.json` | V | – |
| E33 | S1: 1 / 2 unstable; 0.4225 vs 0.4232, 0.6055 vs 0.6061, 0.6029 vs 0.6035; largest 0.0019 (4B Python) (l.587–591) | A | own identical; 0.0019 is a three-way tie (2B, 4B and cascade Python each 1/529) | VN | "largest difference 1/529 = 0.0019 (Python, 2B/4B/cascade alike)" |
| E34 | All SHA256SUMS entries verified with `sha256sum -c` against the data root (l.593–594) | B | from the data root, 8/9 OK and `numbers_e_tpu.json` fails to open (repo-relative path); from the repo root it matches | VN | Should-fix list |
| E35 | Tables unedited; CIs are source-problem cluster bootstrap, 10,000, `default_rng(0)` (l.598–602) | D | `tables_e_tpu.tex` hash matches; meta n_boot 10,000, seed 0; all clusters are singletons within a domain | VN | Should-fix list |
| E36 | A1 pass rates incl. Coder Python 0.647 vs 4B 0.675 (l.608–611) | A | own 342/529 = 0.6465; 357/529 = 0.6749 (all nine cells in `tab:e-pass` match) | V | – |
| E37 | A9: math truncation 0.400/0.320/0.152; code ≤0.053 (2B Python) (l.613–615) | A | own 0.4000/0.3200/0.1520; 2B Python 0.0529 | V | – |
| E38 | No truncated generation passed; truncated math has no \boxed and is UNVERIFIED except one 2B FAILED; lower bounds (l.617–620) | A | 0 truncated pass (243/167/78 truncated). 5 truncated math texts contain "\boxed" (1 complete → the FAILED one). Lower bound is valid under greedy | VN | "no complete \boxed{}" |
| E39 | A2: conf 0.898 (2B Py)–0.938 (4B math); ECE 0.252–0.598; 4B 0.927/0.930/0.938 vs 0.675/0.529/0.612 (l.622–626) | A | own ECE (15 equal-mass bins, both floor split and `array_split`) and mean conf match all 9 cells to 3 dp | V | – |
| E40 | AUROC 0.615–0.775; every CI excludes 0.5 (l.628–629) | A | own AUROC (Mann-Whitney, ties ½) matches all 9 points; AUROC CIs not re-bootstrapped; lowest point 0.615 at n=500 is far from 0.5 | VN | – |
| E41 | A3 4B: cov 0.813/0.596, prec 0.830/0.887, CWR 0.325→0.138, 0.471→0.067; abstain 19 %/40 % (l.631–635) | A | own 430/529, 302/507; prec 0.8302/0.8874; CWR 0.1380/0.0671; all-CWR 0.3251/0.4714 | V | – |
| E42 | Math coverage 0; 500/500 empty `gate_payload`; "registration's gate has nothing to execute" (l.637–639) | E | counts V; registered math gate = PoT re-execution (prereg §1) | O | Must-fix 7 |
| E43 | Precisions <0.95; analogue, not H7 (l.641–642) | D | – | V | – |
| E44 | A4: matched CWR 0.214/0.180; gate 0.138/0.067; Δ −0.076 (−0.098, −0.057), −0.112 (−0.134, −0.093) (l.644–648) | A | own 0.2136/0.1795; Δ −0.0756 [−0.0983, −0.0567], −0.1124 [−0.1322, −0.0927] (3,000 resamples, seed 20261009) | V | – |
| E45 | "This favours the log-prob baseline" (l.655–656; captions; D35; l.1045) | D | see Must-fix 5 | O | Must-fix 5 |
| E46 | "Not H1 … coverage difference within ±3 points" (l.659–660) | E | prereg: the coverage-difference **CI** must lie within ±3 pts | V | – |
| E47 | Pooled −0.042 "mixes in a domain where the comparison is empty" (l.662–663) | D | global threshold; baseline answers 280 math tasks; stratified pooled −0.063 | W | Must-fix 1 |
| E48 | Fig. `e-rc` caption (risk among answered, red dot = 1−precision, dotted = answer-all, no math dot) (l.667–672) | E | figure viewed; gate dots at 0.170 and 0.113 match | V | – |
| E49 | Cascade: gate-verified 2B final, else the 4B's answer is final (l.677–680) | E | arm logic verified (escalated ⇔ 2B gate ≠ VERIFIED, correct = where(esc, 4B, 2B), cost = sum of tiers); but the arm abstains on 757 tasks | VN | Should-fix list |
| E50 | 0.604 vs 0.606; −0.003; McNemar 40/44 p=0.744, Holm 0.744 (l.683–685) | A | own 40/44, exact two-sided p 0.7436; Holm 0.7436 | V | – |
| E51 | Python −0.026 (−0.053, −0.002); verified-but-wrong 2B never escalated (l.685–687) | A | own −0.0265 [−0.0510, 0.0000]; 74 Python 2B gate-VERIFIED hidden-FAIL; cascade-only 17 vs 4B-only 31 | VN | Note CI upper bound ≈ 0 |
| E52 | Rust +0.020 (−0.004, 0.043) (l.687–688) | A | own +0.0197 [−0.0039, 0.0434] | V | – |
| E53 | vs 2B +0.180; 295 vs 18; Holm 2.0e-65; gain comes from the 4B (l.688–689) | A | own 295/18, p 1.007e-65, Holm 2.013e-65 | V | – |
| E54 | Ratio 1.357 (1.329–1.385); Python 0.997 (0.925–1.071); Rust 1.221; math 1.514; escalated 0.675 / 0.408 / 0.633 / 1.000; cost per correct 0.821 (0.759–0.889) vs 0.603 (0.563–0.646) (l.690–695) | A | own 1.3569 [1.3289, 1.3837]; 0.9965 [0.9238, 1.0731]; 1.2211; 1.5138; 0.6751/0.4083/0.6331/1.0; 0.8212 [0.7586, 0.8865]; 0.6026 [0.5635, 0.6438] | V | – |
| E55 | Oracle 0.644 (0.620–0.668); 0.707/0.574/0.648 vs 0.675/0.529/0.612 (l.696–700) | A | own 0.6439 [0.6204, 0.6680]; 0.7070/0.5740/0.6480 | V | – |
| E56 | Cost caveat directions; 144 of 297 s vs 192 of 561 s (l.706–711) | D | arithmetic and directions correct; batch numbers see E14 | V | – |
| E57 | No warm-up-excluded cost; gap size not claimed robust (l.712–714) | D | auditor sensitivity (chunk-1 wall replaced by the median steady rate): pooled ratio 1.24, Python **0.89**, Rust 1.09, math 1.40. The pooled direction holds; Python would favour the cascade | VN | Optionally report these numbers; keep the hedge |
| E59 | A7: 54 night Py/Rust tasks are in E with identical prompts; same prompts, prefill, stop (l.728–731) | B | 54/54 task_ids present, prompts and checker payloads byte-equal | V | – |
| E60 | Python 11 vs 10, 27/28; Rust 6 vs 8, 22/26, p=0.625; pooled 17 vs 18, 49/54, p=1.0 (l.733–735) | A | own from night rows vs overlay-applied TPU rows: discordant 1/0, 1/3, pooled 2/3 | V | – |
| E61 | Math excluded post hoc; including it gave a defect-driven p (l.741–742) | A | CPU math 0/26 vs TPU 15/26 recomputed | V | – |
| E62 | Table `tab:dev34` D31–D41 | E | matches DEVIATIONS.md rows (D33 wording see E14) | V | – |

### Sections 7–11, Limitations, Next steps, Appendix

| ID | Claim (line) | Cl. | Evidence | Status | Fix |
|---|---|---|---|---|---|
| V1 | Prior column = red-team priors (H1 0.45 … H12 0.12) (l.767–792) | E | DEVIATIONS "Hypothesis status" table; PROPOSAL_REVIEW l.197 (H1 0.45), l.209 (H13 0.75) | V | – |
| V2 | No exploratory analysis is a verdict on a registered hypothesis (l.770–771) | E | whole text read: A3/A4/A5/A7 are each labelled "not H7/H1/H3/H8"; no sentence says the system works | V | – |
| V3 | H7 row: precision 0.830 / 0.887 (l.789) | A | E41 | V | – |
| V4 | H13: $2.5289; launcher design; `drive.sh` dry-run by default with ledger + worst-case check (l.795–807) | B | ledger; `drive.sh` l.7–47 read (cap 45, `--yes-spend` + `GWAYA_CONFIRM_SPEND=1`) | V | – |
| V5 | Second E run's driver log shows ledger $1.9809, worst $2.5, cap $45 (l.806–807) | B | driver log not found locally or in the lake listing. Values are arithmetically consistent: sum of first 7 lines = 1.9809; 120/60·1.20+0.1 = 2.5 | VN | Name the log path or commit it |
| V6 | Rule "no stage exceeds 1.25× its cap twice" (l.796) | D | TPU lines have no registered stage caps, so this half of the rule cannot be evaluated | VN | Say so |
| V7 | `runs/LEDGER.jsonl` does not exist (l.810–811) | B | `gcloud storage ls -r …/runs/`: only `runs/FIXTURE/` | V | – |
| V8 | H15 test simulates death with a torn line, no SIGKILL (l.814–816) | B | `test_resume_after_kill_mid_run` read; passes | V | – |
| T1 | Smoke #1: 208.9 s load, 442 tokens in 0.61 s; gate UNVERIFIED (l.822–826) | A | not re-checked this round (prior audit; text unchanged) | VN | – |
| T2 | Qwen3.5 via PyTorch fallback with `SKIP_JAX_PRECOMPILE=1`; 2B/4B all 1,536 E tasks (l.827–837) | B | E logs; 1,536 rows each | V | – |
| T3 | 4B "hybrid attention/Mamba" on 16 GiB chip; fails at 96, starts at 32 (l.838–839) | C | logs hbm 15.75 GiB; HF config: 24 linear + 8 full attention layers | VN | Should-fix list + E14 |
| T4 | Generator requests `logprobs=1`, fails fast (l.840–842) | B | `gen_batch.py` | V | – |
| T5 | No TPU LoRA path for Qwen3.5 dense / Qwen3.8; Tunix 0.1.7, MaxText lists (l.843–847) | C | not re-checked live this round | NV | – |
| H1x | Run #3 table and reading (l.851–877) | A | unchanged from v3.3.1; prior audit verified counts, 22/28 and 24/26; summary.json hash matches | V | – |
| S1 | Ledger: 8 lines with times, seconds and USD as tabled (l.886–898) | A | every line read; seconds·price/3600 reproduces each USD | V | – |
| S2 | Last written 22:20:47; read at 23:30 CEST (l.882–883) | A | mtime 22:20:47.51 (V); the read time is not checkable | VN | – |
| S3 | List prices: spot 0.494237, on-demand 1.20 USD/chip-h (l.900–902) | C | match each line's `price_source`; live pricing page not retrievable (WebFetch returned truncated content); Billing Catalog not queried | NV | – |
| S4 | E lines total $0.989; ≈$0.44 unused (l.904–905) | A | 0.989; 0.441 | V | – |
| S5 | All non-zero spend is TPU v5litepod-1 (l.900) | A | ledger | V | – |
| P1 | Commits: v3.2.0 = 825d1ba; v3.3.0 → 200c7f4; v3.3.1 = 4914b5c; 8cb37ad, 2c1f6b1; branch commits and contents (l.916–929) | B | `git rev-parse`, `git show --stat` | V | – |
| P2 | llama.cpp b11476, model revisions and shas in L0_env.json (l.930–933) | B | all 9 fragments present in L0_env.json; GitHub tag b11476 exists (2026-10-07) | V | – |
| P3 | Hash table, 24 rows (l.940–964) | B | all recomputed with sha256sum: repo files, data-disk files (tasks_E_night/primary, L2score, L2fix, `etpu/tiers/gens.jsonl`, raw_*) and lake raw_* (downloaded); all match | V | Mark data-disk-only rows |
| P4 | HF dataset revisions in the freeze addendum (l.967–968) | B | FREEZE_ADDENDUM_NIGHT l.16–19 | V | – |
| P5 | Data-lake prefixes and contents (l.971–983); `results/` mirror still open | B | read-only listings: night, tpu_smoke (3 runs), tpu_runs (2), e_tpu_final (tiers, coder, overlay2, audit, audit_pass); no `results/` prefix | V | – |
| P6 | Command lines (l.989–1013) | B | every flag exists in the argparse of `analyze_e_tpu.py`, `recheck_failures.py`, `serial_overlay.py`; `run_tiers.sh` matches the `run_study` line | V | – |
| P7 | Did not re-run analyze_e_tpu; output hash matches; counts recomputed (l.1015–1017) | B | `git show f80ac16:… \| sha256sum` = c76bc506 | V | – |
| L1 | Limitations: single sample, amendments, Coder 10 s; no math gate; ≤40 % truncation; five timeouts; dirty trees; contamination; night caveats; spend (l.1039–1064) | E | consistent with the body, except the omissions in Must-fix 3 (E freeze) and Must-fix 6 (batch) | V | Add the two items |
| NS1 | "run GL, B3 and B5 on the same E generations, to test H1 and H3 as registered" (l.1070–1071) | D | prereg §6 freeze rule | W | Must-fix 3 |
| AP1 | Prior audit: 61/29/2/4/3 = 99; classes 45/20/19/15; 14 counts; 0 mismatches over 80+240 (l.1091–1105) | E | CLAIM_AUDIT.md summary | V | – |
| AP2 | Audit gave v3.3.0 as `ebd032f`; tag resolves to `200c7f4`; hash same (l.1107–1110) | B | CLAIM_AUDIT l.31; `git rev-parse v3.3.0^{commit}` = 200c7f4 | V | – |

### Bibliography (all `\cite` keys resolve; `.blg` has 0 warnings)

| ID | Entry | Cl. | Evidence (live) | Status |
|---|---|---|---|---|
| B1 | evalplus | C | arXiv 2305.01210: title and 4 authors match, 2023 | V |
| B2 | multiple | C | Crossref DOI 10.1109/TSE.2023.3267446 | V |
| B3 | math | C | arXiv 2103.03874: title and 8 authors, 2021 | V |
| B4 | letsverify | C | arXiv 2305.20050: title and 10 authors | V |
| B5 | selective | C | arXiv 1705.08500: Geifman, El-Yaniv, 2017 | V |
| B6 | guo | C | arXiv 1706.04599: 4 authors, 2017 | V |
| B7 | conformal | C | arXiv 2107.07511: Angelopoulos, Bates, 2021 | V |
| B8 | holm | C | OpenAlex DOI 10.2307/4615733: Scand. J. Stat. 6, 65–70, 1979 | V |
| B9 | mcnemar | C | Crossref 10.1007/BF02295996: Psychometrika 12(2) 153–157, 1947 | V |
| B10 | vllm | C | arXiv 2309.06180: title and 9 authors, 2023 | V |
| B11 | llamacpp b11476 | C | GitHub API tag b11476, published 2026-10-07 | V |
| B12 | gwayav3 | C | Zenodo API 23121788: title, Callens, v3.1.0 | V |

(Venues such as NeurIPS, ICLR, ICML and SOSP were not fetched live. Titles, authors, years and identifiers
were.)

### Other documents

| ID | Claim | Cl. | Evidence | Status | Fix |
|---|---|---|---|---|---|
| DOC1 | RESULTS_VERDICT l.37–38: "The only scored data are 80 items from one unregistered arm" | E | contradicts the 3.4.0 section of the same file | O | Must-fix 8 |
| DOC2 | DEVIATIONS l.136–137: D31, D32, D34–D37 fixed "before the 2B/4B rows were scored (22:36:38)" | E | scoring started 22:21:55 | O | As Must-fix 4 |
| DOC3 | STAGE_LOG 3.4.0: hashes "verified with `sha256sum -c`" | B | mixed roots, see E34 | VN | – |
| DOC4 | RESULTS_VERDICT, DEVIATIONS and STAGE_LOG E-set numbers equal the paper's | E | cross-read: pass rates, Δ, McNemar, ratio, escalation, truncation, A7, stability, ledger | V | – |
| DOC5 | DEVIATIONS D39: 1,536 re-checked, 3 flips; D40: 963 s, 0.321 USD; orphan 360 s, 0.12 | A | audit JSONs; ledger | V | – |

## Method and limits

- **Own code.** The scripts `audit340.py` and `boot340.py` live in the session scratchpad, not the repo, and
  read the raw files directly.
  - Overlay rule: the last row per (arm, model, domain, task_id) wins. The `prev` links of all three
    hash-chained files are intact (0 breaks).
  - Correct = `score == "VERIFIED"`. gate_only answered = `row.answered`, which equals `gate == "VERIFIED"`
    on every row.
  - Cascade cost = the row's `gpu_s`. This was checked to equal the `gens_summary` 2B cost plus the 4B cost
    when escalated (max error 9e-16).
  - Confidence = exp(`mean_logprob`) from `gens_summary`. It equals the `raw_confidence` rows.
- **Metrics.**
  - ECE: 15 equal-mass bins over a stable sort by confidence. Both split rules (floor index and
    `numpy.array_split`) were computed and agree to 4 dp.
  - AUROC: rank-based, ties counted ½.
  - AURC: mean selective risk over all coverages, ties broken pessimistically.
  - McNemar: exact two-sided binomial. Holm: over the two A5 contrasts.
- **Bootstrap.** 3,000 resamples, seed 20261009 (within-domain) and 20261010 (cross-domain), percentile
  95 %. Two schemes:
  - Clusters resampled within each domain (the paper's scheme). Within a domain every cluster is a single
    item.
  - Clusters resampled jointly across Python and Rust, stratified by domain signature, which keeps the 457
    shared source problems together.

  The matched-coverage baseline is recomputed in each resample with k = the gate's count in that resample.
  No CI endpoint differs from the paper by more than 0.005. The differences are Monte-Carlo noise:
  3,000 vs 10,000 resamples, different seed.
- **External checks run live:**
  - Zenodo, Crossref, arXiv, OpenAlex and GitHub APIs; PyPI (`vllm-tpu` 0.31.0); HF model API and
    `config.json` for Qwen3.5-4B.
  - Read-only `gcloud storage ls/cat` of the data lake: raw files and logs of both E passes, `runs/`,
    `e_tpu_final/`, `tpu_smoke/`.
  - The Google TPU pricing page could not be read (NV). The Cloud Billing Catalog was not queried.
- **Tests re-run:**
  - `tests/test_import_remote_gens.py` + `tests/test_analyze_e_tpu.py`: 22 passed.
  - `tests/test_run_study.py` + `tests/test_rust_harness_hardening.py`: 39 passed.
  - No sandboxed checker was re-run on stored generations this round. The stored statuses were taken as
    given, and the stability claims were checked only against the audit JSONs.
- **Not checked:**
  - The cause of the run #3 host failure and the smoke #1 timings (carried from the prior audit).
  - TPU LoRA availability in Tunix/MaxText.
  - The "about half the time" frequency for `rs/mbpp_130`.
  - The 22:21:55–22:22:55 window: whether anyone looked at outputs then. The log prints none.
  - The driver log behind "ledger 1.9809, worst 2.5, cap 45".
  - Whether "re-split at N concurrent requests" in tpu-inference strictly caps the running batch. This is
    inferred from the log's own wording and the mamba slot counts (433/8 = 54, 89/8 = 11).
- **Counting.** Each table row is one claim. Rows that bundle several numbers count once and are VERIFIED
  only if every number in them reproduces.
