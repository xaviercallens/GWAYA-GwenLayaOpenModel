# Claim audit of papers/gwenlaya_v4.tex revision 3.10.0 (branch release/3.10.0): SKELETON

Status: **skeleton written by the author; verification is the auditor's.** Each row names a claim added or changed in 3.10.0,
where it sits in the paper, and the committed file (and key) it was read from. The "Verified" column is empty on purpose.
Everything generated is written by `scripts/make_tables_3100.py` (`GWAYA_DATA_ROOT=/mnt/data/home/xavkal/gwaya-data`), which
reads only committed JSON/rows files plus `spend_ledger.jsonl`; prose numbers were copied from its tables and from
`scripts/print_F_tables.py` output. Paths are relative to the repository root. Sections are labels in the paper.

Recompute from raw rows where the file is a summary: F rows are in `results/gwenlaya_v4/eval_sets_F/scored/<tier>/rows.jsonl`
(+ `recheck/overlay_<tier>.jsonl`, later rows win), Lean rows in `.../lean_scored/scored_raw_<tier>.jsonl` and
`results/gwenlaya_v4/lean_a18/scored_greedy_v3/scored_raw_<tier>.jsonl`, Laya inputs in `eval_sets_F/laya_rows/`.

## A. Fresh sets F: construction (Section sec:F, Table F-build)

| # | Claim | Source | Verified |
|---|---|---|---|
| A1 | F = Python 407 (LeetCodeDataset 205, FullStackBench 112, BigCodeBench 90), math 501 (MATH-test 200, GSM8K 130, competition 171), Lean 229; 1,137 tasks | `eval_sets_F/manifest_F.json` `counts`, `total_tasks`; `tasks_F_*.jsonl` (row counts; sha256 in manifest) | |
| A2 | candidates -> sanity -> decontam -> final: LeetCode 228/205/205/205; FSB 359/146/112/112; BCB 364/275/275/90; MATH-test 4,500/4,500/4,298/200; GSM8K 1,319 -> 130; competition 183/183/171/171; miniF2F-valid 244/229/229/229 | `manifest_F.json` `reports.*` | |
| A3 | decontamination removed 202 MATH-test, 12 competition, 34 FSB, 0 Lean | `manifest_F.json` `reports.math.*.decontam_drops`, `reports.python.sources.FullStackBench.decontam_drops`, `reports.lean.decontam_dropped` | |
| A4 | n-gram hits discounted: Python 164/164, math 82/690, Lean 3/3 | `manifest_F.json` `reports.*.ngram_filter` | |
| A5 | gate test visible in prompt for 89/407 (22%), not visible 304, undecidable 14 (LeetCode 44/161/0, FSB 27/79/6, BCB 18/64/8) | `manifest_F.json` `counts.python.gate_visible_in_prompt_by_source`; `numbers_F_pm.json` `meta.slice_n` | |
| A6 | hidden suites: LeetCode mean 98 assertions, FSB 3.4, BCB 5.6 test methods; 149 references did not verify | `manifest_F.json` `counts.python.hidden_tests_per_task_by_source`, `reports.python.sources.*.sanity_drops` | |
| A7 | order of events: task files 219178d 17:47:33, plan d60bdc3 17:48:53, launcher fc01629 17:49:58, generation VM names 185104 | `git log`; ledger item names | |

## B. F descriptive results (Section sec:F, Tables F-acc, F-gate, F-casc)

| # | Claim | Source | Verified |
|---|---|---|---|
| B1 | pooled accuracy 0.313 [0.283,0.341] / 0.427 [0.395,0.459] / 0.472 [0.441,0.504] (2B/4B/9B); Python 0.189/0.329/0.396; math 0.413/0.507/0.535 | `numbers_F_pm.json` `per_slice.{all_python_math,python,math}.<tier>.acc` | |
| B2 | GSM8K 0.785/0.915/0.954; MATH-test 0.515/0.655/0.695; competition 0.012/0.023/0.029 with 0.982/0.971/0.965 cut at 1,024 tokens | `numbers_F_pm.json` `per_slice.math/*.acc`, `.truncated` | |
| B3 | LeetCode accuracy 0.122/0.176/0.273, cut 0.507/0.273/0.400; BCB 4B 0.533 vs 9B 0.478 | `numbers_F_pm.json` `per_slice.python/LeetCode`, `python/BCB` | |
| B4 | gate CWR Python 0.170/0.135/0.147 vs log-prob at same coverage 0.258/0.216/0.224; math 0.020/0.010/0.004 vs 0.080/0.094/0.078 | `numbers_F_pm.json` `per_slice.{python,math}.<tier>.{gate_confident_wrong_rate,logprob_matched_confident_wrong_rate}` | |
| B5 | Python gate precision 0.527/0.709/0.729; BCB 0.533/0.706/0.652 (CWR 0.233/0.222/0.256); LeetCode 2B 0.455; E 4B Python precision 0.830 (3.5.0) | `numbers_F_pm.json` `...gate_precision`; E value from `results/gwenlaya_v4/e_tpu/numbers_e_tpu.json` / paper Table e-gate | |
| B6 | visible vs not visible CWR: 4B 0.124 vs 0.128, 9B 0.135 vs 0.135, 2B 0.067 vs 0.184 | `numbers_F_pm.json` `per_slice.python/gate_{visible,not_visible}` | |
| B7 | math gate 9B CWR 0.004 [0.000,0.010], precision 0.991 [0.977,1.000]; competition coverage 0.000/0.006/0.018 | `numbers_F_pm.json` `per_slice.math`, `math/competition` | |
| B8 | cascade pooled: acc 0.466 vs 0.472, diff -0.007 [-0.026,+0.012]; cost ratio 1.737 [1.649,1.832], with program 1.247 [1.196,1.297]; McNemar 37 vs 43, p 0.576; chip-s 2,349.8 / 1,352.5 / 1,884.4 | `numbers_F_pm.json` `cascade.ci["all_python_math|warm_included|*"]`, `cascade.mcnemar_vs_9b_pooled`, `cascade.total_cost_chip_s` | |
| B9 | Python cascade cost 0.947 [0.869,1.028] warm, 1.440 no warm-up; math w/o competition 0.844 [0.724,0.968] (no warm-up, with program), 1.157 warm; competition 3.383 / 2.006 | `numbers_F_pm.json` `cascade.ci` | |
| B10 | serial re-check changed one verdict per tier (2B Python gate FAILED->VERIFIED BCB/843; 4B and 9B math gate MATH-test/algebra/829 UNVERIFIED->VERIFIED); 1,247/1,018/947 checks | `eval_sets_F/recheck/recheck_<tier>.json` | |
| B11 | pooled generations cut at 1,024: 0.420/0.318/0.337 | `numbers_F_pm.json` `per_slice.all_python_math.<tier>.truncated` | |

## C. Frozen Laya on F (Section sec:Flaya, Tables F-laya, F-strata, F-router)

| # | Claim | Source | Verified |
|---|---|---|---|
| C1 | 9B H1 -0.104 [-0.132,-0.078], p 1e-4 one-sided, Holm 0.0002, supported; 4B -0.132 [-0.162,-0.102]; 2B -0.230 [-0.261,-0.199] | `numbers_F_laya.json` `primary.inference`, `per_tier.<tier>.inference.H1_dAURC_laya_minus_logprob` | |
| C2 | 9B H3 -0.002 [-0.022,+0.016], Holm 0.428, not supported; 4B +0.007 [-0.010,+0.029]; 2B 0.000 [-0.045,+0.035]; gate coverage 0.485 | `numbers_F_laya.json` `primary.inference`, `per_tier.*.inference.H3_*`, `.gate_coverage` | |
| C3 | 9B AUROC Laya 0.955, raw log-prob 0.812, gate 0.876, LR 0.960; AURC 0.207/0.311/0.414/0.196 | `numbers_F_laya.json` `per_tier.qwen3.5-9b-bf16.metrics` | |
| C4 | Laya - LR pooled: 9B AUROC -0.004 [-0.015,+0.006], AURC +0.010 [+0.003,+0.019]; 4B -0.011 [-0.021,-0.002], +0.014 [+0.005,+0.023]; 2B +0.010 [-0.002,+0.023], -0.003 [-0.017,+0.011] | `numbers_F_laya.json` `per_tier.*.inference.X_*` | |
| C5 | by domain Laya - LR AUROC: math 9B +0.011 [+0.002,+0.022], 2B +0.035 [+0.012,+0.059]; Python -0.018/-0.040/-0.033; Python H1 9B -0.036 [-0.092,+0.019], 4B -0.056 [-0.113,+0.001]; math H1 9B -0.088 [-0.116,-0.062] | `numbers_F_laya.json` `secondary.{python,math}.<tier>.inference` | |
| C6 | router tau 0.5: acc 0.476 [0.443,0.509] vs always-9B 0.472 [0.441,0.504], diff +0.003 [-0.009,+0.015]; cost ratio vs 9B 0.889 [0.864,0.915], vs ladder 0.713 [0.691,0.737]; CWR 0.079 vs ladder 0.131 vs 9B 0.528 | `numbers_F_laya.json` `router_policy` | |
| C7 | E' comparison values quoted: H1 -0.078 [-0.106,-0.053], H3 -0.012 [-0.023,+0.011], Laya-LR AUROC +0.013 [+0.001,+0.026], cost 0.79 / 0.895 | `results/gwenlaya_v4/laya/laya_eprime.json`, `laya_eprime_v2.json`, `router_nowarm.json` (3.6.0; unchanged) | |
| C8 | leak check: 0 label changes / 0 prompt differences / 0 unexpected fields, 908 + 229 rows; 2,724 calibrator rows, 908 router rows | `eval_sets_F/laya_rows/manifest_F_laya_rows.json` | |
| C9 | Laya prediction on F: 444 s, $0.148 (ledger item tpu_f-laya-152622) | `results/gwenlaya_v4/spend_summary_3100.json`; `spend_ledger.jsonl` | |

## D. Lean (Section sec:lean3100, Tables lean-a18, lean-F)

| # | Claim | Source | Verified |
|---|---|---|---|
| D1 | A18 test (223): proved 22/41/59/69 = 0.099 [0.063,0.139], 0.184 [0.135,0.238], 0.265 [0.206,0.323], 0.309 [0.251,0.368] | `results/gwenlaya_v4/lean_a18/scored_greedy_v3/summary_a18.json` `tiers.*.greedy_proof_rate`, `greedy_proved` | |
| D2 | F valid (229): proved 33/65/76 = 0.144 [0.100,0.192], 0.284 [0.227,0.345], 0.332 [0.271,0.393] | `eval_sets_F/lean_scored/summary_a18.json` | |
| D3 | verified_match exact = all VERIFIED, defeq 0, unknown 0 (both sets) | same two files `tiers.*.verified_match`; `scored_raw_*.jsonl` `match` field | |
| D4 | reasons (summary buckets) A18: 0.8B lean_error 41, forbidden 89, truncated 66, timeout 4, stmt 1; 2B 64/77/31/10; 4B 108/0/37/19; 9B 95/1/38/16 + stmt 3, import 1; F: 2B 90/75/24/4 + stmt 3; 4B 108/1/40/15; 9B 100/2/37/11 + stmt 2, import 1 | `summary_a18.json` `tiers.*.not_verified_reasons` | |
| D5 | every forbidden_construct row fires on `sorry` (A18 89/77/0/1, F 76/1/2 rows of reason forbidden) | `results/gwenlaya_v4/lean_a18/lean_forbidden_breakdown_3100.json` (recompute with `gwaya.lean_gate.lexical_flaws` on `text`) | |
| D6 | timeouts inside cut generations (A18 2/1/4/8; F 0/3/5) | `scored_raw_*.jsonl` rows with `reason=timeout`, `finish_reason=length` | |
| D7 | cascade vs always-9B: A18 81 vs 69 at 1.643x; F 95 vs 76 at 1.758x; overlaps (4B in 9B: 50) | `summary_a18.json` `cascade_greedy`, `greedy_overlap` | |
| D8 | scoring history: pass 1 timeouts 4/10/22/18 (120 s), pass 2 4/10/19/16 (300 s, match unknown), pass 3 reported | `results/gwenlaya_v4/lean_a18/history/*.json`, `scored_greedy_v3/summary_a18.json`; D66-D68, D75 | |
| D9 | k=8 sampling produced no rows | `$GWAYA_DATA_ROOT/lean_a18_tpu/greedy/out_*/raw_sample_*.jsonl` (0 bytes); `summary_a18.json` `any_of_k_proof_rate: null` | |
| D10 | A18 tokens and list-price USD per tier (0.0534, 0.0537, 0.0795, 0.2246; F 0.0263, 0.0441, 0.0789) | `summary_a18.json` `greedy_tokens`, `greedy_usd` | |

## E. A15.3 Rust selection (Section sec:a153, Tables A153, A153-diff)

| # | Claim | Source | Verified |
|---|---|---|---|
| E1 | E: 505 usable tasks; `first` CWR 4B 0.067/0.042/0.022, 9B 0.061/0.032/0.014 at k=1,2,3 | `results/gwenlaya_v4/edition/rust_gate_selection.json` `sets.E.tiers.*.first.k{1,2,3}.confident_wrong` | |
| E2 | E: no paired difference from first excludes zero at k=2,3; first_last k=1 2B +0.030 [+0.004,+0.057]; points within +-0.012 at k=2,3 | same, `*.confident_wrong_minus_first` | |
| E3 | chosen on E: first_last 0.0211 vs first 0.0238, longest 0.0271, spread 0.0221 | same, `chosen_on_E.mean_cw_k2_to_K` | |
| E4 | R' (68 usable): first k=1 CW 15/12/8 (0.221, 0.176, 0.118); first_last 11/6/5; longest 7/6/3; spread 8/10/3 | same, `sets.Rfresh.tiers.*.<rule>.k1.confident_wrong_count` | |
| E5 | R' differences excluding zero: 2B longest k1 [-0.206,-0.044], spread k1 [-0.191,-0.029]; 9B all rules k1..3 except first_last k1; 4B upper bounds 0.000 | same, `sets.Rfresh` | |
| E6 | 0 sanity violations; one stored-2015 difference each: E 4B mbpp_90_len_log, E 9B mbpp_619_move_num, R' 4B allergies | same, `*.sanity_violations`, `*.stored_2015_differences` | |

## F. Runtime, A19, A20 (Section sec:runtime3100, Tables A19, A20)

| # | Claim | Source | Verified |
|---|---|---|---|
| F1 | A19 Python 28 pairs: mean diff +0.096 [-0.020,+0.230], mean abs 0.162 [0.054,0.287] (max 0.913), Spearman 0.64 [0.28,0.90]; gate differs 5/28 (4 F->V, 1 V->F); identical text 9/28, abs diff there 0.030 | `results/gwenlaya_v4/laya_runtime/quant_shift_probe.json` `python`, `table.python` | |
| F2 | A19 Rust 26 pairs: -0.024 [-0.174,+0.126], abs 0.177, Spearman 0.65, gate differs 5/26; math 26 pairs protocol-confounded -0.182 [-0.286,-0.081] | same, `rust`, `math_protocol_confounded` | |
| F3 | A20 E': 828/828 agree on all items; cost 0.7879 runtime = registered; +0.97 points; answered accuracy 0.621 vs 0.581 (+3.99 points); verdicts 547/235/46; R' 69/69, 0.987, 0 | `results/gwenlaya_v4/laya_runtime/system_replay_eprime.json` `eprime.agreement`, `eprime.aggregates`, `rfresh.*` | |
| F4 | A20 0.8946 without first chunk | `docs/LAYA_RUNTIME.md` replay table; `results/gwenlaya_v4/laya/router_nowarm.json` | |
| F5 | parity: router max 0.0038, calibrator max 0.0029, 9/20 calibrator rows truncated | `results/gwenlaya_v4/laya_runtime/parity.json` | |
| F6 | PoT runtime wiring consistency 40/40 | `results/gwenlaya_v4/math_pot_runtime/consistency.json` `agree`, `agreement` | |
| F7 | live smoke n=3, 12.2 / 14.0 / 97.3 s, p 1.00 / 0.43 / 0.97 | `results/gwenlaya_v4/laya_runtime/live_smoke.json`; `docs/LAYA_RUNTIME.md` | |
| F8 | not-validated list (Q4 calibration, end-to-end accuracy, thresholds, 4B/9B paths) | `docs/LAYA_RUNTIME.md` "Not validated"; D77 | |

## G. A22 (Section sec:a22, Table A22)

| # | Claim | Source | Verified |
|---|---|---|---|
| G1 | E' pooled dAUROC v2-v1 -0.0096 [-0.0161,-0.0036] (v1 0.941, v2 0.932); R' pooled +0.0041 [-0.0270,+0.0339] (0.862 vs 0.867); v2 not better under the registered rule | `results/gwenlaya_v4/laya/layout_a22.json` `eprime.pooled.paired.v2_minus_v1.dAUROC`, `rfresh.pooled...`, `decision` | |
| G2 | per tier E' 2B -0.004, 4B -0.015 [-0.025,-0.006], 9B -0.009 [-0.018,+0.000]; R' 2B +0.026, 4B +0.044, 9B -0.009 | same `per_tier` | |
| G3 | v2 - LR: E' +0.003 [-0.009,+0.015]; R' -0.054 [-0.104,-0.010] | same `*.pooled.paired.v2_minus_hand_lr` | |
| G4 | window fractions not computed (no tokenizer in the environment) | absence of any such key in `layout_a22.json`; commit message b98a627; D74 | |
| G5 | single seed, same 9,243 training rows | `results/gwenlaya_v4/laya/layout_a22.json` `meta`; `layout_v2/run_manifest.json`, `head_config.json`; plan A22 | |

## H. Spend (Section spend, Table spend3100, abstract, H13)

| # | Claim | Source | Verified |
|---|---|---|---|
| H1 | ledger 31 lines, total 22.0055 USD (44.0% of the 50 USD cap; macro `SpendCapShare`); 3.9.0 had 27 lines, 19.2801 | `results/gwenlaya_v4/spend_summary_3100.json`; `$GWAYA_DATA_ROOT/spend_ledger.jsonl`; `spend_summary_390.json` | |
| H2 | 3.10.0 lines: tpu_f-9b-185104 1,016 s 1.3547 (v5litepod-4); tpu_f-small-185104 2,139 s 0.7130; tpu_f-laya-152622 444 s 0.1480; tpu_laya-layout-152621 1,529 s 0.5097; sum 2.7254 | same | |
| H3 | the two A18 Lean lines (1.5220) belong to 3.9.0 | `spend_summary_390.json` | |

## I. Corrections and deviations (Section corr3100, Table dev3100, `docs/DEVIATIONS.md` D66-D77)

| # | Claim | Source | Verified |
|---|---|---|---|
| I1 | "no Lean result", "generation in progress", "sampling is being re-run" of 3.9.0 are superseded; the sample files are empty | paper 3.9.0 text (`git show fa14857:papers/gwenlaya_v4.tex`); D75 | |
| I2 | A15.3 plan f4af1e4 07:35:49, code e556339 07:38:01, results b2920c8 08:20:49; 3.9.0 paper commit 2800549 08:10 | `git log` | |
| I3 | A19 plan 1bb7543 14:18:21 / results 73d1a15 14:41:33; A20 plan cca2049 15:01:14 / results 294054e 15:04:15; A22 plan 2a76e2f 15:43:58 / result b98a627 2026-10-11 15:59:42 | `git log` | |
| I4 | 27B probe not run (no result file, no ledger line) | absence in `results/` and ledger; D76 | |
| I5 | pyproject.toml still says version 3.4.0 and README keeps no version badge; neither was changed (as for 3.9.0) | `pyproject.toml`; `git show fa14857 --stat` | |

## Notes for the auditor

- Fix-forward expectations: nothing registered changed. Items most likely to need wording care: C5 (domain-level statements), D6/D4 (bucket
  precedence of `truncated`), E5 (R' intervals touching zero), F1 (in-sample, 2B only), H1 percentage (the spend share in the abstract
  is the macro `\SpendCapShare`).
- Known unverifiable by the author: the data-disk raw generation files for A18 (`raw_all/`) were not copied (D75); only scored rows are committed.
