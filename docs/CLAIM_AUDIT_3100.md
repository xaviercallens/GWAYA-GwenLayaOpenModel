# Claim audit of papers/gwenlaya_v4.tex revision 3.10.0 (branch release/3.10.0)

Status: **audited by an independent auditor (recomputation from raw rows with separate code, not the author's scripts).** Each row names a claim added or changed in 3.10.0,
where it sits in the paper, and the committed file (and key) it was read from. The "Verified" column is filled by the auditor: V = recomputed and matches to printed precision; VN = verified with the stated caveat; X = mismatch (correct value given). Totals over the 64 rows: 51 V, 11 VN, 2 X (B5 wording, B6 reading) plus the omission in E5 and the undisclosed stale gate verdicts (C2, C8), all listed below.
Everything generated is written by `scripts/make_tables_3100.py` (`GWAYA_DATA_ROOT=/mnt/data/home/xavkal/gwaya-data`), which
reads only committed JSON/rows files plus `spend_ledger.jsonl`; prose numbers were copied from its tables and from
`scripts/print_F_tables.py` output. Paths are relative to the repository root. Sections are labels in the paper.

Recompute from raw rows where the file is a summary: F rows are in `results/gwenlaya_v4/eval_sets_F/scored/<tier>/rows.jsonl`
(+ `recheck/overlay_<tier>.jsonl`, later rows win), Lean rows in `.../lean_scored/scored_raw_<tier>.jsonl` and
`results/gwenlaya_v4/lean_a18/scored_greedy_v3/scored_raw_<tier>.jsonl`, Laya inputs in `eval_sets_F/laya_rows/`.

## A. Fresh sets F: construction (Section sec:F, Table F-build)

| # | Claim | Source | Verified |
|---|---|---|---|
| A1 | F = Python 407 (LeetCodeDataset 205, FullStackBench 112, BigCodeBench 90), math 501 (MATH-test 200, GSM8K 130, competition 171), Lean 229; 1,137 tasks | `eval_sets_F/manifest_F.json` `counts`, `total_tasks`; `tasks_F_*.jsonl` (row counts; sha256 in manifest) | V. Counts 407/501/229, 1,137 from manifest `counts`; source sub-counts 205/112/90, 200/130/171 recomputed from the task files via strata. |
| A2 | candidates -> sanity -> decontam -> final: LeetCode 228/205/205/205; FSB 359/146/112/112; BCB 364/275/275/90; MATH-test 4,500/4,500/4,298/200; GSM8K 1,319 -> 130; competition 183/183/171/171; miniF2F-valid 244/229/229/229 | `manifest_F.json` `reports.*` | V. Every stage count in `manifest_F.json` matches (Lean 244/229 per the table; not re-derived from the builder). |
| A3 | decontamination removed 202 MATH-test, 12 competition, 34 FSB, 0 Lean | `manifest_F.json` `reports.math.*.decontam_drops`, `reports.python.sources.FullStackBench.decontam_drops`, `reports.lean.decontam_dropped` | V. 202 / 12 / 34 / 0 read from the manifest `decontam_drops`. |
| A4 | n-gram hits discounted: Python 164/164, math 82/690, Lean 3/3 | `manifest_F.json` `reports.*.ngram_filter` | V. 164/164, 82/690, 3/3. |
| A5 | gate test visible in prompt for 89/407 (22%), not visible 304, undecidable 14 (LeetCode 44/161/0, FSB 27/79/6, BCB 18/64/8) | `manifest_F.json` `counts.python.gate_visible_in_prompt_by_source`; `numbers_F_pm.json` `meta.slice_n` | V. 44/161/0, 27/79/6, 18/64/8; sums 89/304/14 recomputed from `gate_visible_in_prompt` in the task files. |
| A6 | hidden suites: LeetCode mean 98 assertions, FSB 3.4, BCB 5.6 test methods; 149 references did not verify | `manifest_F.json` `counts.python.hidden_tests_per_task_by_source`, `reports.python.sources.*.sanity_drops` | V. Hidden-suite means 98.0 / 3.4 / 5.6; 20+49+80 = 149 `reference_fails_hidden`. |
| A7 | order of events: task files 219178d 17:47:33, plan d60bdc3 17:48:53, launcher fc01629 17:49:58, generation VM names 185104 | `git log`; ledger item names | V. All four commit timestamps (17:47:33, 17:48:53, 17:49:58) and VM name suffix 185104 from `git log` and the ledger. |

## B. F descriptive results (Section sec:F, Tables F-acc, F-gate, F-casc)

| # | Claim | Source | Verified |
|---|---|---|---|
| B1 | pooled accuracy 0.313 [0.283,0.341] / 0.427 [0.395,0.459] / 0.472 [0.441,0.504] (2B/4B/9B); Python 0.189/0.329/0.396; math 0.413/0.507/0.535 | `numbers_F_pm.json` `per_slice.{all_python_math,python,math}.<tier>.acc` | V. Recomputed from raw rows + overlay: 0.313/0.427/0.472; Python 0.189/0.329/0.396; math 0.413/0.507/0.535. CIs [0.283,0.341] etc. are from the author JSON (own bootstrap not run on this block; the Laya blocks below were). |
| B2 | GSM8K 0.785/0.915/0.954; MATH-test 0.515/0.655/0.695; competition 0.012/0.023/0.029 with 0.982/0.971/0.965 cut at 1,024 tokens | `numbers_F_pm.json` `per_slice.math/*.acc`, `.truncated` | V. GSM8K 0.785/0.915/0.954; MATH 0.515/0.655/0.695; competition 0.012/0.023/0.029 with cut 0.982/0.971/0.965 recomputed. |
| B3 | LeetCode accuracy 0.122/0.176/0.273, cut 0.507/0.273/0.400; BCB 4B 0.533 vs 9B 0.478 | `numbers_F_pm.json` `per_slice.python/LeetCode`, `python/BCB` | V. LeetCode 0.122/0.176/0.273, cut 0.507/0.273/0.400; BCB 0.533 vs 0.478. |
| B4 | gate CWR Python 0.170/0.135/0.147 vs log-prob at same coverage 0.258/0.216/0.224; math 0.020/0.010/0.004 vs 0.080/0.094/0.078 | `numbers_F_pm.json` `per_slice.{python,math}.<tier>.{gate_confident_wrong_rate,logprob_matched_confident_wrong_rate}` | V. Python gate CWR 0.170/0.135/0.147 vs matched log-prob 0.258/0.216/0.224; math 0.020/0.010/0.004 vs 0.080/0.094/0.078 (ties wrong first, as documented). |
| B5 | Python gate precision 0.527/0.709/0.729; BCB 0.533/0.706/0.652 (CWR 0.233/0.222/0.256); LeetCode 2B 0.455; E 4B Python precision 0.830 (3.5.0) | `numbers_F_pm.json` `...gate_precision`; E value from `results/gwenlaya_v4/e_tpu/numbers_e_tpu.json` / paper Table e-gate | X (wording only; numbers V). Precision 0.527/0.709/0.729; BCB 0.533/0.706/0.652 with CWR 0.233/0.222/0.256; LeetCode 2B 0.455; E 4B 0.830. **Wording slip X-text (see S2):** the paper said "lowest on BigCodeBench", which is false for 2B and 4B on precision (LeetCode 0.455, 0.590 are lower). |
| B6 | visible vs not visible CWR: 4B 0.124 vs 0.128, 9B 0.135 vs 0.135, 2B 0.067 vs 0.184 | `numbers_F_pm.json` `per_slice.python/gate_{visible,not_visible}` | X. Numbers V (0.124/0.128, 0.135/0.135, 0.067/0.184); **the reading is wrong (S1):** "does not show an effect" for the 2B; the marginal CIs [0.022,0.125] and [0.142,0.227] do not overlap. |
| B7 | math gate 9B CWR 0.004 [0.000,0.010], precision 0.991 [0.977,1.000]; competition coverage 0.000/0.006/0.018 | `numbers_F_pm.json` `per_slice.math`, `math/competition` | V. 9B math CWR 0.004, precision 0.991, competition coverage 0.000/0.006/0.018. |
| B8 | cascade pooled: acc 0.466 vs 0.472, diff -0.007 [-0.026,+0.012]; cost ratio 1.737 [1.649,1.832], with program 1.247 [1.196,1.297]; McNemar 37 vs 43, p 0.576; chip-s 2,349.8 / 1,352.5 / 1,884.4 | `numbers_F_pm.json` `cascade.ci["all_python_math|warm_included|*"]`, `cascade.mcnemar_vs_9b_pooled`, `cascade.total_cost_chip_s` | V. Cascade accuracy 0.4659 vs 0.4725, diff -0.0066; cost ratios 1.7374 and 1.2470; McNemar 37 vs 43, exact p 0.576; chip-s 2,349.8 / 1,352.5 / 1,884.4 recomputed from gens_summary (incl. program seconds). |
| B9 | Python cascade cost 0.947 [0.869,1.028] warm, 1.440 no warm-up; math w/o competition 0.844 [0.724,0.968] (no warm-up, with program), 1.157 warm; competition 3.383 / 2.006 | `numbers_F_pm.json` `cascade.ci` | V. Python 0.947 warm and 1.440 without chunk 0 (chunk-0 set recomputed from the raw TPU files, 384 tasks); math w/o competition 1.157 / 0.844 (no-warm-up, with program); competition 3.383 / 2.006. |
| B10 | serial re-check changed one verdict per tier (2B Python gate FAILED->VERIFIED BCB/843; 4B and 9B math gate MATH-test/algebra/829 UNVERIFIED->VERIFIED); 1,247/1,018/947 checks | `eval_sets_F/recheck/recheck_<tier>.json` | VN. 1,247/1,018/947 re-checked rows and one flip per tier read from `recheck_*.json` and the three overlay rows match the stated task ids; the serial re-run itself was not repeated (it needs the sandboxed scorer). |
| B11 | pooled generations cut at 1,024: 0.420/0.318/0.337 | `numbers_F_pm.json` `per_slice.all_python_math.<tier>.truncated` | V. 0.420/0.318/0.337. |

## C. Frozen Laya on F (Section sec:Flaya, Tables F-laya, F-strata, F-router)

| # | Claim | Source | Verified |
|---|---|---|---|
| C1 | 9B H1 -0.104 [-0.132,-0.078], p 1e-4 one-sided, Holm 0.0002, supported; 4B -0.132 [-0.162,-0.102]; 2B -0.230 [-0.261,-0.199] | `numbers_F_laya.json` `primary.inference`, `per_tier.<tier>.inference.H1_dAURC_laya_minus_logprob` | V. Own implementation (cluster = task `cluster`, within domain, 10,000, default_rng(0)): H1 9B -0.1044 [-0.1316,-0.0783]; 4B -0.1321 [-0.1618,-0.1019]; 2B -0.2302 [-0.2612,-0.1994]; one-sided p 1e-4, Holm 2e-4. |
| C2 | 9B H3 -0.002 [-0.022,+0.016], Holm 0.428, not supported; 4B +0.007 [-0.010,+0.029]; 2B 0.000 [-0.045,+0.035]; gate coverage 0.485 | `numbers_F_laya.json` `primary.inference`, `per_tier.*.inference.H3_*`, `.gate_coverage` | V. H3 9B -0.0023 [-0.0219,+0.0158], p 0.428; 4B +0.0073 [-0.0099,+0.0291]; 2B 0.0000 [-0.0451,+0.0348]; coverage 0.4846. VN caveat (S4): the gate verdict in the Laya rows is pre-re-check for 3 rows; with re-checked verdicts the 2B point is -0.0035 (others unchanged). |
| C3 | 9B AUROC Laya 0.955, raw log-prob 0.812, gate 0.876, LR 0.960; AURC 0.207/0.311/0.414/0.196 | `numbers_F_laya.json` `per_tier.qwen3.5-9b-bf16.metrics` | V. AUROC Laya 0.9552, log-prob 0.8117, gate 0.8758, LR 0.9597; AURC 0.2069/0.3113/0.4140/0.1965 (LR refitted by me on the 9,243 training rows, matches). |
| C4 | Laya - LR pooled: 9B AUROC -0.004 [-0.015,+0.006], AURC +0.010 [+0.003,+0.019]; 4B -0.011 [-0.021,-0.002], +0.014 [+0.005,+0.023]; 2B +0.010 [-0.002,+0.023], -0.003 [-0.017,+0.011] | `numbers_F_laya.json` `per_tier.*.inference.X_*` | V. 9B AUROC -0.0045 [-0.0150,+0.0059], AURC +0.0104 [+0.0026,+0.0185]; 4B -0.0114 [-0.0209,-0.0020], +0.0139 [+0.0054,+0.0228]; 2B +0.0105 [-0.0025,+0.0233], -0.0026 [-0.0169,+0.0110]. |
| C5 | by domain Laya - LR AUROC: math 9B +0.011 [+0.002,+0.022], 2B +0.035 [+0.012,+0.059]; Python -0.018/-0.040/-0.033; Python H1 9B -0.036 [-0.092,+0.019], 4B -0.056 [-0.113,+0.001]; math H1 9B -0.088 [-0.116,-0.062] | `numbers_F_laya.json` `secondary.{python,math}.<tier>.inference` | V. math 9B +0.0114 [+0.0019,+0.0217], 2B +0.0350 [+0.0120,+0.0588]; Python -0.0176/-0.0397/-0.0332 (2B/4B/9B; 4B and 9B exclude zero); Python H1 9B -0.0362 [-0.0923,+0.0191], 4B -0.0562 [-0.1126,+0.0012]; math H1 9B -0.0879 [-0.1162,-0.0623]. Tier order of the Python triple was ambiguous in the text; labelled (S6). |
| C6 | router tau 0.5: acc 0.476 [0.443,0.509] vs always-9B 0.472 [0.441,0.504], diff +0.003 [-0.009,+0.015]; cost ratio vs 9B 0.889 [0.864,0.915], vs ladder 0.713 [0.691,0.737]; CWR 0.079 vs ladder 0.131 vs 9B 0.528 | `numbers_F_laya.json` `router_policy` | V. Router at tau 0.5 recomputed from `router/preds_F.jsonl`: acc 0.4758, ladder 0.4659, 9B 0.4725; CWR 0.0793/0.1311/0.5275; cost ratios 0.7131 and 0.8892; diff +0.0033. CIs from the author JSON. |
| C7 | E' comparison values quoted: H1 -0.078 [-0.106,-0.053], H3 -0.012 [-0.023,+0.011], Laya-LR AUROC +0.013 [+0.001,+0.026], cost 0.79 / 0.895 | `results/gwenlaya_v4/laya/laya_eprime.json`, `laya_eprime_v2.json`, `router_nowarm.json` (3.6.0; unchanged) | V. Read from `laya_eprime.json`, `router_nowarm.json` (0.8946); not recomputed (E' rows are 3.6.0). |
| C8 | leak check: 0 label changes / 0 prompt differences / 0 unexpected fields, 908 + 229 rows; 2,724 calibrator rows, 908 router rows | `eval_sets_F/laya_rows/manifest_F_laya_rows.json` | V. Manifest values; label-correct of all 2,724 rows equals the scored `base` verdict (0 mismatches). **VN (S4):** 3 rows carry the pre-re-check gate verdict. |
| C9 | Laya prediction on F: 444 s, $0.148 (ledger item tpu_f-laya-152622) | `results/gwenlaya_v4/spend_summary_3100.json`; `spend_ledger.jsonl` | V. 444 s x 1 chip x 1.20/3600 = 0.148; ledger line present. |

## D. Lean (Section sec:lean3100, Tables lean-a18, lean-F)

| # | Claim | Source | Verified |
|---|---|---|---|
| D1 | A18 test (223): proved 22/41/59/69 = 0.099 [0.063,0.139], 0.184 [0.135,0.238], 0.265 [0.206,0.323], 0.309 [0.251,0.368] | `results/gwenlaya_v4/lean_a18/scored_greedy_v3/summary_a18.json` `tiers.*.greedy_proof_rate`, `greedy_proved` | V. 22/41/59/69 recomputed from rows (status VERIFIED, all `exact`). Own bootstrap CIs agree within 0.005 (4B [0.211,0.323] vs [0.206,0.323]; 9B [0.251,0.372] vs [0.251,0.368]); the author CIs were not reproduced exactly (different resampling order). |
| D2 | F valid (229): proved 33/65/76 = 0.144 [0.100,0.192], 0.284 [0.227,0.345], 0.332 [0.271,0.393] | `eval_sets_F/lean_scored/summary_a18.json` | V. 33/65/76 of 229; CIs agree to the third decimal. |
| D3 | verified_match exact = all VERIFIED, defeq 0, unknown 0 (both sets) | same two files `tiers.*.verified_match`; `scored_raw_*.jsonl` `match` field | V. All VERIFIED rows have match `exact` (22+41+59+69, 33+65+76). |
| D4 | reasons (summary buckets) A18: 0.8B lean_error 41, forbidden 89, truncated 66, timeout 4, stmt 1; 2B 64/77/31/10; 4B 108/0/37/19; 9B 95/1/38/16 + stmt 3, import 1; F: 2B 90/75/24/4 + stmt 3; 4B 108/1/40/15; 9B 100/2/37/11 + stmt 2, import 1 | `summary_a18.json` `tiers.*.not_verified_reasons` | V. All buckets reproduced by re-bucketing raw reasons (finish_reason length moves lean_error, timeout, statement_not_preserved and unknown_import rows into truncated). **VN (S8):** the truncated bucket also absorbs 1 statement and 2 import rows of the F 9B; the raw counts are 3 and 3. |
| D5 | every forbidden_construct row fires on `sorry` (A18 89/77/0/1, F 76/1/2 rows of reason forbidden) | `results/gwenlaya_v4/lean_a18/lean_forbidden_breakdown_3100.json` (recompute with `gwaya.lean_gate.lexical_flaws` on `text`) | V. `lexical_flaws` rerun on the stored text: every forbidden_construct row (89/77/0/1; 76/1/2) is `sorry`, 0 unexplained. |
| D6 | timeouts inside cut generations (A18 2/1/4/8; F 0/3/5) | `scored_raw_*.jsonl` rows with `reason=timeout`, `finish_reason=length` | V. Timeouts with finish_reason length: A18 2/1/4/8; F 0/3/5; outside cut generations 4/10/19/16 and 4/15/11. |
| D7 | cascade vs always-9B: A18 81 vs 69 at 1.643x; F 95 vs 76 at 1.758x; overlaps (4B in 9B: 50) | `summary_a18.json` `cascade_greedy`, `greedy_overlap` | V. Union 81 (A18) and 95 (F); costs recomputed per task with chips 1/1/1/4: 0.3690 vs 0.2246 = 1.6432, 0.1386 vs 0.0789 = 1.7579; overlaps 16/18/19/35/35/50 and 25/27/50. |
| D8 | scoring history: pass 1 timeouts 4/10/22/18 (120 s), pass 2 4/10/19/16 (300 s, match unknown), pass 3 reported | `results/gwenlaya_v4/lean_a18/history/*.json`, `scored_greedy_v3/summary_a18.json`; D66-D68, D75 | VN. Pass 1 and pass 2 timeout buckets 4/10/22/18 and 4/10/19/16 read from the history summaries (bucket convention excludes cut generations); the pass-1 and pass-2 raw rows are not committed, so the totals including cut generations cannot be checked. Pass 2 match `unknown` for all VERIFIED rows confirmed in the history file. |
| D9 | k=8 sampling produced no rows | `$GWAYA_DATA_ROOT/lean_a18_tpu/greedy/out_*/raw_sample_*.jsonl` (0 bytes); `summary_a18.json` `any_of_k_proof_rate: null` | V. Four `raw_sample_*.jsonl` files are 0 bytes; `any_of_k_proof_rate` null. |
| D10 | A18 tokens and list-price USD per tier (0.0534, 0.0537, 0.0795, 0.2246; F 0.0263, 0.0441, 0.0789) | `summary_a18.json` `greedy_tokens`, `greedy_usd` | V. Tokens 101,627/77,356/102,785/97,222 and 72,756/96,326/94,852 recomputed; USD from accel_s x chips x 1.20/3600 equals the summary (0.0534, 0.0537, 0.0795, 0.2246; 0.0263, 0.0441, 0.0789). |

## E. A15.3 Rust selection (Section sec:a153, Tables A153, A153-diff)

| # | Claim | Source | Verified |
|---|---|---|---|
| E1 | E: 505 usable tasks; `first` CWR 4B 0.067/0.042/0.022, 9B 0.061/0.032/0.014 at k=1,2,3 | `results/gwenlaya_v4/edition/rust_gate_selection.json` `sets.E.tiers.*.first.k{1,2,3}.confident_wrong` | VN. 505 usable and `first` CWR 0.067/0.042/0.022 and 0.061/0.032/0.014 read from the JSON, whose counts and rates are internally consistent (count/n = rate for all 3 sets x 3 tiers x 4 rules x 3 k). The raw verdict cache of the Rust compiles was not found on the data disk, so the compiles were not re-run. |
| E2 | E: no paired difference from first excludes zero at k=2,3; first_last k=1 2B +0.030 [+0.004,+0.057]; points within +-0.012 at k=2,3 | same, `*.confident_wrong_minus_first` | VN. As E1. First_last k=1 2B +0.030 [+0.004,+0.057] is the only k=1 interval excluding zero on E; k=2,3 intervals all include zero, points within +-0.012. |
| E3 | chosen on E: first_last 0.0211 vs first 0.0238, longest 0.0271, spread 0.0221 | same, `chosen_on_E.mean_cw_k2_to_K` | VN. 0.0211/0.0238/0.0271/0.0221 recomputed from the per-tier rates. |
| E4 | R' (68 usable): first k=1 CW 15/12/8 (0.221, 0.176, 0.118); first_last 11/6/5; longest 7/6/3; spread 8/10/3 | same, `sets.Rfresh.tiers.*.<rule>.k1.confident_wrong_count` | VN. 15/12/8, 11/6/5, 7/6/3, 8/10/3 and rates read from the JSON. |
| E5 | R' differences excluding zero: 2B longest k1 [-0.206,-0.044], spread k1 [-0.191,-0.029]; 9B all rules k1..3 except first_last k1; 4B upper bounds 0.000 | same, `sets.Rfresh` | VN, with an omission fixed (S3). 2B longest k=2 [-0.162,-0.015] also excludes zero and was not mentioned; the other statements hold (4B upper bounds exactly 0.000). |
| E6 | 0 sanity violations; one stored-2015 difference each: E 4B mbpp_90_len_log, E 9B mbpp_619_move_num, R' 4B allergies | same, `*.sanity_violations`, `*.stored_2015_differences` | VN. 0 sanity violations; the three stored-2015 differences are as listed (E 4B mbpp_90_len_log, E 9B mbpp_619_move_num, R' 4B allergies). |

## F. Runtime, A19, A20 (Section sec:runtime3100, Tables A19, A20)

| # | Claim | Source | Verified |
|---|---|---|---|
| F1 | A19 Python 28 pairs: mean diff +0.096 [-0.020,+0.230], mean abs 0.162 [0.054,0.287] (max 0.913), Spearman 0.64 [0.28,0.90]; gate differs 5/28 (4 F->V, 1 V->F); identical text 9/28, abs diff there 0.030 | `results/gwenlaya_v4/laya_runtime/quant_shift_probe.json` `python`, `table.python` | V. Recomputed from `per_task`: Python 28 pairs mean +0.096 [-0.020,+0.230], mean abs 0.162 [0.054,0.287], max 0.913, Spearman 0.642 [0.28,0.90]; gate differs 5/28 (4 F->V, 1 V->F); identical text 9/28, abs diff 0.030. |
| F2 | A19 Rust 26 pairs: -0.024 [-0.174,+0.126], abs 0.177, Spearman 0.65, gate differs 5/26; math 26 pairs protocol-confounded -0.182 [-0.286,-0.081] | same, `rust`, `math_protocol_confounded` | V. Rust 26 pairs -0.024 [-0.174,+0.126], abs 0.177, Spearman 0.647, gate differs 5/26; math -0.182 [-0.286,-0.081]. |
| F3 | A20 E': 828/828 agree on all items; cost 0.7879 runtime = registered; +0.97 points; answered accuracy 0.621 vs 0.581 (+3.99 points); verdicts 547/235/46; R' 69/69, 0.987, 0 | `results/gwenlaya_v4/laya_runtime/system_replay_eprime.json` `eprime.agreement`, `eprime.aggregates`, `rfresh.*` | V. 828/828 on five items; cost 0.78786; +0.966 points; answered accuracy 0.6208 vs 0.5809 (+3.99); verdicts 547/235/46; R' 69/69, 0.9872, 0. |
| F4 | A20 0.8946 without first chunk | `docs/LAYA_RUNTIME.md` replay table; `results/gwenlaya_v4/laya/router_nowarm.json` | V. 0.8946 in `system_replay_eprime.json` and `router_nowarm.json`. |
| F5 | parity: router max 0.0038, calibrator max 0.0029, 9/20 calibrator rows truncated | `results/gwenlaya_v4/laya_runtime/parity.json` | V. Router max 0.0038, calibrator max 0.0029, 9 of 20 truncated. |
| F6 | PoT runtime wiring consistency 40/40 | `results/gwenlaya_v4/math_pot_runtime/consistency.json` `agree`, `agreement` | V. 40 of 40 agree, 0 unexplained. |
| F7 | live smoke n=3, 12.2 / 14.0 / 97.3 s, p 1.00 / 0.43 / 0.97 | `results/gwenlaya_v4/laya_runtime/live_smoke.json`; `docs/LAYA_RUNTIME.md` | V. 12.22 / 13.97 / 97.29 s; p 1.00 / 0.43 / 0.97. |
| F8 | not-validated list (Q4 calibration, end-to-end accuracy, thresholds, 4B/9B paths) | `docs/LAYA_RUNTIME.md` "Not validated"; D77 | VN. Statement of limits, checked against `docs/LAYA_RUNTIME.md` and D77; not a number. |

## G. A22 (Section sec:a22, Table A22)

| # | Claim | Source | Verified |
|---|---|---|---|
| G1 | E' pooled dAUROC v2-v1 -0.0096 [-0.0161,-0.0036] (v1 0.941, v2 0.932); R' pooled +0.0041 [-0.0270,+0.0339] (0.862 vs 0.867); v2 not better under the registered rule | `results/gwenlaya_v4/laya/layout_a22.json` `eprime.pooled.paired.v2_minus_v1.dAUROC`, `rfresh.pooled...`, `decision` | V. Raw-score AUROC difference recomputed with my own bootstrap: E' -0.0096 [-0.0161,-0.0036] (v1 0.941, v2 0.932); R' +0.0041 [-0.0270,+0.0339] (0.862 vs 0.867). v2 isotonic map not needed (AUROC is on raw scores by design). |
| G2 | per tier E' 2B -0.004, 4B -0.015 [-0.025,-0.006], 9B -0.009 [-0.018,+0.000]; R' 2B +0.026, 4B +0.044, 9B -0.009 | same `per_tier` | V. E' 2B -0.0035, 4B -0.0152 [-0.0254,-0.0059], 9B -0.0085 [-0.0178,+0.0001]; R' 2B +0.0265, 4B +0.0436, 9B -0.0093. (My 2B R' interval is NaN-affected: constant resamples; the author reports [-0.004,+0.071].) |
| G3 | v2 - LR: E' +0.003 [-0.009,+0.015]; R' -0.054 [-0.104,-0.010] | same `*.pooled.paired.v2_minus_hand_lr` | VN. v2 - LR from the JSON (hand LR refit not repeated for this set). |
| G4 | window fractions not computed (no tokenizer in the environment) | absence of any such key in `layout_a22.json`; commit message b98a627; D74 | V. No window-fraction key in `layout_a22.json`. |
| G5 | single seed, same 9,243 training rows | `results/gwenlaya_v4/laya/layout_a22.json` `meta`; `layout_v2/run_manifest.json`, `head_config.json`; plan A22 | V. `n_train_rows` 9,243, single seed in meta. |

## H. Spend (Section spend, Table spend3100, abstract, H13)

| # | Claim | Source | Verified |
|---|---|---|---|
| H1 | ledger 31 lines, total 22.0055 USD (44.0% of the 50 USD cap; macro `SpendCapShare`); 3.9.0 had 27 lines, 19.2801 | `results/gwenlaya_v4/spend_summary_3100.json`; `$GWAYA_DATA_ROOT/spend_ledger.jsonl`; `spend_summary_390.json` | V. Ledger has 31 lines; sum of `usd_estimate` 22.0055 (44.01%, macro 44.0); first 27 lines 19.2801. |
| H2 | 3.10.0 lines: tpu_f-9b-185104 1,016 s 1.3547 (v5litepod-4); tpu_f-small-185104 2,139 s 0.7130; tpu_f-laya-152622 444 s 0.1480; tpu_laya-layout-152621 1,529 s 0.5097; sum 2.7254 | same | V. 1.3547 / 0.7130 / 0.1480 / 0.5097 = 2.7254; seconds 1,016 / 2,139 / 444 / 1,529. |
| H3 | the two A18 Lean lines (1.5220) belong to 3.9.0 | `spend_summary_390.json` | V. Lines 26-27 are 1.044 + 0.478 = 1.522, in the 19.2801. |

## I. Corrections and deviations (Section corr3100, Table dev3100, `docs/DEVIATIONS.md` D66-D77)

| # | Claim | Source | Verified |
|---|---|---|---|
| I1 | "no Lean result", "generation in progress", "sampling is being re-run" of 3.9.0 are superseded; the sample files are empty | paper 3.9.0 text (`git show fa14857:papers/gwenlaya_v4.tex`); D75 | V. 3.9.0 text quoted in `git show fa14857`; D75; sample files empty. |
| I2 | A15.3 plan f4af1e4 07:35:49, code e556339 07:38:01, results b2920c8 08:20:49; 3.9.0 paper commit 2800549 08:10 | `git log` | V. f4af1e4 07:35:49, e556339 07:38:01, b2920c8 08:20:49, 2800549 08:10:11. |
| I3 | A19 plan 1bb7543 14:18:21 / results 73d1a15 14:41:33; A20 plan cca2049 15:01:14 / results 294054e 15:04:15; A22 plan 2a76e2f 15:43:58 / result b98a627 2026-10-11 15:59:42 | `git log` | V. 1bb7543 14:18:21, 73d1a15 14:41:33, cca2049 15:01:14, 294054e 15:04:15, 2a76e2f 15:43:58 (10-10), b98a627 2026-10-11 15:59:42. |
| I4 | 27B probe not run (no result file, no ledger line) | absence in `results/` and ledger; D76 | VN. No 27B result file and no ledger line; D76. |
| I5 | pyproject.toml still says version 3.4.0 and README keeps no version badge; neither was changed (as for 3.9.0) | `pyproject.toml`; `git show fa14857 --stat` | V. pyproject.toml `version = "3.4.0"` unchanged; README has no version badge. |

## Notes for the auditor

- Fix-forward expectations: nothing registered changed. Items most likely to need wording care: C5 (domain-level statements), D6/D4 (bucket
  precedence of `truncated`), E5 (R' intervals touching zero), F1 (in-sample, 2B only), H1 percentage (the spend share in the abstract
  is the macro `\SpendCapShare`).
- Known unverifiable by the author: the data-disk raw generation files for A18 (`raw_all/`) were not copied (D75); only scored rows are committed.


## Slips found and fixed (auditor, 3.10.0)

| # | Where | What was written | Correct | Fix |
|---|---|---|---|---|
| S1 | Section sec:F, gate paragraph (ii); claim B6 | "the 2B is 0.067 against 0.184 on 89 and 304 tasks, which with these intervals does not show an effect" | marginal 95% CIs are [0.022,0.125] (test in prompt) and [0.142,0.227] (not in prompt): disjoint. The direction is opposite to a leak reading, the groups differ in source mix, 4B and 9B show no difference | sentence rewritten: reported as a difference between task groups, no visibility effect claimed |
| S2 | Section sec:F, gate paragraph (i); claim B5 | precision "lowest on BigCodeBench (0.533, 0.706, 0.652)" | BCB has the lowest precision only for the 9B; LeetCode is lower for the 2B (0.455) and 4B (0.590). BCB has the highest confident-wrong rate on every tier | rewritten with both statements |
| S3 | Section sec:a153, R' paragraph; claim E5 | intervals excluding zero for 2B: longest and spread at k=1 only | 2B longest at k=2 is also below zero: -0.088 [-0.162,-0.015] (Table A153-diff) | added |
| S4 | Laya rows (C2, C8); Table F-laya `gate coverage` and H3 | gate coverage 0.485 / H3 k = 440 for the 9B presented as the F gate verdicts; Table F-gate has 0.486 | `calibrator_F.jsonl` holds the pre-re-check gate verdict for 3 of 2,724 rows (2B BCB/843 FAILED, 4B and 9B MATH/829 UNVERIFIED; re-check made all VERIFIED). With re-checked verdicts: 9B coverage 0.4857 (441), H3 unchanged -0.0023, 4B +0.0073 unchanged, 2B 0.0000 -> -0.0035; labels unaffected; no verdict changes | disclosure sentence in Section sec:Flaya, D73 note in DEVIATIONS.md, table row relabelled in the generator (tables_flaya3100.tex regenerated) |
| S5 | Abstract, Section sec:Flaya (i), README | H1 "supported" / "now holds on 908 fresh Python and math tasks" without the Python-alone caveat in the abstract and in the (i) sentence; README "supported" with no qualifiers; README opening still said no test had completed and the Laya "does not exist yet" | within Python alone the 9B H1 interval includes zero (-0.036 [-0.092,+0.019]); pooled result is carried by math; README opening was a 3.4.0 statement | caveats added in abstract, (i) and README; README opening corrected |
| S6 | Abstract, Section sec:Flaya | Python AUROC Laya - LR "(-0.018, -0.040 and -0.033)" without tier labels (the previous clause lists 9B first) | 2B -0.018, 4B -0.040, 9B -0.033 | labelled in the abstract |
| S7 | Section sec:lean390 (3.9.0 text) | "has produced no result in any revision"; "No proof has yet been scored with it" | stale in 3.10.0 | marked as superseded with a pointer to sec:lean3100 and sec:corr3100 |
| S8 | Section sec:lean3100, "Why proofs were not verified" | statement-not-preserved and unknown-import counts given as if raw | the `truncated` bucket takes precedence: F 9B raw reasons are 3 and 3, of which 1 and 2 are cut generations | sentence added |
| S9 | Section spend, last bullet | "the three newest items" list-price estimates | four items are new in 3.10.0 | reworded to all ledger items |

Not changed, but noted: (a) A15.3 raw verdicts (the compile cache) were not available, so E and E4-E6 are verified for internal consistency only; (b) D8 pass-1 and pass-2 raw rows are not committed; (c) my bootstrap CIs for the Lean proof rates differ from the author's by up to 0.005 in one bound (resampling order), the points agree; (d) my R' 2B A22 interval came out NaN in a plain implementation (some resamples have one class); the author's [-0.004,+0.071] was not reproduced.
