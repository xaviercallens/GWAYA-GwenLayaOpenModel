# Claims register of papers/gwenlaya_v4.tex revision 3.12.0 (branch release/3.12.0), written by the author

Status: **author's register, NOT an audit.** Each row names a claim added or changed in 3.12.0, where it sits in the paper, the committed file and key it was read from,
and how the author checked it. The auditor recomputes everything independently; the "Author check" column says only what the author did.
Paths are relative to the repository root. Generated tables are written by `scripts/paper/tables_phase2.py` (`GWAYA_DATA_ROOT=/mnt/data/home/xavkal/gwaya-data`),
which reads committed JSON files plus the spend ledger. Prose numbers were copied from those tables or read from the JSON keys named below.
`TL` = `results/gwenlaya_v4/trust_loop`, `LY` = `results/gwenlaya_v4/laya`. Wilson intervals are the files' own. Section names are labels in the paper.

Known weak points the auditor should look at first: (a) A29/A30 numbers that exist only as plan text or in a rerun of the committed script on data-disk files (rows C9, C12, C13);
(b) the A31 reading that precision on E is at least as high as on development (row F9: the file contradicts it); (c) the plan's D3 "0.83" (row C7);
(d) the 21% truncation counts (row C13); (e) the spend lines (section H).

## A. Status, order of events and labels (Sections phase2, p2-status, p2-order)

| # | Claim | Source | Author check |
|---|---|---|---|
| A1 | Every phase-2 item is exploratory with a rule written before its number; no registered hypothesis tested | `docs/ANALYSIS_PLAN_E_TPU.md` addenda A23-A32 | read the plan text of each addendum; the rule paragraphs precede the "Outcome" headings |
| A2 | A23 plan only: pool-O builder committed, no generation/label/v3 score | `scripts/data/build_pool_py_oci.py`, `tests/test_build_pool_py_oci.py`; ledger has no pool-O generation item | `git ls-files results` has no pool-O result; ledger item names listed |
| A3 | A24 plan only; A27 plan only (no result file for A27.1) | plan text; no result files | `find results -newer` / grep for A24/A27 outputs returned none |
| A4 | A25: KV arms have no result file; refusal readout is descriptive and has a result file | `results/refusal_F_python_laya_v1.json`; no kv result file anywhere | `find . -name "*kv*"` empty |
| A5 | A25 refusal, 9B at share answered 0.4: Laya 0.748, gate-then-lp 0.798, raw log-prob 0.656; 212 of 407 tasks wrong for all tiers | `results/refusal_F_python_laya_v1.json` `by_tier.qwen3.5-9b-bf16.scorers.{laya,gate_then_logprob,mean_logprob}."0.4".accuracy_answered`, `n_hard_all_tiers_wrong` | printed from the JSON |
| A6 | A26 amendment 0.90 -> 0.80 made by the user before any lever/synthesis/sealed result was read (as the plan states), after the first loop run had started | plan "Amendment to A26"; commits d443d6c 09:31:43, a77f4ee 09:50:23 | quoted; the claim "before results were read" is the plan's own statement, not independently verifiable from files |
| A7 | A26 correction (9B exists on E; primary tier 9B) before any loop result was read | commit 403adc9 09:50:38; `TL/build_summary.json` `test_by_bench_tier` lists 9B rows | read |
| A8 | A29 amendment 0.85 -> 0.84 before any sample was scored or read (scoring ~55% done, no result) | plan "Amendment to A29"; commit 3a5ac77 07:12:53, outcome 2a2a6da 07:28:17 | quoted; same status as A6 |
| A9 | A30 goal redefinition made post hoc by the user after A29 was read | plan A30 first paragraph; commit 3e5ea6c 08:36:08 after 2a2a6da | quoted |
| A10 | Commit times: A28 plan 19:50:02, outcome 21:27:55; A29 plan 21:28:30; A31 freeze 21:08:03, scorer 21:08:12, outcome 21:08:40; A32 plan 21:09:27, outcome 21:11:13 | `git log --format='%h %ad' origin/main..HEAD` | printed |
| A11 | VM name suffixes 200303, 221616, 084710 | ledger items `tpu_laya-v2-200303`, `tpu_f-sample-221616`, `tpu_f-sample-9b-084710` | ledger |

## B. A26 dev loop: U (Table p2-U, Section p2-loop)

| # | Claim | Source | Author check |
|---|---|---|---|
| B1 | Dev table: F 407 + E' MBPP 169 per tier, 1,728 rows, 5 folds grouped by cluster, seed 0, 0 task overlap with 2,116 test rows | `TL/build_summary.json` `dev_rows`, `test_rows`, `n_overlap`; `TL/lever_U_prompt_only.json` `config` | read |
| B2 | Base rates of "tier solves the task": 2B 0.248, 4B 0.389, 9B 0.457 | `TL/lever_U_prompt_only.json` `tiers.<tier>.base_rate` | read |
| B3 | Table p2-U cells: AUROC / bal. acc. [cluster CI] / accuracy for prior, tfidf_lr, hand_lr, hand_gbm, emb_lr, ensemble per tier | same file `tiers.<tier>.models.<m>.{auroc, max_bal_acc_point.{bal_acc,acc,bal_acc_cluster_boot95}}` | table generated, spot-checked against JSON |
| B4 | Best bal. acc.: 2B 0.644, 4B 0.687 (tfidf_lr), 9B 0.654 (ensemble); ensemble AUROC 0.70/0.75/0.70; optimistic threshold at most 0.696 (4B) | same file; `optimistic_best_thr_bal_acc` max over models | python max over keys |
| B5 | U not met: target 0.80 (amended), no model reaches | same | comparison |
| B6 | Router p AUROC 0.671/0.690/0.638 (F only); Laya calibrator p AUROC 0.942/0.926/0.918 (not prompt-only) | `tiers.<tier>.router_p_reference_F_only.auroc`, `laya_p_NOT_prompt_only_reference.auroc` | read |

## C. A26 D, consensus, and best-of-8 (Tables p2-D, p2-cons, p2-bok; Sections p2-loop, p2-bok)

| # | Claim | Source | Author check |
|---|---|---|---|
| C1 | D3 `lr|core` pooled rank 0.5: 0.726 [0.695, 0.754]; hand rule gate-then-log-prob 0.729 [0.699, 0.758]; log-prob alone 0.481 [0.448, 0.515] | `TL/lever_D_stacked_calibrated.json` `lr|core.ALL.curve."0.5"`, `hand_rule.ALL.{gate_then_logprob."0.5", logprob_only_cov50}` | read |
| C2 | Per tier hand rule 0.624/0.757/0.799 vs D3 0.497/0.771/0.819 at rank 0.5 | `hand_rule.<tier>.gate_then_logprob."0.5"`, `lr|core.<tier>.curve."0.5"` | read; intervals in Table p2-D; no paired test exists |
| C3 | Pooled at coverage 0.3: D3 0.861 [0.829, 0.888]; hand rule 0.788 [0.751, 0.821] | `lr|core.ALL.curve."0.3"`, `hand_rule.ALL.gate_then_logprob."0.3"` | read |
| C4 | D3 at CV threshold target 0.85: pooled cov 0.24 prec 0.883 [0.848, 0.910]; 4B 0.27/0.891 [0.832, 0.931]; 9B 0.28/0.909 [0.855, 0.944]; 2B 0.15/0.820 [0.728, 0.886] | `lr|core.<scope>.at_cv_threshold."0.85"` | read |
| C5 | 9B at rank 0.5 meets 0.80 point estimate with Wilson LB 0.771 ("met with confidence" under the amended definition: point >= 0.80, LB >= 0.75); 4B 0.771 and 2B 0.497 do not; 0.90 not met | `lr|core.qwen3.5-9b-bf16.curve."0.5"` | author's reading of the amended criterion on dev data; the amended text is in the plan |
| C6 | AUROC pooled `lr|core` 0.947, `+laya` 0.952, `full` 0.950; file warns laya_p on F may be optimistic | `lr|core.ALL.auroc`, `lr|core+laya.ALL.auroc`, `lr|full.ALL.auroc`, `meta.note` | read |
| C7 | Plan's "D3 0.83 for the 9B at coverage 0.5" is not the D3 `lr|core` value (0.819); 0.828 is `baseline_dev.json` top half of the 338 gate-verified 9B rows (169 accepted, overall cov 0.29) | `TL/baseline_dev.json` `qwen3.5-9b-bf16|ALL.D_gate_verified_then_lp_cov50` (n 169, k 140, precision 0.8284, coverage field 0.5); `lr|core.qwen3.5-9b-bf16.curve."0.5"` | read; the source of the plan's 0.83 is not stated in the plan (hypothesis only) |
| C8 | Consensus: 4B precision at 0.5 base 0.767 -> base+agree 0.774 (three-tier), 0.778 (two-tier); 9B 0.830 -> 0.826; 2B 0.497 -> 0.497; agreement-only weaker | `TL/lever_D_consensus_agreement.json` `learned_grouped_cv.{two_tier,three_tier}.<tier>.{base_only,agree_only,base+agree}.oof_cov50` | read |
| C9 | A29 4B: greedy+gate 189 of 407 (0.46) at 0.709 [0.641, 0.769]; S1 307 (0.75) at 0.619 [0.563, 0.671]; S1@0.5 0.701 [0.635, 0.760]; D3 0.771; 2B S1@0.5 0.475 [0.408, 0.544] vs 0.497 | `LY/phase2_A29_best_of_8_F_python.json` `<tier>.{G_gate,S1_all,S1_cov50,D3_core_cov50_FplusEprime}` | read |
| C10 | A30 9B: greedy pass 0.396, mean sampled 0.373, some correct 0.617; G 221 (0.54) 0.729 [0.666, 0.783]; S1 325 (0.80) 0.646 [0.593, 0.696]; S1@0.5 0.750 [0.686, 0.804] (boot 0.691-0.809); D3 0.819; S1 A30 coverage 0; D3 reference 0.40 / 0.28; lever false | `LY/phase2_A30_best_of_8_F_python_9b.json` `qwen3.5-9b-bf16.*`, `A30_decision` | read |
| C11 | Gate-verified samples failing hidden check: 9B 494 of 1,698 (29.1%) | same file `verified_samples` | arithmetic |
| C12 | 4B 490 of 1,503 (32.6%) and 2B 503 of 954 (52.7%); 4B identical-text 0.745 [0.681, 0.800] at 0.5, no A30 prefix, 56 tasks with group >= 2; 2B 0.505, 17 groups; "some correct" 0.339/0.541 and greedy 0.189/0.329 | **not in `phase2_A29_best_of_8_F_python.json`** (committed before those fields existed); in `LY/phase2_A29_rerun_A30script.json` `<tier>.{verified_samples, posthoc_identical_text}`, which is the output of `scripts/phase2/analyze_best_of_k.py` rerun on `$GWAYA_DATA_ROOT/eval_sets_F_sample/scored_samples.jsonl` (not committed) with the committed D3 file | rerun by the author; every key of the committed A29 file is identical in the rerun (asserted in the table script); the plan text quotes 490/1,503 and 0.745 as well |
| C13 | 9B: 690 of 3,256 samples (21%) hit the 1,024-token cap, 26 of them pass the gate, 1 the hidden check; greedy 85 of 407 | **plan text only** (A30 outcome); no committed key | not recomputed; the sample file is on the data disk |
| C14 | 9B identical-text: 0.765 [0.702, 0.818] at 0.5; largest A30 prefix cov 0.30 (n 124) 0.871 [0.801, 0.919]; 55 tasks with group >= 2 | `LY/phase2_A30_...9b.json` `posthoc_identical_text` | read |
| C15 | A29 outcome's 0.740 / 0.898-at-0.12 for the 4B were from an uncommitted computation and are superseded | plan A30 outcome text; C12 rerun | stated by the plan; the rerun gives 0.745 and no A30 prefix |
| C16 | Oracle "some candidate correct" 0.617 (9B) vs greedy 0.396 | C10 | read |
| C17 | Samples: 2B and 4B 6,512 (2 x 3,256); 9B 3,256; T=0.8, top-p 0.95 | plan A29/A30 text; `LY/..._9b.json` `n_samples_scored`; `scripts/tpu/launch_f_sample*.sh` | read; temperature/top-p from plan text and launcher |

## D. A28 Laya v2 (Table p2-v2, Section p2-v2)

| # | Claim | Source | Author check |
|---|---|---|---|
| D1 | Fold 0: 200 tasks, 600 rows, 46 confident-wrong | `LY/phase2_v1_vs_v2_F_fold0.json` `n_tasks`, `n_rows`, `v1.n_cw` | read |
| D2 | Python AUROC 0.915 -> 0.949, +0.034 [+0.001, +0.069]; math 0.974 -> 0.951, -0.023 [-0.048, -0.000]; pooled -0.007 [-0.028, +0.014] | `v1/v2/delta_v2_minus_v1.{python,math,all}_auroc` | read |
| D3 | ECE 0.042 -> 0.147, Brier 0.084 -> 0.136 (pooled); v2 raw, no isotonic | same keys `all_ece`, `all_brier`; plan outcome text for "no isotonic map fitted" | read; the raw/calibrated status of the v1 column is NOT stated in the file or script (paper says the rows are not like-for-like) |
| D4 | confident-wrong catch below median: v1 0.00, v2 0.20 [0.00, 0.44] | `cw_catch` | read |
| D5 | Rule not met (math falls by more than 0.01) | plan A28 decision rule vs D2 | comparison |
| D6 | Calibrator 1,968 steps in 1,522 s; router 581 s; v2 trained on v1 rows + F folds 1-4; 154 CW rows in train folds | plan A28 outcome/plan text only | not recomputed |
| D7 | T2 probe: 0.8B and 2B did not finish a first step in 900 s (rc 124), PyTorch linear-attention fallback | plan A28 outcome text only (no committed log) | not recomputed |

## E. A32 (Table p2-sel, Section p2-sel)

| # | Claim | Source | Author check |
|---|---|---|---|
| E1 | 9B lr: B0 cov 0.30 prec 0.846 [0.771, 0.899]; B1 cov 0.32 prec 0.884 [0.817, 0.928]; coverage gain +0.015; paired correct-accepted diff @0.30 +5.7 [-0.02, +12]; a_criterion true, b false, c false | `LY/phase2_A32_selector_over_pool_F_python.json` `qwen3.5-9b-bf16.lr.{B0,B1,paired_correct_diff_cov30,rule}` | read |
| E2 | @0.5 B1 vs B0: 9B 0.804 vs 0.740; 4B 0.755 vs 0.652; 2B 0.500 vs 0.377 | `...lr.{B0,B1}.prec_cov50.precision` | read |
| E3 | 4B B1 cov 0.27 prec 0.835 [0.754, 0.893], gain +0.086, (a) not met; GBM changes no verdict (all `lever` false) | `qwen3.5-4b-bf16.lr.*`, `.gbm.rule.lever` | read |
| E4 | B0 is F Python alone, not the D3 number | plan A32 text; D3 9B 0.909 at cov 0.28 (C4) | stated |

## F. A31 sealed E (Tables p2-sealed, p2-devE; Section p2-sealed)

| # | Claim | Source | Author check |
|---|---|---|---|
| F1 | 529 tasks per tier (HumanEval+ 156, MBPP+ 373); 1,587 rows; Coder-1.5B skipped | `LY/sealed_E_D3_lr_core.json` `n_rows`, `<tier>.ALL.n`; `TL/build_summary.json` `no_gate` | read |
| F2 | Frozen 0.85 threshold 0.798, dev OOF coverage 0.243; policy columns gate, mlp, minlp, logtok, trunc, nvis; 1,728 rows | `LY/frozen_policy_D/policy_thresholds.json`, `sealed_E_D3_lr_core.json` `policy` | read |
| F3 | 9B pooled frozen 0.85: cov 0.73, prec 0.886 [0.850, 0.914]; rank 0.5 0.906 [0.864, 0.935]; HumanEval+ 0.71/0.927 [0.863, 0.963]; MBPP+ 0.74/0.869 [0.824, 0.904] | `qwen3.5-9b-bf16.{ALL,HumanEval+,MBPP+}.at_frozen_threshold."0.85"`, `rank_curve."0.5"` | read |
| F4 | 4B 0.852 [0.812, 0.884] at cov 0.70; HumanEval+ 0.841 [0.760, 0.898] LB < 0.80; 2B 0.871 [0.820, 0.908] at cov 0.42; HumanEval+ 0.825 [0.714, 0.900] | `qwen3.5-4b-bf16`, `qwen3.5-2b-bf16` same keys | read |
| F5 | Base rates 0.726 / 0.673 / 0.450 (9B/4B/2B) | `<tier>.ALL.base_rate` | read |
| F6 | Gate only: 9B cov 0.87 prec 0.833 [0.796, 0.864]; 4B 0.828 at 0.81; 2B 0.760 at 0.59; policy buys +0.053 precision for 0.14 coverage on 9B | `<tier>.ALL.gate_only` | arithmetic |
| F7 | Both rules met on the 9B pooled (rule (i) prec >= 0.84 & LB >= 0.80; (ii) prec >= 0.80 & LB >= 0.75) | plan A31 rules vs F3 | comparison |
| F8 | Largest label-using A30 rank prefix on E, 9B: coverage 0.86 prec 0.842 [0.805, 0.872] | `qwen3.5-9b-bf16.ALL.A30_rank_curve_largest` | read |
| F9 | **Plan's reading "precision on E at least as high as dev at each threshold" is not true**: at 0.85, 4B 0.852 vs dev 0.891, 9B 0.886 vs 0.909, pooled 0.869 vs 0.883; at 0.90, 4B 0.911 vs 0.934 | E: `<tier>.ALL.at_frozen_threshold`; dev: `TL/lever_D_stacked_calibrated.json` `lr|core.<scope>.at_cv_threshold` | table p2-devE generated; pooled E = sum of `round(n_accept*precision)` over tiers / sum of `n_accept` (computed in the script, not a file key) |
| F10 | Dev vs E coverage 0.28 (dev 9B at CV thr.) vs 0.73; dev base rate 0.457 vs 0.726 | `lr|core.qwen3.5-9b-bf16.at_cv_threshold."0.85".coverage`, `TL/baseline_dev.json` | read |
| F11 | On E the gate payload includes the extended HumanEval+/MBPP+ tests; on F one visible assertion | plan A31 "Reading" text only | not verified against task files |
| F12 | Sealed table read once, scorer refuses to overwrite | `scripts/loop/score_sealed_D.py` (`sys.exit` if OUT exists); plan | code read; "read once" cannot be verified from files |

## G. Departures table and system smoke (Section p2-dev)

| # | Claim | Source | Author check |
|---|---|---|---|
| G1 | Endpoint smoke: 8 of 8 VERIFIED round 1, cold start 427.1 s, warm 1.8-4.2 s (36.09 s first row), model gwenlaya-4b | `results/assist_smoke_2026-10-13.json` `rows[*].{verdict,rounds,wall_s}`, `cold_start_s`; `docs/IMPROVEMENT_PLAN_ASSISTANT.md` | rows printed: 7 rows 1.78-4.17 s, first row 36.09 s (includes warm-up); the "1.8 to 4.2 s" is the plan's phrase and holds for 7 of 8 rows |
| G2 | `docs/IMPROVEMENT_PLAN_ASSISTANT.md` names the repair-loop evaluation "A23", the plan's A23 is data scale | both files | read |
| G3 | DEVIATIONS.md last entry is D77; not extended for phase 2 | `docs/DEVIATIONS.md` | grep |

## H. Spend (abstract, Section spend, Table spend3120, macros numbers3120.tex)

| # | Claim | Source | Author check |
|---|---|---|---|
| H1 | Ledger 35 lines, sum 26.3314, 52.7% of $50; first 31 lines sum 22.0055 (3.10.0) | `/mnt/data/home/xavkal/gwaya-data/spend_ledger.jsonl`; `papers/numbers3100.tex`; `results/gwenlaya_v4/spend_summary_3120.json` | script asserts first 31 lines = 22.0055 |
| H2 | New lines: `tpu_endpoint_assist_smoke` 520 s $0.1733 (seconds approximate, ts 2026-10-13T00:00); `tpu_laya-v2-200303` 4,386 s $1.462; `tpu_f-sample-221616` 3,304 s $1.1013; `tpu_f-sample-9b-084710` v5litepod-4 1,192 s $1.5893; total 4.3259 | ledger lines 32-35 | read |
| H3 | Purpose column and "A28 T2 probe is not a separate ledger line" | inferred from item names, plan text | stated as inferred in the caption |
| H4 | Cost of A30: ledger line vs plan "1.59 USD, ledger $26.33" | plan A30 outcome | consistent |
| H5 | 3.10.0 macros (31 lines, 22.0055, 44.0%) unchanged | `papers/numbers3100.tex` untouched | `git diff` empty for the file |

## I. Build

| # | Claim | Source | Author check |
|---|---|---|---|
| I1 | 97 pages (3.10.0: 85); no undefined references or citations; 13 overfull boxes, the same 13 as a 3.10.0 build of the committed tex | `latexmk -pdf` from `papers/`; baseline built from `git archive HEAD` before the edit | counts compared |
