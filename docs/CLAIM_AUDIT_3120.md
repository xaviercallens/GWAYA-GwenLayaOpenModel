# Claim audit of papers/gwenlaya_v4.tex revision 3.12.0 (branch release/3.12.0)

Status: **audited by an independent auditor (recomputation from raw rows with separate code, not from the author's summary JSONs).** The rows follow the author's register
`docs/CLAIM_AUDIT_3120_author.md` (same ids). V = recomputed and matches to printed precision; VN = verified with a stated caveat or only partly verifiable; X = mismatch or unsupported statement (correct value given).
Totals over the 82 rows: 66 V, 12 VN, 4 X (B-row coverage of the hand rule, F11 gate payload, C13 label "plan text only" is V now, F9 plan reading, the "fresh set F" label) plus the omissions listed in "Slips found and fixed".
Recomputation code: `scripts/audit/audit3120_recompute.py` (D3 CV re-implemented; writes `results/gwenlaya_v4/trust_loop/audit3120_recomputed.json`); the other checks were run as throw-away scripts that read
`dev_table.jsonl`, `test_table.jsonl`, `eval_sets_F_sample*/scored_samples.jsonl`, `laya_v2/*` prediction files and `spend_ledger.jsonl` from `/mnt/data/home/xavkal/gwaya-data`, with my own Wilson, bootstrap, logistic-CV and selector code.
No GCP/TPU command was run; the sealed table was not re-read for any new policy: the frozen `policy.pkl` was applied unchanged to the committed E rows (verification of the already-read result), and the freezer was rerun into a scratch directory.

Method notes. (1) A31: `freeze_policy_D.py` rerun on `dev_table.jsonl` into a scratch directory gives byte-identical `policy_thresholds.json`, identical logistic coefficients and isotonic map to the committed pickle (max abs difference 0.0); the dev table holds only sets F and Eprime (no E row, 0 task-id, 0 cluster, 0 prompt overlap with the 2,116 test rows), so the policy was fitted on dev only. Scoring `test_table.jsonl` (Coder-1.5B dropped) with my own feature code and Wilson reproduces every figure in the paper's sealed table to the printed precision. (2) A26: grouped CV (md5 fold hash, C=0.3, inner OOF isotonic, Wilson-LB threshold) re-implemented; `lever_U_prompt_only.py` rerun from the cached embeddings into scratch reproduces the committed U JSON to within 0.006 in every number.

## A. Status, order of events and labels

| # | Claim | Verified |
|---|---|---|
| A1 | every phase-2 item has a rule committed before its number; no registered hypothesis tested | V. Every addendum's rule paragraph is in a commit earlier than its outcome commit (A28 bc0375f 19:50:02 < f843d32 21:27:55; A29 e590b17 21:28:30 < 2a2a6da 07:28:17 next day; A30 3e5ea6c 08:36:08 < 72a6c43 21:01:40; A31 d3d2c5b 21:08:03 and 4fb2106 21:08:12 < 7a9c75d 21:08:40; A32 02dd07a 21:09:27 < c71d182 21:11:13). A26 outcome subsection is retrospective and says so. |
| A2 | A23 plan only | VN. Builder and test are committed, no pool-O result file, no ledger item; absence verified in `results/` and the ledger. |
| A3 | A24, A27 plan only | VN. No result file and no ledger line; absence only. |
| A4 | A25 KV arms no result; refusal readout descriptive | V. No kv result file; readout file exists. |
| A5 | A25 refusal 9B at share 0.4: Laya 0.748, gate-then-lp 0.798, log-prob 0.656; 212 of 407 wrong for all tiers | V. 0.7485 / 0.7975 / 0.6564 from the JSON; 212 recomputed from `dev_table.jsonl` (F rows, 3 tiers all wrong). |
| A6 | A26 amendment before any result | VN. Commit order d443d6c 09:31:43, a77f4ee 09:50:23, 403adc9 09:50:38 verified; "before any lever result was read" is the plan's statement. |
| A7 | 9B correction before any loop result was read | VN. Same status. |
| A8 | A29 amendment 0.85 to 0.84 before scoring | VN. Commit 3a5ac77 07:12:53 < outcome 2a2a6da 07:28:17 verified; "55% scored" is plan text. |
| A9 | A30 post hoc | V. 3e5ea6c 08:36:08 after 2a2a6da 07:28:17. |
| A10 | commit times (A28 19:50:02/21:27:55; A29 21:28:30; A31 21:08:03/21:08:12/21:08:40; A32 21:09:27/21:11:13; 28 s; 106 s) | V. All from `git log --format='%h %ci %s'`; 21:08:40 - 21:08:12 = 28 s; 21:11:13 - 21:09:27 = 106 s. |
| A11 | VM suffixes 200303, 221616, 084710 | V. Ledger item names. |

## B. A26 dev loop, U

| # | Claim | Verified |
|---|---|---|
| B1 | 1,728 rows, 576 per tier, F 407 + E' MBPP 169, 5 folds, 0 overlap with 2,116 test rows | V. 1,221 F rows + 507 Eprime rows; 0 overlap by task id, cluster and prompt; 4 x 529 = 2,116 test rows. |
| B2 | base rates 0.248 / 0.389 / 0.457 | V. 0.2483 / 0.3889 / 0.4566 from `correct`. |
| B3 | Table p2-U cells | V. Rerun of `lever_U_prompt_only.py` from the cached embeddings (scratch output) differs from the committed JSON by at most 0.006 in every number; table generator reads the JSON keys named. Not re-derived independently of the author's feature code. |
| B4 | best bal. acc. 0.644 / 0.687 / 0.654, ensemble AUROC 0.70 / 0.75 / 0.70 | V. From the rerun. |
| B5 | U not met | V. |
| B6 | router p AUROC 0.671 / 0.690 / 0.638; Laya p 0.942 / 0.926 / 0.918 | V (JSON; unchanged in rerun). |

## C. D, consensus, best-of-8

| # | Claim | Verified |
|---|---|---|
| C1 | pooled rank 0.5: D3 0.726 [0.695,0.754]; hand 0.729 [0.699,0.758]; log-prob only 0.481 [0.448,0.515] | V. Own CV: 0.7257 [0.695,0.754]; hand 0.729 (864 rows); log-prob 0.481. |
| C2 | per tier hand 0.624 / 0.757 / 0.799 vs D3 0.497 / 0.771 / 0.819; abstract "ahead on the 2B" | **X (S1).** Values V, but the 2B hand rule has only 229 gate-verified rows (coverage 0.40) and cannot reach 0.5; D3 on its top 229 rows is 0.624 (0.619 at its 0.4 point): level. 4B and 9B hand rules are at 0.5. |
| C3 | pooled at 0.3: D3 0.861 [0.829,0.888]; hand 0.788 [0.751,0.821] | V. |
| C4 | D3 at CV threshold 0.85: pooled 0.24 / 0.883; 4B 0.27 / 0.891; 9B 0.28 / 0.909; 2B 0.15 / 0.820 | V. Own CV: 0.2367 / 0.8826 [0.848,0.910]; 0.2708 / 0.8910 [0.832,0.931]; 0.2847 / 0.9085 [0.855,0.944]; 0.1545 / 0.8202 [0.728,0.886]. |
| C5 | 9B at 0.5 0.819, LB 0.771; 4B 0.771; 0.90 not met | V. |
| C6 | AUROC pooled lr core 0.947, +laya 0.952, full 0.950 | VN. Read from the JSON; AUROCs not recomputed (the 0.947 core model reproduces by my CV score ranking, not printed). |
| C7 | plan "D3 0.83" is not lr core (0.819); 0.828 is baseline_dev top half of 338 verified rows (169, coverage 0.29) | V, with an addition (S6): 169 of 576 = 0.293, k = 140, 0.8284 recomputed; the file's `coverage: 0.5` field is wrong. The consensus lever's base-only model gives exactly 0.830 for the 9B (and 0.767 / 0.496 for 4B / 2B, the plan's 0.77 / 0.50), so that is a second and likelier source of the plan's 0.83. |
| C8 | consensus 0.767 to 0.774 / 0.778, 0.830 to 0.826, 0.497 to 0.497 | VN. Read from `lever_D_consensus_agreement.json`; script not rerun (execution cache). |
| C9 | A29 4B G 189 (0.46) 0.709 [0.641,0.769]; S1 307 (0.75) 0.619; S1@0.5 0.701 [0.635,0.760]; 2B 0.475 | V. From raw sample rows + dev greedy: 189 / 0.709; 307 / 0.619 [0.563,0.671]; 0.701; 2B 0.475 [0.408,0.544]. |
| C10 | A30 9B greedy 0.396, some correct 0.617, G 221 (0.54) 0.729 [0.666,0.783], S1 325 (0.80) 0.646, S1@0.5 0.750 [0.686,0.804], no prefix | V. All recomputed. Table row "mean sampled pass rate 0.144 / 0.313 / 0.373" X (S5): it includes the greedy candidate; eight samples alone 0.139 / 0.311 / 0.370. |
| C11 | 9B verified-but-wrong 494 of 1,698 (29.1%) | V. |
| C12 | 4B 490 of 1,503 (32.6%), 2B 503 of 954 (52.7%); identical-text 4B 0.745 [0.681,0.800], 56 tasks, no A30 prefix; 2B 0.505, 17; some correct 0.339 / 0.541; greedy 0.189 / 0.329 | V. All from raw `scored_samples.jsonl` + dev rows with my own code. |
| C13 | 9B 690 of 3,256 (21%) truncated, 26 pass gate, 1 hidden, greedy 85 of 407 | V (was plan text only). 690 / 26 / 1 / 85 recomputed from `finish` and `finish_reason`. For reference: 2B 585 of 3,256 (18%), greedy 121 of 407; 4B 341 (10.5%), greedy 56. |
| C14 | 9B identical-text 0.765 [0.702,0.818]; A30 prefix 0.30 / 0.871 [0.801,0.919] n 124; 55 tasks | V. 0.765 [0.702,0.818]; n 124, 0.871 [0.801,0.919]; 55. |
| C15 | 0.740 / 0.898 superseded | V. Rerun gives 0.745 and no 4B prefix. |
| C16 | oracle 0.617 vs greedy 0.396 | V. |
| C17 | 6,512 and 3,256 samples, 8 per task | V. 407 tasks x 8 per tier. T=0.8 / top-p from plan and launcher (VN). |
| C-grid | "D3 rank-curve reference 0.30 / 0.40; pooled 0.3" | VN (S4): values V on the 0.1-step curve as the A30 rule states; every-prefix largest meeting prefixes are 0.33 (4B), 0.47 (9B), 0.34 pooled. S1 has none either way. |
| C-F | "on the fresh set F the goal is met only at coverage 0.24 to 0.40 for the 4B and 9B" | **X (S2).** Those are the F + E' MBPP development table. On F Python rows alone: pooled cov 0.14 / 0.865 [0.806,0.909]; 9B 0.17 / 0.915 [0.828,0.961] (met); 4B 0.18 / 0.861 [0.763,0.923] (not met); 2B 0.07 / 0.750 (not met). E' MBPP alone 0.47 / 0.895. |

## D. A28 Laya v2

| # | Claim | Verified |
|---|---|---|
| D1 | fold 0: 200 tasks, 600 rows, 46 confident-wrong | V. `calibrator_F_heldout.jsonl` 600 rows, 200 tasks, 46 gate-verified and wrong. |
| D2 | AUROC Python 0.915 to 0.949, +0.034 [+0.001,+0.069]; math 0.974 to 0.951, -0.023 [-0.048,-0.000]; pooled -0.007 [-0.028,+0.014] | VN. Points from the prediction files: 0.9154 / 0.9492 / +0.0337; 0.9736 / 0.9506 / -0.0230; 0.9546 / 0.9478 / -0.0068. My task-level bootstrap (different seed): Python [0.000,+0.070], math [-0.048,-0.000], pooled [-0.029,+0.014]; the Python lower bound is at zero to three decimals, so "barely above zero" is fragile. The rule outcome (math falls by more than 0.01) does not depend on it. |
| D3 | ECE 0.042 to 0.147, Brier 0.084 to 0.136 | V. 0.0421 / 0.1467, 0.0836 / 0.1360. |
| D4 | confident-wrong catch v1 0.00, v2 0.196 [0.00,0.44] | V. 46 rows, 0 and 9. |
| D5 | rule not met | V. |
| D6 | 1,968 steps, 1,522 s, router 581 s; 154 hard negatives in train | V. status.jsonl 1522 / 581 s; `feedback_report.json` 154 train, 46 held out; manifest 1,968 steps (train wall 1,458 s inside the 1,522 s step). The plan's "183 of 2,724" is the Python count only (S6). |
| D7 | T2 probes timed out rc 124 at 900 s | V. status.jsonl rc 124 at 902 s and 901 s. |

## E. A32

| # | Claim | Verified |
|---|---|---|
| E1 | 9B B0 cov 0.30 / 0.846 [0.771,0.899]; B1 0.32 / 0.884 [0.817,0.928]; gain +0.015; paired +5.7 [-0.02,+12]; (a) true (b) false (c) false | V. Own nested-CV logistic selector: B0 0.302 / 0.846, B1 0.317 / 0.884 [0.817,0.928], gain 0.015; paired difference at cov 0.30 point +5 correct (bootstrap mean 5.7), CI [0.0, 12.0] (my seed): touches zero. Sign and (a),(b),(c) agree. |
| E2 | @0.5 B1 vs B0 0.804/0.740, 0.755/0.652, 0.500/0.377 | V. |
| E3 | 4B B1 0.27 / 0.835 [0.754,0.893]; gain +0.086; (a) not met | V (4B point -1 [-8,6], 2B +3 [-7,13]). GBM arm not rerun (VN). |
| E4 | B0 is F Python alone, not D3 | V. |

## F. A31

| # | Claim | Verified |
|---|---|---|
| F1 | 529 tasks per tier, HumanEval+ 156, MBPP+ 373, 1,587 rows | V. |
| F2 | frozen 0.85 threshold 0.798, dev OOF coverage 0.243, columns, 1,728 rows | V. Freezer rerun into scratch: identical thresholds (0.6747 / 0.7984 / 0.9219), identical model and isotonic. |
| F3 | 9B pooled 0.73 / 0.886 [0.850,0.914]; rank 0.5 0.906 [0.864,0.935]; HumanEval+ 0.71 / 0.927 [0.863,0.963]; MBPP+ 0.74 / 0.869 [0.824,0.904] | V. Own scoring: 385 of 529 accepted (0.728), 0.8857; 0.906; 110 accepted 0.927; 275 accepted 0.869. |
| F4 | 4B 0.70 / 0.852 [0.812,0.884]; HumanEval+ 0.841 [0.760,0.898]; 2B 0.42 / 0.871 [0.820,0.908]; HumanEval+ 0.825 [0.714,0.900] | V. |
| F5 | base rates 0.726 / 0.673 / 0.450 | V. |
| F6 | gate only 9B 0.87 / 0.833 [0.796,0.864]; 4B 0.828 at 0.81; 2B 0.760 at 0.59; +0.053 for 0.14 | V. |
| F7 | both rules met on 9B pooled | V. |
| F8 | A30 rank prefix largest on E, 9B 0.86 / 0.842 [0.805,0.872] | V. |
| F9 | plan reading false at 0.85 (4B 0.852 v 0.891; 9B 0.886 v 0.909; pooled 0.869 v 0.883) and at 0.90 for 4B (0.911 v 0.934) | V for the numbers (dev recomputed: 0.891, 0.9085, 0.8826, 0.934; E recomputed). **Incomplete (S3):** at 0.80 the 9B is equal (0.8636 vs 0.8637) so "higher at 0.80" was too strong; at 0.90 pooled E 0.9129 is below dev 0.9135 (level). Paper and plan corrected. |
| F10 | dev coverage 0.28 vs 0.73; base 0.457 vs 0.726 | V. |
| F11 | "on E the gate payload includes the extended HumanEval+/MBPP+ tests; on F one visible assertion", hence "stronger gate" | **X (S7).** `n_visible_tests` = 1 on all 2,116 E rows; the 3.9.0 section already states every VERIFIED on stored E rests on one visible test; the extended tests are the hidden check. Gate verifies more (0.87 vs 0.54) at higher precision (0.833 vs 0.729) but adds less over the base rate (+0.107 vs +0.333). Statement removed everywhere. |
| F12 | read once, refuses to overwrite | VN. Code read; "once" not verifiable from files; commit gap 28 s consistent. |

## G, H, I

| # | Claim | Verified |
|---|---|---|
| G1 | smoke 8 of 8 VERIFIED, cold start 427.1 s, 1.8 to 4.2 s | VN. 7 rows 1.78 to 4.17 s, first row 36.09 s (warm-up); wording "warm 1.8 to 4.2 s" holds for 7 of 8. |
| G2, G3 | A23 name clash; D77 last | V. |
| H1 | 35 lines, 26.3314, 52.7% (52.66), first 31 lines 22.0055 | V. Recounted from the ledger: 35 lines; sum 26.3314; first 31 22.0055; new 4.3259. |
| H2 | 4 new lines: 520 s 0.1733; 4,386 s 1.4620; 3,304 s 1.1013; 1,192 s v5litepod-4 1.5893 | V. Each recomputed at 1.20 USD per chip-hour (1,192 x 4 chips). |
| H3 to H5 | inferred purposes; 1.59 A30; 3.10.0 macros unchanged | V / VN. Prose sums: 4.9136 + 11.0472 + 1.2813 + 0.5160 + 1.5220 + 2.7254 + 4.3259 = 26.3314. |
| I1 | 97 pages; 13 overfull | V for the author's build. After this audit: 98 pages, 13 overfull, no undefined references; the "Float too large" warning (hypothesis table) pre-exists in the 3.10.0 build. |

## Slips found and fixed

| # | Where | What was written | Correct | Fix |
|---|---|---|---|---|
| S1 | Abstract (3.12.0 paragraph); Section p2-loop; Table p2-D; plan A26 outcome (claim C2) | hand rule gate-then-log-prob "ahead on the 2B" (0.624 against 0.497 at "coverage 0.5"), "clearly lower on the 2B" for D3 | the hand rule can only accept the 229 gate-verified 2B answers (coverage 0.40); D3 on its top 229 rows is 0.624 and 0.619 at its 0.4 point: level. The gap was a coverage difference | abstract and section rewritten; table cell labelled "at coverage 0.40" and caption explains (generator regenerated); plan audit note |
| S2 | Abstract; Section p2-sealed; caveats; follow-ups; README; plan A31 | "on the fresh set F the goal is met only at coverage 0.24 to 0.40 (4B, 9B)" | those figures are the development table, F Python + E' MBPP (507 easier rows). On F Python alone: 9B met at coverage 0.17 (0.915 [0.828,0.961]); 4B not met (0.861, LB 0.763); pooled 0.14 / 0.865 | wording changed to "development table"; F-alone figures added; Table p2-D gets two subset rows (F only, E' only); audit script and JSON committed |
| S3 | Section p2-sealed ("higher at 0.80 and for the 2B"); plan A31 reading (known slip) | precision on E at least as high as dev at each threshold (plan); "higher at 0.80 and for the 2B" (paper) | at 0.85: 4B 0.852 v 0.891, 9B 0.886 v 0.909, pooled 0.869 v 0.883 (2B higher); at 0.90: 4B 0.911 v 0.934, pooled 0.913 v 0.914, 2B and 9B higher; at 0.80: 2B, 4B, pooled higher, 9B equal (0.864 v 0.864) | plan sentence corrected in place with a dated note and the original quoted; paper sentence and Table p2-devE caption now state every cell |
| S4 | Table p2-D, p2-bok, text ("D3 reference 0.40") | "rank-curve largest" reference without saying it is the 0.1-step grid | the A30 rule is on the 0.1-step curve; on every rank prefix the largest meeting prefixes are 0.33 (4B), 0.47 (9B), 0.34 pooled. No verdict changes (S1 has none) | grid stated in captions and text with the every-prefix values |
| S5 | Table p2-bok row "mean sampled pass rate"; plan A30 outcome | 0.144 / 0.313 / 0.373 called the mean of the samples | mean over nine candidates incl. greedy; eight samples alone 0.139 / 0.311 / 0.370 | row relabelled in the generator; plan note |
| S6 | Table p2-departures, plan A26 and A28 text | listed contradictions incomplete: D3 "0.83" source unstated; A28 "183 of 2,724"; A29 arm S2; A32 "tier indicators"; 2B hand-rule coverage | consensus base-only 0.830 is a second source of 0.83; 183 is Python only (200 all domains: 154 train, 46 held out); S2 arm never ran; script has no tier indicators; 2B coverage 0.40; 2B rank-0.5 point 0.842 meets 0.84 with LB 0.793 | rows added to Table p2-departures; plan notes |
| S7 | Section p2-sealed ("stronger gate", "gate payload includes the extended tests"), Table p2-devE caption, Section limitations, README, plan A31 | E's gate is stronger because the extended tests are its payload | `n_visible_tests` is 1 for all 2,116 E rows, the 3.9.0 paper says the same; extended tests are the hidden check. The gate verifies more answers on E at higher precision but adds less over the base rate | statement removed or reworded in all five places; departure row added |
| S8 | Section p2-bok, caveat on truncation | "(plan text)" for 690 / 26 / 1 / 85 | recomputed from rows: same values | sentence says recomputed by the audit; 2B and 4B truncation rates given |

## Not changed but noted

- The A28 Python AUROC lower bound (+0.001 in the committed file, 0.000 with another bootstrap seed) is at zero: "passes the first condition barely" should be read as "not distinguishable from the boundary". The rule outcome is carried by math.
- A32 paired difference "+5.7" is the bootstrap mean; the point estimate is +5 correct accepted tasks.
- D3 rank-0.5 comparison of hand rule and D3 has no paired test in any committed file; the "level pooled" statement rests on overlapping Wilson intervals.
- The consensus lever (C8) and the GBM arms (E3) were read from the JSON, not rerun; the U rerun used the author's feature code and cached embeddings.
- The sealed table "read once" and "before any result was read" for the A26 and A29 amendments rest on commit order and the plan text; they cannot be verified from files.
- The E gate-only baseline and the A30 prefix on E use labels to pick coverage and are reported as such.
- The claim that 3.11.x were tooling-only releases could not be checked (no tags after v3.9.0 in the repository; the commits between 3.10.0 and this branch are tooling).
- Two A31 secondary cells (4B and 2B HumanEval+ at the frozen threshold) do not meet rule (i) and are printed as such; the paper's wording "met, barely" for the 4B pooled is accurate (0.852 [0.812,0.884]).
- Build: 98 pages, 13 overfull boxes (as in 3.10.0 and the author's 3.12.0 build), one pre-existing "float too large" warning on the hypothesis table.
