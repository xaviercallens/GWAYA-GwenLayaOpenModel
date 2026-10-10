# Independent claim audit of papers/gwenlaya_v4.tex revision 3.9.0 (commit 2800549, branch feat/py-depth)

Scope: every claim that changed between 3.8.0 (b4025ee) and 2800549 (`git diff b4025ee 2800549 -- papers/ docs/`). Later commits on the branch (A15.3 merge 3b9e9df, D66-D68 feb03b6, greedy Lean generations) are out of scope; the paper text is unchanged since 2800549 (`git diff 2800549 -- papers/gwenlaya_v4.tex` is empty).

Own code only (scratchpad aud/: kgate.py, math_aud.py, vis.py, policy.py); make_tables_390.py, python_gate_depth.py, rescore_math_stored.py and verdict_policy_exploratory.py were not run or imported. Raw data: `GWAYA_DATA_ROOT=/mnt/data/home/xavkal/gwaya-data` (night/results/tasks_E_primary.d25.jsonl, etpu/{tiers,nine,mathgate}/{gens,rows}.jsonl, laya_data/, spend_ledger.jsonl) and results/gwenlaya_v4/. Paper not edited.

## Verdict

**No conclusion-changing error; one wording item to fix before release (W1), plus six optional notes.** Every number that the revision adds or changes was recomputed and agrees with the paper to the digit, except for the one-task differences noted in items 2 and 3 below, which come from running the hidden tests in a plain subprocess instead of the study's isolated harness. Count over the 34 numbered rows of the Verified table: **V 29, VN 4, W 1** (the VN rows are 2, 8, 13, 28; the W row is 10, expanded as W1). N1 to N6 elaborate those VN rows plus two optional edits.

## Must-fix

| # | Where | Paper says | Evidence | Fix |
|---|-------|-----------|----------|-----|
| W1 | D63, corr390 item "Python answers ... no tests", A15.2 "Reading" limit (l.~1393), table tab:dev39 row D63 | The 3.7.0 limit that Rust answers "were produced from prompts that showed one test" "was also inaccurate"; "no Rust prompt contains the gate assertion" | Literally true that the `assert_eq!` line is never in a Rust prompt (0/507). But the gate's first test is shown as a doc-comment example in most HumanEval-Rust prompts: for 89 of 155 prompts the line `/// >>> f(<gate args>)` is present, and in 87 of them the next line is the gate's expected value (the two others, HumanEval_128 and HumanEval_116, show a different result). MBPP-Rust (0/352) shows nothing. So the 3.7.0 sentence was true of about 56% of HumanEval-Rust (about 17% of E Rust), and D63 over-corrects it. | Replace "was also inaccurate" with: "was inaccurate for MBPP-Rust (none of 352 prompts shows a test) and for the gate assertion itself (0 of 507), but the gate's first call and expected value appear as a doc-comment example in about 87 of the 155 HumanEval-Rust prompts". Same edit in D63 (docs/DEVIATIONS.md) and tab:dev39. No number changes. |

## Notes (VN, optional edits)

| # | Where | Note |
|---|-------|------|
| N1 | A17 table, abstract (1) | My re-run (own ast splitter, own subprocess runner, 524 tasks x {1,2,4,8,full,stored} x 2 tiers, 6,288 executions) reproduces every stored-gate and coverage figure and all 9B figures but one cell to the digit. Differences are one task each: 4B accuracy 0.6737 vs 0.6756, 4B precision/CWR at every k 0.1412 vs 0.1393 at k=1 (0.0382 vs 0.0363 at k=8), 9B k=4 coverage 0.8168 vs 0.8149. 9B CWR is identical at all k. The paper's CIs agree with my 4,000-resample problem-level bootstrap to +-0.003 (4B k=1 [0.113, 0.172] vs [0.111, 0.170]; k=8 [0.023, 0.055] vs [0.021, 0.053]; 9B k=1 [0.118, 0.179] vs [0.116, 0.177]; k=8 [0.029, 0.065] vs [0.029, 0.065]). Every statement (monotone fall, non-overlap of k=1 and k=8) holds on mine. The one-task difference is a harness difference (my runner has no sandbox nonce or memory cap), not a paper error. |
| N2 | D63 "344 / 6 / 18" | With my simpler `assert f(args) == x` regex, 341 equal, 6 differ, 21 unparsed (368 total). The paper's 6 agrees; the 3 extra "parsed" cases use a form (e.g. a trailing comment) that the author's parser handles. Not material. |
| N3 | D63 "71 of the 156 HumanEval prompts carry doctest examples" | True (71 prompts contain `>>>`). It slightly understates exposure: in 88 of 156 prompts the gate's first input call appears in some form (43 of them as `>>>`, 45 as an inline example such as `f(x) == y`). The JSON note already says doctest examples are not necessarily the gate's test. Consider adding "the gate's first call appears in about 88". |
| N4 | tab:mathfix gate columns (149/230/236 verified, 5/3/2 comparison flips) | I re-scored all 1,500 labels with both scorers (old from `git show 78cf88b`, new from HEAD) and checked the VERIFIED-but-wrong counts against the stored gate verdicts: 7/13/10 -> 2/0/0, and the same two wrong items (prealgebra/1512, 1761). I did not re-execute the programs (the 3 and 5 new VERIFIED gate verdicts, and the 4B intermediate_algebra/1555 timeout, depend on the author's re-run), so the 149/230/236 and 5/3/2 are consistency-checked only (stored 144/227/234 + flips). |
| N5 | corr390/limits "94 tests passed" | The log has two pytest runs: 81 passed + 13 skipped, and the 13 real-project tests (13 passed, 81 deselected): 94 in total, none failed, but "94 passed" is the sum across two invocations. Fine as stated ("in my run ... all 94 tests passed, including the 13"). |
| N6 | DEVIATIONS D62 | Cites first version c4e560d, which is not an ancestor of 2800549 (its rebased copy on this branch is f837c9f, same subject). The paper cites f837c9f. Harmless; say "c4e560d (f837c9f on this branch)". |

## Verified

| # | Claim | Recomputed | Result |
|---|-------|-----------|--------|
| 1 | A17 task set: 524 of 529 usable (2 with no lists, 3 with fewer than 8 pairs) | own ast check on the checker payloads: 524, none 2, short 3 | V |
| 2 | A17 point estimates, gate coverage/precision/CWR at k=1,2,4,8 and stored, 4B and 9B; sanity violations none | see N1 | VN |
| 3 | A17 reading: CWR monotone 0.139, 0.115, 0.082, 0.036 and 0.147, 0.126, 0.090, 0.046; coverage 0.815 -> 0.712 and 0.872 -> 0.771; precision 0.829 -> 0.949 and 0.832 -> 0.941 | JSON internal consistency plus my re-run (N1) | V |
| 4 | A17 against Rust dial: 0.067 -> 0.024 (4B), 0.061 -> 0.014 (9B) at k=1 -> 3 | rust_gate_depth.json points 0.0673, 0.0238, 0.0614, 0.0139 | V |
| 5 | "stored one-assertion gate gives almost the same numbers at k=1" | stored 0.813/0.831/0.137 vs k=1 0.815/0.829/0.139 (4B); 0.870/0.831/0.147 vs 0.872/0.832/0.147 (9B) | V |
| 6 | MBPP prompts: all 373 print the gate's assert verbatim (368 in the A17 set) | assert line of every MBPP gate payload found in its prompt, 373/373 (10 gates also carry an `import math` line before the assert) | V |
| 7 | HumanEval: 71 of 156 prompts carry `>>>` examples | count 71 | V |
| 8 | A17 MBPP first-test equality 344/6/18 | see N2 | VN |
| 9 | Rust: 152 of 155 HumanEval-Rust prompts carry doc-comment examples; 0 of 352 MBPP-Rust; no Rust prompt contains the gate assertion (0/507) | own count: 152/155, 0/352, assert line 0/507 | V (but see W1) |
| 10 | "Prompts showed one test" in 3.7.0 was inaccurate | see W1 | W |
| 11 | Math accuracies 0.514/0.612/0.642 registered (reproduces all 1,500 stored base labels, 500/500 per tier) and 0.558/0.664/0.692 re-scored | own driver: old scorer from `git show 78cf88b`, new from HEAD, extractor of my own | V |
| 12 | Flips 22/26/25 wrong to right, 0 right to wrong; all notational | flip sets identical to the JSON's lists; I read all 28 distinct (gold, extracted) pairs: brace-less fractions, `\sqrt2`, units on gold only, `x =`/`x \in`, base subscript, tuple spacing, `\left(`, pmatrix, mixed number, `12^{\text{th}}` | V |
| 13 | Gate VERIFIED-but-wrong 7/13/10 -> 2/0/0 of 500 and rule-of-three 0.006 (3/500) and 0.013 (3/230) | see N4; arithmetic checked | VN |
| 14 | cwr_all 0.486/0.388/0.358 -> 0.442/0.336/0.308 | (N - correct)/N under both scorers | V |
| 15 | Answer-if-boxed 0.088 -> 0.044, 0.068 -> 0.016, 0.052 -> 0.002 (counts 44/22, 34/8, 26/1); coverage 0.602/0.680/0.694 | own boxed extractor, wrong = not VERIFIED, over 500; boxed 301/340/347 | V |
| 16 | "160 of the 4B's 194 wrong rows" have no boxed answer, "mostly truncated" | 194 wrong, 160 without `\boxed`, and all 160 have done_reason = length (so "all", not "mostly", is also true) | V |
| 17 | Headline 0.026 against 0.068, gate coverage 0.454, answer-if-boxed coverage 0.680, difference 0.042, coverage cost 0.226; re-score 0.016 against 0 of 500 (230 verified) | 13/500, 227/500, 34/500, 340/500; 230 see N4 | V |
| 18 | Re-score exploratory/post hoc, registered primary, not propagated to A4/A5/A11/A12/Laya labels | read every place it appears (abstract, corr390, A10 paragraph, A11 item, A12 caveat, limitations, H7 row); no sentence presents it as confirmatory | V |
| 19 | Verdict-policy table (9 rows: tasks, verified, 1 test, wrong among verified, kept at k_min=2) | own counter (comments/strings stripped; input/result pair count for Python, `#[test]` or assert macros for Rust) on stored rows: all 9 rows identical, including R' 4B 39 of 40 / 1 kept and 9B 43 of 45 / 2 kept | V |
| 20 | 4B Python wrong-among-verified 0.172 here vs 0.170 from precision 0.830 | 74/430 = 0.172; 1 - 0.830 = 0.170 | V |
| 21 | Weak-gate policy opt-in and off by default, no k_min chosen; no published result uses it | `parse_min_visible_tests(None)` with the env var unset returns `{}`; policy text and D64 agree | V |
| 22 | Lean never LIKELY_CORRECT; Lean task without formal statement never VERIFIED | gwaya/gwenlaya.py lines 13, 29-31, 456-474 | V |
| 23 | Spend: 27 lines, total 19.2801, 25-line total 17.7581, 24-line total 17.2421, 2 new lines 1.5220 (1.0440 + 0.4780), last ts 2026-10-10 07:59:02, 38.6% of 50 | own sum over the ledger | V |
| 24 | Stale 3.8.0 spend ($17.24, 24 lines in abstract, H13 row and H13 paragraph) | present in b4025ee l.184, 1479, 1491; ledger then had 25 lines/17.7581 | V |
| 25 | Ledger sha 9f97c3bb... as "27 lines, as read for 3.9.0" | `head -27 spend_ledger.jsonl` and whole file both hash to it (file still 27 lines) | V |
| 26 | Every sha256 in papers/sha390.tex (32 entries) | `sha256sum` of working tree and of `git show 2800549:<path>` for the 30 tracked files, and the data-root task file and ledger: 32/32 match; all 32 also present in the built PDF text | V |
| 27 | Lean sanity: 223 of 244 pass, 17 commented out, 4 do not elaborate | lean_sanity.json (excluded list of 21 by class 17 + 4), tasks_lean.jsonl 223 distinct ids, disjoint from the excluded | V |
| 28 | Lean gate tests ("94 passed", 13 real-project tests, genuine proofs 12 to 21 s at load ~3 on 8 cores) | see N5; slowest genuine proofs 12.4 to 20.7 s; load 3.2 in the log | VN |
| 29 | Cited commit hashes and times (6d229b1 21:52:55, 499b9eb 21:54:18, f19e283 22:53:02, da7d708 22:53:02, dfd41aa 23:57:40, aa572bb, d97e390, 5b53ce8, cc3969e, ff97b0b, 16a7abd, f318f71 00:33:22, 24bc5c0 01:01:28, 302ab95, 2305d90, 5e14bd8, becb57f 01:18:58) | `git log`; all are ancestors of 2800549; each A17/A18 plan commit precedes the commits that use it; f837c9f/c4e560d, see N6 | V |
| 30 | A18 "generation in progress, no Lean result" is the only Lean status claimed | read all 21 non-comment mentions of Lean/A18/miniF2F in the .tex (l.140, 206, 314-315, 480, 539, 565, 1443-1457, 1527-1528, 1572, 1578, 1591, 1667, 1755, 2072-2075, 2090) and lean_sanity_summary_390.json (`generation_run: false`); no proof rate, no VERIFIED count; the 3.9.0 test log is a gate unit-test log, not a result | V |
| 31 | D62 to D65 in tab:dev39 match docs/DEVIATIONS.md | read side by side: D62 (post hoc, registered primary, not propagated), D63 (visibility counts, A17 numbers unchanged), D64 (opt-in, off by default, k_min not chosen, no published result uses it), D65 (second process, exact or defeq recorded, binder annotations not compared, red team not completed); no number differs | V |
| 32 | Lean gate limits stated: binder annotations not compared, definitional match counts as VERIFIED (A18 reports exact/defeq separately), red team not completed, lexical rules over-reject, slow | present in sec:lean390 (v) items i-v, D65, limitations, next steps; code confirms `match` in {exact, defeq} and no binder comparison (gwaya/lean_gate.py l.230, 274) | V |
| 33 | H7 and H11 rows: H7 "under the exploratory D62 re-score no 4B math-verified answer is wrong; A17 4B Python precision 0.949 with 8 tests"; H11 "dropped; descriptive A18 pre-registered, no result" | consistent with items 3, 13, 30; labelled exploratory | V |
| 34 | PDF builds | `latexmk -pdf -interaction=nonstopmode -outdir=<scratch>` from the working tree: rc 0, 66 pages, no undefined references or citations, no "??", text identical to the committed PDF; warnings only hyperref unicode-in-bookmark (6) and one float 24 pt too large (l.1574). Build artifacts in papers/ remain untracked and were not touched | V |

## Wording checks

- Exploratory presented as confirmatory: none found. The D62 re-score, the policy counts and A17 are each marked exploratory or "dial, not a hypothesis test" wherever they appear (abstract, section headers, captions, H7).
- Weak-gate policy: described as opt-in, off by default, with no k_min chosen, in the abstract, sec:policy390, D64 and limitations; code confirms.
- One sentence to watch (optional): abstract (3) "under it the 4B and 9B math gates have no confident-wrong answer on E" is followed by "rule-of-three 0.006" and "registered numbers remain primary", which is adequate; the A10 paragraph adds that E is easy, so zero is not evidence of no confident-wrong in general.
- The A17 order-of-events paragraph and the D63 erratum are honest about the false "no tests" sentence; the A17 plan text is left unchanged with an erratum line.
