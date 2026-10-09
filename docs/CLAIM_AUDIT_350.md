# Independent claim audit of papers/gwenlaya_v4.tex, revision 3.5.0 (uncommitted)

Auditor method: all numbers recomputed from raw rows (results/gwenlaya_v4/e_tpu_mathgate/rows_*.jsonl, data-disk etpu/{tiers,overlay2/tiers,nine,mathgate}, gens.jsonl, spend_ledger.jsonl) with separate code, point estimates only. Result JSONs were used only for the spot checks of CIs.

## Recomputed and CONFIRMED (no discrepancy)
- Pass rates: 2B 0.423, 4B 0.606, 9B 0.653 pooled; 9B 0.726/0.588/0.642, 4B 0.675/0.529/0.612 (Python/Rust/math).
- 4B math gate: coverage 0.454, precision 0.943, CWR 0.026, answer-all 0.388. 2B math: 0.288, 0.951, 0.014. 4B pooled 0.624/0.875/0.078. 9B pooled 0.666/0.885/0.077.
- Cascades (accuracy diff vs reference, cost ratio with warm-up, without warm-up (first 192 tasks of tasks_E_primary.d25 dropped), escalation, discordant counts): all 16 domain-by-cascade cells match Table a12-casc and text to 3 decimals, e.g. 4B->9B pooled +0.037, 1.381, 1.724, esc 0.376, 86/29; 2B->9B pooled -0.008, 1.344, 1.526, 40/53; ladder pooled +0.018, 1.565, 1.905, 91/64; 2B->4B pooled -0.001, 2.021, 2.023, 45/46; cascade vs 2B alone 293/13, +0.182.
- 2B->4B math: 1.784 vs 0.686 chip-s/task, escalation 0.712; pooled cost per correct 1.219 vs 0.603; selective view 0.669/0.113, math 0.496/0.030, Python 0.854/0.206, Rust 0.645/0.097; oracle 2-tier 0.644 and 3-tier 0.726 (per domain 0.784/0.708/0.682).
- 9B math truncation 0.306 (answers only; 153/500), pooled 0.0996; matched-baseline deltas (4B -0.076/-0.112/-0.084/-0.090; 9B -0.053/-0.126/-0.082); 9B raw ECE 0.207/0.344/0.304 and 4B 0.252/0.402/0.326.
- Ledger: 11 lines, sum 4.9136 (last three lines 2.3847; 9B lines 2.0547; 587 s and 954 s; 4 chips x $1.20/h reproduces 0.7827 and 1.272).
- A11 (abstention_a11.json), 12 numbers checked: AUROC 0.851/0.708/0.931, AURC 0.256/0.135, d_auroc 0.223 [0.197,0.249] and 0.079 [0.064,0.095], per-domain M3-M0 0.094/0.029/0.119 (CIs above 0), 2B AUROC 0.958/0.703, M3 ECE 0.045/0.025/0.033, M2 ECE python 0.056 [0.057,0.111] and rust 0.045 [0.047,0.101], operating points (M3@0.05 pooled 0.275 / 0.052; @0.10 0.585 / 0.101; M1 0.033 / 0.220; gate 0.624 / 0.125; M3@0.05 risks 0.043/0.037/0.066 with CIs covering 0.05). All match.
- A12 reading rule: applied correctly in every cell of Table a12-casc and in the text (accuracy condition = CI not wholly below 0; cost condition = point ratio < 1 in both readings). 4B->9B pays: Python, Rust; 2B->9B: Rust only; ladder: Rust by point estimates only (text hedges that CIs include 1, consistent with the rule being on points).
- No leftover current-tense "math gate coverage 0": every remaining occurrence (l.233, 275, 776, 836-839, 1103) is explicitly labelled 3.4.0 / superseded. No sentence claims H1/H3 support; H1/H3/H4-H6 rows say "not run", and Laya is mentioned only to say that nothing here involves it.
- sha256: all 24 values in the "Revision 3.5.0" block match sha256sum of the files (repo files and data-disk files); both figure PNGs equal their papers/figures_v4/*_a10.png copies.
- PDF builds (latexmk rc 0, 36 pages), no undefined references or citations.

## Discrepancies and defects
1. **PDF layout defect, reproducibility sha256 table (l.~1340-1380, page 29)**: the table is a single unbreakable tabular (Overfull \vbox 541 pt, log l.1039). In the PDF the table is cut off after the `rows_9b.jsonl` row. The last 18 rows (both figure PNGs, math_gate.py, 3 scripts, make_tables_350.py, 3 table parts, plan, DEVIATIONS, spend_ledger, nine/gens, mathgate/gens, 4 raw files) do not appear anywhere in the PDF (checked with pdftotext; e.g. 75fce2e3, 6fe1d5a3, ff882615, 3abe365a are absent). Source LaTeX is correct; the rendered paper loses those hashes. Fix: use longtable or split the table.
2. **Stale spend sentence, l.182 (correction A28, in the 3.3.1 corrections block)**: "Section spend now gives the full ledger as read for this revision: 8 lines, $2.5289 in total." Current ledger is 11 lines, $4.9136 (l.246-247 and l.1258 say so). Paper value 8 lines / 2.5289; recomputed 11 / 4.9136. The sentence is in an older correction list, so it reads as a present-tense claim that contradicts Section Spend.
3. Minor, no numeric error: l.~1050 rule paragraph does not say "point estimates" for the cost condition (the Table a12-casc caption and the Limitations item do). Plan A12 (e) promised "A11 repeated with the 9B"; the paper states it was not run (D46, l.1510), which is disclosed, but the plan text and the paper differ.
4. Minor rounding: l.~788 3.4.0 text says baseline CWR on Rust 0.180; recomputed 0.179 (0.1795 boundary; JSON delta -0.112 is consistent). Not a 3.5.0 claim.
5. Not verifiable by me: the ECE bootstrap statement, bootstrap CIs generally (not recomputed), and "bubblewrap" sandbox details.

No error found in any 3.5.0 headline number.
