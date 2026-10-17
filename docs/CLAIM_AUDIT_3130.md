# Independent audit of the phase-3 section (3.13.0 draft)

Recomputed from raw files with separate code (no verifier_v2.py / tables_phase3.py).

## Confirmed
- F greedy / any-correct: 2B .189/.339 (77/138 of 407), 4B .329/.541 (134/220), 9B .396/.617 (161/251), 27B .550/.715 (224/291). 9 candidates per task (2B/4B/9B), 5 for the 27B, every task.
- G: 218 tasks (BigCodeBench 184, FullStackBench 34), 27B greedy 139/218 = 0.638 [0.572, 0.699], oracle 161/218 = 0.739 (last-wins copy, see slip 1), gate-only 171 accepted / 139 correct = 0.813 [0.748, 0.864], coverage 0.784.
- Ladder p-values (exact binomial, risk 0.10) match: V2 39/3 p=.444, 45/3 .329, 51/3 .236; B0 4/0 .656, 54/6 .707, 55/6 .690. Certifying with 0 errors needs n>=22, so 4 accepted cannot pass; V2 accepts 0 at 0.95. Certificate cannot pass for reported counts.
- Rank-curve counts implied by the precisions (61/66, 77/88, 95/109; 59/66, 78/88, 91/109) give the quoted Wilson intervals exactly. The scores themselves were not refitted (no dumped G scores), so the ranking cannot be independently reproduced.
- Certificate arithmetic: max errors 1, 4, 9, 18, 31, 50 at n = 50, 88, 150, 250, 400, 600; certify probabilities at r=.05 (.55,.78,.95,.99) and r=.08 (.16,.23,.37,.47) correct; r=.10 is .05-.08 ("about 0.06" is fine), r=.12 .004-.015.
- Ledger: 40 lines, 41.7526 total, 35 before (26.3314), 5 after totalling 15.4212; table rows match; T5 VMs 200529 (us-west4-a) and 200546 (us-west1-c) are both in the ledger, 9204 s and 9193 s. 8.25 and 5.25 USD headroom correct.
- Commits and times all match git log (798bb24 18:35, 6a59278 19:35, 4c7d02c 19:36, cd91a8b 20:05:07, c03dcd9 07:46:54, 7ccf98c 09:56:43, a4202ab 10:22:51, ce47e16 10:30:11, 805bfc5 07:47). Raw 27B counts 625/2500/1250 match.

## Slips
1. **"G was read once" and the sealed-run mechanics are contradicted by the file.** scored27_G.jsonl has 2179 rows for 1090 unique (task, cand): 1089 pairs are scored twice, and exec27_G.jsonl has 1184 rows. The hidden tests ran twice on nearly every candidate, not "one row then resume". Greedy labels agree across copies; one candidate label flips (py/BigCodeBench/720 cand 2, FAILED vs VERIFIED), which moves the oracle between 161 (0.739, reported) and 162 (0.743). Suggested text: "The first sealed run was started without detaching; the relaunch did not skip the rows already written, so most candidates were scored twice (2,179 rows for 1,090 candidates). Greedy labels are identical across copies; one sampled candidate's hidden label differs between copies, so the oracle is 0.739 (161/218) or 0.743 (162/218) depending on the copy. The reported 0.739 uses the later copy. No policy or threshold was changed after the first rows." Same change for the process note in NEXT_STEPS 4 ("died after one row").
2. NEXT_STEPS 4: "the 200546 line booked the full 150 minutes" - both T5 lines book about 153 minutes (9204 s and 9193 s); the 200529 line is equally full. Correct to "both T5 lines book about 153 minutes of VM time".
3. NEXT_STEPS 1 says Wilson [0.79, 0.94]; V2 is [0.790, 0.929], B0 [0.803, 0.937]. Write "[0.79, 0.93] (V2), [0.80, 0.94] (B0)".
4. Paper: "tables are written from the committed JSON files", but results/gwenlaya_v4/phase3/v2_dev_oof.json is untracked (git status). Commit it or drop the mention of the out-of-fold dump as committed.
5. "Verifier V2 is not a lever for any tier" holds by rule (a), but the 2B has a paired interval above zero ([8, 34]); the section does say this, so wording is acceptable, only keep "by rule (a)".

## Not verified
V2/B0 rank-curve scores (no G score dump); dev certified-coverage and bootstrap values; 23 spot / 10 on-demand attempt counts; "idle about 1.5 hours"; "identical job, same failure of the 35B" for the second VM; the "pipeline check" certified coverage; the v2 run cost of 1.46 USD; pool-O greedy 6,000 per tier; no claims about 35B or 2B/4B/9B on G were found beyond the stated absences.

## Author follow-up (2026-10-17)
Slip 1 confirmed and fixed in the paper and `NEXT_STEPS_PHASE4.md`: the first sealed run had not stopped, both runs wrote the file. Robustness check (first-copy vs last-copy dedupe, same sealed code): identical fixed-sequence ladders for V2 and B0, goal not met in both; oracle 0.743 (first) vs 0.739 (last, the reported figure) from one sampled candidate whose hidden check is nondeterministic. Slips 2 and 3 fixed as worded. Slip 4: `v2_dev_oof.json` committed. Slip 5: wording kept ("by rule (a)"). Items listed as not verified by the auditor remain unverified by an independent party.
