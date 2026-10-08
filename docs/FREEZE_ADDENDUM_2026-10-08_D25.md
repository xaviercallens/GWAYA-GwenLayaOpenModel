# Freeze addendum, 2026-10-08: gate payload correction (D25) and math extraction rule (D24)

Written 2026-10-08 after the L2 generations and after the external review of the interim paper,
before any further generation. It corrects, without editing, section 2 of
`docs/FREEZE_ADDENDUM_NIGHT_2026-10-07.md` (whose sha256 is registered and therefore left as is).
Deviations: `docs/DEVIATIONS.md` D24, D25, D26.

## 1. What section 2 of the night addendum said, and what was true

Section 2 says the HumanEval+ gate is "the first input/expected pair of the combined evalplus input
list (`zip(inputs, results)` sliced to `[:1]`)". That described what the gate **executes**, not what
it **contains**. In the files frozen that night (`tasks_E_night.jsonl`
`e842edd1...d679`, `tasks_E_primary.jsonl` `971f44c6...7097`) every HumanEval+ `gate_payload`
contained the complete hidden `inputs` and `results` literals (11/11 E-night rows, 161/161 primary
rows), and four of them also contained a `ref_func` reference implementation. The sentence "the gate
never sees the scoring checks" (prereg sections 0-1) was therefore **false for HumanEval+ in the
night files**. It was true for MBPP+ (`test_list[0]` only), Rust (first `assert_eq!` only) and Math
(empty gate payload).

Consequence for reported numbers: none. The L2 run executed no gate (`gate` is null in all 80 scored
rows), so no recorded result used a gate payload.

## 2. Corrected files (prompts and hidden checks unchanged)

Built by `scripts/data/regen_gate_payloads_d25.py` from the night files with the ast-based
`scripts/data/build_eval_manifest.py::python_gate_tests_plus`, which rewrites each of the two
literals to a one-element list holding its original first element (source text kept verbatim) and
fails closed. Only `gate_payload.tests` of
`py/HumanEval/*` rows changes; every other field of every row is copied unchanged. Report:
`$GWAYA_DATA_ROOT/night/results/gate_regen_d25_report.json`.

| File | Rows | sha256 |
|---|---|---|
| `tasks_E_night.d25.jsonl` | 80 (11 gates rewritten, 0 excluded) | `504909fc74dc2af8fb3f2491a475df07de8e95e6fdf0f4f69bc22352e95e0952` |
| `tasks_E_primary.d25.jsonl` | 1,536 (156 rewritten, 5 excluded fail-closed) | `2143110c78a3a9f57599dce77021cf4571fc57cf2584017f16ebdd08d5acba66` |
| `gate_regen_d25_report.json` | - | `e4cfd6f1fd786dc1c38e73555c87f23448178001d531da864ca6d4cae298546a` |

Excluded from primary E (none is in E-night): HumanEval/14, /83, /100, /130 (`ref_func` in the test
module: the gate cannot be made free of the hidden answers by literal rewriting) and HumanEval/149
(the leak check flags a hidden input literal that also occurs inside the first expected output; this
is probably a false positive of the substring rule, and the item is excluded anyway, fail-closed).
Primary E for future runs is therefore 1,536 items, not 1,541. `manifest_eval.json` is not
regenerated; its per-item `gate_tests_sha256` fields describe the v1 gate text.

Check performed: on the 11 E-night HumanEval+ L2 candidates, the v1 gate and the corrected gate
gave the same verdict for every item (both execute only the first pair).

From now on runs use the `.d25` task files. The night files stay as the record of what L2 used.

## 3. Math answer extraction, frozen before the math regeneration (D24)

The registered rule applies: the answer is the content of the last `\boxed{}`; output without a
`\boxed{}` is UNVERIFIED (fail-closed). The `####` and last-number fallbacks of
`gwaya/domains/math_check.py::extract_final_answer` are not registered and may only be reported as
a labelled sensitivity analysis.
