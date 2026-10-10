#!/usr/bin/env python
"""Write the generated tables and summaries of revision 3.9.0; no value is typed by hand.

Inputs (all read-only):
  results/gwenlaya_v4/edition/python_gate_depth.json          A17 Python depth dial (commit f19e283)
  results/gwenlaya_v4/e_tpu_mathgate/rescore_exploratory.json  D62 exploratory math re-score (f837c9f, 2305d90, 5e14bd8)
  results/gwenlaya_v4/lowtier/verdict_policy_exploratory.json  weak-gate verdict policy, exploratory (dfd41aa)
  results/gwenlaya_v4/lean_a18/lean_sanity.json                A18 pre-generation sanity (16a7abd)
  $GWAYA_DATA_ROOT/night/results/tasks_E_primary.d25.jsonl     E task file (prompts and gate payloads)
  $GWAYA_DATA_ROOT/spend_ledger.jsonl                          spend ledger (25 lines as of 3.8.0; later lines = A18 generation)
Outputs:
  papers/tables_pydepth390.tex (tab:pydepth), tables_mathfix390.tex (tab:mathfix), tables_policy390.tex (tab:policy390); papers/numbers390.tex (spend macros, read from the ledger); papers/sha390.tex (sha256 rows; the .tex files and this script are hashed as they are on disk)
  results/gwenlaya_v4/edition/prompt_test_visibility_390.json  what the E prompts show of the tests (D63)
  results/gwenlaya_v4/spend_summary_390.json
  results/gwenlaya_v4/lean_a18/lean_sanity_summary_390.json
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = Path(os.environ.get("GWAYA_DATA_ROOT", "/mnt/data/home/xavkal/gwaya-data"))
TASKS = DATA / "night/results/tasks_E_primary.d25.jsonl"
LEDGER = DATA / "spend_ledger.jsonl"
RES = ROOT / "results/gwenlaya_v4"
TN = {"qwen3.5-2b-bf16": "2B", "qwen3.5-4b-bf16": "4B", "qwen3.5-9b-bf16": "9B"}
K_DIAL = 8  # A17: K = 8, the dial uses tasks with at least 8 input/result pairs


def num(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def pci(v: dict, nd: int = 3) -> str:
    return f"{num(v['point'], nd)} [{num(v['lo'], nd)}, {num(v['hi'], nd)}]"


# ── prompt visibility of the tests (pure helpers, tested in tests/test_make_tables_390.py) ──────────────────────

def gate_asserts(gate_tests: str) -> list[str]:
    """The assert lines of a gate payload (stripped)."""
    return [l.strip() for l in (gate_tests or "").splitlines() if l.strip().startswith("assert")]


def gate_assert_in_prompt(prompt: str, gate_tests: str) -> bool:
    """True iff the gate payload has at least one assert line and every one of them occurs verbatim in the prompt."""
    a = gate_asserts(gate_tests)
    return bool(a) and all(x in prompt for x in a)


def prompt_assert_args(gate_tests: str):
    """Literal arguments of the call on the left of `assert f(args) == x` (None if not of that literal form)."""
    try:
        node = ast.parse(gate_tests.strip()).body[0]
        call = node.test.left
        return [ast.literal_eval(a) for a in call.args]
    except Exception:  # noqa: BLE001 - any other form is counted as "not parsed"
        return None


def first_pair_matches(gate_tests: str, first_input) -> bool | None:
    """Does the first hidden input equal the arguments of the gate assert? None when either side cannot be read."""
    args = prompt_assert_args(gate_tests)
    if args is None:
        return None
    try:
        inp = ast.literal_eval(first_input)
    except Exception:  # noqa: BLE001
        return None
    return list(args) == list(inp)


def visibility(tasks: list[dict]) -> dict:
    from scripts.python_gate_depth import find_pairs
    py = [t for t in tasks if t["domain"] == "python"]
    rs = [t for t in tasks if t["domain"] == "rust"]
    mb = [t for t in py if "/MBPP/" in t["task_id"]]
    he = [t for t in py if "/HumanEval/" in t["task_id"]]
    out = {
        "source": str(TASKS),
        "python": {
            "n": len(py), "mbpp_n": len(mb), "humaneval_n": len(he),
            "mbpp_prompt_contains_gate_assert_verbatim": sum(gate_assert_in_prompt(t["prompt"], t["gate_payload"].get("tests", "")) for t in mb),
            "humaneval_prompt_contains_gate_assert_verbatim": sum(gate_assert_in_prompt(t["prompt"], t["gate_payload"].get("tests", "")) for t in he),
            "humaneval_prompt_has_doctest_examples": sum(">>>" in t["prompt"] for t in he),
            "note": "doctest examples = the prompt contains '>>>'; they are examples from the docstring, not necessarily the gate's or a hidden test's input",
        },
        "rust": {
            "n": len(rs),
            "mbpp_n": sum("rs/mbpp" in t["task_id"] for t in rs),
            "humaneval_n": sum("rs/HumanEval" in t["task_id"] for t in rs),
            "prompt_contains_gate_assert_verbatim": sum(gate_assert_in_prompt(t["prompt"], t["gate_payload"].get("tests", "")) for t in rs),
            "mbpp_prompt_has_doctest_examples": sum(">>>" in t["prompt"] for t in rs if "rs/mbpp" in t["task_id"]),
            "humaneval_prompt_has_doctest_examples": sum(">>>" in t["prompt"] for t in rs if "rs/HumanEval" in t["task_id"]),
            "note": "Rust gate payloads are an assert_eq! inside fn main, never in the prompt verbatim; HumanEval-Rust prompts carry '>>>' doc-comment examples",
        },
    }
    dial = []
    for t in py:
        p = find_pairs(t["checker_payload"]["tests"])
        if p and len(p[0].elts) >= K_DIAL:
            dial.append((t, p))
    mb_dial = [(t, p) for t, p in dial if "/MBPP/" in t["task_id"]]
    seg = lambda node, src: ast.get_source_segment(src, node)  # noqa: E731
    matches = [first_pair_matches(t["gate_payload"].get("tests", ""), seg(p[0].elts[0], t["checker_payload"]["tests"])) for t, p in mb_dial]
    out["a17_task_set"] = {
        "n": len(dial), "mbpp_n": len(mb_dial), "humaneval_n": len(dial) - len(mb_dial),
        "mbpp_prompt_contains_gate_assert_verbatim": sum(gate_assert_in_prompt(t["prompt"], t["gate_payload"].get("tests", "")) for t, _ in mb_dial),
        "humaneval_prompt_has_doctest_examples": sum(">>>" in t["prompt"] for t, _ in dial if "/HumanEval/" in t["task_id"]),
        "mbpp_first_hidden_input_equals_prompt_assert_args": sum(m is True for m in matches),
        "mbpp_first_hidden_input_differs": sum(m is False for m in matches),
        "mbpp_not_parsed_by_this_check": sum(m is None for m in matches),
        "method": "prompt assert `assert f(args) == x` parsed with ast.literal_eval and compared with the literal first element of the hidden `inputs` list; other assert forms (e.g. math.isclose, set(...)) are 'not parsed'",
    }
    return out


# ── tables ──────────────────────────────────────────────────────────────────────────────────────────────────────

def table_pydepth(D: dict) -> str:
    v = D["validation"]
    exc = v["excluded_reasons"]
    t = ("\\begin{table}[!htbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
         f"\\caption{{A17, the Python visible-test depth dial on the stored E answers (\\code{{python_gate_depth.json}}). "
         f"{v['usable_with_at_least_K_pairs']} of {v['tasks']} Python tasks have at least {v['K']} input/result pairs and are used at every $k$ "
         f"(excluded: {exc['no_pairs']} with no pairs, {exc['fewer_than_k_pairs']} with fewer than {v['K']}). The $k$-gate is the hidden test text with both lists cut to their first $k$ elements; "
         "``stored'' is the one-assertion gate as stored in E, on the same tasks. Coverage = share of tasks verified; precision = share of verified answers that pass the hidden tests; "
         "CWR = verified and wrong, over all tasks. 95\\% cluster-bootstrap CIs (10,000 resamples, seed 0; cluster = problem), marginal per $k$, not a paired test. "
         "Descriptive; no hypothesis is tested. Most of these prompts showed the model a test (Section~\\ref{sec:a17}).}\\label{tab:pydepth}\n"
         "\\begin{tabular}{llccc}\\toprule\nTier & gate & coverage [CI] & precision [CI] & CWR [CI]\\\\\\midrule\n")
    for i, tr in enumerate(("qwen3.5-4b-bf16", "qwen3.5-9b-bf16")):
        e = D["tiers"][tr]
        if e["sanity_violations"]:
            raise SystemExit(f"sanity violations for {tr}: {e['sanity_violations']}")
        for g, lab in (("stored", "stored (1)"), ("k1", "$k=1$"), ("k2", "$k=2$"), ("k4", "$k=4$"), ("k8", "$k=8$")):
            t += (f"{TN[tr] if g == 'stored' else ''} & {lab} & {pci(e[g + '.coverage'])} & {pci(e[g + '.precision'])} & "
                  f"{pci(e[g + '.confident_wrong'])}\\\\\n")
        if i == 0:
            t += "\\midrule\n"
    t += "\\bottomrule\\end{tabular}\\end{table}\n"
    return t


def table_mathfix(M: dict) -> str:
    t = ("\\begin{table}[!htbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{2.5pt}\n"
         "\\caption{D62, exploratory post-hoc re-score of the stored E math answers (500 tasks per tier; \\code{rescore_exploratory.json}). "
         "Each cell is registered scorer $\\to$ re-score. The registered values remain the primary record. "
         "``Gate V-wrong'' = math gate VERIFIED but wrong on the hidden answer (count, and rate over all 500). "
         "``Answer all'' counts every task with no final \\code{\\boxed{}} (mostly generations cut at 1,024 tokens) as confident-wrong; "
         "``answer if boxed'' answers only when a boxed answer exists (coverage = boxed share) and counts the rest as abstentions. "
         "No model output changed; only the scorer did.}\\label{tab:mathfix}\n"
         "\\begin{tabular}{lcccccccc}\\toprule\n"
         "Tier & accuracy & flips w$\\to$r / r$\\to$w & gate verified & gate V-wrong & gate CWR & no boxed & answer-all CWR & boxed: cov., CWR\\\\\\midrule\n")
    for k in ("2B", "4B", "9B"):
        e = M["tiers"][k]
        g = e["gate"]
        b = e["answer_if_boxed_confident_wrong"]
        n = e["n"]
        t += (f"{k} & {num(e['accuracy']['old'])} $\\to$ {num(e['accuracy']['new'])} & "
              f"{e['label_flips']['wrong_to_right']} / {e['label_flips']['right_to_wrong']} & "
              f"{g['verified']['old']} $\\to$ {g['verified']['new']} & "
              f"{g['verified_but_wrong']['old']} $\\to$ {g['verified_but_wrong']['new']} & "
              f"{num(g['cwr_rate_over_n']['old'])} $\\to$ {num(g['cwr_rate_over_n']['new'])} & "
              f"{e['cwr_all']['n_no_boxed']} & {num(e['cwr_all']['old'])} $\\to$ {num(e['cwr_all']['new'])} & "
              f"{num(b['n_boxed'] / n)}, {num(b['old'])} $\\to$ {num(b['new'])}\\\\\n")
    t += "\\bottomrule\\end{tabular}\\end{table}\n"
    return t


def table_policy(P: dict) -> str:
    t = ("\\begin{table}[!htbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
         "\\caption{Exploratory, not pre-registered: what the weak-gate policy would do to the stored gate verdicts (\\code{verdict_policy_exploratory.json}, key \\code{stored_gate}, Rust counts after the counting fix). "
         "``1 test'' = verified answers whose gate payload has exactly one visible test. ``Wrong among verified'' = share of verified answers that fail the hidden tests (no policy). "
         "``Kept at $k_{\\min}=2$'' = verified answers that would stay VERIFIED; the others would be demoted to the calibrator path, which is not counted as wrong. "
         "The policy is off by default and no $k_{\\min}$ is recommended; E is an evaluation set.}\\label{tab:policy390}\n"
         "\\begin{tabular}{llrrrcr}\\toprule\nSet & Tier & tasks & verified & 1 test & wrong among verified & kept at $k_{\\min}=2$\\\\\\midrule\n")
    names = {"E.python": "E Python", "E.rust": "E Rust", "Rprime.rust": "R$'$ Rust"}
    for s, nm in names.items():
        for tr in ("qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16"):
            e = P["stored_gate"][s][tr]
            a = e["after_fix"]
            k2 = next(b for b in a["by_k_min"] if b["k_min"] == 2)
            one = a["visible_tests_hist_verified"].get("1", 0)
            t += (f"{nm} & {TN[tr]} & {e['n_tasks']} & {e['n_verified']} & {one} & "
                  f"{num(e['confident_wrong_among_verified_no_policy'])} & {k2['remaining_verified']}\\\\\n")
        if s != "Rprime.rust":
            t += "\\midrule\n"
    t += "\\bottomrule\\end{tabular}\\end{table}\n"
    return t


HASHED = ["results/gwenlaya_v4/edition/python_gate_depth.json", "scripts/python_gate_depth.py", "tests/test_python_gate_depth.py",
          "results/gwenlaya_v4/edition/prompt_test_visibility_390.json", "gwaya/domains/strength.py", "gwaya/domains/checkers.py",
          "gwaya/gwenlaya.py", "tests/test_gate_strength.py", "tests/test_gwenlaya_verdict_policy.py",
          "results/gwenlaya_v4/lowtier/verdict_policy_exploratory.json", "scripts/verdict_policy_exploratory.py",
          "results/gwenlaya_v4/e_tpu_mathgate/rescore_exploratory.json", "scripts/rescore_math_stored.py", "gwaya/domains/math_check.py",
          "tests/test_math_check_normalization.py", "gwaya/lean_gate.py", "tests/test_lean_gate.py",
          "results/gwenlaya_v4/lean_a18/lean_sanity.json", "results/gwenlaya_v4/lean_a18/tasks_lean.jsonl",
          "results/gwenlaya_v4/lean_a18/lean_gate_tests_390.log", "results/gwenlaya_v4/lean_a18/lean_sanity_summary_390.json",
          "results/gwenlaya_v4/spend_summary_390.json", "scripts/make_tables_390.py", "tests/test_make_tables_390.py",
          "papers/tables_pydepth390.tex", "papers/tables_mathfix390.tex", "papers/tables_policy390.tex", "papers/numbers390.tex",
          "docs/ANALYSIS_PLAN_E_TPU.md", "docs/DEVIATIONS.md"]


def write_hashes() -> None:
    """papers/sha390.tex: sha256 rows of the 3.9.0 inputs and outputs, computed when the tables are written."""
    import hashlib
    rows = []
    for rel in HASHED:
        rows.append((rel, hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()))
    rows.append((f"tasks_E_primary.d25.jsonl (data root)", hashlib.sha256(TASKS.read_bytes()).hexdigest()))
    n = sum(1 for l in open(LEDGER) if l.strip())
    rows.append((f"spend_ledger.jsonl ({n} lines, as read for 3.9.0; a living file)", hashlib.sha256(LEDGER.read_bytes()).hexdigest()))
    out = "% generated by scripts/make_tables_390.py - do not edit\n"
    for rel, h in rows:
        out += f"\\url{{{rel}}} & \\texttt{{\\seqsplit{{{h}}}}}\\\\\n"
    (ROOT / "papers/sha390.tex").write_text(out)


def main() -> None:
    D = json.load(open(RES / "edition/python_gate_depth.json"))
    M = json.load(open(RES / "e_tpu_mathgate/rescore_exploratory.json"))
    P = json.load(open(RES / "lowtier/verdict_policy_exploratory.json"))
    L = json.load(open(RES / "lean_a18/lean_sanity.json"))
    tasks = [json.loads(l) for l in open(TASKS) if l.strip()]

    head = "% generated by scripts/make_tables_390.py - do not edit\n"
    (ROOT / "papers/tables_pydepth390.tex").write_text(head + table_pydepth(D))
    (ROOT / "papers/tables_mathfix390.tex").write_text(head + table_mathfix(M))
    (ROOT / "papers/tables_policy390.tex").write_text(head + table_policy(P))

    vis = visibility(tasks)
    (RES / "edition/prompt_test_visibility_390.json").write_text(json.dumps(vis, indent=1) + "\n")

    # A17 differences of printed points (k=1 -> k=8), for the text
    for tr in ("qwen3.5-4b-bf16", "qwen3.5-9b-bf16"):
        e = D["tiers"][tr]
        print(TN[tr], "CWR k1->k8", round(e["k1.confident_wrong"]["point"], 3), "->", round(e["k8.confident_wrong"]["point"], 3),
              "coverage", round(e["k1.coverage"]["point"], 3), "->", round(e["k8.coverage"]["point"], 3),
              "precision", round(e["k1.precision"]["point"], 3), "->", round(e["k8.precision"]["point"], 3))

    Lsum = {"n": L["n"], "elaborating": L["elaborating"], "excluded_by_class": L["excluded_by_class"],
            "mathlib": L["mathlib"], "tasks_sha256": L["tasks_sha256"], "generation_run": False,
            "note": "A18 generation has not been run; no Lean result exists"}
    assert L["n"] - L["elaborating"] == sum(L["excluded_by_class"].values())
    (RES / "lean_a18/lean_sanity_summary_390.json").write_text(json.dumps(Lsum, indent=1) + "\n")

    Lg = [json.loads(l) for l in open(LEDGER) if l.strip()]
    spend = {"lines_total": len(Lg), "total_usd": round(sum(x["usd_estimate"] for x in Lg), 4),
             "lines_3_8": 25, "total_3_8": round(sum(x["usd_estimate"] for x in Lg[:25]), 4),
             "lines_new": len(Lg) - 25, "total_new": round(sum(x["usd_estimate"] for x in Lg[25:]), 4),
             "last_ts": Lg[-1]["ts"], "last_item": Lg[-1]["item"],
             "new_items": {x["item"]: {"ts": x["ts"], "seconds": x["seconds"], "usd": x["usd_estimate"], "accel": x["accel"]} for x in Lg[25:]},
             "note": "lines after 25 were added after 3.8.0 (A18 Lean generation, in progress; no Lean result is reported in 3.9.0)"}
    (RES / "spend_summary_390.json").write_text(json.dumps(spend, indent=1) + "\n")
    items = ", ".join(f"\\code{{{k}}} ({v['accel'].split()[0]}, {v['seconds']:,} s, \\${v['usd']:.4f})" for k, v in spend["new_items"].items())
    macros = (head + f"\\newcommand{{\\SpendLines}}{{{spend['lines_total']}}}\n"
              f"\\newcommand{{\\SpendTotal}}{{{spend['total_usd']:.4f}}}\n"
              f"\\newcommand{{\\SpendNewLines}}{{{spend['lines_new']}}}\n"
              f"\\newcommand{{\\SpendNewTotal}}{{{spend['total_new']:.4f}}}\n"
              f"\\newcommand{{\\SpendNewItems}}{{{items or 'none'}}}\n"
              f"\\newcommand{{\\SpendLastTs}}{{{spend['last_ts'].replace('T', ' ')[:19]}}}\n"
              f"\\newcommand{{\\SpendCapShare}}{{{100 * spend['total_usd'] / 50:.1f}}}\n")
    (ROOT / "papers/numbers390.tex").write_text(macros)
    write_hashes()
    print(json.dumps(vis, indent=1))
    print(Lsum)
    print(spend)


if __name__ == "__main__":
    main()
