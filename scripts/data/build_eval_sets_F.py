#!/usr/bin/env python3
"""A21: build the fresh evaluation sets F (python, math, lean4) for the next TPU run; deterministic and seeded.

"Fresh" = not in anything the frozen Laya (docs/LAYA_FREEZE.txt) was trained, calibrated or evaluated on:
E (tasks_E_primary.d25.jsonl), P_all (train/calib/E'), Rust_fresh, and the A18 miniF2F-test tasks. Every candidate is
screened against ALL of them with gwaya.decontaminate (exact / AST / n-gram / jaccard / math / test_io / id methods);
any hit removes the candidate. No model is run here.

Sources (schemas as in tasks_E_primary.d25.jsonl: domain, task_id, prompt, checker_payload, gate_payload, cluster;
extra bookkeeping keys: source, stratum, gate_visible_in_prompt, n_hidden_tests):
  python  newfacade/LeetCodeDataset (test split, 2024-08..2025-03): LeetCode contests, reference = dataset completion
          ByteDance/FullStackBench (en, python/python): the assert-style subset with stdlib+numpy tests
          bigcode/bigcodebench (v0.1.4): stdlib+numpy tasks, unittest converted to one asserted suite run
  math    EleutherAI/hendrycks_math TEST minus HuggingFaceH4/MATH-500 (answer = last \\boxed of the solution)
          openai/gsm8k TEST (seeded sample), MathArena AIME/HMMT/BRUMO 2025-2026 (competition stratum)
  lean4   cat-searcher/minif2f-lean4 split "validation" via scripts/lean/build_minif2f_tasks.py
Not built (reasons in the manifest): LiveCodeBench (stdin/stdout harness incompatible; functional subset is contained
in LeetCodeDataset), Rust (gate-only now).

  GWAYA_DATA_ROOT=... python scripts/data/build_eval_sets_F.py --out results/gwenlaya_v4/eval_sets_F
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

SEED = 20261010
MATH_SUFFIX = "\n\nPut the final answer in \\boxed{}."   # identical to build_eval_manifest.MATH_SUFFIX
N_FSB = 130
N_BCB = 90
N_MATH_TEST = 200
N_GSM8K = 130
MAX_REF_MS = 10000.0         # a reference slower than a third of the 30 s hidden budget would flip under load
HIDDEN_TIMEOUT_S, GATE_TIMEOUT_S = 30.0, 10.0
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]
ARENA = {  # MathArena set -> (task prefix); CMIMC-2025 and SMT-2025 are left out for the generation budget
    "MathArena/aime_2025": "aime2025", "MathArena/aime_2026": "aime2026",
    "MathArena/hmmt_feb_2025": "hmmt-feb2025", "MathArena/hmmt_feb_2026": "hmmt-feb2026",
    "MathArena/hmmt_nov_2025": "hmmt-nov2025", "MathArena/brumo_2025": "brumo2025",
}
ARENA_SKIPPED = {"MathArena/cmimc_2025": "40 problems left out to bound generation cost (can be added later)",
                 "MathArena/smt_2025": "53 problems left out to bound generation cost (can be added later)"}
ALLOWED_TEST_MODULES = set(sys.stdlib_module_names) | {"numpy"}
PY_HEAD = ("Complete the following Python code. Reply with the complete code (every import it needs, and the class or "
           "function) in one ```python block.\n\n")


# ---------------------------------------------------------------------------------------------------------------
# small helpers

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _imports_of(tree: ast.AST) -> set[str]:
    mods: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            mods.add(n.module.split(".")[0])
    return mods


def _has_assert(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Assert) for n in ast.walk(node))


def count_asserts(src: str) -> int:
    """Number of top-level statements containing an assert: what the harness counts as tests."""
    return sum(1 for s in ast.parse(src).body if _has_assert(s))


_WRAPPER = re.compile(r"^assert (?:check\(.*\) is None|_bcb_result\.wasSuccessful\(\))\s*$", re.M)


def count_assertions(src: str) -> int:
    """Individual assertions of a hidden suite: `assert` nodes plus `self.assert*` calls, without the one-line
    wrapper assert that runs `check(...)` / the unittest suite (the study's harness counts statements instead)."""
    tree = ast.parse(src)
    n = 0
    for x in ast.walk(tree):
        if isinstance(x, ast.Assert):
            n += 1
        elif isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr.startswith("assert") \
                and isinstance(x.func.value, ast.Name) and x.func.value.id == "self":
            n += 1
    return max(0, n - len(_WRAPPER.findall(src)))


# ---------------------------------------------------------------------------------------------------------------
# prompt visibility of the gate test (lesson of audit D63): is the gate's first call/expected value shown in the prompt?

def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s).replace("'", '"')


def _call_items(test_expr: ast.AST) -> list[str]:
    """Squashed source of every argument / expected value of the calls and comparisons in an assertion."""
    items: list[str] = []
    for n in ast.walk(test_expr):
        vals: list[ast.AST] = []
        if isinstance(n, ast.Call):
            for a in list(n.args) + [k.value for k in n.keywords]:
                if isinstance(a, ast.Starred) and isinstance(a.value, (ast.List, ast.Tuple)):
                    vals.extend(a.value.elts)
                elif isinstance(a, ast.Starred):
                    vals.append(a.value)
                else:
                    vals.append(a)
        elif isinstance(n, ast.Compare):
            vals.extend(n.comparators)
        for v in vals:
            if isinstance(v, (ast.Name, ast.Attribute, ast.Call)):
                continue
            s = _squash(ast.unparse(v))
            if s:
                items.append(s)
    return sorted(set(items))


def _item_shown(item: str, squashed_prompt: str, raw_prompt: str) -> bool:
    if len(item) >= 2:
        return item in squashed_prompt
    return re.search(r"(?<![\w.])" + re.escape(item) + r"(?![\w.])", raw_prompt) is not None


def first_assert_node(src: str) -> ast.Assert | None:
    tree = ast.parse(src)
    best = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assert) and (best is None or (n.lineno, n.col_offset) < (best.lineno, best.col_offset)):
            best = n
    return best


def gate_visible_in_prompt(prompt: str, gate_src: str) -> dict[str, Any]:
    """Heuristic: every argument and expected value of the gate's first assertion occurs verbatim in the prompt
    (whitespace and quote-style insensitive; one-character items as whole tokens). `visible` is None when the
    assertion has no such item."""
    node = first_assert_node(gate_src)
    if node is None:
        return {"visible": None, "n_items": 0, "n_shown": 0}
    items = _call_items(node.test)
    if not items:
        return {"visible": None, "n_items": 0, "n_shown": 0}
    p = _squash(prompt)
    shown = [i for i in items if _item_shown(i, p, prompt)]
    return {"visible": len(shown) == len(items), "n_items": len(items), "n_shown": len(shown)}


def unittest_gate_visible(prompt: str, gate_src: str) -> dict[str, Any]:
    """Same idea for the unittest-converted BigCodeBench gate: literals (>= 2 chars) of the kept test method."""
    tree = ast.parse(gate_src)
    items: set[str] = set()
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for m in cls.body:
            if isinstance(m, ast.FunctionDef) and m.name.startswith("test"):
                for n in ast.walk(m):
                    if isinstance(n, ast.Constant) and isinstance(n.value, (str, int, float)) \
                            and not isinstance(n.value, bool):
                        s = _squash(str(n.value))
                        if len(s) >= 2:
                            items.add(s)
    if not items:
        return {"visible": None, "n_items": 0, "n_shown": 0}
    p = _squash(prompt)
    shown = [i for i in items if i in p]
    return {"visible": len(shown) == len(items), "n_items": len(items), "n_shown": len(shown)}


# ---------------------------------------------------------------------------------------------------------------
# python: LeetCodeDataset

_LC_IMPORT_END = "inf = float('inf')"


def _lc_defs(prelude: str) -> tuple[str, str, str]:
    """(import block, ListNode class, TreeNode class) of the dataset's shared prelude."""
    i = prelude.find(_LC_IMPORT_END)
    imports = prelude[: i + len(_LC_IMPORT_END)] if i >= 0 else ""
    ln = re.search(r"class ListNode:\n(?:    .*\n|\n(?=    ))+", prelude)
    tn = re.search(r"class TreeNode:\n(?:    .*\n|\n(?=    ))+", prelude)
    return imports.strip(), (ln.group(0).rstrip() if ln else ""), (tn.group(0).rstrip() if tn else "")


def trim_check_to_first_assert(test_src: str) -> str | None:
    """`def check(candidate): ...` -> the same function with only its first assertion (and the setup before it)."""
    tree = ast.parse(test_src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            for i, st in enumerate(node.body):
                if _has_assert(st):
                    node.body = node.body[: i + 1]
                    return ast.unparse(node)
    return None


def leetcode_row(r: dict[str, Any]) -> dict[str, Any] | None:
    starter = (r.get("starter_code") or "").rstrip()
    test = r.get("test") or ""
    ep = r.get("entry_point") or ""
    if not starter or not test or not ep or not (r.get("completion") or "").strip():
        return None
    imports, listnode, treenode = _lc_defs(r.get("prompt") or "")
    shown = [imports] if imports else []
    if "ListNode" in starter and listnode:
        shown.append(listnode)
    if "TreeNode" in starter and treenode:
        shown.append(treenode)
    shown.append(starter)
    prompt = (PY_HEAD + (r.get("problem_description") or "").strip() + "\n\nStarter code:\n```python\n"
              + "\n\n".join(shown) + "\n```")
    gate_check = trim_check_to_first_assert(test)
    if gate_check is None:
        return None
    prelude = (r.get("prompt") or "").rstrip() + "\n\n"
    hidden = prelude + test.rstrip() + f"\n\nassert check({ep}) is None\n"
    gate = prelude + gate_check + f"\n\nassert check({ep}) is None\n"
    ref = "\n\n".join(([imports] if imports else []) + ([listnode] if listnode else [])
                      + ([treenode] if treenode else []) + [r["completion"].rstrip()])
    return {"domain": "python", "task_id": f"py/LeetCodeDataset/{r['task_id']}", "prompt": prompt,
            "checker_payload": {"tests": hidden, "timeout_s": HIDDEN_TIMEOUT_S},
            "gate_payload": {"tests": gate, "timeout_s": GATE_TIMEOUT_S},
            "cluster": f"LeetCodeDataset/{r['task_id']}", "source": "newfacade/LeetCodeDataset:test",
            "stratum": "python/LeetCodeDataset", "_reference": ref, "_date": str(r.get("estimated_date", ""))}


# ---------------------------------------------------------------------------------------------------------------
# python: FullStackBench

def fsb_row(r: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """-> (row | None, drop reason). Only python/python tasks whose test is a plain module of top-level asserts."""
    lab = r.get("labels") or {}
    if lab.get("programming_language") != "python" or lab.get("execution_language") != "python":
        return None, "not_python_python"
    t = r.get("test") or {}
    code = t.get("code") or ""
    if "#<INSERT>" not in code:
        return None, "no_insert_marker"
    if t.get("asset"):
        return None, "needs_asset_files"
    body = code.replace("#<INSERT>", "")
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return None, "test_unparsable"
    mods = _imports_of(tree)
    if mods - ALLOWED_TEST_MODULES:
        return None, "test_needs_third_party:" + ",".join(sorted(mods - ALLOWED_TEST_MODULES))[:40]
    if "unittest" in mods or "pytest" in mods:
        return None, "test_uses_unittest_or_pytest"
    # the gate runs the setup and the FIRST executed assertion; asserts that live only inside a function the test
    # calls later (`def check(...): assert ...` + `check(f)`) would make the truncated gate a bare definition that
    # passes for anything, so such tests are not converted
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    idx = next((i for i, s in enumerate(tree.body) if _has_assert(s) and not isinstance(s, defs)), None)
    if idx is None:
        if any(_has_assert(s) for s in tree.body):
            return None, "asserts_only_inside_functions"
        return None, "no_top_level_assert"
    gate_tree = ast.Module(body=tree.body[: idx + 1], type_ignores=[])
    ref = (r.get("canonical_solution") or "")
    content = (r.get("content") or "").strip()
    if not content:
        return None, "empty_prompt"
    return {"domain": "python", "task_id": f"py/FullStackBench/{r['id']}",
            "prompt": content,
            "checker_payload": {"tests": body.strip() + "\n", "timeout_s": HIDDEN_TIMEOUT_S},
            "gate_payload": {"tests": ast.unparse(gate_tree) + "\n", "timeout_s": GATE_TIMEOUT_S},
            "cluster": f"FullStackBench/{r['id']}", "source": "ByteDance/FullStackBench:en",
            "stratum": "python/FullStackBench", "_reference": ref}, ""


# ---------------------------------------------------------------------------------------------------------------
# python: BigCodeBench

_BCB_RUN = ("_bcb_result = unittest.TextTestRunner(stream=__import__('io').StringIO(), verbosity=0).run("
            "unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(_c) for _c in ({classes},)]))\n"
            "assert _bcb_result.wasSuccessful()\n")


def _bcb_classes(tree: ast.Module) -> list[str]:
    out = []
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and any(
                (isinstance(b, ast.Attribute) and b.attr == "TestCase") or (isinstance(b, ast.Name) and b.id == "TestCase")
                for b in n.bases):
            out.append(n.name)
    return out


def bcb_row(r: dict[str, Any], idx: int) -> tuple[dict[str, Any] | None, str]:
    libs = r.get("libs")
    libs = json.loads(libs.replace("'", '"')) if isinstance(libs, str) else list(libs or [])
    bad = [m.split(".")[0] for m in libs if m.split(".")[0] not in ALLOWED_TEST_MODULES]
    if bad:
        return None, "needs_third_party:" + ",".join(sorted(set(bad)))[:40]
    test = r.get("test") or ""
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return None, "test_unparsable"
    classes = _bcb_classes(tree)
    if not classes:
        return None, "no_testcase_class"
    if "unittest" not in _imports_of(tree):
        return None, "no_unittest_import"
    # gate = the same module with only the first test_* method of the first class that has one
    gtree = ast.parse(test)
    kept = False
    for n in gtree.body:
        if isinstance(n, ast.ClassDef):
            body = []
            for m in n.body:
                if isinstance(m, ast.FunctionDef) and m.name.startswith("test"):
                    if kept:
                        continue
                    kept = True
                body.append(m)
            n.body = body
    if not kept:
        return None, "no_test_method"
    run = _BCB_RUN.format(classes=", ".join(classes))
    hidden = test.rstrip() + "\n\n" + run
    gate = ast.unparse(gtree) + "\n\n" + run
    prompt = ("Complete the following Python function. Reply with the complete code (every import it needs and the "
              "function) in one ```python block.\n\n" + (r.get("complete_prompt") or "").strip())
    ref = (r.get("code_prompt") or "").rstrip("\n") + "\n" + (r.get("canonical_solution") or "")
    num = str(r.get("task_id", f"BigCodeBench/{idx}")).split("/")[-1]
    return {"domain": "python", "task_id": f"py/BigCodeBench/{num}", "prompt": prompt,
            "checker_payload": {"tests": hidden, "timeout_s": HIDDEN_TIMEOUT_S},
            "gate_payload": {"tests": gate, "timeout_s": GATE_TIMEOUT_S},
            "cluster": f"BigCodeBench/{num}", "source": "bigcode/bigcodebench:v0.1.4",
            "stratum": "python/BigCodeBench", "_reference": ref, "_unittest": True}, ""


def count_test_methods(src: str) -> int:
    """`test*` methods of the TestCase classes in a (unittest-style) suite."""
    return sum(1 for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test"))


def hidden_extends_gate(row: dict) -> bool:
    """True iff the hidden suite holds something the gate's visible test does not (more assertions, or more test
    methods for the unittest-converted tasks). Otherwise the gate would verify exactly what is scored."""
    h, g = row["checker_payload"]["tests"], row["gate_payload"]["tests"]
    if row.get("_unittest"):
        return count_test_methods(h) > count_test_methods(g)
    return count_assertions(h) > count_assertions(g)


def hidden_asserts_in_prompt(prompt: str, tests: str) -> int:
    """Hidden assertion lines (>= 12 chars, whitespace-squashed) that the prompt shows verbatim. A positive count
    means the prompt itself reveals part of the hidden suite (e.g. FullStackBench's 'fix the failing code' variants,
    which paste the failing assert), so the task is not a fair held-out test."""
    p = _squash(prompt)
    n = 0
    for line in tests.splitlines():
        t = _squash(line.strip())
        if t.startswith("assert") and len(t) >= 12 and t in p:
            n += 1
    return n


# ---------------------------------------------------------------------------------------------------------------
# python sanity: reference passes hidden+gate, trivial stub (reference signatures, empty bodies) fails hidden

def stub_of(reference: str) -> str:
    """The reference with every function body replaced by `pass`: same interface, no behaviour."""
    tree = ast.parse(reference)

    class _Strip(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
            node.body = [ast.Pass()]
            return node
        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    return ast.unparse(ast.fix_missing_locations(_Strip().visit(tree)))


def _verdict(row: dict, payload_key: str, code: str) -> tuple[str, float | None, str]:
    from gwaya.domains.checkers import check
    from gwaya.domains.task import Task
    res = check(Task("python", row["task_id"], "", dict(row[payload_key])), "```python\n" + code + "\n```")
    ev = res.evidence or {}
    return res.status, ev.get("latency_ms"), str(ev.get("error", ""))[:160]


def sanity_python(row: dict) -> tuple[str | None, dict[str, Any]]:
    ref = row.get("_reference") or ""
    info: dict[str, Any] = {}
    if hidden_asserts_in_prompt(row["prompt"], row["checker_payload"]["tests"]):
        return "hidden_test_shown_in_prompt", info
    if not hidden_extends_gate(row):
        return "gate_equals_hidden_suite", info
    if not ref.strip() or "exit(0)" in ref:
        # FullStackBench hides the reference of its LeetCode-style tasks (`exit(0)`): borrow a verified one from
        # LeetCodeDataset-train (same method name); the hidden tests themselves must accept it.
        ref = ""
        for i, cand in enumerate(row.get("_ref_candidates") or []):
            if _verdict(row, "checker_payload", cand)[0] == "VERIFIED":
                ref, info["reference_source"] = cand, f"LeetCodeDataset-train#{i}"
                break
        if not ref:
            return "no_reference_solution", info
        row["_reference"] = ref
    try:
        ast.parse(ref)
    except SyntaxError:
        return "reference_unparsable", info
    st, ms, err = _verdict(row, "checker_payload", ref)
    info["ref_ms"] = ms
    if st != "VERIFIED":
        return f"reference_fails_hidden:{st}:{err}", info
    if ms is not None and ms > MAX_REF_MS:
        return f"reference_too_slow_for_a_stable_timeout:{ms:.0f}ms", info
    st, _, err = _verdict(row, "gate_payload", ref)
    if st != "VERIFIED":
        return f"reference_fails_gate:{st}:{err}", info
    stub = stub_of(ref)
    st, _, _ = _verdict(row, "checker_payload", stub)
    info["stub_status"] = st
    if st == "VERIFIED":
        return "stub_passes_hidden", info
    st, _, _ = _verdict(row, "gate_payload", stub)
    info["stub_gate_status"] = st
    if st == "VERIFIED":
        return "stub_passes_gate", info   # a gate that cannot fail is not a gate
    return None, info


# ---------------------------------------------------------------------------------------------------------------
# math

def _last_boxed(text: str) -> str | None:
    from gwaya.domains.math_check import _last_boxed as lb
    return lb(text)


def math_test_row(subject: str, idx: int, r: dict[str, Any]) -> dict[str, Any] | None:
    ans = _last_boxed(r.get("solution") or "")
    prob = (r.get("problem") or "").strip()
    if not ans or not ans.strip() or not prob:
        return None
    return {"domain": "math", "task_id": f"ma/MATHtest/{subject}/{idx}", "prompt": prob + MATH_SUFFIX,
            "checker_payload": {"answer": ans.strip()}, "gate_payload": {},
            "cluster": f"MATH/test/{subject}/{idx}", "source": "EleutherAI/hendrycks_math:test",
            "stratum": "math/MATH-test", "_sub": f"{subject}|{r.get('level', '')}", "_problem": prob}


def gsm8k_test_row(idx: int, r: dict[str, Any]) -> dict[str, Any] | None:
    a = r.get("answer") or ""
    q = (r.get("question") or "").strip()
    if "####" not in a or not q:
        return None
    gold = a.rsplit("####", 1)[1].strip().replace(",", "")
    if not gold:
        return None
    return {"domain": "math", "task_id": f"ma/gsm8k-test/{idx}", "prompt": q + MATH_SUFFIX,
            "checker_payload": {"answer": gold}, "gate_payload": {}, "cluster": f"GSM8K/test/{idx}",
            "source": "openai/gsm8k:test", "stratum": "math/GSM8K", "_problem": q}


def arena_row(prefix: str, repo: str, r: dict[str, Any]) -> dict[str, Any] | None:
    ans = str(r.get("answer", "")).strip()
    q = (r.get("problem") or "").strip()
    if not ans or not q:
        return None
    return {"domain": "math", "task_id": f"ma/comp/{prefix}/{int(r['problem_idx'])}", "prompt": q + MATH_SUFFIX,
            "checker_payload": {"answer": ans}, "gate_payload": {}, "cluster": f"comp/{prefix}/{int(r['problem_idx'])}",
            "source": f"{repo}", "stratum": "math/competition", "_problem": q}


def sanity_math(row: dict) -> str | None:
    from gwaya.domains.math_check import check_math
    gold = row["checker_payload"]["answer"]
    if check_math("\\boxed{" + gold + "}", gold).status != "VERIFIED":
        return "gold_not_self_verifying"
    return None


def stratified_sample(rows: list[dict], n: int, seed: int, key) -> list[dict]:
    """Seeded proportional (largest remainder) sample of up to n rows over strata `key`."""
    if n >= len(rows):
        return sorted(rows, key=lambda r: r["task_id"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    total = sum(len(v) for v in groups.values())
    raw = {k: n * len(v) / total for k, v in groups.items()}
    alloc = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (raw[k] - alloc[k], k), reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    out: list[dict] = []
    for k in sorted(groups):
        g = sorted(groups[k], key=lambda r: r["task_id"])
        random.Random(f"{seed}:{k}").shuffle(g)
        out.extend(g[: alloc[k]])
    return sorted(out, key=lambda r: r["task_id"])


def seeded_sample(rows: list[dict], n: int, seed: int, tag: str) -> list[dict]:
    g = sorted(rows, key=lambda r: r["task_id"])
    random.Random(f"{seed}:{tag}").shuffle(g)
    return sorted(g[:n], key=lambda r: r["task_id"])


# ---------------------------------------------------------------------------------------------------------------
# decontamination against everything the frozen Laya saw

def _strip_suffix(p: str) -> str:
    return p[: -len(MATH_SUFFIX)] if p.endswith(MATH_SUFFIX) else p


def record_to_item(dataset: str, rec: dict[str, Any]):
    """Existing task record (study schema) -> decontaminate.Item (prompt + hidden tests + statement/answer)."""
    from gwaya import decontaminate as dc
    cp = rec.get("checker_payload") or {}
    fields: list[tuple[str, str]] = []
    prompt = rec.get("prompt") or ""
    if rec.get("domain") == "math":
        fields.append(("prompt", _strip_suffix(prompt)))
    else:
        fields.append(("prompt", prompt))
    for k in ("tests",):
        if isinstance(cp.get(k), str) and cp[k].strip():
            fields.append(("tests", cp[k]))
    if isinstance(cp.get("formal_statement"), str):
        fields.append(("statement", cp["formal_statement"]))
    ids = set()
    tid = rec.get("task_id", "")
    row = {"task_id": tid}
    ids |= dc.canonical_ids(dataset, row, fields)
    return dc.Item(dataset=dataset, item_id=tid, domain=rec.get("domain", "unknown"), fields=fields,
                   ids=frozenset(ids))


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:  # not str.splitlines(): U+2028 inside JSON strings would split a record
        return [json.loads(line) for line in fh if line.strip()]


def existing_sets(data_root: Path, lean_a18: Path) -> dict[str, Path]:
    return {
        "E": data_root / "night/results/tasks_E_primary.d25.jsonl",
        "P_all": data_root / "laya_data/tasks_P_all.jsonl",
        "Rust_fresh": data_root / "laya_data/tasks_Rust_fresh.jsonl",
        "A18_minif2f_test": lean_a18,
    }


def build_decontaminator(paths: dict[str, Path]):
    from gwaya import decontaminate as dc
    items, sizes = [], {}
    for name, p in paths.items():
        recs = read_jsonl(p)
        sizes[name] = len(recs)
        items.extend(record_to_item(name, r) for r in recs)
    # the Laya router/calibrator rows (their `prompt` is the model-visible text) as a further net
    for f in sorted(glob.glob(str(paths["E"].parents[2] / "laya_data/laya_*/router.jsonl"))):
        name = "laya:" + Path(f).parent.name
        n = 0
        for line in open(f):
            r = json.loads(line)
            items.append(dc.Item(dataset=name, item_id=str(r.get("id")), domain=r.get("domain", "unknown"),
                                 fields=[("prompt", _strip_suffix(r.get("prompt") or ""))]))
            n += 1
        sizes[name] = n
    return dc.Decontaminator(items), sizes


_TRIVIAL_TOKENS = {"none", "true", "false", "null", "nan", "inf"}


def informative_gram(gram: str) -> bool:
    """A shared 13-gram that is just an alphabet run ('a b c ... m') or a repeated literal ('None None ...') is a
    harness/test artefact, not shared content: it needs >= 5 distinct tokens of which >= 3 are words of >= 3 letters
    that are not bare literals."""
    toks = gram.lower().split()
    words = {t for t in toks if len(t) >= 3 and t.isalpha() and t not in _TRIVIAL_TOKENS}
    return len(set(toks)) >= 5 and len(words) >= 3


def decontaminate_rows(dec, rows: list[dict], dedup: bool = True,
                       ignored: Counter | None = None) -> tuple[list[dict], list[dict]]:
    """Screen candidates (processed in the given order; first wins on cross-source duplicates). Any hit removes the
    candidate, except n-gram hits whose shared gram is not informative (see informative_gram; counted in `ignored`)."""
    kept, drops = [], []
    for r in rows:
        item = record_to_item("F:" + r["source"], {**r, "prompt": r.get("_problem_for_dc") or r["prompt"]})
        if r["domain"] == "math":
            item.fields[:] = [("prompt", r["_problem"])]
        elif r["domain"] == "python":
            item.fields[:] = [("prompt", r["prompt"]), ("tests", r["checker_payload"]["tests"])]
            if r.get("_reference"):
                item.fields.append(("solution", r["_reference"]))
        elif r["domain"] == "lean4":
            item.fields[:] = [("statement", r["checker_payload"]["formal_statement"])]
        hits = dec.check(item)
        # an n-gram hit against item k stands only if at least one gram they share is informative; evaluated over
        # ALL shared grams (Hit.detail is just the first one and depends on the interpreter's hash seed)
        gram_hits = [h for h in hits if h.method.startswith("ngram")]
        if gram_hits:
            from gwaya import decontaminate as dc
            f = dc.features(item, dec.cfg)
            frozen = getattr(dec.index, "_frozen_df", None) or set()
            informative: dict[str, bool] = {}
            for g in sorted(f.grams):
                if g in frozen:
                    continue
                ok = informative_gram(f.gram_text.get(g, ""))
                for k in dc._get(dec.index.grams, g):
                    informative[k] = informative.get(k, False) or ok
            junk = [h for h in gram_hits if not informative.get(h.other, True)]
            if ignored is not None:
                ignored["ngram_hits_total"] += len(gram_hits)
                ignored["ngram_hits_ignored_uninformative"] += len(junk)
            hits = [h for h in hits if h not in junk]
        why = None
        if hits:
            why = {"methods": sorted({h.method.split(":")[0] for h in hits}),
                   "against": sorted({h.other.split("::")[0] for h in hits}),
                   "example": f"{hits[0].method}->{hits[0].other}"}
        elif dedup:
            dh = dec.dedup_check(item)
            if dh:
                why = {"methods": sorted({h.method for h in dh}), "against": ["F(other source)"],
                       "example": f"{dh[0].method}->{dh[0].other}"}
        if why:
            drops.append({"task_id": r["task_id"], "source": r["source"], **why})
        else:
            kept.append(r)
    return kept, drops


# ---------------------------------------------------------------------------------------------------------------
# loading (local HF cache only; revisions from the snapshot directory / hub when reachable)

def _hub(root: Path) -> str:
    return str(root / "hf-home" / "hub")


def snapshot(repo: str, root: Path) -> Path:
    os.environ.setdefault("HF_HOME", str(root / "hf-home"))
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, repo_type="dataset"))


def load_python_sources(root: Path) -> dict[str, Any]:
    import pandas as pd
    src: dict[str, Any] = {"revisions": {}, "files": {}}
    p = snapshot("newfacade/LeetCodeDataset", root)
    f = p / "LeetCodeDataset-test.jsonl"
    src["leetcode"] = read_jsonl(f)
    src["leetcode_train"] = read_jsonl(p / "LeetCodeDataset-train.jsonl")
    src["revisions"]["newfacade/LeetCodeDataset"] = p.name
    src["files"]["LeetCodeDataset-test.jsonl"] = sha256_file(f)
    src["files"]["LeetCodeDataset-train.jsonl"] = sha256_file(p / "LeetCodeDataset-train.jsonl")
    p = snapshot("ByteDance/FullStackBench", root)
    f = p / "en" / "test-00000-of-00001.parquet"
    src["fsb"] = pd.read_parquet(f).to_dict("records")
    src["revisions"]["ByteDance/FullStackBench"] = p.name
    src["files"]["FullStackBench/en/test.parquet"] = sha256_file(f)
    p = snapshot("bigcode/bigcodebench", root)
    f = p / "data" / "v0.1.4-00000-of-00001.parquet"
    src["bcb"] = pd.read_parquet(f).to_dict("records")
    src["revisions"]["bigcode/bigcodebench"] = p.name
    src["files"]["bigcodebench/v0.1.4.parquet"] = sha256_file(f)
    return src


def load_math_sources(root: Path) -> dict[str, Any]:
    import pandas as pd
    from datasets import load_dataset
    cache = str(root / "hf-datasets")
    os.environ.setdefault("HF_HOME", str(root / "hf-home"))
    src: dict[str, Any] = {"revisions": {}, "files": {}, "problems": []}
    test: list[dict] = []
    for sub in MATH_SUBJECTS:
        d = load_dataset("EleutherAI/hendrycks_math", sub, split="test", cache_dir=cache)
        for i, r in enumerate(d):
            test.append({"subject": sub, "idx": i, **dict(r)})
    src["math_test"] = test
    m5 = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=cache)
    src["math500"] = [dict(r) for r in m5]
    src["gsm8k_test"] = [dict(r) for r in load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache)]
    src["arena"] = {}
    for repo in list(ARENA) + list(ARENA_SKIPPED):
        p = snapshot(repo, root)
        f = p / "data" / "train-00000-of-00001.parquet"
        src["arena"][repo] = pd.read_parquet(f).to_dict("records")
        src["revisions"][repo] = p.name
        src["files"][repo] = sha256_file(f)
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for repo in ("EleutherAI/hendrycks_math", "HuggingFaceH4/MATH-500", "openai/gsm8k"):
            src["revisions"][repo] = api.dataset_info(repo).sha
    except Exception as exc:  # noqa: BLE001
        src["problems"].append(f"hub revision lookup failed: {type(exc).__name__}")
    return src


# ---------------------------------------------------------------------------------------------------------------
# driver

def clean(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def dump(path: Path, rows: list[dict]) -> str:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)


def build_python(src: dict, dec, workers: int, fsb_n: int, bcb_n: int, seed: int) -> tuple[list[dict], dict]:
    rep: dict[str, Any] = {"sources": {}}
    cand: dict[str, list[dict]] = {}
    drops_conv: dict[str, Counter] = defaultdict(Counter)
    # LeetCodeDataset: all
    lc = []
    for r in src["leetcode"]:
        row = leetcode_row(r)
        if row is None:
            drops_conv["LeetCodeDataset"]["unusable_row"] += 1
        else:
            lc.append(row)
    cand["LeetCodeDataset"] = lc
    by_method: dict[str, list[str]] = defaultdict(list)
    for r in src.get("leetcode_train", []):
        m = re.match(r"Solution\(\)\.(\w+)$", r.get("entry_point") or "")
        if m and (r.get("completion") or "").strip():
            imports, ln, tn = _lc_defs(r.get("prompt") or "")
            by_method[m.group(1)].append("\n\n".join(x for x in (imports, ln, tn, r["completion"].rstrip()) if x))
    fsb = []
    for r in src["fsb"]:
        row, why = fsb_row(r)
        if row is None:
            if why != "not_python_python":
                drops_conv["FullStackBench"][why.split(":")[0]] += 1
        else:
            if "exit(0)" in row["_reference"]:
                names = re.findall(r"def (\w+)\(self", row["prompt"]) or re.findall(
                    r"\bmy_solution\.(\w+)\(", row["checker_payload"]["tests"])
                row["_ref_candidates"] = by_method.get(names[0], [])[:5] if names else []
            fsb.append(row)
    cand["FullStackBench"] = fsb
    bcb = []
    for i, r in enumerate(src["bcb"]):
        row, why = bcb_row(r, i)
        if row is None:
            drops_conv["BigCodeBench"][why.split(":")[0]] += 1
        else:
            bcb.append(row)
    cand["BigCodeBench"] = bcb
    for name, rows in cand.items():
        rep["sources"][name] = {"n_candidates_after_conversion": len(rows),
                                "conversion_drops": dict(drops_conv[name])}
    # sanity (reference + stub) for every candidate, then decontaminate, then the seeded caps
    with ThreadPoolExecutor(workers) as ex:
        for name, rows in cand.items():
            res = list(ex.map(sanity_python, rows))
            ok = []
            drops = Counter()
            dropped_ids: list[dict] = []
            for r, (why, info) in zip(rows, res):
                r["_sanity"] = info
                if why is None:
                    ok.append(r)
                else:
                    drops[why.split(":")[0]] += 1
                    dropped_ids.append({"task_id": r["task_id"], "reason": why})
            rep["sources"][name]["sanity_drops"] = dict(drops)
            rep["sources"][name]["sanity_dropped"] = dropped_ids
            rep["sources"][name]["n_after_sanity"] = len(ok)
            cand[name] = ok
    all_drops: list[dict] = []
    ign: Counter = Counter()
    for name in ("LeetCodeDataset", "FullStackBench", "BigCodeBench"):  # fixed priority for cross-source dedup
        kept, drops = decontaminate_rows(dec, cand[name], ignored=ign)
        cand[name] = kept
        rep["sources"][name]["decontam_drops"] = len(drops)
        all_drops.extend(drops)
        rep["sources"][name]["n_after_decontam"] = len(kept)
    caps = {"LeetCodeDataset": None, "FullStackBench": fsb_n, "BigCodeBench": bcb_n}
    out: list[dict] = []
    for name, rows in cand.items():
        if caps[name] is not None and len(rows) > caps[name]:
            rows = seeded_sample(rows, caps[name], seed, "py:" + name)
        rep["sources"][name]["n_final"] = len(rows)
        out.extend(rows)
    # prompt visibility of the gate test + hidden test counts
    for r in out:
        g = r["gate_payload"]["tests"]
        vis = unittest_gate_visible(r["prompt"], g) if r.get("_unittest") else gate_visible_in_prompt(r["prompt"], g)
        r["gate_visible_in_prompt"] = vis["visible"]
        r["gate_visible_detail"] = {"n_items": vis["n_items"], "n_shown": vis["n_shown"]}
        if r.get("_unittest"):   # BigCodeBench: the unit is the test method
            r["n_hidden_tests"] = count_test_methods(r["checker_payload"]["tests"])
            r["n_gate_assertions"] = count_test_methods(g)
        else:                    # assertion-style suites: individual assertions
            r["n_hidden_tests"] = count_assertions(r["checker_payload"]["tests"])
            r["n_gate_assertions"] = count_assertions(g)
    rep["decontam_dropped"] = all_drops
    rep["ngram_filter"] = dict(ign)
    return sorted(out, key=lambda r: r["task_id"]), rep


def build_math(src: dict, dec, seed: int, test_n: int, gsm_n: int) -> tuple[list[dict], dict]:
    rep: dict[str, Any] = {}
    m500 = {norm_ws(r["problem"]) for r in src["math500"]}
    rows_test, ex500, noans = [], 0, 0
    for r in src["math_test"]:
        if norm_ws(r["problem"]) in m500:
            ex500 += 1
            continue
        row = math_test_row(r["subject"], r["idx"], r)
        if row is None:
            noans += 1
        else:
            rows_test.append(row)
    rep["MATH_test"] = {"n_test": len(src["math_test"]), "excluded_in_MATH500": ex500, "no_boxed_answer": noans,
                        "n_candidates": len(rows_test)}
    gsm = [g for g in (gsm8k_test_row(i, r) for i, r in enumerate(src["gsm8k_test"])) if g]
    comp: list[dict] = []
    for repo, prefix in ARENA.items():
        for r in src["arena"][repo]:
            a = arena_row(prefix, repo, r)
            if a:
                comp.append(a)
    rep["competition_candidates"] = dict(Counter(r["source"] for r in comp))
    rep["competition_skipped"] = ARENA_SKIPPED
    # sanity first (gold self-verifies), then decontaminate, then seeded caps on MATH/GSM8K
    all_drops: list[dict] = []
    sanity_drops: list[dict] = []
    ign: Counter = Counter()
    pools = {"MATH-test": rows_test, "GSM8K": gsm, "competition": comp}
    for name, rows in pools.items():
        ok = []
        for r in rows:
            w = sanity_math(r)
            (sanity_drops.append({"task_id": r["task_id"], "reason": w}) if w else ok.append(r))
        kept, drops = decontaminate_rows(dec, ok, ignored=ign)
        all_drops.extend(drops)
        pools[name] = kept
        rep[name] = {**rep.get(name, {}), "n_after_sanity": len(ok), "n_after_decontam": len(kept),
                     "decontam_drops": len(drops)}
    pools["MATH-test"] = stratified_sample(pools["MATH-test"], test_n, seed, lambda r: r["_sub"])
    pools["GSM8K"] = seeded_sample(pools["GSM8K"], gsm_n, seed, "gsm8k-test")
    out = pools["MATH-test"] + pools["GSM8K"] + pools["competition"]
    for name in pools:
        rep[name]["n_final"] = len(pools[name])
    rep["sanity_dropped"] = sanity_drops
    rep["ngram_filter"] = dict(ign)
    rep["decontam_dropped"] = all_drops
    rep["math_test_final_by_subject"] = dict(Counter(r["_sub"].split("|")[0] for r in pools["MATH-test"]))
    return sorted(out, key=lambda r: r["task_id"]), rep


def _lean_reason(err: str) -> str:
    """The source dataset marks statements it could not typecheck by commenting them out ('-- Error: ...')."""
    if "unexpected token 'sorry'" in err:
        return "statement is commented out in the source dataset (marked '-- Error:' upstream); nothing to prove"
    return "does not elaborate under the pinned Mathlib: " + err[:160].replace("\n", " ")


def build_lean(lean_dir: Path, dec) -> tuple[list[dict], dict]:
    rows = read_jsonl(lean_dir / "tasks_lean.jsonl")
    san = json.loads((lean_dir / "lean_sanity.json").read_text())
    cand = []
    for r in rows:
        name = r["task_id"].rsplit("/", 1)[-1]
        cand.append({**r, "gate_payload": {}, "cluster": f"miniF2F/valid/{name}", "source": "cat-searcher/minif2f-lean4:validation",
                     "stratum": "lean4/miniF2F-valid"})
    ign: Counter = Counter()
    kept, drops = decontaminate_rows(dec, cand, ignored=ign)
    rep = {"ngram_filter": dict(ign), "n_split": san["n"], "n_elaborating": san["elaborating"],
           "excluded_not_elaborating": [{"task_id": e["task_id"], "reason": _lean_reason(e["error"])}
                                        for e in san["excluded"]],
           "decontam_dropped": drops, "n_final": len(kept), "mathlib": san.get("mathlib"),
           "lean_tasks_sha256_raw": san.get("tasks_sha256")}
    return sorted(kept, key=lambda r: r["task_id"]), rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get(
        "GWAYA_DATA_ROOT", os.path.expanduser("~/gwaya-data"))))
    ap.add_argument("--lean-dir", type=Path, default=None,
                    help="output dir of scripts/lean/build_minif2f_tasks.py --split validation "
                         "(default $GWAYA_DATA_ROOT/eval_sets_F_raw/lean)")
    ap.add_argument("--lean-a18", type=Path, default=ROOT / "results/gwenlaya_v4/lean_a18/tasks_lean.jsonl")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--only", choices=["python", "math", "lean"], nargs="*", default=["python", "math", "lean"])
    a = ap.parse_args(argv)
    root = a.data_root
    os.environ.setdefault("HF_HOME", str(root / "hf-home"))  # before huggingface_hub / datasets are imported
    lean_dir = a.lean_dir or (root / "eval_sets_F_raw" / "lean")
    a.out.mkdir(parents=True, exist_ok=True)
    paths = existing_sets(root, a.lean_a18)
    dec, sizes = build_decontaminator(paths)
    manifest: dict[str, Any] = {
        "addendum": "A21", "date": date.today().isoformat(), "seed": a.seed,
        "script": "scripts/data/build_eval_sets_F.py",
        "purpose": "fresh out-of-sample sets for the frozen Laya (docs/LAYA_FREEZE.txt); no model has generated for F",
        "reference_sets_screened_against": {
            n: {"path": str(p), "n_records": sizes[n], "sha256": sha256_file(p)} for n, p in paths.items()},
        "laya_router_rows_screened": {k: v for k, v in sizes.items() if k.startswith("laya:")},
        "decontamination_methods": "gwaya.decontaminate.Decontaminator (exact, py_ast, ngram13, jaccard>=0.8, "
                                   "math_norm/template/nums, test_io, id, lean_stmt) + cross-source dedup; any hit removes",
        "allowed_test_modules": "python stdlib + numpy (test code); references may need whatever the sandbox python has",
        "files": {}, "counts": {}, "reports": {},
    }
    report: dict[str, Any] = {}
    out_rows: dict[str, list[dict]] = {}
    if "python" in a.only:
        psrc = load_python_sources(root)
        manifest.setdefault("source_revisions", {}).update(psrc["revisions"])
        manifest.setdefault("source_files_sha256", {}).update(psrc["files"])
        rows, rep = build_python(psrc, dec, a.workers, N_FSB, N_BCB, a.seed)
        out_rows["python"], report["python"] = rows, rep
    if "math" in a.only:
        msrc = load_math_sources(root)
        manifest.setdefault("source_revisions", {}).update(msrc["revisions"])
        manifest.setdefault("source_files_sha256", {}).update(msrc["files"])
        rows, rep = build_math(msrc, dec, a.seed, N_MATH_TEST, N_GSM8K)
        out_rows["math"], report["math"] = rows, rep
        if msrc["problems"]:
            manifest.setdefault("source_problems", []).extend(msrc["problems"])
    if "lean" in a.only:
        rows, rep = build_lean(lean_dir, dec)
        out_rows["lean"], report["lean"] = rows, rep
        shutil.copyfile(lean_dir / "lean_sanity.json", a.out / "lean_sanity_F.json")
    for dom, rows in out_rows.items():
        ids = [r["task_id"] for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate task ids in {dom}"
        p = a.out / f"tasks_F_{dom}.jsonl"
        sha = dump(p, [clean(r) for r in rows])
        manifest["files"][p.name] = {"rows": len(rows), "sha256": sha}
        manifest["counts"][dom] = {"n": len(rows), "per_source": dict(Counter(r["source"] for r in rows)),
                                   "per_stratum": dict(Counter(r["stratum"] for r in rows))}
        if dom == "python":
            vis: dict[str, Counter] = defaultdict(Counter)
            for r in rows:
                vis[r["source"]][str(r["gate_visible_in_prompt"])] += 1
            manifest["counts"][dom]["gate_visible_in_prompt_by_source"] = {k: dict(v) for k, v in sorted(vis.items())}
            per_src: dict[str, list[int]] = defaultdict(list)
            for r in rows:
                per_src[r["source"]].append(r["n_hidden_tests"])
            manifest["counts"][dom]["hidden_tests_per_task_by_source"] = {
                k: {"min": min(v), "max": max(v), "mean": round(sum(v) / len(v), 1),
                    "unit": "test methods" if k.startswith("bigcode") else "assertions"}
                for k, v in sorted(per_src.items())}
            manifest["counts"][dom]["gate_tests_per_task"] = "the first assertion statement (LeetCodeDataset, FullStackBench) " \
                "or the first test method (BigCodeBench); the hidden suite is strictly larger for every kept task"
            manifest["counts"][dom]["stub_fails_hidden"] = all(
                (r.get("_sanity") or {}).get("stub_status") not in (None, "VERIFIED") for r in rows)
            manifest["counts"][dom]["stub_fails_gate"] = all(
                (r.get("_sanity") or {}).get("stub_gate_status") not in (None, "VERIFIED") for r in rows)
    if "lean" in out_rows:
        manifest["files"]["lean_sanity_F.json"] = {
            "note": "raw output of scripts/lean/build_minif2f_tasks.py --split validation (every statement elaborated "
                    "with sorry under the pinned Mathlib; excluded items with the Lean error)",
            "sha256": sha256_file(a.out / "lean_sanity_F.json")}
    # compact reports into the manifest; full drop lists in a separate file
    full = {d: rep for d, rep in report.items()}
    for d, rep in report.items():
        manifest["reports"][d] = json.loads(json.dumps(rep, default=str))
        for k in ("decontam_dropped", "sanity_dropped", "excluded_not_elaborating"):
            if k in manifest["reports"][d]:
                manifest["reports"][d][k] = f"{len(manifest['reports'][d][k])} items; see decontamination_report_F.json"
        if d == "python":
            for s in manifest["reports"][d]["sources"].values():
                if "sanity_dropped" in s:
                    s["sanity_dropped"] = f"{len(s['sanity_dropped'])} items; see decontamination_report_F.json"
    (a.out / "decontamination_report_F.json").write_text(json.dumps(full, indent=1, ensure_ascii=False, default=str))
    manifest["total_tasks"] = sum(len(r) for r in out_rows.values())
    (a.out / "manifest_F.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"counts": manifest["counts"], "files": manifest["files"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
