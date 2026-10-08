#!/usr/bin/env python3
"""C0-lite: build the primary-E manifest (python 542, rust 510, math 500) and the E-night tasks.

Stages (all CPU, no GCP):
  fetch    load the four datasets (HF cache), record HF revision SHA and license
  build    items with sha256, gate-visible vs hidden split, source-problem clusters
  sanity   harness-sanity exclusion (reference solution vs own checks; Rust: see D17)
  size     fix E-night n_d from measured tok/s and a seeded order
Everything is deterministic given the HF revisions. Pure helpers are unit tested offline in
tests/test_eval_manifest.py. Nothing here generates with a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVAL_ORDER_SEED = 20261007
PROPORTION = {"python": 542, "rust": 510, "math": 500}
MATH_SUFFIX = "\n\nPut the final answer in \\boxed{}."


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def item_sha(statement: str, tests: str) -> str:
    return sha256_text(normalize(statement) + "\n--tests--\n" + normalize(tests))


# ---- Python gate-visible split -------------------------------------------------------------

class _Src:
    """Fast ast node -> source text (ast.get_source_segment re-splits the whole file on every call)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.starts = [0]
        for ln in self.lines:
            self.starts.append(self.starts[-1] + len(ln))

    def off(self, lineno: int, col: int) -> int:  # ast col offsets are UTF-8 byte offsets
        line = self.lines[lineno - 1]
        if line.isascii():
            return self.starts[lineno - 1] + col
        return self.starts[lineno - 1] + len(line.encode("utf-8")[:col].decode("utf-8"))

    def span(self, node) -> tuple[int, int]:
        return self.off(node.lineno, node.col_offset), self.off(node.end_lineno, node.end_col_offset)

    def seg(self, node) -> str:
        a, b = self.span(node)
        return self.text[a:b]


def python_gate_tests_plus(test: str) -> str | None:
    """HumanEval+ `test` text whose `check()` CONTAINS ONLY the first input/expected pair.

    D25: the v1 builder only rewrote the loop header (`zip(inputs[:1], results[:1])`) and left the
    complete `inputs = [...]` / `results = [...]` literals in the gate text, so every hidden input and
    expected output was handed to the gate (and readable by a candidate via frame inspection). This
    version parses the test with `ast` and rewrites the two literals themselves to their first
    element, keeping the source text of that element verbatim. It fails closed (returns None, so the
    item is excluded as `tests_not_splittable`) when:
      * the text does not parse, or there is not exactly one top-level `def check`;
      * `check` does not assign `inputs` AND `results` exactly once each, as list literals with >= 1
        element and equal length;
      * the module defines `ref_func` or any other top-level function besides `is_floats`, `assertion`
        and `check` (a reference implementation would itself leak the hidden answers).
    """
    import ast
    try:
        mod = ast.parse(test)
    except SyntaxError:
        return None
    allowed_defs = {"is_floats", "assertion", "check"}
    defs = [n for n in mod.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if any(n.name not in allowed_defs for n in defs):
        return None
    checks = [n for n in defs if n.name == "check"]
    if len(checks) != 1:
        return None
    assigns: dict[str, list[ast.Assign]] = {"inputs": [], "results": []}
    for node in ast.walk(checks[0]):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in assigns:
                    if not isinstance(node, ast.Assign):
                        return None
                    assigns[t.id].append(node)
    if any(len(v) != 1 for v in assigns.values()):
        return None
    lits = {k: v[0].value for k, v in assigns.items()}
    if not all(isinstance(v, ast.List) and v.elts for v in lits.values()):
        return None
    if len(lits["inputs"].elts) != len(lits["results"].elts):
        return None
    src = _Src(test)
    edits = []
    for lst in lits.values():
        a, b = src.span(lst)
        edits.append((a, b, f"[{src.seg(lst.elts[0])}]"))
    out = test
    for a, b, rep in sorted(edits, reverse=True):
        out = out[:a] + rep + out[b:]
    # self-check: the rewritten text parses and both literals now hold exactly one element
    try:
        chk = [n for n in ast.parse(out).body if isinstance(n, ast.FunctionDef) and n.name == "check"][0]
    except (SyntaxError, IndexError):
        return None
    for node in ast.walk(chk):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in assigns:
            if not isinstance(node.value, ast.List) or len(node.value.elts) != 1:
                return None
    return out


def python_gate_leaks(gate_text: str, full_test: str, min_len: int = 8) -> list[str]:
    """Hidden elements of `inputs[1:]` / `results[1:]` (from the full test) whose repr appears in
    the gate text. Elements whose source is shorter than `min_len` characters or equal to the first
    element are skipped (e.g. `True`, `0`, which appear anyway). Empty list = no leak found."""
    import ast
    mod = ast.parse(full_test)
    src = _Src(full_test)
    chk = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "check"][0]
    leaks: list[str] = []
    for node in ast.walk(chk):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("inputs", "results") and isinstance(node.value, ast.List) and node.value.elts:
            first = src.seg(node.value.elts[0])
            for e in node.value.elts[1:]:
                s = src.seg(e)
                if len(s) >= min_len and s != first and s in gate_text:
                    leaks.append(s)
    return leaks


# ---- Rust gate-visible split -----------------------------------------------------------------

def _split_statements(body: str) -> list[str]:
    """Split a Rust block body into top-level statements on `;` (outside brackets, strings, chars)."""
    out, depth, i, start, n = [], 0, 0, 0, len(body)
    char_lit = re.compile(r"'(?:\\.[^']*|[^'\\])'")
    while i < n:
        c = body[i]
        if c == '"':
            i += 1
            while i < n and body[i] != '"':
                i += 2 if body[i] == "\\" else 1
        elif c == "'":
            m = char_lit.match(body, i)
            if m:
                i = m.end() - 1
        elif c == "/" and body.startswith("//", i):
            while i < n and body[i] != "\n":
                i += 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth == 0:
            out.append(body[start:i + 1].strip())
            start = i + 1
        i += 1
    tail = body[start:].strip()
    if tail:
        out.append(tail)
    return out


def rust_split_tests(tests: str) -> dict[str, Any] | None:
    """MultiPL-E Rust tests are `}\\n\\nfn main() {...}` (the leading `}` closes the prompt's
    open function). Returns {preamble, asserts, gate_tests, n_asserts} or None if unparsable."""
    m = re.search(r"\bfn\s+main\s*\(\s*\)\s*\{", tests)
    if not m:
        return None
    depth, end = 0, None
    for i in range(m.end() - 1, len(tests)):
        if tests[i] == "{":
            depth += 1
        elif tests[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None
    stmts = _split_statements(tests[m.end():end])
    asserts = [s for s in stmts if re.match(r"assert(?:_eq|_ne)?!", s)]
    pre = [s for s in stmts if not re.match(r"assert(?:_eq|_ne)?!", s)]
    if not asserts:
        return None
    head = tests[:m.start()]
    gate = head + "fn main() {\n    " + "\n    ".join(pre + asserts[:1]) + "\n}\n"
    return {"preamble": pre, "asserts": asserts, "gate_tests": gate, "n_asserts": len(asserts)}


def rust_stub_program(prompt: str, tests: str) -> str:
    """prompt (open fn) + unimplemented body + tests: used only for a typecheck of the harness."""
    return prompt.rstrip() + "\n    unimplemented!()\n" + tests


# ---- Item construction -----------------------------------------------------------------------

def cluster_of(domain: str, native_id: str) -> str:
    """Python task and its MultiPL-E translation share a source-problem cluster."""
    if domain == "python":
        return native_id                       # 'HumanEval/N' or 'MBPP/N'
    if domain == "rust":
        m = re.match(r"HumanEval_(\d+)_", native_id)
        if m:
            return f"HumanEval/{m.group(1)}"
        m = re.match(r"mbpp_(\d+)_", native_id)
        if m:
            return f"MBPP/{m.group(1)}"
        return f"rust:{native_id}"
    return f"MATH-500/{native_id}"


def python_item_he(r: dict[str, Any]) -> dict[str, Any]:
    ep = r["entry_point"]
    full = f"{r['test']}\nassert check({ep}) is None\n"
    gt = python_gate_tests_plus(r["test"])
    prompt = "Complete the following Python function. Reply with the full function.\n\n" + r["prompt"].rstrip()
    return {"domain": "python", "native_id": r["task_id"], "prompt": prompt, "statement": r["prompt"],
            "tests": full, "gate_tests": None if gt is None else f"{gt}\nassert check({ep}) is None\n",
            "reference": r["prompt"] + r["canonical_solution"], "source": "evalplus/humanevalplus"}


def python_item_mbpp(r: dict[str, Any]) -> dict[str, Any]:
    imports = "\n".join(r.get("test_imports") or [])
    pre = imports + "\n" if imports else ""
    first = r["test_list"][0]
    prompt = (f"{r['prompt'].strip()}\nYour code should pass this test:\n{first}\n"
              "Reply with the full Python function.")
    return {"domain": "python", "native_id": f"MBPP/{r['task_id']}", "prompt": prompt, "statement": r["prompt"],
            "tests": pre + r["test"], "gate_tests": pre + first,
            "reference": r["code"], "source": "evalplus/mbppplus"}


def rust_item(r: dict[str, Any], config: str) -> dict[str, Any]:
    sp = rust_split_tests(r["tests"])
    prompt = ("Complete the following Rust function. Reply with the full function including its "
              "signature, and no main.\n\n" + r["prompt"].rstrip())
    return {"domain": "rust", "native_id": r["name"], "prompt": prompt, "statement": r["prompt"],
            "tests": r["tests"], "gate_tests": None if sp is None else sp["gate_tests"],
            "n_asserts": 0 if sp is None else sp["n_asserts"], "reference": None,
            "source": f"nuprl/MultiPL-E:{config}"}


def math_item(r: dict[str, Any]) -> dict[str, Any]:
    return {"domain": "math", "native_id": r["unique_id"], "prompt": r["problem"].strip() + MATH_SUFFIX,
            "statement": r["problem"], "tests": str(r["answer"]), "gate_tests": "",
            "reference": r["solution"], "gold": str(r["answer"]), "source": "HuggingFaceH4/MATH-500"}


def task_id_of(it: dict[str, Any]) -> str:
    return {"python": "py/", "rust": "rs/", "math": "ma/"}[it["domain"]] + it["native_id"]


# ---- Sizing ----------------------------------------------------------------------------------

def apportion(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Largest-remainder apportionment of `total` proportional to weights."""
    s = sum(weights.values())
    raw = {k: total * v / s for k, v in weights.items()}
    out = {k: int(math.floor(x)) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (raw[k] - out[k], k), reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def size_n_total(tok_s: dict[str, float], mean_tokens_per_item: float, hours: dict[str, float],
                 margin: float = 0.85) -> dict[str, Any]:
    """Largest n such that every stage's expected generation time <= margin * its limit.
    tok_s: stage -> measured sequential tokens/s; hours: stage -> wall-clock limit (h)."""
    per = {s: int(math.floor(margin * hours[s] * 3600.0 * tok_s[s] / mean_tokens_per_item)) for s in hours}
    binding = min(per, key=per.get)
    return {"n_by_stage": per, "binding_stage": binding, "n_total": per[binding]}


# ---- Pooling ---------------------------------------------------------------------------------

def seeded_order(ids: list[str], seed: int = EVAL_ORDER_SEED, tag: str = "") -> list[str]:
    rng = random.Random(f"{seed}:{tag}")
    out = sorted(ids)
    rng.shuffle(out)
    return out


# ---- Driver ----------------------------------------------------------------------------------

def load_primary_e() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Primary E items (python 542, rust 510, math 500) plus per-source HF revision and license."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    meta: dict[str, Any] = {}
    for repo in ("evalplus/humanevalplus", "evalplus/mbppplus", "nuprl/MultiPL-E", "HuggingFaceH4/MATH-500"):
        info = api.dataset_info(repo)
        lic = (info.card_data.get("license") if info.card_data else None)
        if isinstance(lic, list):
            lic = ",".join(lic)
        meta[repo] = {"hf_revision": info.sha,
                      "license": lic if lic else "none on card (derived from MIT MATH); eval only"}
    items: list[dict[str, Any]] = []
    for r in load_dataset("evalplus/humanevalplus", split="test"):
        items.append(python_item_he(dict(r)))
    for r in load_dataset("evalplus/mbppplus", split="test"):
        items.append(python_item_mbpp(dict(r)))
    for cfg in ("humaneval-rs", "mbpp-rs"):
        for r in load_dataset("nuprl/MultiPL-E", cfg, split="test"):
            items.append(rust_item(dict(r), cfg))
    for r in load_dataset("HuggingFaceH4/MATH-500", split="test"):
        items.append(math_item(dict(r)))
    return items, meta


def sanity_one(it: dict[str, Any]) -> tuple[str, str | None]:
    """(task_id, exclusion reason or None). Uses the repo's own checkers (bwrap sandbox / rustc)."""
    import subprocess
    import tempfile
    from gwaya.domains.task import Task
    tid = task_id_of(it)
    d = it["domain"]
    if d == "python":
        from gwaya.domains.checkers import check_python
        if it["gate_tests"] is None:
            return tid, "tests_not_splittable"
        fenced = "```python\n" + it["reference"] + "\n```"
        for label, tests in (("hidden", it["tests"]), ("gate", it["gate_tests"])):
            res = check_python(Task("python", tid, "", {"tests": tests, "timeout_s": 30.0}), fenced)
            if res.status != "VERIFIED":
                return tid, f"reference_fails_{label}_checks:{res.status}"
        return tid, None
    if d == "rust":
        if it["gate_tests"] is None or it.get("n_asserts", 0) < 1:
            return tid, "tests_not_splittable"
        if it["n_asserts"] < 2:
            return tid, "single_assert_no_gate_hidden_split"
        for label, tests in (("hidden", it["tests"]), ("gate", it["gate_tests"])):
            with tempfile.TemporaryDirectory() as td:
                src = Path(td) / "m.rs"
                src.write_text(rust_stub_program(it["statement"], tests))
                pr = subprocess.run(["rustc", "--edition", "2021", "--emit=metadata", "-A", "warnings",
                                     "-o", str(Path(td) / "m.rmeta"), str(src)],
                                    capture_output=True, text=True, timeout=120)
            if pr.returncode != 0:
                return tid, f"harness_typecheck_fails_{label}"
        return tid, None
    from gwaya.domains.math_check import _last_boxed, answers_equivalent, check_math
    gold = it["gold"]
    if check_math("\\boxed{" + gold + "}", gold).status != "VERIFIED":
        return tid, "gold_not_self_verifying"
    sol = _last_boxed(it["reference"])
    if sol is None:
        return tid, "no_boxed_in_reference_solution"
    ok, _ = answers_equivalent(sol, gold)
    return tid, None if ok else "reference_solution_answer_differs_from_gold"


def mcnemar_power(n: int, p10: float, p01: float, alpha: float) -> float:
    """Power of the one-sided exact McNemar test (H1: B has the extra confident-wrong items) when
    each of n pairs falls in b10 w.p. p10, b01 w.p. p01. Exact enumeration, clustering ignored.
    Reproduces the registered table's method (docs/GWENLAYA_PREREGISTRATION.md section 9)."""
    from scipy.stats import binom
    pd = p10 + p01
    power = 0.0
    for d in range(1, n + 1):
        pd_d = binom.pmf(d, n, pd)
        if pd_d < 1e-15:
            continue
        # smallest x with P(Bin(d, .5) >= x) <= alpha
        xc = next((x for x in range(d + 1) if binom.sf(x - 1, d, 0.5) <= alpha), d + 1)
        power += pd_d * binom.sf(xc - 1, d, p10 / pd)
    return float(power)


def interleave(rows_by_domain: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge per-domain seeded lists so every prefix keeps the domain proportions
    (position = (rank + 0.5) / n_domain), ties broken by domain name."""
    keyed = []
    for dom, rows in rows_by_domain.items():
        for r, row in enumerate(rows):
            keyed.append(((r + 0.5) / len(rows), dom, row))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in keyed]


if __name__ == "__main__":
    raise SystemExit("driver: run scripts/data/run_c0_lite.py")
