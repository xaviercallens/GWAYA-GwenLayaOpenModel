#!/usr/bin/env python3
"""GWAYA v3 oracle-gate evaluation on a small, author-constructed labelled corpus.

The gate under test is: ZeroStubAudit (Python/Lean) AND language oracle (python ast / rustc / lean).
`label` is the ground truth: should a *correct* verifier accept this candidate?
Cases tagged `limitation` are included deliberately to measure known blind spots
(semantic errors that a syntax/type checker cannot see, lexical false positives).

Writes results/gwaya_v3_oracle_eval.json. Requires real `lean` and `rustc` on PATH (else exits 2).
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import statistics
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anse.gwaya.oracles import (  # noqa: E402
    Lean4CompilerOracle,
    PythonCompilerOracle,
    RustCompilerOracle,
)
from anse.orchestration.dichotomic_decomposer import ZeroStubAudit  # noqa: E402

CORPUS: list[dict] = [
    # ---- Python ----
    {"id": "py_ok_sum", "lang": "python", "label": "accept", "tag": "valid",
     "code": "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"},
    {"id": "py_ok_walrus", "lang": "python", "label": "accept", "tag": "valid",
     "code": "def first_long(xs):\n    for x in xs:\n        if (n := len(x)) > 3:\n            return n\n    return 0\n"},
    {"id": "py_stub_pass", "lang": "python", "label": "reject", "tag": "stub",
     "code": "def solve(x):\n    pass\n"},
    {"id": "py_stub_ellipsis", "lang": "python", "label": "reject", "tag": "stub",
     "code": "def solve(x):\n    ...\n"},
    {"id": "py_stub_mock", "lang": "python", "label": "reject", "tag": "stub",
     "code": "def mock_solver(x):\n    return 42\n"},
    {"id": "py_empty", "lang": "python", "label": "reject", "tag": "empty", "code": ""},
    {"id": "py_comment_only", "lang": "python", "label": "reject", "tag": "empty",
     "code": "# TODO implement\n"},
    {"id": "py_docstring_only", "lang": "python", "label": "reject", "tag": "empty",
     "code": '"""Solution goes here."""\n'},
    {"id": "py_syntax_error", "lang": "python", "label": "reject", "tag": "syntax",
     "code": "def f(x)\n    return x\n"},
    {"id": "py_semantic_bug", "lang": "python", "label": "reject", "tag": "limitation",
     "code": "def add(a, b):\n    return a - b\n"},
    {"id": "py_spec_passed", "lang": "python", "label": "accept", "tag": "test_spec",
     "code": "def square(x):\n    return x * x\n",
     "test_spec": "assert square(4) == 16\nassert square(-3) == 9\n"},
    {"id": "py_spec_failed", "lang": "python", "label": "reject", "tag": "test_spec",
     "code": "def square(x):\n    return x * 2\n",
     "test_spec": "assert square(4) == 16\n"},
    # ---- Rust ----
    {"id": "rs_ok_add", "lang": "rust", "label": "accept", "tag": "valid",
     "code": "pub fn add(a: i32, b: i32) -> i32 { a + b }"},
    {"id": "rs_ok_generic", "lang": "rust", "label": "accept", "tag": "valid",
     "code": "pub fn largest<T: PartialOrd + Copy>(v: &[T]) -> Option<T> {\n"
             "    let mut it = v.iter();\n    let mut m = *it.next()?;\n"
             "    for &x in it { if x > m { m = x; } }\n    Some(m)\n}"},
    {"id": "rs_borrow_move", "lang": "rust", "label": "reject", "tag": "type",
     "code": "pub fn f() { let s = String::new(); let t = s; println!(\"{} {}\", s, t); }"},
    {"id": "rs_type_mismatch", "lang": "rust", "label": "reject", "tag": "type",
     "code": "pub fn f() -> i32 { \"text\" }"},
    {"id": "rs_syntax", "lang": "rust", "label": "reject", "tag": "syntax", "code": "pub fn f( {"},
    {"id": "rs_semantic_bug", "lang": "rust", "label": "reject", "tag": "limitation",
     "code": "pub fn add(a: i32, b: i32) -> i32 { a - b }"},
    # ---- Lean 4 (core only, no Mathlib) ----
    {"id": "lean_ok_comm", "lang": "lean", "label": "accept", "tag": "valid",
     "code": "theorem t (a b : Nat) : a + b = b + a := Nat.add_comm a b"},
    {"id": "lean_ok_simp", "lang": "lean", "label": "accept", "tag": "valid",
     "code": "theorem t (n : Nat) : n + 0 = n := by simp"},
    {"id": "lean_ok_decide", "lang": "lean", "label": "accept", "tag": "valid",
     "code": "theorem t : 2 + 2 = 4 := by decide"},
    {"id": "lean_false", "lang": "lean", "label": "reject", "tag": "type",
     "code": "theorem t : 1 = 2 := rfl"},
    {"id": "lean_sorry", "lang": "lean", "label": "reject", "tag": "escape",
     "code": "theorem t : 1 = 2 := by sorry"},
    {"id": "lean_axiom", "lang": "lean", "label": "reject", "tag": "escape",
     "code": "axiom bad : False\ntheorem t : 1 = 2 := bad.elim"},
    {"id": "lean_native_decide", "lang": "lean", "label": "reject", "tag": "escape",
     "code": "theorem t : 2 + 2 = 4 := by native_decide"},
    {"id": "lean_axiom_lexical_bypass", "lang": "lean", "label": "reject", "tag": "escape",
     "code": 'def s : String := "--" axiom bad : False\ntheorem t : 1 = 2 := bad.elim'},
    {"id": "lean_example_bypass", "lang": "lean", "label": "reject", "tag": "escape",
     "code": 'def s : String := "--" axiom bad : False\nexample : 1 = 2 := bad.elim'},
    # Found as a false rejection by the first run of this script; fixed by stripping comments
    # before the lexical scan. Kept as a regression case (not independent evidence).
    {"id": "lean_comment_fp", "lang": "lean", "label": "accept", "tag": "regression",
     "code": "-- this proof uses no axiom\ntheorem t (a b : Nat) : a + b = b + a := Nat.add_comm a b"},
]


def gate(case: dict, oracles: dict) -> tuple[bool, str]:
    lang, code = case["lang"], case["code"]
    test_spec = case.get("test_spec")
    if lang == "python":
        audit = ZeroStubAudit.audit_python_code(code)
        if not audit.is_clean:
            return False, "stub_audit: " + "; ".join(audit.violations)[:160]
        if test_spec:
            res = oracles["python"].verify_with_test(code, test_spec)
            return res.success, ("ok" if res.success else res.error_message.splitlines()[0][:160] if res.error_message else "fail")
    if lang == "lean":
        audit = ZeroStubAudit.audit_lean_code(code)
        if not audit.is_clean:
            return False, "stub_audit: " + "; ".join(audit.violations)[:160]
    res = oracles[lang].verify_snippet(code)
    return res.success, ("ok" if res.success else res.error_message.splitlines()[0][:160] if res.error_message else "fail")


def tool_version(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()  # nosec B603


def main() -> int:
    if not (shutil.which("lean") and shutil.which("rustc")):
        print("lean and rustc are required; refusing to report unverified results", file=sys.stderr)
        return 2
    repeats = 5
    oracles = {"python": PythonCompilerOracle(), "rust": RustCompilerOracle(), "lean": Lean4CompilerOracle()}
    rows, lat = [], {"python": [], "rust": [], "lean": []}
    for case in CORPUS:
        verdicts = set()
        for _ in range(repeats):
            t0 = time.perf_counter()
            ok, reason = gate(case, oracles)
            lat[case["lang"]].append((time.perf_counter() - t0) * 1000.0)
            verdicts.add(ok)
        accepted = verdicts == {True}
        rows.append({"id": case["id"], "lang": case["lang"], "tag": case["tag"], "label": case["label"],
                     "accepted": accepted, "deterministic": len(verdicts) == 1,
                     "correct": accepted == (case["label"] == "accept"), "reason": reason})

    def cm(sel):
        tp = sum(r["accepted"] and r["label"] == "accept" for r in sel)
        tn = sum(not r["accepted"] and r["label"] == "reject" for r in sel)
        fa = sum(r["accepted"] and r["label"] == "reject" for r in sel)
        fr = sum(not r["accepted"] and r["label"] == "accept" for r in sel)
        return {"n": len(sel), "true_accept": tp, "true_reject": tn, "false_accept": fa, "false_reject": fr}

    def pct(xs, q):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))], 2)

    in_scope = [r for r in rows if r["tag"] != "limitation"]
    out = {
        "corpus_sha256": hashlib.sha256(json.dumps(CORPUS, sort_keys=True).encode()).hexdigest(),
        "repeats_per_case": repeats,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "lean": tool_version(["lean", "--version"]), "rustc": tool_version(["rustc", "--version"])},
        "in_scope": cm(in_scope),
        "limitation_cases": cm([r for r in rows if r["tag"] == "limitation"]),
        "all_deterministic": all(r["deterministic"] for r in rows),
        "latency_ms": {k: {"n": len(v), "median": round(statistics.median(v), 2), "p95": pct(v, 0.95)}
                       for k, v in lat.items()},
        "cases": rows,
    }
    dest = ROOT / "results" / "gwaya_v3_oracle_eval.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("in_scope", "limitation_cases", "all_deterministic", "latency_ms")}, indent=2))
    for r in rows:
        print(f"{'OK ' if r['correct'] else 'ERR'} {r['id']:<20} accepted={r['accepted']!s:<5} {r['reason'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
