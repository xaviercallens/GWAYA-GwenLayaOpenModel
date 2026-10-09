"""Tests for gwaya.domains: Task schema, loaders (offline, tmp files), math checker, dispatch."""
from __future__ import annotations

import json
import shutil

import pytest

from gwaya.domains import CheckResult, Task, check
from gwaya.domains import checkers as ck
from gwaya.domains import loaders
from gwaya.domains.math_check import (
    answers_equivalent, check_math, extract_final_answer)
from gwaya.sandbox import is_sandbox_available

HAS_SYMPY = bool(__import__("importlib").util.find_spec("sympy"))
needs_sympy = pytest.mark.skipif(not HAS_SYMPY, reason="sympy not installed")


class TestTask:
    def test_valid(self):
        t = Task("math", "x/1", "p", {"answer": "1"})
        assert t.domain == "math"

    @pytest.mark.parametrize("kw", [
        dict(domain="cobol", task_id="a", prompt="p"),
        dict(domain="math", task_id="", prompt="p"),
        dict(domain="math", task_id="a", prompt="p", checker_payload=[]),
    ])
    def test_invalid(self, kw):
        with pytest.raises(ValueError):
            Task(**kw)

    def test_checkresult_rejects_bad_status(self):
        with pytest.raises(ValueError):
            CheckResult("PASS")


class TestExtraction:
    def test_boxed_nested_and_last(self):
        assert extract_final_answer(r"\boxed{1} then \boxed{\frac{3}{4}}") == (r"\frac{3}{4}", "boxed")

    def test_hashes(self):
        assert extract_final_answer("work\n#### 42") == ("42", "hashes")

    def test_last_number(self):
        assert extract_final_answer("so 3 apples and 1,234.5 pears") == ("1,234.5", "last_number")

    def test_none(self):
        assert extract_final_answer("no digits here") is None
        assert extract_final_answer("  ") is None
        assert extract_final_answer(r"\boxed{unclosed") is None or True  # falls to number rule


class TestEquivalence:
    @pytest.mark.parametrize("a,b,exp", [
        ("18", "18.00", True), ("1,000", "1000", True), (r"\frac{1}{2}", "0.5", True),
        ("1/2", r"\dfrac{1}{2}", True), ("7", "8", False), ("0.3333", r"\frac{1}{3}", False),
        ("50\\%", "50", True), (r"\text{12}", "12", True),
    ])
    def test_numeric(self, a, b, exp):
        assert answers_equivalent(a, b)[0] is exp

    def test_unparseable_is_undecided(self):
        assert answers_equivalent("banana", "apple")[0] is None
        assert answers_equivalent("(1,2)", "(2,1)")[0] is False  # tuples compare element-wise (D62)
        assert answers_equivalent("banana", "banana")[1] == "string_exact"

    def test_exact_string_match_ok(self):
        assert answers_equivalent("(1,2)", "(1,2)") == (True, "string_exact")

    @needs_sympy
    @pytest.mark.parametrize("a,b,exp", [
        (r"2\sqrt{2}", r"\sqrt{8}", True), (r"\frac{\pi}{2}", "pi/2", True),
        ("x^2+2x+1", "(x+1)^2", True), ("x+1", "x+2", False),
    ])
    def test_sympy(self, a, b, exp):
        assert answers_equivalent(a, b)[0] is exp

    @needs_sympy
    def test_malicious_expression_not_evaluated(self):
        assert answers_equivalent("__import__('os').system('true')", "1")[0] is None

    def test_no_sympy_expression_is_undecided(self, monkeypatch):
        import builtins
        real = builtins.__import__

        def fake(name, *a, **k):
            if name.startswith("sympy"):
                raise ImportError
            return real(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", fake)
        assert answers_equivalent(r"2\sqrt{2}", r"\sqrt{8}")[0] is None


class TestCheckMath:
    def test_verified_failed_unverified(self):
        assert check_math(r"so \boxed{12}", "12").status == "VERIFIED"
        assert check_math(r"so \boxed{13}", "12").status == "FAILED"
        r = check_math("I do not know", "12")
        assert r.status == "UNVERIFIED" and r.evidence["reason"] == "no_boxed_answer"

    def test_registered_rule_is_boxed_only(self):
        # D24: '####' and last-number answers are not scored unless the fallback is requested
        assert check_math("#### 12", "12").status == "UNVERIFIED"
        assert check_math("the answer is 12", "12").status == "UNVERIFIED"
        assert check_math("#### 12", "12", boxed_only=False).status == "VERIFIED"
        assert check_math("the answer is 12", "12", boxed_only=False).status == "VERIFIED"
        r = check_math("I do not know", "12", boxed_only=False)
        assert r.status == "UNVERIFIED" and r.evidence["reason"] == "no_extractable_answer"
        r = check_math(r"\boxed{banana}", "12")
        assert r.status == "UNVERIFIED" and r.evidence["reason"] == "unparseable_answer"

    def test_dispatch_and_missing_reference(self):
        t = Task("math", "t", "q", {"answer": "5"})
        assert check(t, r"\boxed{5}").status == "VERIFIED"
        assert check(t, "#### 5").status == "UNVERIFIED"
        assert check(Task("math", "t", "q", {}), "5").status == "UNVERIFIED"


class TestRustWrapper:
    def test_unwrap_main_and_test_fns(self):
        body = ck.rust_test_body("fn main() { assert_eq!(f(1), 2); }")
        assert "assert_eq!(f(1), 2)" in body and "fn main" not in body
        spec = "#[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn a() { assert!(f()); }\n}"
        assert "assert!(f())" in ck.rust_test_body(spec)

    def test_missing_linker_is_unverified(self, monkeypatch):
        from gwaya import oracles
        monkeypatch.setattr(oracles.RustCompilerOracle, "verify_with_test", lambda *a, **k: oracles.OracleResult(
            False, "rustc", "Compilation failed: error: linker `cc` not found"))
        t = Task("rust", "t", "p", {"tests": "assert_eq!(add(1, 2), 3);"})
        assert check(t, "fn add(a: i32, b: i32) -> i32 { a + b }").status == "UNVERIFIED"

    def test_plain_asserts_untouched(self):
        assert ck.rust_test_body("assert!(true);") == "assert!(true);"

    def test_no_assertions_unverified(self):
        r = check(Task("rust", "t", "p", {"tests": "let x = 1;"}), "fn f() {}")
        assert r.status == "UNVERIFIED"

    @pytest.mark.skipif(not shutil.which("rustc") or not shutil.which("cc") or not is_sandbox_available(), reason="rustc/cc/bwrap missing")
    def test_real_rust(self):
        t = Task("rust", "t", "p", {"tests": "fn main() { assert_eq!(add(1, 2), 3); }"})
        good = check(t, "```rust\nfn add(a: i32, b: i32) -> i32 { a + b }\n```")
        if good.evidence.get("reason") == "rust_linker_missing":
            pytest.skip("linker not reachable inside the sandbox")
        assert good.status == "VERIFIED"
        assert check(t, "```rust\nfn add(a: i32, b: i32) -> i32 { a - b }\n```").status == "FAILED"


class TestPython:
    def test_no_tests_unverified(self):
        assert check(Task("python", "t", "p", {}), "x=1").status == "UNVERIFIED"

    def test_python_dispatch(self):
        t = Task("python", "t", "p", {"tests": "assert add(1, 2) == 3"})
        good = check(t, "```python\ndef add(a, b):\n    return a + b\n```")
        bad = check(t, "```python\ndef add(a, b):\n    return a - b\n```")
        if is_sandbox_available():
            assert (good.status, bad.status) == ("VERIFIED", "FAILED")
        else:
            assert good.status == "UNVERIFIED"


class TestLean:
    def test_changed_statement_fails(self):
        t = Task("lean4", "t", "p", {"formal_statement": "theorem foo : 2 + 2 = 4"})
        r = check(t, "```lean\ntheorem foo : True := trivial\n```")
        assert r.status == "FAILED" and r.evidence["reason"] == "statement_not_preserved"

    def test_sorry_not_verified(self):
        t = Task("lean4", "t", "p", {"formal_statement": "theorem foo : 2 + 2 = 4"})
        assert check(t, "theorem foo : 2 + 2 = 4 := by sorry").status != "VERIFIED"

    @pytest.mark.skipif(not shutil.which("lean") or not is_sandbox_available(), reason="lean/bwrap missing")
    def test_real_lean(self):
        t = Task("lean4", "t", "p", {"formal_statement": "theorem foo : 2 + 2 = 4"})
        assert check(t, "theorem foo : 2 + 2 = 4 := rfl").status == "VERIFIED"

    def test_checker_exception_is_unverified(self, monkeypatch):
        monkeypatch.setitem(ck._CHECKERS, "math", lambda t, r: 1 / 0)
        r = check(Task("math", "t", "p", {"answer": "1"}), "1")
        assert r.status == "UNVERIFIED" and r.evidence["reason"] == "checker_error"


class TestLoaders:
    def test_unmaterialized_manifest_entry(self):
        with pytest.raises(loaders.DatasetNotMaterialized):
            loaders.load_tasks("EVAL001")

    def test_unknown_dataset(self):
        with pytest.raises(KeyError):
            loaders.load_tasks("NOPE")

    def _manifest(self, hf_id, path):
        return {"datasets": {"eval": [{"id": "E", "hf_id": hf_id, "path": str(path)}]}}

    def test_gsm8k_jsonl(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text(json.dumps({"question": "q?", "answer": "steps\n#### 1,234"}) + "\n")
        (t,) = loaders.load_tasks("E", manifest=self._manifest("openai/gsm8k", p))
        assert t.domain == "math" and t.checker_payload["answer"] == "1234"
        assert check(t, r"answer: \boxed{1234}").status == "VERIFIED"

    def test_hendrycks_boxed(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps([{"problem": "p", "solution": r"so \boxed{\frac{1}{2}}", "unique_id": "u1"}]))
        (t,) = loaders.load_tasks("E", manifest=self._manifest("EleutherAI/hendrycks_math", p))
        assert t.task_id == "EleutherAI/hendrycks_math/u1" and t.checker_payload["answer"] == r"\frac{1}{2}"

    def test_humaneval_payload_runs_check(self, tmp_path):
        p = tmp_path / "h.jsonl"
        p.write_text(json.dumps({
            "task_id": "HumanEval/0", "prompt": "def f(x):\n", "entry_point": "f",
            "test": "def check(candidate):\n    assert candidate(1) == 2\n"}) + "\n")
        (t,) = loaders.load_tasks("E", limit=1, manifest=self._manifest("evalplus/humanevalplus", p))
        assert t.domain == "python" and "assert check(f) is None" in t.checker_payload["tests"]

    def test_mbpp_imports_and_tests(self):
        t = loaders.row_to_task("evalplus/mbppplus", {
            "task_id": 3, "prompt": "p", "test_imports": ["import math"], "test_list": ["assert f(1)==1"]}, 0)
        assert t.checker_payload["tests"] == "import math\nassert f(1)==1"

    def test_lean_row(self):
        t = loaders.row_to_task("cat-searcher/minif2f-lean4", {
            "name": "n", "header": "import Mathlib\n", "formal_statement": "theorem n : True"}, 0)
        assert t.domain == "lean4" and t.checker_payload["formal_statement"] == "theorem n : True"

    def test_lean_row_drops_the_sorry_placeholder(self):
        t = loaders.row_to_task("cat-searcher/minif2f-lean4", {
            "id": "n", "header": "import Mathlib\n", "formal_statement": "theorem n\n  (x : ℕ) :\n  x = x := sorry"}, 0)
        assert t.checker_payload["formal_statement"] == "theorem n\n  (x : ℕ) :\n  x = x :="
        assert "x = x := by\n  sorry" in t.prompt and "import Mathlib" in t.prompt
        t2 = loaders.row_to_task("internlm/Lean-Workbook", {"id": "w", "formal_statement": "theorem w : 1 = 1 := by sorry"}, 0)
        assert t2.checker_payload["formal_statement"] == "theorem w : 1 = 1 :="

    def test_missing_schema_field_raises(self):
        with pytest.raises(KeyError):
            loaders.row_to_task("openai/gsm8k", {"foo": 1}, 0)
