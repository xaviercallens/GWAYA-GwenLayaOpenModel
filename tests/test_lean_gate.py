"""Hardened Lean gate (gwaya/lean_gate.py, A18 amendment).

Unit tests need no Lean. TestRealGate re-creates every exploit of the internal Lean-gate audit against the real pinned
project (v4.7.0 + Mathlib a45ae637) and checks genuine miniF2F proofs; it skips without the project or bwrap.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from gwaya import lean_gate, lean_project, oracles
from gwaya.domains.checkers import check
from gwaya.domains.task import Task
from gwaya.lean_gate import (interpret_trusted, lexical_flaws, normalize_statement, parse_trusted,
                             statement_preserved, strip_lean, target_name)
from gwaya.sandbox import is_sandbox_available

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "results" / "gwenlaya_v4" / "lean_a18" / "tasks_lean.jsonl"
HEADER = (
    "import Mathlib.Algebra.BigOperators.Basic\nimport Mathlib.Data.Real.Basic\n"
    "import Mathlib.Data.Complex.Basic\nimport Mathlib.Data.Nat.Log\nimport Mathlib.Data.Complex.Exponential\n"
    "import Mathlib.NumberTheory.Divisors\nimport Mathlib.Data.ZMod.Defs\nimport Mathlib.Data.ZMod.Basic\n"
    "import Mathlib.Topology.Basic\nimport Mathlib.Data.Nat.Digits\n\nopen BigOperators\nopen Real\nopen Nat\nopen Topology"
)
S_HARD = "theorem numbertheory_4x3m7y3neq2003\n  (x y : ℤ) :\n  4 * x^3 - 7 * y^3 ≠ 2003 :="
S_AIME = ("theorem aime_1983_p1 (x y z w : ℕ) (ht : 1 < x ∧ 1 < y ∧ 1 < z) (hw : 0 ≤ w)\n"
          "    (h0 : Real.log w / Real.log x = 24) (h1 : Real.log w / Real.log y = 40)\n"
          "    (h2 : Real.log w / Real.log (x * y * z) = 12) : Real.log w / Real.log z = 60 :=")
S_TOY = "theorem toy_lt : (5:ℕ) < 3 :="
EVAL_EXIT0 = ('def a : String := "/-"\n#eval (do let h ← IO.FS.Handle.mk "/dev/stdout" IO.FS.Mode.append; '
              'h.putStrLn "\'numbertheory_4x3m7y3neq2003\' does not depend on any axioms"; h.flush; '
              'IO.Process.exit 0 : IO Unit)\ndef b : String := "-/"\n')

# Each audit exploit: (name, statement, candidate body). Bodies are answers without imports (the header is prepended).
EXPLOITS = {
    "comment_smuggle": (S_HARD, f"/- {S_HARD} -/\ntheorem easy : True := trivial"),
    "string_smuggle": (S_HARD, f'def s : String := "{S_HARD}"\ntheorem easy : True := trivial'),
    "variable_false": (S_HARD, f"variable (hF : False)\n{S_HARD} by\n  exact hF.elim"),
    "local_notation": (S_HARD, f'local notation:50 (priority := high) a:51 " ≠ " b:51 => (a = a ∧ b = b)\n{S_HARD} by\n'
                               "  exact ⟨rfl, rfl⟩"),
    "local_instance": (S_TOY, "def badLT : LT ℕ := ⟨fun _ _ => True⟩\nattribute [local instance high] badLT\n"
                              f"{S_TOY} trivial"),
    "priority_instance": (S_TOY, f"instance (priority := high) badLT : LT ℕ := ⟨fun _ _ => True⟩\n{S_TOY} trivial"),
    # aime_1983_p1 does not elaborate under the pinned header (no `Real.log` in scope); the shadow supplied one
    "namespace_shadow": (S_AIME, f"namespace Foo\ndef Real.log (_x : ℕ) : ℝ := 0\n{S_AIME} by\n  simp [Foo.Real.log] at h0"),
    "namespace_shadow_toy": (S_TOY, "namespace Foo\ninstance badLT : LT ℕ := ⟨fun _ _ => True⟩\n"
                                    f"{S_TOY} trivial\nend Foo"),
    "string_hidden_eval_exit0": (S_HARD, EVAL_EXIT0 + f"{S_HARD} by\n  intro h\n  omega"),
}


def task(stmt: str | None, **extra) -> Task:
    payload = {"header": HEADER, **extra}
    if stmt is not None:
        payload["formal_statement"] = stmt
    return Task("lean4", "t", "p", payload)


def fence(code: str) -> str:
    return f"```lean4\n{code}\n```"


# ── lexer / rules (no Lean) ───────────────────────────────────────────────────

def test_strip_handles_nested_comments_strings_and_chars():
    src = 'a /- x /- y -/ z -/ b -- c\n"s /- \\" -/" \'"\' h\' d'
    out, flaws = strip_lean(src)
    assert flaws == []
    assert "x" not in out and "z" not in out and "c" not in out and "s /-" not in out
    assert out.split() == ["a", "b", '"', '"', "'", "'", "h'", "d"]
    assert len(out) == len(src) and out.count("\n") == 1  # positions and lines kept


def test_strip_raw_strings_and_flags():
    out, _ = strip_lean('x r#"a "eval" b"# y r"\\" z')
    assert out.split() == ["x", '""', "y", '""', "z"]
    assert "unterminated_comment" in strip_lean("/- open")[1]
    assert "unterminated_string" in strip_lean('"open')[1]
    assert "interpolated_string" in strip_lean('s!"{x}"')[1]


def test_string_hidden_eval_is_now_seen():
    flaws = lexical_flaws(EVAL_EXIT0)
    assert "hash_command" in flaws and "io" in flaws
    # the old regex stripping removed `/-` ... `-/` across the strings and hid the #eval
    assert oracles.Lean4CompilerOracle._lexical_flaws(EVAL_EXIT0)


@pytest.mark.parametrize("bad, flaw", [
    ("variable (hF : False)", "variable"), ("local notation \"x\" => 1", "local"), ("notation \"x\" => 1", "notation"),
    ("infixl:65 \" +' \" => HAdd.hAdd", "infixl"), ("attribute [local instance] foo", "attribute"),
    ("instance (priority := high) badLT : LT ℕ := ⟨fun _ _ => True⟩", "instance"), ("namespace Foo", "namespace"),
    ("section", "section"), ("open Foo in", "open_in"), ("axiom a : False", "axiom"), ("opaque o : Nat", "opaque"),
    ("@[implemented_by f] def g := 1", "implemented_by"), ("@[extern \"c\"] def g := 1", "extern"),
    ("example : 1 = 1 := by native_decide", "native_decide"), ("Lean.ofReduceBool", "ofReduceBool"),
    ("unsafe def f := 1", "unsafe"), ("set_option debug.skipKernelTC true", "set_option debug.skipKernelTC"),
    ("#eval 1", "hash_command"), ("#exit", "hash_command"), ("run_cmd pure ()", "run_cmd"), ("elab \"x\" : term => 1", "elab"),
    ("macro \"x\" : term => `(1)", "macro"), ("notation3 \"x\" => 1", "notation3"), ("syntax \"x\" : term", "syntax"), ("by sorry", "sorry"), ("by admit", "admit"),
    ("open Lean Elab", "lean_meta"), ("export Nat (succ)", "export"), ("by_elab pure q(1)", "by_elab"), ("prelude", "prelude"),
])
def test_lexical_rejects(bad, flaw):
    assert flaw in lexical_flaws(f"{bad}\ntheorem t : 1 = 1 := rfl"), lexical_flaws(bad)


@pytest.mark.parametrize("proof", [
    "decide", "norm_num", "nlinarith [sq_nonneg (a - b), sq_nonneg (a + b)]", "omega", "simp", "field_simp", "ring_nf",
    "interval_cases n <;> simp_all", "induction n with\n  | zero => simp\n  | succ k ih => simp [ih]",
    "have h : (2:ℝ) = 2 := rfl\n  obtain ⟨a, b⟩ := h₀\n  exact h", "calc a = b := by rfl\n    _ = c := by rfl",
    "elab_as_elim_lemma", "set_option maxHeartbeats 400000 in\n  norm_num", "exact inferInstance", "positivity",
    "rcases h with ⟨x, hx⟩ <;> linarith -- a comment with sorry and #eval\n  /- also axiom here -/",
    "exact (by decide : ('a' : Char) ≠ 'b')", "simp only [Finset.sum_range_succ] at h ⊢", "exact h'",
])
def test_lexical_accepts_genuine_tactics(proof):
    assert lexical_flaws(f"{HEADER}\n\ntheorem t (a b : ℝ) : a = a := by\n  {proof}") == []


def test_named_priority_instance_is_not_anonymous_in_snippet_mode():
    code = "instance (priority := high) foo : Inhabited Nat := ⟨0⟩"
    assert "anonymous_declaration" not in oracles.Lean4CompilerOracle._lexical_flaws(code)
    assert "anonymous_declaration" in oracles.Lean4CompilerOracle._lexical_flaws("instance : Inhabited Nat := ⟨0⟩")
    assert lexical_flaws(code) == ["instance"]  # the hardened gate names the real reason


def test_target_name_and_statement_normalization():
    assert target_name(S_HARD) == "numbertheory_4x3m7y3neq2003"
    assert target_name("lemma foo.bar : True :=") == "foo.bar"
    assert target_name("/- theorem fake : True -/ theorem real' (x : ℕ) : x = x :=") == "real'"
    assert target_name("example : True :=") is None and target_name("") is None
    assert normalize_statement("theorem t : True := by sorry") == "theorem t : True :="
    assert normalize_statement("theorem t : True := sorry") == "theorem t : True :="
    assert normalize_statement("theorem t : True") == "theorem t : True :="


def test_statement_check_ignores_comments_and_strings():
    for name in ("comment_smuggle", "string_smuggle"):
        stmt, body = EXPLOITS[name]
        assert not statement_preserved(stmt, body), name
    assert statement_preserved(S_HARD, f"{S_HARD} by\n  omega")


def test_imports_and_header_detection():
    assert lean_gate.imports_of("-- import Fake\nimport Mathlib.Tactic\nimport «Foo».Bar") == ["Mathlib.Tactic", "Foo.Bar"]
    assert not lean_gate.has_imports("/- import Mathlib -/ theorem t : True := trivial")


# ── nonce parsing and reason mapping (no Lean) ────────────────────────────────

NONCE = "a" * 32


def _out(**kv: str) -> str:
    return "".join(f"GWAYA-{NONCE} {k} {v}\n" for k, v in kv.items())


def test_parse_trusted_reads_only_this_nonce():
    spoof = "GWAYA-" + "b" * 32 + " done 1\n'x' does not depend on any axioms\n"
    f = parse_trusted(spoof + _out(kind="theorem", levels="true", match="exact", axioms="propext", done="1"), NONCE)
    assert f == {"kind": "theorem", "levels": "true", "match": "exact", "axioms": "propext", "done": "1"}
    assert parse_trusted(_out(done="1") + _out(done="1"), NONCE) is None  # repeated key
    assert parse_trusted(f"GWAYA-{NONCE} bogus 1\n", NONCE) is None


@pytest.mark.parametrize("fields, status, reason", [
    (dict(kind="theorem", levels="true", match="exact", axioms="propext,Quot.sound", done="1"), "VERIFIED", "verified"),
    (dict(kind="theorem", levels="true", match="defeq", axioms="", done="1"), "VERIFIED", "verified"),
    (dict(kind="theorem", levels="true", match="none", axioms="", done="1"), "FAILED", "statement_mismatch"),
    (dict(kind="theorem", levels="false", match="exact", axioms="", done="1"), "FAILED", "statement_mismatch"),
    (dict(kind="other", levels="true", match="exact", axioms="", done="1"), "FAILED", "target_not_theorem"),
    (dict(kind="theorem", levels="true", match="exact", axioms="propext,sorryAx", done="1"), "FAILED", "nonstandard_axioms"),
    (dict(kind="theorem", levels="true", match="exact", axioms=""), "UNVERIFIED", "trusted_check_incomplete"),
    (dict(error="target_missing"), "FAILED", "target_theorem_missing"),
    (dict(error="replay"), "FAILED", "kernel_replay_failed"),
    (dict(error="import"), "UNVERIFIED", "trusted_check_error"),
    (dict(error="ref_missing"), "UNVERIFIED", "trusted_check_error"),
    ({}, "UNVERIFIED", "trusted_check_incomplete"),
])
def test_interpret_trusted(fields, status, reason):
    assert interpret_trusted(parse_trusted(_out(**fields), NONCE))[:2] == (status, reason)
    assert interpret_trusted(None)[:2] == ("UNVERIFIED", "trusted_check_malformed")


class _FakeProject:
    def __init__(self, tmp: Path):
        self.lean_bin = tmp / "tc" / "bin" / "lean"
        lib = tmp / "lib"
        for mod in ("Mathlib", "Mathlib/Data/Real/Basic"):
            (lib / mod).parent.mkdir(parents=True, exist_ok=True)
            (lib / f"{mod}.olean").write_bytes(b"")
        self.lean_path = (str(lib),)
        self.lean_path_env = str(lib)
        self.ro_binds = (str(tmp),)


def test_pipeline_verdict_comes_only_from_the_trusted_run(tmp_path, monkeypatch):
    """Candidate stdout carrying fake audit lines (even with a guessed tag) cannot verify; only the trusted run's
    nonce lines count, and a timeout is UNVERIFIED."""
    calls = []

    def fake_run(cmd, timeout_s, cpu_s, work_dir, files=None, ro_binds=None, env=None):
        calls.append(cmd)
        if "-o" in cmd:
            (Path(work_dir) / Path(cmd[cmd.index("-o") + 1]).name).write_bytes(b"olean")
            return True, "GWAYA-" + "0" * 32 + " done 1\n", "", False
        return True, fake["out"].replace(NONCE, cmd[3]), "", fake.get("timeout", False)

    good = _out(kind="theorem", levels="true", match="exact", axioms="propext", done="1")
    # a forged report under a guessed nonce, then the real (mismatching) one
    fake = {"out": good.replace(NONCE, "f" * 32) + _out(kind="theorem", levels="true", match="none", axioms="", done="1")}
    monkeypatch.setattr(lean_gate.sandbox, "run_in_sandbox", fake_run)
    lean_gate._REF_CACHE.clear()
    proj = _FakeProject(tmp_path)
    code = f"import Mathlib\n{S_TOY} by decide"
    v = lean_gate.verify_theorem(proj, code, S_TOY, "import Mathlib")
    assert (v.status, v.reason) == ("FAILED", "statement_mismatch")
    assert calls[-1][1:3] == ["--run", "/work/Check.lean"] and calls[-1][4] == "toy_lt"
    fake["out"] = good
    v = lean_gate.verify_theorem(proj, code, S_TOY, "import Mathlib")
    assert (v.status, v.reason, v.evidence["match"]) == ("VERIFIED", "verified", "exact")
    fake["timeout"] = True
    assert lean_gate.verify_theorem(proj, code, S_TOY, "import Mathlib").reason == "timeout"
    assert len([c for c in calls if "/work/Ref.olean" in c and "-o" in c]) == 1  # reference compiled once, then cached


def test_unknown_import_failed_but_missing_header_module_unverified(tmp_path, monkeypatch):
    monkeypatch.setattr(lean_gate.sandbox, "run_in_sandbox", lambda *a, **k: pytest.fail("must not run Lean"))
    proj = _FakeProject(tmp_path)
    v = lean_gate.verify_theorem(proj, f"import Mathlib.Tactic.NotARealModule\n{S_TOY} by decide", S_TOY, "import Mathlib")
    assert (v.status, v.reason, v.evidence["missing"]) == ("FAILED", "unknown_import", ["Mathlib.Tactic.NotARealModule"])
    v = lean_gate.verify_theorem(proj, f"import Mathlib.Gone\n{S_TOY} by decide", S_TOY, "import Mathlib.Gone")
    assert (v.status, v.reason) == ("UNVERIFIED", "lean_environment_missing_dependency")
    v = lean_gate.verify_theorem(None, f"{S_TOY} by decide", S_TOY)
    assert (v.status, v.reason) == ("UNVERIFIED", "lean_project_missing")


def test_check_lean4_without_statement_or_project(monkeypatch):
    assert check(task(None), fence("theorem easy : True := trivial")).evidence["reason"] == "no_formal_statement"
    assert check(task(None), fence("theorem easy : True := trivial")).status == "UNVERIFIED"
    assert check(task("   "), fence("theorem easy : True := trivial")).status == "UNVERIFIED"
    monkeypatch.setattr(lean_project, "default_project_dir", lambda: None)
    r = check(task(S_TOY), fence(f"{S_TOY} by decide"))
    assert (r.status, r.evidence["reason"]) == ("UNVERIFIED", "lean_project_missing")


@pytest.mark.parametrize("name", sorted(EXPLOITS))
def test_every_audit_exploit_is_rejected_before_lean(name, monkeypatch):
    """The full gate rejects each exploit lexically / by the stripped statement check (no Lean needed)."""
    monkeypatch.setattr(lean_gate.sandbox, "run_in_sandbox", lambda *a, **k: pytest.fail("must not run Lean"))
    stmt, body = EXPLOITS[name]
    r = check(task(stmt), fence(body))
    assert r.status == "FAILED" and r.evidence["reason"] in ("forbidden_construct", "statement_not_preserved"), r.evidence


# ── real Lean ─────────────────────────────────────────────────────────────────

_PROJ = lean_project.discover()
needs_mathlib = pytest.mark.skipif(_PROJ is None or not is_sandbox_available(),
                                   reason="pinned Lean+Mathlib project or bwrap not available")


def _real_task(name: str) -> Task:
    for line in TASKS.read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        if t["task_id"].endswith("/" + name):
            return Task("lean4", t["task_id"], t["prompt"], t["checker_payload"])
    raise KeyError(name)


@needs_mathlib
class TestRealGate:
    @pytest.mark.parametrize("name, proof", [
        ("mathd_algebra_478", "rw [h₁, h₂, h₃]\n  norm_num"),
        ("mathd_numbertheory_207", "norm_num"),
        ("mathd_numbertheory_342", "decide"),
        ("mathd_numbertheory_34", "omega"),
    ])
    def test_genuine_minif2f_proofs_verify(self, name, proof):
        t = _real_task(name)
        t0 = time.perf_counter()
        r = check(t, fence(f"{t.checker_payload['formal_statement']} by\n  {proof}"))
        print(f"{name}: {r.status} {time.perf_counter() - t0:.1f}s")
        assert r.status == "VERIFIED", r.evidence
        assert r.evidence["details"]["match"] == "exact"
        assert set(r.evidence["details"]["axioms"]) <= lean_gate.STANDARD_AXIOMS

    def test_wrong_proof_fails(self):
        t = _real_task("mathd_algebra_478")
        r = check(t, fence(f"{t.checker_payload['formal_statement']} by\n  norm_num"))
        assert (r.status, r.evidence["reason"]) == ("FAILED", "lean_error")

    def test_invented_import_and_timeout(self):
        r = check(task(S_TOY), fence(f"import FooBarLib\n{S_TOY} by decide"))
        assert (r.status, r.evidence["reason"]) == ("FAILED", "unknown_import")
        r = check(task(S_HARD, timeout_s=0.5), fence(f"{S_HARD} by\n  omega"))
        assert (r.status, r.evidence["reason"]) == ("UNVERIFIED", "timeout")

    @pytest.mark.parametrize("name, reason", [
        ("comment_smuggle", "target_theorem_missing"),
        ("variable_false", "statement_mismatch"),
        ("local_notation", "statement_mismatch"),
        ("local_instance", "statement_mismatch"),
        ("namespace_shadow", "reference_not_elaborated"),
        ("namespace_shadow_toy", "target_theorem_missing"),
        ("string_hidden_eval_exit0", "lean_error"),
    ])
    def test_trusted_check_alone_rejects_exploits(self, name, reason):
        """Defence in depth: with the lexical rules bypassed, the trusted second process still rejects each exploit."""
        stmt, body = EXPLOITS[name]
        v = lean_gate._trusted_pipeline(_PROJ, f"{HEADER}\n\n{body}", stmt, HEADER, target_name(stmt), 300.0, 4096,
                                        lambda s, r, log="", **ev: lean_gate.GateVerdict(s, r, ev, log))
        want = "UNVERIFIED" if reason == "reference_not_elaborated" else "FAILED"
        assert (v.status, v.reason) == (want, reason), (v.evidence, v.log[-600:])
