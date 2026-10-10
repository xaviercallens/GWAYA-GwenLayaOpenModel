"""Math PoT gate wired into GwenLaya.answer: fake tiers, a fake sandbox runner, one real-sandbox test (no network)."""
import pytest

from gwaya import gwenlaya as g
from gwaya import math_pot_runtime as mpr
from gwaya.domains.task import Task
from gwaya.sandbox import is_sandbox_available

needs_sandbox = pytest.mark.skipif(not is_sandbox_available(), reason="needs a working bwrap sandbox")

Q = r"What is 2+2? Put the final answer in \boxed{}."
BOXED4 = r"Two plus two is \boxed{4}"
PROG = "```python\nprint(4)\n```"


class FakeTier:
    """Answers math from `answer` and math_pot from `pot` (a str or an Exception); records every call."""

    def __init__(self, name, answer=BOXED4, pot=PROG):
        self.name, self.answer, self.pot, self.calls = name, answer, pot, []

    def generate(self, prompt, domain):
        self.calls.append((prompt, domain))
        v = self.pot if domain == "math_pot" else self.answer
        if isinstance(v, Exception):
            raise v
        return v

    def tier(self):
        return g.Tier(self.name, self.generate)

    def n(self, domain):
        return sum(d == domain for _, d in self.calls)


class StepClock:
    def __init__(self, dt=1.0):
        self.t, self.dt = 0.0, dt

    def __call__(self):
        self.t += self.dt
        return self.t


@pytest.fixture
def fake_sandbox(monkeypatch):
    """Replace the bwrap runner used by check_math_gate; the fixture's .result is (ok, stdout, stderr, timed_out)."""
    import gwaya.sandbox as sb

    class R:
        result, progs = (True, "4\n", "", False), []

    def run(cmd, **kw):
        R.progs.append(kw["files"]["pot.py"])
        return R.result
    monkeypatch.setattr(sb, "run_in_sandbox", run)
    R.progs = []
    return R


def system(*tiers, **kw):
    return g.GwenLaya([t.tier() for t in tiers], clock=StepClock(), **kw)


def test_pot_prompt_strips_only_the_trailing_boxed_instruction():
    assert mpr.pot_prompt(Q) == "What is 2+2?"
    assert mpr.pot_prompt("Find \\boxed{x} usage. Put the final answer in \\boxed{}.\n") == "Find \\boxed{x} usage."
    assert mpr.pot_prompt("no instruction") == "no instruction"


def test_verified_without_gold_answer_and_evidence(fake_sandbox):
    t = FakeTier("t1")
    out = system(t).answer(Q, "math")
    assert out["verdict"] == "VERIFIED" and out["answer"] == BOXED4 and out["tier"] == "t1"
    assert t.calls == [(Q, "math"), ("What is 2+2?", "math_pot")]
    step = out["evidence"]["trajectory"][0]
    assert step["gate"]["status"] == "VERIFIED"
    ev = step["gate"]["evidence"]
    assert ev["gate"] == "program_of_thought" and ev["answer"] == "4" and ev["program_output"] == "4"
    assert step["pot"] == {"gen_s": 1.0, "status": "VERIFIED", "reason": None}
    assert step["gen_s"] == 1.0
    assert out["cost_s"] == 2.0  # answer generation + PoT generation, both charged to the tier
    assert fake_sandbox.progs == ["print(4)"]


def test_mismatch_is_unverified_not_failed(fake_sandbox):
    fake_sandbox.result = (True, "5\n", "", False)
    out = system(FakeTier("t1")).answer(Q, "math")
    step = out["evidence"]["trajectory"][0]
    assert out["verdict"] == "ABSTAIN" and out["answer"] is None
    assert step["gate"]["status"] == "UNVERIFIED" and step["gate"]["evidence"]["reason"] == "mismatch"
    assert step["gate"]["evidence"]["answer"] == "4" and step["gate"]["evidence"]["program_output"] == "5"
    assert step["pot"]["reason"] == "mismatch" and step["pot"]["status"] == "UNVERIFIED"


def test_no_boxed_answer_is_lazy_and_never_generates_a_program(fake_sandbox):
    t = FakeTier("t1", answer="The answer is 4")
    out = system(t).answer(Q, "math")
    step = out["evidence"]["trajectory"][0]
    assert out["verdict"] == "ABSTAIN"
    assert step["gate"]["evidence"] == {"gate": "program_of_thought", "reason": "no_boxed_answer"}
    assert t.n("math") == 1 and t.n("math_pot") == 0 and fake_sandbox.progs == []
    assert step["pot"] == {"gen_s": 0.0, "status": "UNVERIFIED", "reason": "no_boxed_answer"}
    assert out["cost_s"] == 1.0


@pytest.mark.parametrize("res,reason", [((False, "", "Traceback ... boom", False), "program_error"),
                                         ((False, "", "", True), "program_timeout")])
def test_program_crash_or_timeout_is_unverified(fake_sandbox, res, reason):
    fake_sandbox.result = res
    out = system(FakeTier("t1")).answer(Q, "math")
    step = out["evidence"]["trajectory"][0]
    assert out["verdict"] == "ABSTAIN"
    assert step["gate"]["status"] == "UNVERIFIED" and step["gate"]["evidence"]["reason"] == reason


def test_no_program_in_generation_is_unverified(fake_sandbox):
    out = system(FakeTier("t1", pot="   ")).answer(Q, "math")
    assert out["evidence"]["trajectory"][0]["gate"]["evidence"]["reason"] == "no_program"
    assert fake_sandbox.progs == []


def test_pot_generation_exception_is_unverified_and_costed(fake_sandbox):
    t = FakeTier("t1", pot=RuntimeError("ollama down"))
    out = system(t).answer(Q, "math")
    step = out["evidence"]["trajectory"][0]
    assert out["verdict"] == "ABSTAIN"
    assert step["gate"]["evidence"]["reason"] == "pot_generation_error"
    assert step["gate"]["evidence"]["gate"] == "program_of_thought"
    assert "RuntimeError" in step["gate"]["evidence"]["error"]
    assert step["pot"]["reason"] == "pot_generation_error" and step["pot"]["gen_s"] == 1.0
    assert out["cost_s"] == 2.0


def test_sandbox_exception_is_unverified(monkeypatch):
    import gwaya.sandbox as sb

    def boom(*a, **k):
        raise OSError("no bwrap")
    monkeypatch.setattr(sb, "run_in_sandbox", boom)
    out = system(FakeTier("t1")).answer(Q, "math")
    assert out["evidence"]["trajectory"][0]["gate"]["evidence"]["reason"] == "gate_error"
    assert out["verdict"] == "ABSTAIN"


def test_gold_answer_path_is_unchanged_and_generates_no_program(fake_sandbox):
    t = FakeTier("t1")
    out = system(t).answer(Q, "math", checker_payload={"answer": "4"})
    step = out["evidence"]["trajectory"][0]
    assert out["verdict"] == "VERIFIED" and t.n("math_pot") == 0 and "pot" not in step
    assert "gate" not in step["gate"]["evidence"] or step["gate"]["evidence"].get("gate") != "program_of_thought"
    assert fake_sandbox.progs == [] and out["cost_s"] == 1.0


def test_non_math_domain_never_uses_pot(fake_sandbox):
    t = FakeTier("t1", answer="```python\nx=1\n```")
    system(t, checker=lambda task, r: g.CheckResult("VERIFIED", {})).answer("p", "python")
    assert t.n("math_pot") == 0


def test_escalation_continues_across_tiers_when_pot_gate_fails(fake_sandbox):
    t1 = FakeTier("small", answer=r"\boxed{5}", pot="```python\nprint(4)\n```")   # wrong answer, mismatch
    t2 = FakeTier("large")
    out = system(t1, t2).answer(Q, "math")
    assert out["verdict"] == "VERIFIED" and out["tier"] == "large" and out["answer"] == BOXED4
    traj = out["evidence"]["trajectory"]
    assert traj[0]["gate"]["evidence"]["reason"] == "mismatch" and traj[0]["escalated_to"] == "large"
    assert traj[0]["pot"]["gen_s"] == 1.0 and traj[1]["pot"]["status"] == "VERIFIED"
    assert t1.n("math_pot") == 1 and t2.n("math_pot") == 1
    assert out["cost_s"] == 4.0
    # each tier's program is written by that tier, never by another one
    assert t2.calls[1][1] == "math_pot"


def test_budget_counts_pot_generation(fake_sandbox):
    fake_sandbox.result = (True, "9\n", "", False)
    out = system(FakeTier("a"), FakeTier("b"), budget_s=2.0).answer(Q, "math")
    assert out["verdict"] == "ESCALATE" and out["cost_s"] == 2.0  # answer 1s + PoT 1s exhausts the budget


def test_math_pot_false_restores_previous_behaviour(fake_sandbox):
    t = FakeTier("t1")
    out = system(t, math_pot=False).answer(Q, "math")
    step = out["evidence"]["trajectory"][0]
    assert out["verdict"] == "ABSTAIN"
    assert step["gate"] == {"status": "UNVERIFIED", "evidence": {"reason": "no_reference_answer"}}
    assert "pot" not in step and t.n("math_pot") == 0 and out["cost_s"] == 1.0


def test_injected_checker_takes_precedence(fake_sandbox):
    t = FakeTier("t1")
    out = system(t, checker=lambda task, r: g.CheckResult("VERIFIED", {"by": "custom"})).answer(Q, "math")
    assert out["verdict"] == "VERIFIED" and t.n("math_pot") == 0


def test_system_from_env_switch(monkeypatch):
    monkeypatch.setenv("GWENLAYA_TIERS", "m1")
    assert g.system_from_env().math_pot is True
    monkeypatch.setenv("GWENLAYA_MATH_POT", "0")
    assert g.system_from_env().math_pot is False


def test_applies_requires_missing_answer_key():
    assert mpr.applies(Task("math", "x", "p", {}))
    assert not mpr.applies(Task("math", "x", "p", {"answer": "1"}))
    assert not mpr.applies(Task("python", "x", "p", {}))


@needs_sandbox
@pytest.mark.parametrize("program,verdict,reason", [
    ("print(2+2)", "VERIFIED", None),
    ("print(5)", "UNVERIFIED", "mismatch"),
    ("raise SystemExit(3)", "UNVERIFIED", "program_error"),
])
def test_real_sandbox(program, verdict, reason):
    t = FakeTier("t1", pot=f"```python\n{program}\n```")
    out = system(t).answer(Q, "math")
    step = out["evidence"]["trajectory"][0]
    assert step["gate"]["status"] == verdict and step["gate"]["evidence"].get("reason") == reason
    assert step["gate"]["evidence"]["gate"] == "program_of_thought"
    assert (out["verdict"] == "VERIFIED") == (verdict == "VERIFIED")
