"""Program-of-thought math gate: unit tests (fake runner), real-sandbox tests, importer and Study integration."""
import inspect
import json
import shutil
import types

import pytest

from gwaya.domains import math_gate as mg
from gwaya.domains.task import CheckResult, Task
from gwaya.sandbox import is_sandbox_available
from scripts import import_remote_gens as irg
from scripts import run_study as rs

needs_sandbox = pytest.mark.skipif(not is_sandbox_available(), reason="needs a working bwrap sandbox")


def runner_returning(ok=True, out="", err="", timed_out=False):
    def run(cmd, **kw):
        return ok, out, err, timed_out
    return run


def gate(resp, pot="```python\nprint(1)\n```", **kw):
    return mg.check_math_gate(resp, pot, runner=runner_returning(**kw))


def test_verified_when_program_output_matches_boxed_answer_in_any_equivalent_form():
    assert gate(r"so \boxed{3/4}", out="0.75\n").status == "VERIFIED"
    assert gate(r"so \boxed{12}", out="working...\n12\n").status == "VERIFIED"  # last line is the answer
    assert gate(r"so \boxed{\frac{1}{2}}", out="1/2").status == "VERIFIED"


def test_mismatch_is_unverified_not_failed():
    r = gate(r"\boxed{5}", out="6")
    assert r.status == "UNVERIFIED" and r.evidence["reason"] == "mismatch"  # a wrong program is not a refutation


@pytest.mark.parametrize("resp,pot,kw,reason", [
    ("no boxed answer here", "```python\nprint(1)\n```", {}, "no_boxed_answer"),
    (r"\boxed{1}", None, {}, "no_program"),
    (r"\boxed{1}", "", {}, "no_program"),
    (r"\boxed{1}", "```python\nprint(1)\n```", {"ok": False, "err": "Traceback"}, "program_error"),
    (r"\boxed{1}", "```python\nprint(1)\n```", {"timed_out": True}, "program_timeout"),
    (r"\boxed{1}", "```python\nprint(1)\n```", {"out": "\n\n"}, "no_program_output"),
    (r"\boxed{1}", "```python\nprint(1)\n```", {"out": "the answer"}, "unparseable_output"),
])
def test_everything_else_is_unverified_with_a_reason(resp, pot, kw, reason):
    r = mg.check_math_gate(resp, pot, runner=runner_returning(**kw))
    assert r.status == "UNVERIFIED" and r.evidence["reason"] == reason


def test_the_gate_never_receives_a_gold_answer():
    assert list(inspect.signature(mg.check_math_gate).parameters)[:2] == ["response", "pot_text"]
    assert "gold" not in inspect.getsource(mg.check_math_gate).split("def check_math_gate")[1]


def test_unboxed_fallbacks_do_not_count():
    # the registered rule is boxed-only: a last-number answer must not be gated
    assert gate("the answer is 12", out="12").status == "UNVERIFIED"


@needs_sandbox
@pytest.mark.parametrize("program,resp,status", [
    ("print(2+3)", r"\boxed{5}", "VERIFIED"),
    ("from fractions import Fraction\nprint(Fraction(3,4))", r"\boxed{0.75}", "VERIFIED"),
    ("import sympy\nprint(sympy.Rational(1,3)*3)", r"\boxed{1}", "VERIFIED"),
    ("print(7)", r"\boxed{5}", "UNVERIFIED"),
    ("while True:\n    pass", r"\boxed{5}", "UNVERIFIED"),
    ("raise SystemExit(0)", r"\boxed{5}", "UNVERIFIED"),
])
def test_real_sandbox(program, resp, status):
    r = mg.check_math_gate(resp, "```python\n" + program + "\n```", timeout_s=4.0)
    assert r.status == status, r.evidence


@needs_sandbox
def test_program_cannot_read_host_files_or_reach_the_network():
    prog = ("import os\n"
            "try:\n    print(open('/etc/hostname').read().strip() or 99)\nexcept Exception:\n    print(5)\n")
    r = mg.check_math_gate(r"\boxed{5}", "```python\n" + prog + "\n```", timeout_s=4.0)
    assert r.status == "VERIFIED"  # the host file is invisible, so the fallback value was printed
    net = "import socket\ns=socket.socket()\ns.settimeout(1)\ntry:\n    s.connect(('1.1.1.1',80)); print(1)\nexcept Exception:\n    print(5)\n"
    assert mg.check_math_gate(r"\boxed{5}", "```python\n" + net + "\n```", timeout_s=6.0).status == "VERIFIED"


# ── importer ────────────────────────────────────────────────────────────────────────────────

def write(tmp_path):
    tasks = [{"domain": "math", "task_id": f"m{i}", "prompt": f"2+{i}? Put the final answer in \\boxed{{}}.",
              "checker_payload": {"answer": str(2 + i)}} for i in range(3)]
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    raw = [{"domain": "math", "task_id": f"m{i}", "prompt_tokens": 10, "completion_tokens": 5, "chunk": 0,
            "chunk_wall_s": 3.0, "text": f"print(2+{i})", "finish_reason": "stop", "mean_logprob": -0.1,
            "token_logprobs": [-0.1] * 5} for i in range(3)]
    (tmp_path / "raw_pot.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    ans = [{**r, "text": f"so \\boxed{{{2 + i}}}"} for i, r in enumerate(raw)]
    (tmp_path / "raw_ans.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ans))
    return tasks


def test_pot_import_uses_a_distinct_key_call_index_and_fenced_python(tmp_path):
    write(tmp_path)
    irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw_ans.jsonl", "small-bf16", "bf16", tmp_path / "run")
    irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw_pot.jsonl", "small-bf16", "bf16", tmp_path / "run", kind="pot")
    recs = rs.ChainedLog(tmp_path / "run" / "gens.jsonl").records
    pots = [r for r in recs if r.get("kind") == "pot"]
    ans = [r for r in recs if r.get("kind") != "pot"]
    assert len(pots) == len(ans) == 3 and all(r["key"].endswith("|pot") for r in pots)
    assert all(r["call_index"] == 1 and r["text"].startswith("```python\nprint(") for r in pots)
    for a in ans:
        p = next(r for r in pots if r["task_id"] == a["task_id"])
        assert p["seed"] == a["seed"] + 1 and p["key"] != a["key"]


# ── Study integration ───────────────────────────────────────────────────────────────────────

PLAN = {"stages": [], "seeds": {"eval_order": 1}, "generation_settings": {"max_new_tokens": {"math": 64}}}
STAGE = {"id": "T", "models": ["small-bf16", "big-bf16"], "quant": ["bf16"], "arms": []}


def build_study(tmp_path, arms, checker_runner):
    write(tmp_path)
    out = tmp_path / "study"
    for m, wrong in (("small-bf16", {"m1"}), ("big-bf16", set())):
        raw = [json.loads(l) for l in (tmp_path / "raw_ans.jsonl").read_text().splitlines()]
        for r in raw:
            if r["task_id"] in wrong:
                r["text"] = "so \\boxed{99}"  # small tier answers m1 wrongly
        (tmp_path / f"a_{m}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
        irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / f"a_{m}.jsonl", m, "bf16", out)
        irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw_pot.jsonl", m, "bf16", out, kind="pot")
    tasks = rs.load_tasks_jsonl(tmp_path / "tasks.jsonl")
    from gwaya.domains.checkers import check as base_check
    return rs.Study(stage=STAGE, plan=PLAN, tasks=tasks, models=["small-bf16", "big-bf16"], quants=["bf16"], arms=arms,
                    backend=types.SimpleNamespace(), out_dir=out, mode="score", checker=base_check)


@needs_sandbox
def test_study_gate_and_cascade_use_the_program_of_the_tier_that_answered(tmp_path):
    st = build_study(tmp_path, ["base", "gate_only", "gwenlaya"], None)
    st.run()
    rows = st.rows.records
    gate = {(r["model"], r["task_id"]): r["gate"] for r in rows if r["arm"] == "gate_only"}
    # the program prints 2+i; small tier answered m1 with 99 -> gate refuses it; everything else verifies
    assert gate[("small-bf16", "m0")] == "VERIFIED" and gate[("small-bf16", "m1")] == "UNVERIFIED"
    assert all(gate[("big-bf16", f"m{i}")] == "VERIFIED" for i in range(3))
    casc = {r["task_id"]: r for r in rows if r["arm"] == "gwenlaya"}
    assert casc["m0"]["tiers_invoked"] == ["small-bf16"] and casc["m0"]["answered"] is True
    assert casc["m1"]["tiers_invoked"] == ["small-bf16", "big-bf16"] and casc["m1"]["score"] == "VERIFIED"
    # program records are never scored as answers
    assert {r["model"] for r in rows if r["arm"] == "base"} == {"small-bf16", "big-bf16"}
    assert len([r for r in rows if r["arm"] == "base"]) == 6


@needs_sandbox
def test_prefetch_covers_math_gate_tasks_and_ignores_pot_records(tmp_path):
    st = build_study(tmp_path, ["gate_only"], None)
    n = st.prefetch_checks(2)
    # identical (task, payload, answer) checks are shared: the tiers answered m0 and m2 identically and m1 differently
    # -> 4 distinct scorer checks + 4 distinct gate checks; program-of-thought records add no check of their own
    assert n == 8 and st.checker.cache_size() == 8
    st.run()
    assert all(r["gate"] in ("VERIFIED", "UNVERIFIED") for r in st.rows.records)


def test_without_pot_records_the_math_gate_stays_unverified(tmp_path):
    write(tmp_path)
    out = tmp_path / "np"
    irg.import_gens(tmp_path / "tasks.jsonl", tmp_path / "raw_ans.jsonl", "small-bf16", "bf16", out)
    st = rs.Study(stage=STAGE, plan=PLAN, tasks=rs.load_tasks_jsonl(tmp_path / "tasks.jsonl"), models=["small-bf16"],
                  quants=["bf16"], arms=["gate_only"], backend=types.SimpleNamespace(), out_dir=out, mode="score",
                  checker=lambda t, x: CheckResult("VERIFIED" if t.checker_payload.get("answer") else "UNVERIFIED"))
    st.run()
    assert all(r["gate"] == "UNVERIFIED" for r in st.rows.records)


@needs_sandbox
def test_gated_arms_are_charged_for_the_program_generation(tmp_path):
    st = build_study(tmp_path, ["base", "gate_only", "gwenlaya"], None)
    st.run()
    gens = {(r["model"], r["task_id"]): r for r in st.gens.records if r.get("kind") != "pot"}
    pots = {(r["model"], r["task_id"]): r for r in st.gens.records if r.get("kind") == "pot"}
    for r in st.rows.records:
        if r["arm"] == "base":  # ungated arms pay nothing for the gate
            assert r.get("pot_gpu_s", 0.0) == 0.0
            assert r["gpu_s"] == pytest.approx(gens[(r["model"], r["task_id"])]["gpu_s"])
        if r["arm"] == "gate_only":
            k = (r["model"], r["task_id"])
            assert r["gpu_s"] == pytest.approx(gens[k]["gpu_s"] + pots[k]["gpu_s"])
        if r["arm"] == "gwenlaya":
            exp = sum(gens[(m, r["task_id"])]["gpu_s"] + pots[(m, r["task_id"])]["gpu_s"] for m in r["tiers_invoked"])
            assert r["gpu_s"] == pytest.approx(exp)
