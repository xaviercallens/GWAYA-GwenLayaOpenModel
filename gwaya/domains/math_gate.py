"""Math gate: program-of-thought re-execution (registered design, docs/GWENLAYA_PREREGISTRATION.md l.166).

A math answer is VERIFIED by the gate iff a program written by the same model, run in the bwrap sandbox,
prints a value equivalent to the answer's final \\boxed{} value. The gold answer is never used.

Fail-closed: no boxed answer, no program, a crash, a timeout or an output that cannot be parsed are
UNVERIFIED; a mismatch is also UNVERIFIED (not FAILED), because a wrong or sloppy program is not evidence
that the answer is wrong. Agreement between two independent derivations is evidence of correctness, not proof:
the precision of this gate is measured, not assumed.
"""
from __future__ import annotations

import sys

from gwaya.domains.math_check import answers_equivalent, extract_final_answer
from gwaya.domains.task import CheckResult

MAX_OUTPUT_CHARS = 2000


def program_answer(stdout: str) -> str | None:
    """The program's answer = its last non-empty output line (programs may print working first)."""
    lines = [ln.strip() for ln in (stdout or "").strip().splitlines() if ln.strip()]
    return lines[-1][:MAX_OUTPUT_CHARS] if lines else None


def _extract_program(pot_text: str) -> str:
    from gwaya.low_tier_engine import extract_code_block
    return extract_code_block(pot_text or "", "python")


def check_math_gate(response: str, pot_text: str | None, *, timeout_s: float = 10.0, runner=None) -> CheckResult:
    ev: dict = {"gate": "program_of_thought"}
    ext = extract_final_answer(response, boxed_only=True)
    if ext is None:
        return CheckResult("UNVERIFIED", {**ev, "reason": "no_boxed_answer"})
    answer = ext[0]
    program = _extract_program(pot_text or "").strip()
    if not program:
        return CheckResult("UNVERIFIED", {**ev, "reason": "no_program"})
    if runner is None:
        from gwaya.sandbox import run_in_sandbox as runner
    ok, stdout, stderr, timed_out = runner([sys.executable, "-I", "/work/pot.py"], timeout_s=timeout_s, mem_mb=1024,
                                           cpu_s=int(timeout_s) + 5, files={"pot.py": program})
    if timed_out:
        return CheckResult("UNVERIFIED", {**ev, "reason": "program_timeout"})
    if not ok:
        return CheckResult("UNVERIFIED", {**ev, "reason": "program_error", "error": (stderr or "")[-200:]})
    out = program_answer(stdout)
    if out is None:
        return CheckResult("UNVERIFIED", {**ev, "reason": "no_program_output"})
    same, method = answers_equivalent(out, answer)
    ev.update(answer=answer, program_output=out, method=method)
    if same is True:
        return CheckResult("VERIFIED", ev)
    return CheckResult("UNVERIFIED", {**ev, "reason": "mismatch" if same is False else "unparseable_output"})
