"""Runtime wiring of the registered math gate (A10: program-of-thought re-execution) for answers without a gold answer.

What a VERIFIED verdict means here: the tier's boxed answer and the value printed by a program that THE SAME tier
wrote (greedy, sandboxed, no gold answer) are equivalent. It is agreement between two derivations by one model,
NOT a proof and not independent ground truth; a mismatch, a crash, a timeout or a missing boxed answer is
UNVERIFIED (never FAILED), because a sloppy program is not evidence that the answer is wrong.

Measured precision (descriptive, MATH-500 subset E, n = 500 math tasks, Qwen3.5 tiers only; nothing is known for
other models, other benchmarks or other prompt styles; E is easy for these tiers):
  * registered scorer (results/gwenlaya_v4/e_tpu_mathgate/numbers_e_tpu_mathgate.json, A3.*, and
    rescore_exploratory.json gate.verified / verified_but_wrong "old"):
      2B: coverage 0.288, precision 0.951 (137/144)   4B: coverage 0.454 [0.410, 0.498], precision 0.943 [0.911, 0.971]
      (214/227)   9B: coverage 0.468, precision 0.957 (224/234)
  * exploratory post hoc re-score with the corrected answer scorer (D62; rescore_exploratory.json gate "new"):
      2B 149 verified, 2 wrong (0.987)   4B 230 verified, 0 wrong   9B 236 verified, 0 wrong
      (rule-of-three 95% upper bound on the wrong-among-verified rate: 0.013 for 4B, 0.013 for 9B).
    Most registered VERIFIED-but-wrong rows were scorer notation misses, not wrong answers; the re-score was
    designed after seeing the outputs and is exploratory.

Lazy PoT (deliberate deviation from the registered cost model): the study charged one PoT generation per
consulted tier even when the answer had no boxed value and the gate could not run (160 of the 4B's 500 answers
were unboxed, mostly truncated). Here no program is generated when there is no boxed answer, so the runtime
cost is lower than the registered charge. Verdicts are unchanged: those rows were UNVERIFIED either way.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from gwaya.domains.math_check import extract_final_answer
from gwaya.domains.math_gate import check_math_gate
from gwaya.domains.task import CheckResult, Task

POT_DOMAIN = "math_pot"   # gwaya.generators._POT_DOMAIN: fenced python, own system prompt
# Same strip as scripts/tpu/gen_batch.py --kind pot: the program is asked for, not a boxed answer.
_BOXED_INSTRUCTION = re.compile(r"\s*Put the final answer in \\boxed\{\}\.?\s*$")


def pot_prompt(problem: str) -> str:
    """The problem text without the trailing boxed-answer instruction (the program prints its answer)."""
    return _BOXED_INSTRUCTION.sub("", problem)


def applies(task: Task) -> bool:
    """The PoT gate is for math tasks without a gold answer; the gold-answer path (benchmarks) is untouched."""
    return task.domain == "math" and "answer" not in task.checker_payload


def check_math_pot(task: Task, response: str, generate: Callable[[str, str], str], *, timeout_s: float = 10.0,
                   clock: Callable[[], float] = time.perf_counter) -> tuple[CheckResult, dict[str, Any]]:
    """Gate ``response`` with a program written by ``generate`` (the response's own tier).
    Returns (result, pot) with pot = {"gen_s", "status", "reason"}; gen_s is 0.0 when no program was generated.
    Never raises: generation or sandbox failures are UNVERIFIED."""
    pot: dict[str, Any] = {"gen_s": 0.0, "status": "UNVERIFIED", "reason": None}
    if extract_final_answer(response, boxed_only=True) is None:  # lazy: no program for an answer the gate cannot compare
        res = check_math_gate(response, None)
        pot["reason"] = res.evidence.get("reason")
        return res, pot
    t0 = clock()
    try:
        pot_text = generate(pot_prompt(task.prompt), POT_DOMAIN)
    except Exception as exc:  # noqa: BLE001 - a dead generator is an unverified answer, not an error
        pot["gen_s"] = max(0.0, clock() - t0)
        pot["reason"] = "pot_generation_error"
        return CheckResult("UNVERIFIED", {"gate": "program_of_thought", "reason": "pot_generation_error",
                                          "error": f"{type(exc).__name__}: {exc}"[:300]}), pot
    pot["gen_s"] = max(0.0, clock() - t0)
    try:
        res = check_math_gate(response, pot_text, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 - fail closed on any sandbox/checker fault
        res = CheckResult("UNVERIFIED", {"gate": "program_of_thought", "reason": "gate_error",
                                         "error": f"{type(exc).__name__}: {exc}"[:300]})
    pot["status"], pot["reason"] = res.status, res.evidence.get("reason")
    return res, pot
