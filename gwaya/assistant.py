"""Local coding assistant: remote generator (any OpenAI-format endpoint, e.g. the TPU gateway) + local gate + repair.

The generator is remote; verification stays in this process (bwrap sandbox, your tests never leave the machine).
Per round: generate with the study prompt protocol -> extract code -> run the visible tests with
``gwaya.test_harness.run_candidate_tests`` -> on failure send the failing test indices back and ask for a fix.

Honest verdicts (no calibrated probability is claimed here):
  VERIFIED     the candidate passed every visible test in the sandbox (a weak gate: see docs on gate strength)
  FAILED       tests ran and the candidate still fails after ``max_rounds`` rounds; the last code is returned for reading
  UNVERIFIED   no tests were supplied, the sandbox was unavailable, or the generator died: the code is a suggestion
The mean/min log-prob of the first generation is reported as a *signal* only; the frozen Laya was calibrated on bf16
vLLM-TPU log-probs of Qwen3.5 and is not applied here.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from gwaya.llama_tier import LlamaServerClient, LlamaServerError, signals_from_logprobs
from gwaya.low_tier_engine import extract_code_block
from gwaya.test_harness import HarnessResult, run_candidate_tests

REPAIR_TEMPLATE = (
    "{task}\n\nYour previous solution:\n```python\n{code}\n```\n\n"
    "It failed the visible tests: {why}\nReturn the complete corrected solution only."
)


@dataclass
class Round:
    n: int
    gen_s: float
    code: str
    completion_tokens: int = 0
    finish_reason: str = ""
    passed: int | None = None
    total: int | None = None
    error: str = ""
    gate_s: float = 0.0


@dataclass
class AssistResult:
    verdict: str
    code: str
    rounds: list[Round] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _why(r: HarnessResult) -> str:
    bits = [r.error or "failed"]
    idx = (r.details or {}).get("failed_idx")
    if idx:
        bits.append(f"failing assert indices {idx[:8]}")
    fails = (r.details or {}).get("failures")
    if fails:
        bits.append("; ".join(str(f)[:160] for f in fails[:3]))
    return " | ".join(bits)[:600]


class Assistant:
    def __init__(self, client: LlamaServerClient, domain: str = "python", max_rounds: int = 3,
                 run_tests: Callable[..., HarnessResult] = run_candidate_tests, timeout_s: float = 10.0,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.client, self.domain, self.max_rounds = client, domain, int(max_rounds)
        self.run_tests, self.timeout_s, self.clock = run_tests, timeout_s, clock

    def solve(self, task: str, tests: str = "") -> AssistResult:
        res = AssistResult("UNVERIFIED", "")
        prompt, tests = task, tests or ""
        for n in range(1, self.max_rounds + 1):
            t0 = self.clock()
            try:
                text, lps = self.client.complete(prompt, self.domain, logprobs=n == 1)
            except LlamaServerError as exc:
                res.note = f"generator unavailable: {exc}"[:300]
                return res
            rd = Round(n, self.clock() - t0, extract_code_block(text, self.domain),
                       int(self.client.last.get("completion_tokens", 0)), str(self.client.last.get("finish_reason", "")))
            res.rounds.append(rd)
            res.code = rd.code
            if n == 1:
                res.signals = signals_from_logprobs(lps)
            if not tests.strip():
                res.note = "no tests supplied: nothing was checked"
                return res
            if not rd.code:
                res.verdict, rd.error = "FAILED", "empty code block"
                prompt = REPAIR_TEMPLATE.format(task=task, code="", why="no code was returned")
                continue
            g0 = self.clock()
            hr = self.run_tests(rd.code, tests, timeout_s=self.timeout_s)
            rd.gate_s, rd.passed, rd.total, rd.error = self.clock() - g0, hr.passed, hr.total, hr.error
            if hr.success:
                res.verdict = "VERIFIED"
                return res
            if hr.isolation == "none" or (hr.details or {}).get("reason") in {"no_isolation", "invalid_spec", "no_asserts"}:
                res.note = f"gate could not run: {hr.error}"[:300]
                res.verdict = "UNVERIFIED"
                return res
            res.verdict = "FAILED"
            prompt = REPAIR_TEMPLATE.format(task=task, code=rd.code, why=_why(hr))
        return res
