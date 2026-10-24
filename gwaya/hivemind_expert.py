"""GWAYA as the ``Code`` expert of Project HiveMind (https://github.com/xaviercallens/HiveMind).

HiveMind routes a prompt to Top-k domain experts (``Season3DomainExpertRegistry.route_domain_query``). This module is the
serving-layer expert for the Python-coding branch: ``score`` is its routing weight, ``handle`` generates with a Q4 tier of
GWAYA-Lite, runs the visible tests in the local sandbox (``gwaya.assistant``) and escalates from the small tier to the large
one only when the tests fail. Verdicts stay honest: VERIFIED means "passed the visible tests", never "correct".

Standard library plus gwaya only, so HiveMind can ``pip install`` this repo and import it without torch. ``HIVEMIND_SPEC``
has the shape of one ``SEASON_3_DOMAIN_EXPERT_SPECS`` entry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from gwaya.assistant import Assistant, AssistResult
from gwaya.llama_tier import LlamaServerClient

DOMAIN_NAME = "Python_Code_GWAYA"
KEYWORDS = ("def ", "import ", "python", "pytest", "unittest", "traceback", "assert ", "lambda", "class ", "numpy",
            "pandas", "list comprehension", "decorator", "generator", "dataclass", "pip ", ".py", "typing")

HIVEMIND_SPEC: dict[str, Any] = {
    "dataset_id": "gwaya/python-verified-tasks",
    "lora_rank_r": 16,
    "lora_alpha": 32,
    "keywords": list(KEYWORDS),
    "description": "Python function-level generation and repair gated by AST stub audit and sandboxed tests (GWAYA-Lite).",
}


def keyword_hit(keyword: str, text_lower: str) -> bool:
    """Whole-word/phrase match, as HiveMind's ``hivemind.core.btm_moe._keyword_hit`` (no hit inside a longer word)."""
    kw = keyword.strip().lower()
    if not kw:
        return False
    head = r"(?<![a-z0-9_])" if (kw[0].isalnum() or kw[0] == "_") else ""
    tail = r"(?![a-z0-9_])" if (kw[-1].isalnum() or kw[-1] == "_") else ""
    return re.search(head + re.escape(kw) + tail, text_lower) is not None


def score(prompt: str) -> float:
    """Routing score on the same scale as ``route_domain_query`` (1.0 + 2.5 per whole-word keyword hit)."""
    low = prompt.lower()
    return 1.0 + 2.5 * sum(1 for kw in KEYWORDS if keyword_hit(kw, low))


@dataclass
class ExpertResult:
    verdict: str
    code: str
    tier: str
    escalated: bool
    attempts: list[AssistResult] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "code": self.code, "tier": self.tier, "escalated": self.escalated,
                "note": self.note, "attempts": [a.to_dict() for a in self.attempts]}


class GwayaCodeExpert:
    """``small`` and ``large`` are llama-server clients (e.g. 3B and 7B Q4). With ``small_first`` False (the default until
    the P1 measurement shows the cascade pays off) every request goes to ``large``."""

    domain_name = DOMAIN_NAME

    def __init__(self, small: LlamaServerClient | None, large: LlamaServerClient, small_first: bool = False,
                 max_rounds: int = 3, assistant_factory: Callable[..., Assistant] = Assistant) -> None:
        self.small, self.large, self.small_first = small, large, bool(small_first and small is not None)
        self._mk = lambda client: assistant_factory(client, max_rounds=max_rounds)

    score = staticmethod(score)

    def handle(self, task: str, tests: str = "") -> ExpertResult:
        ladder = [self.small, self.large] if self.small_first and tests.strip() else [self.large]
        attempts: list[AssistResult] = []
        for client in ladder:
            res = self._mk(client).solve(task, tests)
            attempts.append(res)
            if res.verdict != "FAILED":  # VERIFIED, or UNVERIFIED (generator down, gate unavailable): a bigger tier cannot help
                break
        last = attempts[-1]
        return ExpertResult(last.verdict, last.code, ladder[len(attempts) - 1].name, len(attempts) > 1, attempts, last.note)
