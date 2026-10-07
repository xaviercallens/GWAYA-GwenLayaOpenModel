"""Common task schema and checker verdict for the multi-domain GwenLaya v4 plan."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DOMAINS = ("python", "rust", "lean4", "math")
STATUSES = ("VERIFIED", "FAILED", "UNVERIFIED")


@dataclass(frozen=True)
class Task:
    """One benchmark/training item. `checker_payload` holds what the domain checker needs
    (python/rust: `tests`; lean4: `formal_statement`; math: `answer`)."""
    domain: str
    task_id: str
    prompt: str
    checker_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"unknown domain {self.domain!r}; expected one of {DOMAINS}")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.prompt, str):
            raise ValueError("prompt must be a string")
        if not isinstance(self.checker_payload, dict):
            raise ValueError("checker_payload must be a dict")


@dataclass
class CheckResult:
    """status is VERIFIED (checker proved it), FAILED (checker refuted it) or UNVERIFIED
    (the checker could not decide: missing toolchain, unparseable answer, ...). UNVERIFIED is
    never counted as a pass."""
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"invalid status {self.status!r}")

    @property
    def verified(self) -> bool:
        return self.status == "VERIFIED"
