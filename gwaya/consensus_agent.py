"""
gwaya/consensus_agent.py
========================
GWAYA v4 pipeline for small local code models: sample, ground, test, repair, or abstain.

What it adds over the v3 best-of-N + repair optimizer (each piece can be switched off so the
benchmark can ablate it):

* ``use_selftests``  (H1) – candidates are ranked by agreement with model-written asserts that several
  independent candidates also pass, not only by the single public example;
* ``full_context_repair`` (H2) – the repair prompt shows the whole previous code and the *actual value*
  the failing assert produced, instead of 200 characters of code and a bare error;
* ``grounding`` (H3) – candidates that use undefined names, unresolvable imports or non-existent stdlib
  attributes are rejected before they can be selected;
* abstention (H4) – nothing is "verified" without a passing check; the answer then carries
  ``verified=False`` and the best attempt is returned as an unverified attempt, never as an answer.

The pipeline never reads hidden tests. ``AgentAnswer.level`` states exactly what was checked.
"""

from __future__ import annotations

import ast
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gwaya.grounding import grounding_flags
from gwaya.low_tier_engine import LowTierModelOptimizer, extract_code_block
from gwaya.oracles import PythonCompilerOracle
from gwaya.self_tests import build_selftest_prompt, consensus_asserts, parse_asserts
from gwaya.ast_audit import ZeroStubAudit

GeneratorFn = Callable[..., str]
LEVEL_PUBLIC_AND_CONSENSUS = "PUBLIC_TEST_AND_CONSENSUS"
LEVEL_PUBLIC = "PUBLIC_TEST"
LEVEL_CONSENSUS = "CONSENSUS_ONLY"
LEVEL_NONE = "UNVERIFIED"


@dataclass
class Candidate:
    code: str
    rejected: list[str] = field(default_factory=list)  # stub / syntax / grounding findings
    public_ok: bool | None = None  # None = no public test supplied or not executed
    public_error: str = ""
    selftest_pass: list[bool] = field(default_factory=list)
    round_index: int = 0

    @property
    def executable(self) -> bool:
        return not self.rejected


@dataclass
class AgentAnswer:
    verified: bool
    level: str
    code: str  # the verified answer, or the best UNVERIFIED attempt when verified is False
    rounds: int
    candidates: int
    trusted_selftests: int
    flags_seen: list[str]
    wall_s: float
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "VERIFIED" if self.verified else "UNVERIFIED"


class ConsensusRepairAgent:
    def __init__(
        self,
        generator_fn: GeneratorFn,
        oracle: PythonCompilerOracle | None = None,
        n_candidates: int = 5,
        repair_candidates: int = 3,
        max_repair_rounds: int = 3,
        max_tokens: int = 512,
        use_selftests: bool = True,
        full_context_repair: bool = True,
        grounding: bool = True,
    ) -> None:
        self.gen = generator_fn
        self.oracle = oracle or PythonCompilerOracle()
        self.n, self.repair_n, self.max_rounds, self.max_tokens = (
            n_candidates,
            repair_candidates,
            max_repair_rounds,
            max_tokens,
        )
        self.use_selftests, self.full_context, self.grounding = (
            use_selftests,
            full_context_repair,
            grounding,
        )
        self.asserts: list[str] = []
        self._seen: set[str] = set()

    # ── generation ────────────────────────────────────────────────────────────────
    def _sample(self, prompt: str, count: int, round_index: int, preamble: str) -> list[Candidate]:
        out: list[Candidate] = []
        for i in range(count):
            raw = self.gen(prompt, temperature=0.2 if i == 0 else 0.8, max_tokens=self.max_tokens)
            code = extract_code_block(raw, "python")
            if self._duplicate(code):
                continue
            out.append(
                Candidate(
                    code=code,
                    rejected=self._static_rejections(code, preamble),
                    round_index=round_index,
                )
            )
        return out

    def _duplicate(self, code: str) -> bool:
        try:
            digest = hashlib.sha256(ast.dump(ast.parse(code)).encode()).hexdigest()
        except SyntaxError:
            return False
        if digest in self._seen:
            return True
        self._seen.add(digest)
        return False

    def _static_rejections(self, code: str, preamble: str) -> list[str]:
        if not code.strip():
            return ["empty code"]
        audit = ZeroStubAudit.audit_python_code(code)
        if not audit.is_clean:
            return list(audit.violations)
        return (
            [f"grounding: {f}" for f in grounding_flags(code, preamble)] if self.grounding else []
        )

    # ── execution ─────────────────────────────────────────────────────────────────
    def _run(self, code: str, spec: str):
        return self.oracle.verify_with_test(code, spec, timeout_s=5.0)

    def _evaluate(self, cands: list[Candidate], public: str | None, preamble: str) -> None:
        for c in cands:
            if not c.executable:
                continue
            if public:
                res = self._run(c.code, public)
                c.public_ok, c.public_error = bool(res.success), res.error_message
            if self.asserts and (public is None or c.public_ok):
                self._run_selftests(c, preamble)

    def _run_selftests(self, c: Candidate, preamble: str) -> None:
        res = self._run(c.code, preamble + "\n".join(self.asserts) + "\n")
        failed = set(res.details.get("failed_idx", []))
        if res.details.get("total", len(self.asserts)) != len(
            self.asserts
        ) or res.error_message.startswith("IMPORT_FAILED"):
            failed = set(range(len(self.asserts)))
        c.selftest_pass = [i not in failed for i in range(len(self.asserts))]

    def _trusted(self, cands: list[Candidate], public: str | None) -> list[int]:
        pool = [
            c for c in cands if c.executable and c.selftest_pass and (public is None or c.public_ok)
        ]
        keep = set(consensus_asserts(self.asserts, [c.selftest_pass for c in pool]))
        return [i for i, a in enumerate(self.asserts) if a in keep]

    def _rank(
        self, cands: list[Candidate], trusted: list[int], public: str | None
    ) -> tuple[Candidate | None, bool]:
        best, best_key = None, (-1, -1)
        for c in cands:
            if not c.executable or (public is not None and not c.public_ok):
                continue
            n_ok = sum(1 for i in trusted if i < len(c.selftest_pass) and c.selftest_pass[i])
            if n_ok == len(trusted) and (public is not None or trusted):
                key = (1, n_ok)
                if key > best_key:
                    best, best_key = c, key
        return (best, True) if best else (None, False)

    # ── repair feedback ───────────────────────────────────────────────────────────
    def _actual_value(self, code: str, failing_assert: str, preamble: str) -> str:
        """Run the left side of ``assert LHS == RHS`` in the sandbox and report what it returned."""
        try:
            node = ast.parse(failing_assert).body[0]
            if not (
                isinstance(node, ast.Assert)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.ops[0], ast.Eq)
            ):
                return ""
            lhs = ast.unparse(node.test.left)
        except (SyntaxError, IndexError):
            return ""
        res = self._run(code, f"{preamble}assert False, repr({lhs})\n")
        text = res.error_message
        marker = "AssertionError: "
        if marker in text:
            return f"{lhs} returned {text.split(marker, 1)[1][:200]}"
        # Try to extract from failure details if available
        failures = res.details.get("failures", [])
        if failures and len(failures) > 0:
            # The first failure message should contain the AssertionError for our injected assert
            failure_msg = failures[0]
            if marker in failure_msg:
                return f"{lhs} returned {failure_msg.split(marker, 1)[1][:200]}"
            return f"evaluating {lhs} failed: {failure_msg[-200:]}"
        return f"evaluating {lhs} failed: {text[-200:]}"

    def _first_failing_assert(self, c: Candidate, public: str | None, trusted: list[int]) -> str:
        if public and c.public_ok is False:
            for line in public.splitlines():
                if line.strip().startswith("assert "):
                    return line.strip()
        for i in trusted:
            if i < len(c.selftest_pass) and not c.selftest_pass[i]:
                return self.asserts[i]
        return ""

    def _repair_prompt(
        self,
        base: str,
        c: Candidate,
        public: str | None,
        trusted: list[int],
        preamble: str,
        attempt: int,
    ) -> str:
        if c.rejected:
            why = "; ".join(c.rejected)[:400]
        elif c.public_ok is False and c.public_error:
            why = c.public_error[:500]
        else:
            why = "a consensus test failed"
        if not self.full_context:  # v3 behaviour, kept for the ablation
            if len(why) > 500:
                why = why[:250] + "\n...[truncated]...\n" + why[-250:]
            snip = c.code[:200] + ("\n...[truncated]..." if len(c.code) > 200 else "")
            return (
                f"{base}\n--- REPAIR FEEDBACK (Attempt {attempt}) ---\nYour previous solution was REJECTED with the "
                f"following error:\n{why}\n\nPrevious rejected snippet:\n```\n{snip}\n```\n"
                "Fix this issue immediately. Do NOT repeat the error and do NOT use any placeholders."
            )
        failing = self._first_failing_assert(c, public, trusted)
        actual = self._actual_value(c.code, failing, preamble) if failing and not c.rejected else ""
        detail = (
            f"Failing check: {failing}\nObserved: {actual}\n"
            if actual
            else (f"Failing check: {failing}\n" if failing else "")
        )
        return (
            f"{base}\n--- REPAIR (attempt {attempt}) ---\nYour previous solution:\n```python\n{c.code}\n```\n"
            f"It was rejected: {why}\n{detail}"
            "Rewrite the complete solution so it satisfies the task and every test. Use only names, imports and "
            "library functions that really exist. No placeholders."
        )

    # ── main entry ────────────────────────────────────────────────────────────────
    def solve(self, goal: str, public_test: str | None = None, preamble: str = "") -> AgentAnswer:
        t0 = time.perf_counter()
        self.asserts, self._seen = [], set()
        base = LowTierModelOptimizer._build_base_prompt(goal, "python", "", public_test)
        spec = (preamble + public_test) if public_test else None
        cands = self._sample(base, self.n, 0, preamble)
        if self.use_selftests:
            raw = self.gen(
                build_selftest_prompt(goal, public_test), temperature=0.4, max_tokens=300
            )
            self.asserts = parse_asserts(raw, public_test)
        self._evaluate(cands, spec, preamble)
        trusted = self._trusted(cands, spec)
        best, ok = self._rank(cands, trusted, spec)
        rounds = 0
        while not ok and rounds < self.max_rounds:
            rounds += 1
            worst = self._repair_target(cands, trusted, spec)
            new = self._sample(
                self._repair_prompt(base, worst, spec, trusted, preamble, rounds),
                self.repair_n,
                rounds,
                preamble,
            )
            self._evaluate(new, spec, preamble)
            cands.extend(new)
            trusted = self._trusted(cands, spec)
            best, ok = self._rank(cands, trusted, spec)
        return self._answer(cands, best, ok, trusted, rounds, spec, time.perf_counter() - t0)

    def _repair_target(
        self, cands: list[Candidate], trusted: list[int], spec: str | None
    ) -> Candidate:
        def score(c: Candidate) -> tuple[int, int, int]:
            n_ok = sum(1 for i in trusted if i < len(c.selftest_pass) and c.selftest_pass[i])
            return (int(c.executable), int(bool(c.public_ok)), n_ok)

        return max(cands, key=score) if cands else Candidate(code="")

    def _answer(self, cands, best, ok, trusted, rounds, spec, wall) -> AgentAnswer:
        flags = sorted({f for c in cands for f in c.rejected})
        if ok and best is not None:
            level = (
                (LEVEL_PUBLIC_AND_CONSENSUS if trusted else LEVEL_PUBLIC)
                if spec
                else LEVEL_CONSENSUS
            )
            code = best.code
        else:
            level = LEVEL_NONE
            attempt = self._repair_target(cands, trusted, spec)
            code = attempt.code
        return AgentAnswer(
            verified=ok and best is not None,
            level=level,
            code=code,
            rounds=rounds,
            candidates=len(cands),
            trusted_selftests=len(trusted),
            flags_seen=flags,
            wall_s=round(wall, 2),
            telemetry={
                "selftests_generated": len(self.asserts),
                "rejected": sum(1 for c in cands if c.rejected),
            },
        )
