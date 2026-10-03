"""
anse/gwaya/low_tier_engine.py
=============================
GWAYA v3 Low-Tier Model Compound AI Optimizer & Verification Engine.

Addresses the core challenge of low-tier models (<=1B parameters, SLMs, or lightweight APIs):
1. Tendency to emit stubs ('pass', '...', 'unimplemented!()', 'sorry').
2. Vulnerability to syntax errors, phantom imports, and algorithmic degeneration.
3. Lack of zero-shot multi-step reasoning.

Architecture:
- Model Tier Profiles: LOW (<=1B/SLM), MID, HIGH.
- Atomic Decompose-Then-Verify: Bounded subtask decomposition to keep prompts strictly <800 tokens.
- Targeted Bounded Self-Repair Loop (N <= 3): Iterative repair using compiler & AST diagnostics.
- Best-of-N Candidate Generation with ANSE Energy Selection:
    E = w_t * latency_ms + w_c * cyclomatic_complexity + 1e6 * violations
  Stubs, syntax failures, or violated conservation laws receive E = 10^6 (Maximum Pain).
- Fail-Closed Guarantee: If no candidate satisfies the verification gate, returns UNVERIFIED
  with full diagnostic telemetry; never issues fake/mock approval.
"""
from __future__ import annotations

import ast
import enum
import hashlib
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gwaya.oracles import (
    Lean4CompilerOracle,
    OracleResult,
    PythonCompilerOracle,
    RustCompilerOracle,
)
from gwaya.ast_audit import ZeroStubAudit

logger = logging.getLogger("GwayaLowTierEngine")


class ModelTier(enum.StrEnum):
    LOW = "low"      # Local <=1B parameter SLMs, ModernBERT-149M, Flash-Lite
    MID = "mid"      # Flash, Qwen2.5-Coder-7B
    HIGH = "high"    # Pro, Claude, GPT-4


@dataclass
class TierBudgetConfig:
    max_tokens_per_leaf: int
    max_repair_attempts: int
    best_of_n_candidates: int
    enforce_atomic_decomposition: bool
    temperature: float


TIER_CONFIGS: dict[ModelTier, TierBudgetConfig] = {
    ModelTier.LOW: TierBudgetConfig(
        max_tokens_per_leaf=600,
        max_repair_attempts=3,
        best_of_n_candidates=3,
        enforce_atomic_decomposition=True,
        temperature=0.1,
    ),
    ModelTier.MID: TierBudgetConfig(
        max_tokens_per_leaf=1200,
        max_repair_attempts=2,
        best_of_n_candidates=2,
        enforce_atomic_decomposition=False,
        temperature=0.2,
    ),
    ModelTier.HIGH: TierBudgetConfig(
        max_tokens_per_leaf=2400,
        max_repair_attempts=1,
        best_of_n_candidates=1,
        enforce_atomic_decomposition=False,
        temperature=0.2,
    ),
}


class VerificationLevel(enum.StrEnum):
    """How much evidence backs a selected candidate (LT0).

    Only VERIFIED_TESTS justifies ``verified=True``: user-supplied tests ran in the
    isolated harness and every assert passed. WELL_FORMED means the candidate parsed /
    compiled, has no stubs and resolves its imports, which says nothing about whether it
    solves the task.
    """

    VERIFIED_TESTS = "verified_tests"    # user-supplied tests executed and all passed
    SELF_TESTS_ONLY = "self_tests_only"  # only model-written tests passed (weak evidence)
    WELL_FORMED = "well_formed"          # compiles / parses, no stubs, imports resolve
    UNVERIFIED = "unverified"            # nothing valid was produced


@dataclass
class CandidateEvaluation:
    code: str
    energy: float
    is_valid: bool
    violations: list[str]
    compiler: str
    oracle_error: str = ""
    latency_ms: float = 0.0
    cyclomatic_complexity: int = 1
    tests_passed: int = 0
    tests_total: int = 0

    @property
    def level(self) -> VerificationLevel:
        if not self.is_valid:
            return VerificationLevel.UNVERIFIED
        if self.tests_total > 0 and self.tests_passed == self.tests_total:
            return VerificationLevel.VERIFIED_TESTS
        return VerificationLevel.WELL_FORMED


@dataclass
class OptimizationResult:
    goal: str
    selected_code: str
    tier: ModelTier
    verified: bool
    energy: float
    repair_attempts: int
    candidates_evaluated: int
    duration_ms: float
    violations_caught: list[str]
    proof_token: str
    telemetry: dict[str, Any] = field(default_factory=dict)
    level: VerificationLevel = VerificationLevel.UNVERIFIED

    @property
    def is_verified(self) -> bool:
        return self.verified


def extract_code_block(raw_text: str, domain: str = "python") -> str:
    """
    Extracts executable code from model responses, stripping markdown fences
    and conversational preamble/postscript.
    """
    text = raw_text.strip()
    if not text:
        return ""

    # Check for fenced code block matching domain
    pattern = rf"```(?:{domain}|py|rust|lean4|lean)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()

    # Check for generic code block
    generic_matches = re.findall(r"```\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if generic_matches:
        return generic_matches[0].strip()

    # If starting with ``` without closing
    if text.startswith("```"):
        lines = text.splitlines()
        return "\n".join(lines[1:]).strip()

    return text


class LowTierModelOptimizer:
    """
    Optimizes and hardens low-tier models using ANSE System 1 physical shields
    and iterative compiler-guided self-repair.
    """

    def __init__(
        self,
        tier: ModelTier = ModelTier.LOW,
        generator_fn: Callable[[str], str] | None = None,
        generator: Any = None,
    ) -> None:
        self.tier = tier
        self.config = TIER_CONFIGS[tier]
        if generator is not None and generator_fn is None:
            if hasattr(generator, "generate"):
                self.generator_fn = lambda p, g=generator: g.generate(p).code
            elif callable(generator):
                self.generator_fn = generator
        else:
            self.generator_fn = generator_fn
        self.python_oracle = PythonCompilerOracle()
        self.rust_oracle = RustCompilerOracle()
        self.lean_oracle = Lean4CompilerOracle()

    def _check_stubs(self, code: str, domain: str) -> list[str]:
        if domain == "lean4":
            audit = ZeroStubAudit.audit_lean_code(code)
        else:
            audit = ZeroStubAudit.audit_python_code(code)
        return [] if audit.is_clean else audit.violations

    def _check_python_syntax_and_complexity(self, code: str) -> tuple[int, str | None]:
        try:
            tree = ast.parse(code)
            branches = sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With))
            )
            return 1 + branches, None
        except SyntaxError as e:
            return 1, f"SyntaxError at line {e.lineno}: {e.msg}"

    def _run_oracle(self, code: str, domain: str) -> OracleResult:
        if domain == "rust":
            return self.rust_oracle.verify_snippet(code)
        elif domain == "lean4":
            return self.lean_oracle.verify_snippet(code)
        return self.python_oracle.verify_snippet(code)

    @staticmethod
    def _rejected(
        code: str, err: str, compiler: str, latency_ms: float, cc: int = 1,
        violations: list[str] | None = None, passed: int = 0, total: int = 0,
    ) -> CandidateEvaluation:
        return CandidateEvaluation(
            code=code,
            energy=1_000_000.0,
            is_valid=False,
            violations=violations if violations is not None else [err],
            compiler=compiler,
            oracle_error=err,
            latency_ms=latency_ms,
            cyclomatic_complexity=cc,
            tests_passed=passed,
            tests_total=total,
        )

    def _static_gate(
        self, code: str, domain: str, duration_ms: float
    ) -> tuple[CandidateEvaluation | None, int, OracleResult | None]:
        """Stub shield, syntax/complexity and compiler oracle. Returns (rejection, cc, oracle)."""
        stub_violations = self._check_stubs(code, domain)
        if stub_violations:
            err = "STUB_DETECTED: " + ", ".join(stub_violations)
            return self._rejected(code, err, "anti_stub_guard", duration_ms, violations=stub_violations), 1, None

        cc_score = 1
        if domain == "python":
            cc_score, syn_err = self._check_python_syntax_and_complexity(code)
            if syn_err:
                return self._rejected(code, syn_err, "python_ast", duration_ms), cc_score, None

        oracle_res = self._run_oracle(code, domain)
        if not oracle_res.success:
            err = oracle_res.error_message or "Compiler verification failed"
            latency = duration_ms + oracle_res.latency_ms
            return self._rejected(code, err, oracle_res.compiler, latency, cc_score), cc_score, oracle_res
        return None, cc_score, oracle_res

    def evaluate_candidate(
        self,
        code: str,
        domain: str = "python",
        duration_ms: float = 0.0,
        test_spec: str | None = None,
    ) -> CandidateEvaluation:
        """Evaluates a candidate snippet under the ANSE Energy Physics model."""
        cleaned_code = extract_code_block(code, domain)

        # 1-3. Anti-stub shield, syntax/complexity, domain compiler oracle
        rejection, cc_score, oracle_res = self._static_gate(cleaned_code, domain, duration_ms)
        if rejection is not None or oracle_res is None:
            return rejection  # type: ignore[return-value]

        # 4. Optional isolated functional tests (Python only for now)
        tests_passed = tests_total = 0
        test_latency = 0.0
        if domain == "python" and test_spec:
            test_res = self.python_oracle.verify_with_test(cleaned_code, test_spec)
            tests_passed = int(test_res.details.get("passed", 0) or 0)
            tests_total = int(test_res.details.get("total", 0) or 0)
            test_latency = test_res.latency_ms
            if not test_res.success:
                err = test_res.error_message or "Test specification assertion failed"
                latency = duration_ms + oracle_res.latency_ms + test_latency
                return self._rejected(
                    cleaned_code, err, test_res.compiler, latency, cc_score,
                    passed=tests_passed, total=tests_total,
                )

        # 5. Thermodynamic Physical Energy Scoring (Valid candidate)
        energy = 0.001 * (duration_ms + oracle_res.latency_ms) + 0.05 * cc_score + 0.1
        return CandidateEvaluation(
            code=cleaned_code,
            energy=round(energy, 4),
            is_valid=True,
            violations=[],
            compiler=oracle_res.compiler,
            latency_ms=round(duration_ms + oracle_res.latency_ms + test_latency, 2),
            cyclomatic_complexity=cc_score,
            tests_passed=tests_passed,
            tests_total=tests_total,
        )

    def _sample_batch(
        self, prompt: str, goal: str, domain: str, test_spec: str | None = None
    ) -> list[CandidateEvaluation]:
        if self.generator_fn is None:
            raise RuntimeError("LowTierModelOptimizer requires a generator_fn; none configured")
        cands: list[CandidateEvaluation] = []
        seen_hashes: set[str] = set()
        for i in range(self.config.best_of_n_candidates):
            t_cand_start = time.perf_counter()
            code_cand = self._generate_one(prompt, temperature=0.2 if i == 0 else 0.8)
            cand_dur = (time.perf_counter() - t_cand_start) * 1000.0
            if self._is_duplicate(code_cand, domain, seen_hashes):
                continue
            eval_res = self.evaluate_candidate(
                code_cand, domain=domain, duration_ms=cand_dur, test_spec=test_spec
            )
            cands.append(eval_res)
            if self._passes_all_public_tests(eval_res):
                break  # early exit
        return cands

    @staticmethod
    def _passes_all_public_tests(ev: CandidateEvaluation) -> bool:
        return ev.is_valid and ev.tests_total > 0 and ev.tests_passed == ev.tests_total

    def _generate_one(self, prompt: str, temperature: float) -> str:
        try:
            return self.generator_fn(  # type: ignore[misc,call-arg]
                prompt,
                temperature=temperature,
                max_tokens=self.config.max_tokens_per_leaf,
            )
        except TypeError:
            return self.generator_fn(prompt)  # type: ignore[misc]

    @staticmethod
    def _is_duplicate(code_cand: str, domain: str, seen_hashes: set[str]) -> bool:
        """True if an AST-identical candidate was already seen. Unparseable code is never a duplicate."""
        if domain != "python":
            return False
        try:
            tree = ast.parse(extract_code_block(code_cand, domain))
        except SyntaxError:
            return False  # evaluated normally; the oracle rejects it with E=1e6
        digest = hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            return True
        seen_hashes.add(digest)
        return False

    def _build_repair_prompt(
        self, base_prompt: str, worst_cand: CandidateEvaluation, attempt: int
    ) -> str:
        # Bound feedback length (approx 150 tokens ~ 600 chars)
        err_msg = worst_cand.oracle_error or "Detected incomplete code or stubs"
        if len(err_msg) > 500:
            err_msg = err_msg[:250] + "\n...[truncated]...\n" + err_msg[-250:]
            
        code_snip = worst_cand.code[:200]
        if len(worst_cand.code) > 200:
            code_snip += "\n...[truncated]..."
            
        return (
            f"{base_prompt}\n"
            f"--- REPAIR FEEDBACK (Attempt {attempt}) ---\n"
            f"Your previous solution was REJECTED with the following error:\n"
            f"{err_msg}\n\n"
            f"Previous rejected snippet:\n"
            f"```\n{code_snip}\n```\n"
            "Fix this issue immediately. Do NOT repeat the error and do NOT use any placeholders."
        )

    def _bound_context(self, context: str) -> tuple[str, bool]:
        char_budget = self.config.max_tokens_per_leaf * 4
        if self.config.enforce_atomic_decomposition and len(context) > char_budget:
            bounded = context[:char_budget] + "\n[Context bounded for low-tier leaf]"
            return bounded, True
        return context, False

    def _select_best_candidate(
        self,
        batch: list[CandidateEvaluation],
        current_best: CandidateEvaluation | None,
    ) -> CandidateEvaluation | None:
        best = current_best
        for cand in batch:
            if not cand.is_valid:
                continue
            
            # If we don't have a best yet, use this one
            if best is None:
                best = cand
                continue
                
            # Ranking logic:
            # 1. More tests passed is better
            if cand.tests_passed > best.tests_passed:
                best = cand
                continue
            elif cand.tests_passed < best.tests_passed:
                continue
                
            # 2. Lower cyclomatic complexity is better
            if cand.cyclomatic_complexity < best.cyclomatic_complexity:
                best = cand
                continue
            elif cand.cyclomatic_complexity > best.cyclomatic_complexity:
                continue
                
            # 3. Lower energy (runtime without generation latency) is better
            # Note: energy includes cc_score, but here we can just use energy since cc is tied
            if cand.energy < best.energy:
                best = cand
                
        return best

    def optimize_and_solve(
        self,
        goal: str,
        domain: str = "python",
        context: str = "",
        test_spec: str | None = None,
    ) -> OptimizationResult:
        """Executes bounded decompose-then-verify and iterative self-repair loop."""
        t0 = time.perf_counter()
        if self.generator_fn is None:
            return self._format_result(
                goal, None, 0.0, 0, 0, ["NO_GENERATOR: no model configured; nothing to verify"]
            )

        bounded_context, decomp_applied = self._bound_context(context)
        base_prompt = self._build_base_prompt(goal, domain, bounded_context, test_spec)
        current_prompt = base_prompt

        caught_violations: list[str] = []
        candidates_count = 0
        repair_attempts = 0
        best_candidate: CandidateEvaluation | None = None

        for attempt in range(self.config.max_repair_attempts):
            repair_attempts = attempt + 1
            batch = self._sample_batch(current_prompt, goal, domain, test_spec=test_spec)
            candidates_count += len(batch)
            best_candidate = self._select_best_candidate(batch, best_candidate)

            if best_candidate is not None:
                break
            if not batch:
                caught_violations.append("EMPTY_BATCH: generator returned no candidates")
                continue

            worst = batch[0]
            caught_violations.extend(worst.violations)
            current_prompt = self._build_repair_prompt(base_prompt, worst, attempt + 1)

        total_duration_ms = (time.perf_counter() - t0) * 1000.0
        res = self._format_result(
            goal, best_candidate, total_duration_ms, repair_attempts, candidates_count, caught_violations
        )
        if decomp_applied:
            res.telemetry["atomic_decomposition"] = True
            res.telemetry["max_tokens_per_leaf"] = self.config.max_tokens_per_leaf
        return res

    @staticmethod
    def _build_base_prompt(goal: str, domain: str, context: str, test_spec: str | None) -> str:
        prompt = f"Goal: {goal}\nDomain: {domain}\nRequirements: Output complete working code without stubs.\n"
        if context:
            prompt += f"Context:\n{context}\n"
        if test_spec:
            prompt += f"Verification Spec / Test:\n{test_spec}\n"
        return prompt

    def _format_result(
        self,
        goal: str,
        best_candidate: CandidateEvaluation | None,
        total_duration_ms: float,
        repair_attempts: int,
        candidates_count: int,
        caught_violations: list[str],
    ) -> OptimizationResult:
        if best_candidate and best_candidate.is_valid:
            level = best_candidate.level
            token_payload = f"{goal}:{best_candidate.energy}:{best_candidate.code[:64]}"
            proof_token = hashlib.sha256(token_payload.encode("utf-8")).hexdigest()
            return OptimizationResult(
                goal=goal,
                selected_code=best_candidate.code,
                tier=self.tier,
                verified=level == VerificationLevel.VERIFIED_TESTS,
                energy=best_candidate.energy,
                repair_attempts=repair_attempts,
                candidates_evaluated=candidates_count,
                duration_ms=round(total_duration_ms, 2),
                violations_caught=caught_violations,
                proof_token=proof_token,
                telemetry={
                    "compiler": best_candidate.compiler,
                    "cyclomatic_complexity": best_candidate.cyclomatic_complexity,
                    "status": "APPROVED" if level == VerificationLevel.VERIFIED_TESTS else "WELL_FORMED",
                    "tests_passed": best_candidate.tests_passed,
                    "tests_total": best_candidate.tests_total,
                },
                level=level,
            )

        token_payload = f"FAILED:{goal}:{time.time()}"
        proof_token = hashlib.sha256(token_payload.encode("utf-8")).hexdigest()
        fallback_code = (
            f"# UNVERIFIED: No valid candidate found after {repair_attempts} attempts\n"
            f"# Violations: {', '.join(caught_violations[:3])}\n"
        )
        return OptimizationResult(
            goal=goal,
            selected_code=fallback_code,
            tier=self.tier,
            verified=False,
            energy=1_000_000.0,
            repair_attempts=repair_attempts,
            candidates_evaluated=candidates_count,
            duration_ms=round(total_duration_ms, 2),
            violations_caught=caught_violations,
            proof_token=proof_token,
            telemetry={
                "status": "UNVERIFIED",
                "reason": "exhausted_repair_attempts",
                "violations": caught_violations,
            },
        )
