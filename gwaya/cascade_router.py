"""
gwaya/cascade_router.py
=======================
Speculative Multi-Tier Cascade Router for GWAYA.

Empirically derived from NVIDIA RTX multi-model benchmark results:
- Qwen2.5-Coder-0.5B runs at 226.9 tok/s (~480 MB VRAM)
- Qwen2.5-Coder-1.5B runs at 135.1 tok/s (~1.2 GB VRAM)
- Qwen2.5-Coder-7.0B runs at 71.4 tok/s (~4.7 GB VRAM, 100% Gate Precision)

Instead of always incurring the latency and VRAM penalty of 7B, the Cascade Router:
1. Speculatively queries lightweight SLMs (0.5B/1.5B) at ultra-high throughput (226 tok/s).
2. Verifies candidates using AST ZeroStubAudit and isolated test harnesses.
3. On verification success, returns immediately (saving 70% VRAM and 3x latency).
4. On failure or stub detection, speculatively escalates to larger parameter tiers
   with full diagnostic feedback transferred to the higher tier model.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gwaya.generators import OllamaGenerator
from gwaya.low_tier_engine import (
    LowTierModelOptimizer,
    ModelTier,
    OptimizationResult,
    VerificationLevel,
)

logger = logging.getLogger("GwayaCascadeRouter")

DEFAULT_CASCADE_LADDER = [
    "qwen2.5-coder:0.5b",
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:7b",
]


@dataclass
class CascadeResult:
    goal: str
    selected_code: str
    verified: bool
    final_model: str
    ladder_models: list[str]
    escalated: bool
    tiers_evaluated: int
    duration_ms: float
    effective_tokens_per_s: float
    energy: float
    trajectory: list[dict[str, Any]]
    proof_token: str
    telemetry: dict[str, Any] = field(default_factory=dict)
    level: VerificationLevel = VerificationLevel.UNVERIFIED


class GwayaCascadeRouter:
    """
    Routes code synthesis across a graduated ladder of models, speculatively
    solving with small fast models and escalating to heavy models only when
    formal or functional verification fails.
    """

    def __init__(
        self,
        ladder: list[str] | None = None,
        generator_factory: Callable[[str], Callable[[str], str]] | None = None,
    ) -> None:
        self.ladder = ladder or list(DEFAULT_CASCADE_LADDER)
        self.generator_factory = generator_factory or self._default_factory

    @staticmethod
    def _default_factory(model_name: str) -> Callable[[str], str]:
        gen = OllamaGenerator(model=model_name)
        return gen

    @staticmethod
    def _model_to_tier(model_name: str) -> ModelTier:
        name_lower = model_name.lower()
        if "0.5b" in name_lower or "1.5b" in name_lower:
            return ModelTier.LOW
        elif "3b" in name_lower or "7b" in name_lower:
            return ModelTier.MID
        return ModelTier.HIGH

    def route_and_solve(
        self,
        goal: str,
        domain: str = "python",
        context: str = "",
        test_spec: str | None = None,
    ) -> CascadeResult:
        """
        Executes speculative cascading across the model ladder until a candidate
        is verified by the GWAYA gate or the ladder is exhausted.
        """
        t0 = time.perf_counter()
        trajectory: list[dict[str, Any]] = []
        last_result: OptimizationResult | None = None
        current_context = context

        for tier_idx, model_name in enumerate(self.ladder):
            tier = self._model_to_tier(model_name)
            gen_fn = self.generator_factory(model_name)
            optimizer = LowTierModelOptimizer(tier=tier, generator_fn=gen_fn)

            t_tier_start = time.perf_counter()
            opt_res = optimizer.optimize_and_solve(
                goal=goal,
                domain=domain,
                context=current_context,
                test_spec=test_spec,
            )
            tier_dur_ms = round((time.perf_counter() - t_tier_start) * 1000.0, 1)

            step_record = {
                "tier_index": tier_idx + 1,
                "model": model_name,
                "tier": str(tier),
                "verified": opt_res.verified,
                "level": str(opt_res.level),
                "energy": opt_res.energy,
                "duration_ms": tier_dur_ms,
                "repair_attempts": opt_res.repair_attempts,
                "candidates_evaluated": opt_res.candidates_evaluated,
                "violations": opt_res.violations_caught,
            }
            trajectory.append(step_record)
            last_result = opt_res

            # Early exit: If candidate passed all verification tests (or is well-formed with no test spec)
            if opt_res.verified or (not test_spec and opt_res.level == VerificationLevel.WELL_FORMED):
                total_duration_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                tok_s = self._estimate_tokens_per_s(model_name)

                return CascadeResult(
                    goal=goal,
                    selected_code=opt_res.selected_code,
                    verified=opt_res.verified,
                    final_model=model_name,
                    ladder_models=self.ladder,
                    escalated=tier_idx > 0,
                    tiers_evaluated=tier_idx + 1,
                    duration_ms=total_duration_ms,
                    effective_tokens_per_s=tok_s,
                    energy=opt_res.energy,
                    trajectory=trajectory,
                    proof_token=opt_res.proof_token,
                    level=opt_res.level,
                    telemetry={
                        "early_exit": True,
                        "saved_tiers": len(self.ladder) - (tier_idx + 1),
                        "speculative_speedup": f"{round(tok_s / 71.4, 2)}x" if tok_s else "1.0x",
                    },
                )

            # Failure: enrich context for the next tier with diagnostic trace
            diagnostic = f"\n[Cascade Note: Previous attempt by {model_name} failed with: {opt_res.violations_caught}]"
            current_context = (context + diagnostic)[:2400]

        total_duration_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        fallback_code = last_result.selected_code if last_result else "# UNVERIFIED: Cascade exhausted"
        fallback_token = last_result.proof_token if last_result else "unverified"
        fallback_energy = last_result.energy if last_result else 1_000_000.0
        fallback_level = last_result.level if last_result else VerificationLevel.UNVERIFIED

        return CascadeResult(
            goal=goal,
            selected_code=fallback_code,
            verified=False,
            final_model=self.ladder[-1] if self.ladder else "unknown",
            ladder_models=self.ladder,
            escalated=len(self.ladder) > 1,
            tiers_evaluated=len(self.ladder),
            duration_ms=total_duration_ms,
            effective_tokens_per_s=self._estimate_tokens_per_s(self.ladder[-1] if self.ladder else ""),
            energy=fallback_energy,
            trajectory=trajectory,
            proof_token=fallback_token,
            level=fallback_level,
            telemetry={"early_exit": False, "ladder_exhausted": True},
        )

    @staticmethod
    def _estimate_tokens_per_s(model_name: str) -> float:
        """RTX 2070 empirical token speed benchmarks."""
        name = model_name.lower()
        if "0.5b" in name:
            return 226.9
        elif "1.5b" in name:
            return 135.1
        elif "3b" in name:
            return 110.2
        elif "7b" in name:
            return 71.4
        return 100.0
