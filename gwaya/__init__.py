"""
GWAYA: Fail-Closed Gate & Real Toolchain Verification for LLM Code Generation.
Open-source repository: https://github.com/xaviercallens/GWAYA-GwenLayaOpenModel
Zenodo DOI: 10.5281/zenodo.23123923
Hugging Face: https://huggingface.co/datasets/callensxavier/gwaya-v3-verified-report
"""
from __future__ import annotations

from gwaya.ast_audit import ZeroStubAudit, ZeroStubAuditResult
from gwaya.exemplar_store import ExemplarStore
from gwaya.generators import (
    GenerationStats,
    GeneratorUnavailableError,
    OllamaGenerator,
)
from gwaya.low_tier_engine import (
    CandidateEvaluation,
    LowTierModelOptimizer,
    ModelTier,
    OptimizationResult,
    TierBudgetConfig,
    VerificationLevel,
    extract_code_block,
)
from gwaya.oracles import (
    Lean4CompilerOracle,
    OracleResult,
    PythonCompilerOracle,
    RustCompilerOracle,
)
from gwaya.test_harness import HarnessResult, isolation_available, run_candidate_tests

__version__ = "3.1.1"

__all__ = [
    "ZeroStubAudit",
    "ZeroStubAuditResult",
    "OracleResult",
    "PythonCompilerOracle",
    "RustCompilerOracle",
    "Lean4CompilerOracle",
    "run_candidate_tests",
    "isolation_available",
    "HarnessResult",
    "LowTierModelOptimizer",
    "ModelTier",
    "TierBudgetConfig",
    "VerificationLevel",
    "CandidateEvaluation",
    "OptimizationResult",
    "extract_code_block",
    "OllamaGenerator",
    "GeneratorUnavailableError",
    "GenerationStats",
    "ExemplarStore",
]
