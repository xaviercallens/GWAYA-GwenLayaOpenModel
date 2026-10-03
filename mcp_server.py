#!/usr/bin/env python3
"""
GWAYA Model Context Protocol (MCP) Server.
==========================================
Provides autonomous AI agents (Claude Code, Antigravity, Cursor, Windsurf, OpenHands)
with direct access to GWAYA's fail-closed verification oracles, AST stub auditing,
bubblewrap sandboxed test execution, and Qwen low-tier model repair loop.

Usage:
    # Run with FastMCP CLI:
    fastmcp run mcp_server.py

    # Or directly with Python:
    python mcp_server.py
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import urllib.request
from typing import Any, Dict

from fastmcp import FastMCP

from gwaya.ast_audit import ZeroStubAudit
from gwaya.generators import OllamaGenerator
from gwaya.low_tier_engine import LowTierModelOptimizer, ModelTier
from gwaya.oracles import (
    Lean4CompilerOracle,
    PythonCompilerOracle,
    RustCompilerOracle,
)
from gwaya.test_harness import isolation_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("GwayaMCPServer")

mcp = FastMCP("gwaya-verification-engine")

_py_oracle = PythonCompilerOracle()
_rust_oracle = RustCompilerOracle()
_lean_oracle = Lean4CompilerOracle()


@mcp.tool()
def gwaya_system_status() -> Dict[str, Any]:
    """
    Check the operational status of all GWAYA verification toolchains, sandboxing, and Ollama server.
    Returns status for Python, Rust (rustc), Lean 4 (lean), Bubblewrap (bwrap), and Ollama.
    """
    status: Dict[str, Any] = {
        "python": {"available": True, "version": sys.version.split()[0]},
        "rustc": {"available": shutil.which("rustc") is not None},
        "lean4": {"available": shutil.which("lean") is not None},
        "bubblewrap_sandbox": {"available": isolation_available()},
        "ollama": {"running": False, "models": []},
    }

    if status["rustc"]["available"]:
        status["rustc"]["path"] = shutil.which("rustc")
    if status["lean4"]["available"]:
        status["lean4"]["path"] = shutil.which("lean")

    # Check Ollama connectivity
    ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags", headers={"User-Agent": "gwaya-mcp/3.1.1"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
            status["ollama"]["running"] = True
            status["ollama"]["models"] = [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        status["ollama"]["running"] = False
        status["ollama"]["error"] = str(exc)

    return status


@mcp.tool()
def gwaya_audit_stubs(language: str, code: str) -> Dict[str, Any]:
    """
    Perform an AST-level Zero-Stub physical hardness audit on code.
    Rejects mock identifiers (mock_*, fake_*, dummy_*, stub_*), pass statements,
    ellipsis literals (...), and formal proof holes (sorry, admit).

    Args:
        language: Language of the code ('python', 'lean', 'rust').
        code: Source code to audit.
    """
    lang = language.lower().strip()
    if lang in ("python", "py"):
        res = ZeroStubAudit.audit_python_code(code)
    elif lang in ("lean", "lean4"):
        res = ZeroStubAudit.audit_lean_code(code)
    elif lang in ("rust", "rs"):
        res = ZeroStubAudit.audit_rust_code(code)
    else:
        return {
            "is_clean": False,
            "violations": [f"Unsupported language '{language}' for stub audit."],
            "penalty_energy": 1e6,
        }

    return {
        "is_clean": res.is_clean,
        "violations": res.violations,
        "penalty_energy": res.penalty_energy,
    }


@mcp.tool()
def gwaya_verify_code(language: str, code: str, test_spec: str = "") -> Dict[str, Any]:
    """
    Verify candidate code using GWAYA's fail-closed verification oracles.
    - Python: Syntactic AST check, non-triviality check, and sandboxed Bubblewrap execution if test_spec is provided.
    - Rust: Fail-closed type, syntax, and borrow checker via 'rustc --emit=metadata'.
    - Lean 4: Fail-closed syntax & Lean 4 kernel with '#print axioms' soundness audit.

    Missing toolchains return UNVERIFIED (fail-closed) rather than false approval.

    Args:
        language: 'python', 'rust', or 'lean'.
        code: Candidate code snippet.
        test_spec: Optional test assertion code (for Python).
    """
    lang = language.lower().strip()

    if lang in ("python", "py"):
        if test_spec.strip():
            res = _py_oracle.verify_with_test(code, test_spec)
        else:
            res = _py_oracle.verify(code)
    elif lang in ("rust", "rs"):
        res = _rust_oracle.verify(code)
    elif lang in ("lean", "lean4"):
        res = _lean_oracle.verify(code)
    else:
        return {
            "success": False,
            "status": "UNSUPPORTED_LANGUAGE",
            "errors": [f"Language '{language}' not supported by GWAYA oracles."],
            "energy": 1e6,
        }

    return {
        "success": res.success,
        "status": res.status,
        "errors": res.errors,
        "duration_ms": res.latency_ms,
        "energy": 0.0 if res.success else 1e6,
        "reason": res.details.get("reason", ""),
        "details": res.details,
    }


@mcp.tool()
def gwaya_repair_code(
    prompt: str,
    failing_code: str,
    test_spec: str,
    model: str = "qwen2.5-coder:1.5b",
) -> Dict[str, Any]:
    """
    Run GWAYA's sample-and-repair loop on a failing Python candidate.
    Uses the local Ollama generator to iteratively fix syntax and failed assertions.

    Args:
        prompt: Description of the task/problem.
        failing_code: Initial candidate code that failed verification.
        test_spec: Public test assertions that the candidate must satisfy.
        model: Ollama model name (e.g., 'qwen2.5-coder:1.5b', 'qwen2.5-coder:7b').
    """
    generator = OllamaGenerator(model=model)
    optimizer = LowTierModelOptimizer(generator=generator, tier=ModelTier.LOW)

    res = optimizer.optimize_and_solve(
        goal=prompt,
        domain="python",
        context=failing_code,
        test_spec=test_spec,
    )

    return {
        "success": res.verified,
        "selected_code": res.selected_code,
        "energy": res.energy,
        "repair_attempts": res.repair_attempts,
        "duration_ms": res.duration_ms,
        "proof_token": res.proof_token,
        "telemetry": res.telemetry,
        "level": str(res.level),
    }


if __name__ == "__main__":
    mcp.run()
