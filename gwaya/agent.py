"""
anse/gwaya/agent.py
===================
``gwaya-agent``: a small local coding agent for a standard NVIDIA RTX GPU (or CPU) using Ollama.

    uv run python -m anse.gwaya.agent doctor
    uv run python -m anse.gwaya.agent ask "write is_prime(n)" --test "assert is_prime(7) and not is_prime(8)"

Contract (what it does and does not claim):
* Code is only reported ``VERIFIED`` if it passed a check that was actually executed in the bwrap sandbox
  (your test, and/or the model-written tests several candidates agree on). ``--level`` says which.
* With no verification possible it says ``UNVERIFIED`` and prints the best attempt marked as such.
  A verified answer is not a proof of correctness: it only passed the checks it names.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from typing import Any

from gwaya.consensus_agent import AgentAnswer, ConsensusRepairAgent
from gwaya.generators import DEFAULT_MODEL, GeneratorUnavailableError, OllamaGenerator
from gwaya.oracles import PythonCompilerOracle
from gwaya.test_harness import isolation_available

# (min free VRAM GiB, model tag). Sized for Q4 weights + ~1.5 GiB of KV cache/overhead. Measured VRAM is
# only claimed in the benchmark results; this table is a conservative recommendation.
MODEL_LADDER = [
    (10.0, "qwen2.5-coder:14b"),
    (6.0, "qwen2.5-coder:7b"),
    (3.5, "qwen2.5-coder:3b"),
    (0.0, "qwen2.5-coder:1.5b"),
]


def detect_vram_gib() -> tuple[str | None, float | None]:
    """(GPU name, total VRAM GiB) of the first NVIDIA GPU, or (None, None)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()[0]
        name, mib = (x.strip() for x in out.split(",", 1))
        return name, round(int(mib) / 1024.0, 1)
    except (subprocess.SubprocessError, IndexError, ValueError):
        return None, None


def recommend_model(vram_gib: float | None) -> str:
    if vram_gib is None:
        return DEFAULT_MODEL  # CPU only: the smallest model
    return next(tag for need, tag in MODEL_LADDER if vram_gib >= need)


def doctor() -> dict[str, Any]:
    name, vram = detect_vram_gib()
    gen = OllamaGenerator(model=DEFAULT_MODEL)
    try:
        import urllib.request

        with urllib.request.urlopen(f"{gen.host}/api/tags", timeout=5) as r:
            models = [m["name"] for m in json.load(r)["models"]]
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        models = []
        ollama = f"unreachable ({exc.__class__.__name__}) at {gen.host}"
    else:
        ollama = f"ok at {gen.host}"
    return {
        "gpu": name or "none detected",
        "vram_gib": vram,
        "recommended_model": recommend_model(vram),
        "ollama": ollama,
        "installed_models": models,
        "sandbox_bwrap": isolation_available(),
        "note": "Verification refuses to run without bwrap isolation."
        if not isolation_available()
        else "ok",
    }


def ask(goal: str, test: str | None, model: str, rounds: int, candidates: int) -> AgentAnswer:
    if not isolation_available():
        raise RuntimeError(
            "bwrap isolation is unavailable: refusing to execute model-written code (fail-closed)"
        )
    agent = ConsensusRepairAgent(
        OllamaGenerator(model=model),
        oracle=PythonCompilerOracle(),
        n_candidates=candidates,
        max_repair_rounds=rounds,
    )
    return agent.solve(goal, public_test=test)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gwaya-agent", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="detect GPU/VRAM, Ollama and sandbox; recommend a model")
    q = sub.add_parser("ask", help="solve one Python task with verification")
    q.add_argument("goal")
    q.add_argument(
        "--test", default=None, help="Python asserts the answer must pass (strongly recommended)"
    )
    q.add_argument("--model", default=None, help="Ollama tag (default: recommended for your VRAM)")
    q.add_argument("--rounds", type=int, default=3)
    q.add_argument("--candidates", type=int, default=5)
    q.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "doctor":
        print(json.dumps(doctor(), indent=2))
        return 0
    model = args.model or recommend_model(detect_vram_gib()[1])
    try:
        ans = ask(args.goal, args.test, model, args.rounds, args.candidates)
    except (GeneratorUnavailableError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"model": model, "status": ans.status, **asdict(ans)}, indent=2))
    else:
        print(
            f"# model={model} status={ans.status} level={ans.level} rounds={ans.rounds} candidates={ans.candidates} wall={ans.wall_s}s"
        )
        if not ans.verified:
            print(
                "# WARNING: UNVERIFIED best attempt. It did not pass any executed check; review before use."
            )
        print(ans.code)
    return 0 if ans.verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
