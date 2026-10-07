"""
gwaya/generators.py
===================
Local low-tier code generators for GWAYA (LT1).

``OllamaGenerator`` talks to a local Ollama server (default ``qwen2.5-coder:1.5b``) and is
shaped for small models on CPU:

* raw mode with the Qwen chat template and the assistant turn pre-filled with an opening
  code fence, so the model starts writing code immediately;
* ``stop`` on the closing fence / end-of-turn, so no tokens are spent on prose after the
  code (on a 1.5B model at ~4 tok/s this is most of the latency);
* an explicit ``seed`` per call (``base_seed + call index`` when none is given), so best-of-N
  samples differ from each other and a run can be replayed.

The callable signature matches ``LowTierModelOptimizer`` (``prompt, temperature=,
max_tokens=``). ``as_tot_generator`` adapts it to ``GwayaTreeOfThoughts`` (``prompt, ctx``).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "qwen2.5-coder:1.5b"
DEFAULT_HOST = "http://localhost:11434"

_FENCE_TAG = {"python": "python", "physics": "python", "rust": "rust", "lean4": "lean", "sql": "sql"}

_SYSTEM_PROMPT = (
    "You are a careful {lang} programmer. Reply with one complete, runnable code block only. "
    "No placeholders, no 'pass', no '...', no 'sorry', no 'todo!()'. Do not explain."
)


class GeneratorUnavailableError(RuntimeError):
    """The generator backend could not be reached or returned an unusable answer."""


@dataclass
class GenerationStats:
    model: str
    seed: int
    temperature: float
    max_tokens: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_ms: float = 0.0
    eval_ms: float = 0.0
    done_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_s(self) -> float:
        return self.completion_tokens / (self.eval_ms / 1000.0) if self.eval_ms > 0 else 0.0


class OllamaGenerator:
    """Seeded, stop-sequence-bounded code generator backed by a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        domain: str = "python",
        base_seed: int = 0,
        timeout_s: float = 600.0,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        if not self.host.startswith("http"):
            self.host = f"http://{self.host}"
        self.domain = domain
        self.base_seed = base_seed
        self.timeout_s = timeout_s
        self.system_prompt = system_prompt
        self.calls = 0
        self.last_stats: GenerationStats | None = None
        self.history: list[GenerationStats] = []

    # ── prompt construction ────────────────────────────────────────────────

    def fence_tag(self, domain: str | None = None) -> str:
        return _FENCE_TAG.get(domain or self.domain, "python")

    def build_raw_prompt(self, prompt: str, domain: str | None = None) -> str:
        tag = self.fence_tag(domain)
        system = self.system_prompt or _SYSTEM_PROMPT.format(lang=tag)
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n```{tag}\n"
        )

    # ── transport (patched in unit tests) ──────────────────────────────────

    def _post(self, path: str, payload: dict[str, Any], retries: int = 2) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310 - local host
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if attempt < retries:
                    time.sleep(1.0)
                    continue
                raise GeneratorUnavailableError(f"Ollama request to {self.host}{path} failed: {exc} ({body})") from exc
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                if attempt < retries:
                    time.sleep(1.0)
                    continue
                raise GeneratorUnavailableError(f"Ollama request to {self.host}{path} failed: {exc}") from exc

    def available(self) -> bool:
        """True if the server answers and has ``self.model`` pulled."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 - local host
                tags = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    # ── generation ─────────────────────────────────────────────────────────

    def __call__(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        seed: int | None = None,
        domain: str | None = None,
    ) -> str:
        call_seed = self.base_seed + self.calls if seed is None else seed
        self.calls += 1
        tag = self.fence_tag(domain)
        data = None
        for retry in range(3):
            actual_seed = int(call_seed) + retry * 37
            actual_temp = min(1.0, float(temperature) + retry * 0.15)
            payload = {
                "model": self.model,
                "prompt": self.build_raw_prompt(prompt, domain),
                "raw": True,
                "stream": False,
                "options": {
                    "temperature": actual_temp,
                    "num_predict": int(max_tokens),
                    "seed": actual_seed,
                    "stop": ["```", "<|im_end|>", "<|endoftext|>"],
                },
            }
            try:
                data = self._post("/api/generate", payload)
                break
            except GeneratorUnavailableError as exc:
                if "token repeat limit reached" in str(exc) and retry < 2:
                    continue
                if "token repeat limit reached" in str(exc):
                    data = {"response": ""}
                    break
                raise
        if not data or "response" not in data:
            raise GeneratorUnavailableError(f"Ollama returned no 'response' field: {str(data)[:200]}")

        stats = GenerationStats(
            model=self.model,
            seed=int(call_seed),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            total_ms=float(data.get("total_duration", 0) or 0) / 1e6,
            eval_ms=float(data.get("eval_duration", 0) or 0) / 1e6,
            done_reason=str(data.get("done_reason", "")),
        )
        self.last_stats = stats
        self.history.append(stats)

        code = str(data["response"]).rstrip()
        # Re-wrap so callers can use the same extract_code_block() path as for chat models.
        return f"```{tag}\n{code}\n```"

    def as_tot_generator(self, temperature: float = 0.2, max_tokens: int = 512) -> Callable[[str, dict], str]:
        """Adapter for GwayaTreeOfThoughts, whose generator signature is ``(prompt, ctx)``."""

        def _gen(prompt: str, _ctx: dict) -> str:
            return self(prompt, temperature=temperature, max_tokens=max_tokens)

        return _gen
