"""GwenLaya tier backed by a running llama.cpp ``llama-server`` (OpenAI-compatible /v1/completions).

Why this exists: Ollama 0.1.44 returns no token log-probs, so the frozen Laya calibrator (trained on rows that always
carried ``mean_logprob`` and ``min_logprob``) could never support LIKELY_CORRECT with Ollama tiers. llama-server does
return them. The request mirrors the study's serving protocol (scripts/run_study.py ``OpenAICompatBackend``): the raw
Qwen prompt of ``OllamaGenerator.build_raw_prompt`` (math prose / code-fence prefill, "math_pot" domain), its stop
sequences, greedy decoding (temperature 0, seed 0), 1024 new tokens, thinking off (the prompt carries the empty think
block; the server is started with ``--reasoning-budget 0``), and ``OllamaGenerator.wrap_output`` on the way back.

Errors (connection refused, timeout, HTTP error, malformed body) raise, so GwenLaya records a generation_error for the
tier. Nothing here fabricates text or log-probs; a response without log-probs yields signals ``{}``."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from gwaya.generators import OllamaGenerator

DEFAULT_MAX_NEW_TOKENS = 1024
API_KEY_ENV = "GWENLAYA_LLAMA_API_KEY"  # bearer key for an authenticated OpenAI-format endpoint (deploy/tpu_endpoint gateway)
DEFAULT_COLD_WAIT_S = 0.0  # seconds to keep retrying HTTP 503 (an endpoint that is still starting); 0 = fail at once
DEFAULT_TIMEOUT_S = 1800.0  # the 4B tier decodes about 1.3 tok/s on the study CPU: 1024 tokens is about 13 min


class LlamaServerError(RuntimeError):
    """The llama-server request failed or its response could not be parsed."""


def parse_completion(data: Any) -> tuple[str, str, list[float], dict[str, Any]]:
    """(text, finish_reason, token_logprobs, usage) from an OpenAI-compatible /v1/completions body.

    Accepts the OpenAI shape (``logprobs.token_logprobs``) and the llama.cpp shape (``logprobs.content[].logprob``).
    Raises RuntimeError when the body has no ``choices[0].text``. Log-probs may legitimately be an empty list."""
    ch = (data.get("choices") or [{}])[0] if isinstance(data, dict) else {}
    if not isinstance(ch, dict) or "text" not in ch:
        raise RuntimeError(f"OpenAI-compatible server returned no text: {str(data)[:200]}")
    lpo = ch.get("logprobs") or {}
    lps = [x for x in (lpo.get("token_logprobs") or []) if isinstance(x, (int, float))]
    if not lps:  # llama.cpp server: logprobs.content[].logprob
        lps = [c["logprob"] for c in (lpo.get("content") or [])
               if isinstance(c, dict) and isinstance(c.get("logprob"), (int, float))]
    return ch["text"], str(ch.get("finish_reason") or ""), lps, (data.get("usage") or {})


def signals_from_logprobs(lps: list[float]) -> dict[str, float]:
    """{"mean_logprob", "min_logprob"} (the definitions of scripts/data/build_laya_data.py), or {} when none."""
    if not lps:
        return {}
    return {"mean_logprob": sum(lps) / len(lps), "min_logprob": min(lps)}


class LlamaServerClient:
    """One tier's connection. ``last`` holds the stats of the most recent call (finish_reason, token counts)."""

    def __init__(self, name: str, base_url: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 timeout_s: float = DEFAULT_TIMEOUT_S, seed: int = 0, api_key: str | None = None,
                 cold_wait_s: float = DEFAULT_COLD_WAIT_S) -> None:
        self.name = name
        self.api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        self.cold_wait_s = float(cold_wait_s)
        url = base_url.rstrip("/")
        self.url = url if url.endswith("/v1") else url + "/v1"
        self.max_new_tokens = int(max_new_tokens)
        self.timeout_s = float(timeout_s)
        self.seed = int(seed)
        self.gen = OllamaGenerator(model=name)  # prompt construction / stops / wrapping only; never contacted
        self.last: dict[str, Any] = {}

    def complete(self, prompt: str, domain: str, logprobs: bool = True) -> tuple[str, list[float]]:
        raw = self.gen.build_raw_prompt(prompt, domain)
        payload: dict[str, Any] = {"model": self.name, "prompt": raw, "temperature": 0.0,
                                   "max_tokens": self.max_new_tokens, "seed": self.seed,
                                   "stop": self.gen.stop_sequences(domain)}
        if logprobs:
            payload["logprobs"] = 1
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(payload).encode("utf-8")
        deadline = time.monotonic() + self.cold_wait_s
        while True:
            req = urllib.request.Request(self.url + "/completions", data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310 - operator-supplied URL
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                text_ = exc.read().decode("utf-8", errors="replace")[:200]
                if exc.code == 503 and time.monotonic() < deadline:  # endpoint still starting: honour Retry-After
                    try:
                        wait = float(exc.headers.get("Retry-After", "") or 10)
                    except ValueError:
                        wait = 10.0
                    time.sleep(max(0.0, min(wait, deadline - time.monotonic())))
                    continue
                raise LlamaServerError(f"llama-server {self.url} returned HTTP {exc.code}: {text_}") from exc
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:  # includes JSONDecodeError
                raise LlamaServerError(f"llama-server {self.url} request failed: {type(exc).__name__}: {exc}") from exc
        try:
            text, finish, lps, usage = parse_completion(data)
        except RuntimeError as exc:
            raise LlamaServerError(str(exc)) from exc
        if not isinstance(text, str):
            raise LlamaServerError(f"llama-server returned non-string text: {type(text).__name__}")
        self.last = {"finish_reason": finish, "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                     "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0), "n_logprobs": len(lps)}
        return self.gen.wrap_output(text, domain), lps


def llama_cpp_tier(name: str, base_url: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                   timeout_s: float = DEFAULT_TIMEOUT_S, api_key: str | None = None,
                   cold_wait_s: float = DEFAULT_COLD_WAIT_S):
    """A GwenLaya ``Tier`` for a running llama-server (network is only touched when the tier is called)."""
    from gwaya.gwenlaya import Tier
    client = LlamaServerClient(name, base_url, max_new_tokens=max_new_tokens, timeout_s=timeout_s,
                               api_key=api_key, cold_wait_s=cold_wait_s)

    def generate(prompt: str, domain: str) -> str:
        return client.complete(prompt, domain, logprobs=False)[0]

    def generate_with_signals(prompt: str, domain: str) -> tuple[str, dict[str, float]]:
        text, lps = client.complete(prompt, domain, logprobs=True)
        return text, signals_from_logprobs(lps)

    tier = Tier(name, generate, generate_with_signals)
    tier.client = client  # type: ignore[attr-defined]  # last-call stats for callers that want them
    return tier
