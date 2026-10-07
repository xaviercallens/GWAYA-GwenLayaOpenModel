"""Thin FastAPI front for the GwenLaya Cloud Run generation service.

Authentication is Cloud Run IAM (--no-allow-unauthenticated); this process does not check tokens.
Routes:
  GET  /healthz            200 only when the local llama-server reports healthy (startup probe), else 503
  POST /v1/chat/completions  pass-through to llama-server (OpenAI-compatible)
  POST /generate           {"prompt","max_tokens","temperature"} -> {"text","usage","elapsed_s"}
  POST /route              forward a JSON body to the existing Laya service (LAYA_SERVICE_URL), using a
                           Google-signed ID token from the metadata server. 503 if LAYA_SERVICE_URL is unset.
The Laya service is only called, never modified. Its request schema is not assumed: /route forwards the body
unchanged to LAYA_SERVICE_URL + LAYA_ROUTE_PATH (default /route; TBD until checked against the live service).
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

LLAMA_URL = os.environ.get("LLAMA_URL", f"http://127.0.0.1:{os.environ.get('LLAMA_PORT', '8081')}")
LAYA_URL = os.environ.get("LAYA_SERVICE_URL", "").rstrip("/")
LAYA_ROUTE_PATH = os.environ.get("LAYA_ROUTE_PATH", "/route")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "gwenlaya-gen")
METADATA_IDTOKEN = ("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/"
                    "identity")
TIMEOUT_S = float(os.environ.get("UPSTREAM_TIMEOUT_S", "300"))

app = FastAPI(title="gwenlaya-gen")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=64_000)
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.0, ge=0.0, le=2.0)


def _client(timeout: float | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT_S if timeout is None else timeout)


@app.get("/healthz")
async def healthz() -> Response:
    try:
        async with _client(2.0) as c:
            r = await c.get(f"{LLAMA_URL}/health")
        if r.status_code == 200:
            return JSONResponse({"status": "ok"})
    except httpx.HTTPError:
        pass
    return JSONResponse({"status": "loading"}, status_code=503)


@app.post("/v1/chat/completions")
async def chat(request: Request) -> Response:
    body = await request.body()
    try:
        async with _client() as c:
            r = await c.post(f"{LLAMA_URL}/v1/chat/completions", content=body,
                             headers={"content-type": "application/json"})
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"llama-server unreachable: {type(exc).__name__}")
    return Response(r.content, status_code=r.status_code, media_type="application/json")


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    t0 = time.perf_counter()
    payload = {"model": MODEL_ALIAS, "messages": [{"role": "user", "content": req.prompt}],
               "max_tokens": req.max_tokens, "temperature": req.temperature, "stream": False}
    try:
        async with _client() as c:
            r = await c.post(f"{LLAMA_URL}/v1/chat/completions", json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"llama-server unreachable: {type(exc).__name__}")
    if r.status_code != 200:
        raise HTTPException(502, f"llama-server returned {r.status_code}")
    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, "malformed llama-server response")
    return {"text": text, "usage": data.get("usage", {}), "elapsed_s": round(time.perf_counter() - t0, 4)}


async def _id_token(c: httpx.AsyncClient, audience: str) -> str:
    r = await c.get(METADATA_IDTOKEN, params={"audience": audience},
                    headers={"Metadata-Flavor": "Google"}, timeout=5.0)
    r.raise_for_status()
    return r.text


@app.post("/route")
async def route(request: Request) -> Response:
    if not LAYA_URL:
        raise HTTPException(503, "LAYA_SERVICE_URL is not configured; routing unavailable (fail closed)")
    body = await request.body()
    try:
        async with _client() as c:
            token = await _id_token(c, LAYA_URL)
            r = await c.post(f"{LAYA_URL}{LAYA_ROUTE_PATH}", content=body,
                             headers={"content-type": "application/json", "authorization": f"Bearer {token}"})
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"laya service unreachable: {type(exc).__name__}")
    return Response(r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))
