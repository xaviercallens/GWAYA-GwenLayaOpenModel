"""GWAYA/GwenLaya serverless endpoint gateway (Cloud Run, CPU, min-instances 0) in front of an on-demand TPU VM.

Cloud Run cannot attach TPUs, so the TPU is a Cloud TPU VM that this gateway creates on the first request and that
deletes itself when idle (see deploy/tpu_endpoint/vm/startup.sh). The gateway holds no state of its own: the TPU node
(name, state, internal IP) IS the state, read from the Cloud TPU API on each request.

Routes (OpenAI format; Authorization: Bearer <key> on everything except /healthz):
  GET  /healthz                 gateway liveness, never wakes the TPU
  GET  /v1/models               static model list, never wakes the TPU
  POST /v1/chat/completions     wakes the TPU if needed, then streams/proxies to vLLM
  POST /v1/completions          same
  POST /v1/warmup               start the TPU now and return immediately (202)
  GET  /v1/status               {"state": "off|starting|ready", ...} from the TPU API (no wake)
  POST /v1/stop                 delete the TPU VM now

Cold-request behaviour: the request is held until the TPU is healthy for at most HOLD_S seconds (default 540, under
the 600 s default client timeout), then 503 with Retry-After while the TPU keeps starting. Send header
`X-Gwaya-Wait: 0` to get that 503 immediately instead of holding.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

PROJECT = os.environ.get("GCP_PROJECT", "gen-lang-client-0625573011")
ZONES = [z for z in re.split(r"[\s,+]+", os.environ.get("TPU_ZONES", "us-west4-a us-central1-a us-east5-b")) if z]
NODE = os.environ.get("TPU_NAME", "gwenlaya-endpoint")
ACCEL = os.environ.get("TPU_ACCEL", "v5litepod-1")
RUNTIME = os.environ.get("TPU_RUNTIME", "v2-alpha-tpuv5-lite")
SERVICE_ACCOUNT = os.environ.get("TPU_SERVICE_ACCOUNT", "")
MODEL_HF = os.environ.get("MODEL_HF", "Qwen/Qwen3.5-4B")
SERVED = os.environ.get("MODEL_SERVED", "gwenlaya-4b")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_KEY = os.environ.get("VLLM_KEY", "")
API_KEYS = [k.strip() for k in os.environ.get("GWAYA_API_KEYS", "").split(",") if k.strip()]
HOLD_S = float(os.environ.get("HOLD_S", "540"))
IDLE_MIN = os.environ.get("IDLE_MIN", "10")
MAX_UP_MIN = os.environ.get("MAX_UP_MIN", "120")
MAX_TOKENS_CAP = int(os.environ.get("MAX_TOKENS_CAP", "2048"))
MAX_BODY = int(os.environ.get("MAX_BODY_BYTES", str(512 * 1024)))
WEIGHTS_GCS = os.environ.get("WEIGHTS_GCS", f"gs://gwenlaya-endpoint-{PROJECT}/models")
_HERE = Path(__file__).resolve().parent
VM_DIR = Path(os.environ.get("VM_SCRIPTS_DIR") or next((p for p in (_HERE / "vm", _HERE.parent / "vm") if p.is_dir()), _HERE / "vm"))
TPU_API = os.environ.get("TPU_API_BASE", "https://tpu.googleapis.com/v2")
METADATA_TOKEN = ("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")

app = FastAPI(title="gwaya-endpoint", docs_url=None, redoc_url=None)
_lock = asyncio.Lock()
_http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
_tok: dict[str, Any] = {"v": "", "exp": 0.0}
_zone_hint: dict[str, str] = {}


def _err(status: int, msg: str, typ: str = "gwaya_error", headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": msg, "type": typ, "code": status}}, status_code=status, headers=headers)


def authorized(request: Request) -> bool:
    if not API_KEYS:
        return False  # fail closed: no keys configured means nobody gets in
    got = request.headers.get("authorization", "")
    got = got[7:] if got.lower().startswith("bearer ") else got
    return any(hmac.compare_digest(got.encode(), k.encode()) for k in API_KEYS)


async def _token() -> str:
    if _tok["v"] and time.time() < _tok["exp"] - 60:
        return _tok["v"]
    r = await _http.get(METADATA_TOKEN, headers={"Metadata-Flavor": "Google"}, timeout=5.0)
    r.raise_for_status()
    d = r.json()
    _tok.update(v=d["access_token"], exp=time.time() + float(d.get("expires_in", 300)))
    return _tok["v"]


async def _tpu(method: str, path: str, **kw: Any) -> httpx.Response:
    h = {"authorization": f"Bearer {await _token()}"}
    return await _http.request(method, f"{TPU_API}/{path}", headers=h, timeout=30.0, **kw)


def node_path(zone: str) -> str:
    return f"projects/{PROJECT}/locations/{zone}/nodes/{NODE}"


async def find_node() -> tuple[str, dict[str, Any]] | None:
    """The node and its zone, if it exists in any configured zone (cached hint first)."""
    order = ([_zone_hint["z"]] if "z" in _zone_hint else []) + [z for z in ZONES if z != _zone_hint.get("z")]
    for z in order:
        r = await _tpu("GET", node_path(z))
        if r.status_code == 200:
            _zone_hint["z"] = z
            return z, r.json()
        if r.status_code not in (404, 403):
            r.raise_for_status()
    _zone_hint.pop("z", None)
    return None


def node_ip(node: dict[str, Any]) -> str:
    eps = node.get("networkEndpoints") or []
    return (eps[0].get("ipAddress") or "") if eps else ""


def _create_body() -> dict[str, Any]:
    md = {"gwaya-model-hf": MODEL_HF, "gwaya-model-served": SERVED, "gwaya-vllm-key": VLLM_KEY,
          "gwaya-vllm-port": str(VLLM_PORT), "gwaya-idle-min": IDLE_MIN, "gwaya-max-up-min": MAX_UP_MIN,
          "gwaya-weights-gcs": WEIGHTS_GCS, "gwaya-project": PROJECT, "gwaya-node": NODE}
    body: dict[str, Any] = {"acceleratorType": ACCEL, "runtimeVersion": RUNTIME, "labels": {"purpose": "gwenlaya-endpoint"}, "tags": ["gwenlaya-endpoint"],
                            "networkConfig": {"enableExternalIps": True}, "metadata": md}
    # scripts travel inline in node metadata (no bucket read needed at boot; the files are baked into this image)
    for key, fname in (("startup-script", "startup.sh"), ("gwaya-watchdog-py", "watchdog.py")):
        p = VM_DIR / fname
        if not p.is_file():  # a node without its scripts would boot and do nothing, burning TPU time: refuse instead
            raise RuntimeError(f"VM script missing: {p}")
        md[key] = p.read_text()
    if SERVICE_ACCOUNT:
        body["serviceAccount"] = {"email": SERVICE_ACCOUNT, "scope": ["https://www.googleapis.com/auth/cloud-platform"]}
    return body


async def start_node() -> tuple[str, str]:
    """Create the node, trying each zone in turn. Returns (zone, note). Idempotent: an existing node is kept."""
    existing = await find_node()
    if existing:
        return existing[0], "exists"
    last = ""
    for z in ZONES:
        r = await _tpu("POST", f"projects/{PROJECT}/locations/{z}/nodes", params={"nodeId": NODE}, json=_create_body())
        if r.status_code in (200, 409):  # 409: created concurrently
            _zone_hint["z"] = z
            return z, "creating"
        last = f"{z}: {r.status_code} {r.text[:200]}"
    raise RuntimeError(f"no zone could create the TPU ({last})")


async def vllm_ready(ip: str) -> bool:
    if not ip:
        return False
    try:
        r = await _http.get(f"http://{ip}:{VLLM_PORT}/health", timeout=3.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


async def status() -> dict[str, Any]:
    found = await find_node()
    if not found:
        return {"state": "off", "model": SERVED}
    zone, node = found
    ip = node_ip(node)
    ready = node.get("state") == "READY" and await vllm_ready(ip)
    return {"state": "ready" if ready else "starting", "tpu_state": node.get("state"), "zone": zone, "model": SERVED}


async def ensure_up(hold_s: float) -> tuple[str | None, dict[str, Any]]:
    """Start the TPU if needed and wait up to hold_s for vLLM health. Returns (ip or None, info)."""
    async with _lock:
        zone, note = await start_node()
    t0 = time.monotonic()
    while True:
        found = await find_node()
        if found:
            zone, node = found
            ip = node_ip(node)
            if node.get("state") == "READY" and await vllm_ready(ip):
                return ip, {"zone": zone, "waited_s": round(time.monotonic() - t0, 1)}
            if node.get("state") in ("PREEMPTED", "TERMINATED", "FAILED"):
                async with _lock:
                    await _tpu("DELETE", node_path(zone))
                return None, {"zone": zone, "state": node.get("state"), "note": "node failed; deleted, retry"}
        if time.monotonic() - t0 >= hold_s:
            return None, {"zone": zone, "state": "starting", "waited_s": round(time.monotonic() - t0, 1)}
        await asyncio.sleep(5.0)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def models(request: Request) -> JSONResponse:
    if not authorized(request):
        return _err(401, "invalid or missing API key", "invalid_request_error")
    return JSONResponse({"object": "list", "data": [{"id": SERVED, "object": "model", "owned_by": "gwenlaya",
                                                    "root": MODEL_HF}]})


@app.get("/v1/status")
async def get_status(request: Request) -> JSONResponse:
    if not authorized(request):
        return _err(401, "invalid or missing API key", "invalid_request_error")
    return JSONResponse(await status())


@app.post("/v1/warmup")
async def warmup(request: Request) -> JSONResponse:
    if not authorized(request):
        return _err(401, "invalid or missing API key", "invalid_request_error")
    try:
        async with _lock:
            zone, note = await start_node()
    except Exception as exc:  # noqa: BLE001 - report, never crash the gateway
        return _err(503, f"could not start TPU: {exc}", headers={"Retry-After": "60"})
    return JSONResponse({"state": "starting", "zone": zone, "note": note,
                         "hint": "poll GET /v1/status until state is 'ready' (typically a few minutes)"}, status_code=202)


@app.post("/v1/stop")
async def stop(request: Request) -> JSONResponse:
    if not authorized(request):
        return _err(401, "invalid or missing API key", "invalid_request_error")
    found = await find_node()
    if not found:
        return JSONResponse({"state": "off"})
    await _tpu("DELETE", node_path(found[0]))
    _zone_hint.pop("z", None)
    return JSONResponse({"state": "stopping", "zone": found[0]})


async def _body_json(request: Request) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    raw = await request.body()
    if len(raw) > MAX_BODY:
        return None, _err(413, f"request body over {MAX_BODY} bytes", "invalid_request_error")
    try:
        d = json.loads(raw)
        if not isinstance(d, dict):
            raise ValueError
    except ValueError:
        return None, _err(400, "body must be a JSON object", "invalid_request_error")
    for k in ("max_tokens", "max_completion_tokens"):
        if isinstance(d.get(k), int):
            d[k] = max(1, min(d[k], MAX_TOKENS_CAP))
    if "max_tokens" not in d and "max_completion_tokens" not in d:
        d["max_tokens"] = min(512, MAX_TOKENS_CAP)
    d["model"] = SERVED
    d["n"] = 1
    return d, None


async def _proxy(request: Request, path: str) -> Any:
    if not authorized(request):
        return _err(401, "invalid or missing API key", "invalid_request_error")
    body, bad = await _body_json(request)
    if bad is not None:
        return bad
    assert body is not None
    hold = 0.0 if request.headers.get("x-gwaya-wait", "") == "0" else HOLD_S
    try:
        ip, info = await ensure_up(hold)
    except Exception as exc:  # noqa: BLE001
        return _err(503, f"TPU could not be started: {exc}", headers={"Retry-After": "60"})
    if ip is None:
        return _err(503, f"TPU is starting ({json.dumps(info)}); retry in about 30 s. "
                         "POST /v1/warmup before a test session to avoid this.", "gwaya_cold_start",
                    headers={"Retry-After": "30"})
    url = f"http://{ip}:{VLLM_PORT}{path}"
    hdr = {"content-type": "application/json", "authorization": f"Bearer {VLLM_KEY}"}
    if body.get("stream"):
        req = _http.build_request("POST", url, json=body, headers=hdr, timeout=httpx.Timeout(600.0, connect=10.0))
        try:
            up = await _http.send(req, stream=True)
        except httpx.HTTPError as exc:
            return _err(502, f"upstream unreachable: {type(exc).__name__}")

        async def gen() -> AsyncIterator[bytes]:
            try:
                async for chunk in up.aiter_raw():
                    yield chunk
            finally:
                await up.aclose()

        return StreamingResponse(gen(), status_code=up.status_code, media_type="text/event-stream")
    try:
        r = await _http.post(url, json=body, headers=hdr)
    except httpx.HTTPError as exc:
        return _err(502, f"upstream unreachable: {type(exc).__name__}")
    return JSONResponse(r.json() if r.headers.get("content-type", "").startswith("application/json") else
                        {"error": {"message": r.text[:500], "type": "upstream_error", "code": r.status_code}},
                        status_code=r.status_code)


@app.post("/v1/chat/completions")
async def chat(request: Request) -> Any:
    return await _proxy(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request) -> Any:
    return await _proxy(request, "/v1/completions")
