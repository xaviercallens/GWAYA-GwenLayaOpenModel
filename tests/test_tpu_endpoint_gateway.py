"""Gateway of the serverless TPU endpoint (deploy/tpu_endpoint): auth, request clamping, TPU lifecycle against a fake
Cloud TPU API and a fake vLLM. No network, no GCP."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

ROOT = Path(__file__).resolve().parents[1]
GW = ROOT / "deploy" / "tpu_endpoint" / "gateway" / "main.py"
WD = ROOT / "deploy" / "tpu_endpoint" / "vm" / "watchdog.py"


def load(env: dict[str, str]):
    for k, v in env.items():
        os.environ[k] = v
    spec = importlib.util.spec_from_file_location("gwaya_gateway_under_test", GW)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeCloud:
    """Fake TPU API + fake vLLM. `ready_after` = number of GET polls before the node is READY and healthy."""

    def __init__(self, ready_after: int = 0, fail_zones: tuple[str, ...] = ()):
        self.node: dict | None = None
        self.zone = ""
        self.polls = 0
        self.ready_after = ready_after
        self.fail_zones = fail_zones
        self.created: list[dict] = []
        self.deleted = 0
        self.upstream: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "metadata.google.internal" in url:
            return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
        if "tpu.googleapis.com" in url:
            zone = url.split("/locations/")[1].split("/")[0]
            if request.method == "GET":
                if self.node and zone == self.zone:
                    self.polls += 1
                    ready = self.polls > self.ready_after
                    n = dict(self.node, state="READY" if ready else "CREATING",
                             networkEndpoints=[{"ipAddress": "10.0.0.7"}] if ready else [])
                    return httpx.Response(200, json=n)
                return httpx.Response(404, json={})
            if request.method == "POST":
                if zone in self.fail_zones:
                    return httpx.Response(429, json={"error": "RESOURCE_EXHAUSTED"})
                body = json.loads(request.content)
                self.created.append(dict(body, _zone=zone))
                self.zone, self.node = zone, {"name": "n"}
                return httpx.Response(200, json={"name": "op"})
            if request.method == "DELETE":
                self.deleted += 1
                self.node = None
                return httpx.Response(200, json={})
        if url.startswith("http://10.0.0.7:8000"):
            if url.endswith("/health"):
                return httpx.Response(200 if self.polls > self.ready_after else 503)
            body = json.loads(request.content)
            self.upstream.append(body)
            assert request.headers["authorization"] == "Bearer vk"
            return httpx.Response(200, json={"id": "x", "object": "chat.completion", "model": body["model"],
                                             "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]})
        return httpx.Response(500, json={"unexpected": url})


@pytest.fixture()
def gw():
    env = {"GWAYA_API_KEYS": "k1,k2", "VLLM_KEY": "vk", "GCP_PROJECT": "p", "TPU_ZONES": "z1+z2", "HOLD_S": "5",
           "MAX_TOKENS_CAP": "100", "MODEL_SERVED": "gwenlaya-4b", "MODEL_HF": "Qwen/Qwen3.5-4B"}
    mod = load(env)
    return mod


def client(mod, cloud: FakeCloud):
    mod._http = httpx.AsyncClient(transport=httpx.MockTransport(cloud.handler))
    mod._tok.update(v="", exp=0.0)
    mod._zone_hint.clear()

    async def no_sleep(_s: float) -> None:
        return None
    mod.asyncio.sleep = no_sleep  # type: ignore[assignment]
    return fastapi_testclient.TestClient(mod.app)


H = {"Authorization": "Bearer k1"}
BODY = {"model": "anything", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 5000, "n": 4}


def test_auth_required_and_fail_closed(gw):
    c = client(gw, FakeCloud())
    assert c.get("/healthz").status_code == 200
    assert c.get("/health").status_code == 200
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.post("/v1/chat/completions", json=BODY).status_code == 401
    r = c.get("/v1/models", headers=H)
    assert r.status_code == 200 and r.json()["data"][0]["id"] == "gwenlaya-4b"
    gw.API_KEYS.clear()  # no keys configured: nobody gets in
    assert c.get("/v1/models", headers=H).status_code == 401


def test_cold_request_creates_tpu_then_proxies_with_clamps(gw):
    cloud = FakeCloud(ready_after=2)
    c = client(gw, cloud)
    r = c.post("/v1/chat/completions", json=BODY, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "hi"
    assert len(cloud.created) == 1
    made = cloud.created[0]
    assert made["acceleratorType"] == "v5litepod-1" and made["tags"] == ["gwenlaya-endpoint"]
    assert made["metadata"]["gwaya-vllm-key"] == "vk" and "vllm serve" in made["metadata"]["startup-script"]
    assert "def delete_self" in made["metadata"]["gwaya-watchdog-py"]
    sent = cloud.upstream[0]
    assert sent["max_tokens"] == 100 and sent["n"] == 1 and sent["model"] == "gwenlaya-4b"


def test_warm_request_does_not_create_again(gw):
    cloud = FakeCloud(ready_after=0)
    c = client(gw, cloud)
    assert c.post("/v1/chat/completions", json=BODY, headers=H).status_code == 200
    assert c.post("/v1/chat/completions", json=BODY, headers=H).status_code == 200
    assert len(cloud.created) == 1


def test_no_wait_header_returns_503_retry_after_and_keeps_starting(gw):
    cloud = FakeCloud(ready_after=10_000)
    c = client(gw, cloud)
    r = c.post("/v1/chat/completions", json=BODY, headers={**H, "X-Gwaya-Wait": "0"})
    assert r.status_code == 503 and r.headers["retry-after"] == "30"
    assert r.json()["error"]["type"] == "gwaya_cold_start"
    assert len(cloud.created) == 1


def test_zone_fallback_when_first_zone_has_no_capacity(gw):
    cloud = FakeCloud(ready_after=0, fail_zones=("z1",))
    c = client(gw, cloud)
    r = c.post("/v1/warmup", headers=H)
    assert r.status_code == 202 and r.json()["zone"] == "z2"


def test_status_stop_and_idempotent_warmup(gw):
    cloud = FakeCloud(ready_after=0)
    c = client(gw, cloud)
    assert c.get("/v1/status", headers=H).json()["state"] == "off"
    c.post("/v1/warmup", headers=H)
    c.post("/v1/warmup", headers=H)
    assert len(cloud.created) == 1
    assert c.get("/v1/status", headers=H).json()["state"] == "ready"
    assert c.post("/v1/stop", headers=H).json()["state"] == "stopping" and cloud.deleted == 1
    assert c.get("/v1/status", headers=H).json()["state"] == "off"


def test_bad_bodies(gw):
    c = client(gw, FakeCloud())
    assert c.post("/v1/chat/completions", content=b"not json", headers=H).status_code == 400
    assert c.post("/v1/chat/completions", content=b"[1]", headers=H).status_code == 400
    big = {"messages": [{"role": "user", "content": "x" * (600 * 1024)}]}
    assert c.post("/v1/chat/completions", json=big, headers=H).status_code == 413


def test_watchdog_decision():
    spec = importlib.util.spec_from_file_location("wd", WD)
    wd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wd)
    # not healthy yet: only the hard TTL applies (load time is never counted as idle)
    assert wd.decide(now=1000, t0=0, max_up_s=7200, idle_s=600, last_change=None) is None
    assert wd.decide(now=8000, t0=0, max_up_s=7200, idle_s=600, last_change=None) == "max_up"
    assert wd.decide(now=1000, t0=0, max_up_s=7200, idle_s=600, last_change=500) is None
    assert wd.decide(now=1200, t0=0, max_up_s=7200, idle_s=600, last_change=500) == "idle"
