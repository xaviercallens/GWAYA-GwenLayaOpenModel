"""Offline tests for deploy/cloudrun (T9): gates, flags, front, smoke-test logic. No gcloud, GPU or network."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
CR = ROOT / "deploy" / "cloudrun"
PLAN = json.loads((ROOT / "experiments" / "plan.json").read_text())


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, CR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def stub_path(tmp_path):
    """PATH with a gcloud stub that records calls."""
    stub = tmp_path / "gcloud"
    log = tmp_path / "calls.log"
    stub.write_text(f'#!/bin/bash\necho "$*" >> {log}\nexit 0\n')
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    env.pop("GWAYA_CONFIRM_SPEND", None)
    return env, log


def _sh(script, args, env):
    return subprocess.run(["bash", str(CR / script), *args], capture_output=True, text=True, env=env)


def test_deploy_dry_run_has_required_flags_and_no_gcloud(stub_path):
    env, log = stub_path
    r = _sh("deploy.sh", ["--model-file", "m.gguf"], env)
    assert r.returncode == 0 and "DRY-RUN" in r.stdout
    for flag in ["--gpu 1", "--gpu-type nvidia-l4", "--min-instances 0", "--max-instances 1",
                 "--no-allow-unauthenticated", "--no-cpu-throttling", "--cpu 4", "--memory 16Gi",
                 "purpose=gwenlaya", "us-central1", "type=cloud-storage", "gwenlaya_v4/quantized"]:
        assert flag in r.stdout.replace("\\", ""), flag
    assert not log.exists()


def test_deploy_flags_match_plan_s7():
    s7 = next(s for s in PLAN["stages"] if s["id"] == "S7")
    text = (CR / "deploy.sh").read_text()
    assert f'REGION="{s7["region"]}"' in text
    assert "--min-instances 0 --max-instances 1" in text and "--no-gpu-zonal-redundancy" in text
    assert PLAN["gcs_output_prefix"].endswith("gwenlaya_v4/")
    assert "gwenlaya_v4/quantized" in text


def test_deploy_real_requires_both_keys(stub_path):
    env, log = stub_path
    r = _sh("deploy.sh", ["--yes-spend", "--model-file", "m.gguf"], env)
    assert r.returncode == 3 and "GWAYA_CONFIRM_SPEND" in r.stderr
    assert not log.exists()
    r = _sh("deploy.sh", ["--yes-spend"], {**env, "GWAYA_CONFIRM_SPEND": "1"})
    assert r.returncode == 2 and "--model-file" in r.stderr and not log.exists()


def test_deploy_aborts_without_quota_before_any_spend(stub_path):
    env, log = stub_path   # stub prints nothing -> quota JSON invalid -> abort
    r = _sh("deploy.sh", ["--yes-spend", "--model-file", "m.gguf"], {**env, "GWAYA_CONFIRM_SPEND": "1"})
    assert r.returncode == 4
    calls = log.read_text()
    assert "quotas info list" in calls and "run deploy" not in calls and "builds submit" not in calls


@pytest.mark.parametrize("script", ["deploy.sh", "teardown.sh"])
def test_refuses_non_gwenlaya_service(stub_path, script):
    env, log = stub_path
    r = _sh(script, ["--service", "anse-serverless-laya"], env)
    assert r.returncode == 2 and not log.exists()


def test_teardown_gates(stub_path):
    env, log = stub_path
    r = _sh("teardown.sh", [], env)
    assert r.returncode == 0 and "services delete gwenlaya-gen" in r.stdout and not log.exists()
    r = _sh("teardown.sh", ["--yes"], env)
    assert r.returncode == 3 and not log.exists()
    r = _sh("teardown.sh", ["--yes"], {**env, "GWAYA_CONFIRM_SPEND": "1"})
    assert r.returncode == 0 and "services delete gwenlaya-gen" in log.read_text()


# ---- front ----
def _front(handler):
    front = _load("front")
    front._client = lambda timeout=None: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return front, TestClient(front.app)


def test_front_health_and_generate():
    def h(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(req.content)
        assert body["messages"][0]["content"] == "hi" and body["stream"] is False
        return httpx.Response(200, json={"choices": [{"message": {"content": "yo"}}], "usage": {"total_tokens": 3}})
    _, c = _front(h)
    assert c.get("/healthz").status_code == 200
    r = c.post("/generate", json={"prompt": "hi", "max_tokens": 4})
    assert r.status_code == 200 and r.json()["text"] == "yo" and r.json()["usage"]["total_tokens"] == 3
    assert c.post("/generate", json={"prompt": ""}).status_code == 422


def test_front_health_503_while_loading_and_generate_502():
    def h(req):
        raise httpx.ConnectError("down")
    _, c = _front(h)
    assert c.get("/healthz").status_code == 503
    assert c.post("/generate", json={"prompt": "x"}).status_code == 502


def test_front_route_fail_closed_without_laya_url():
    front, c = _front(lambda req: httpx.Response(200, json={}))
    front.LAYA_URL = ""
    assert c.post("/route", json={"prompt": "x"}).status_code == 503


def test_front_route_forwards_with_id_token():
    seen = {}
    def h(req):
        if "metadata.google.internal" in str(req.url):
            return httpx.Response(200, text="tok123")
        seen["auth"], seen["url"] = req.headers["authorization"], str(req.url)
        return httpx.Response(200, json={"tier": 1})
    front, c = _front(h)
    front.LAYA_URL, front.LAYA_ROUTE_PATH = "https://laya.example", "/route"
    r = c.post("/route", json={"prompt": "x"})
    assert r.status_code == 200 and r.json() == {"tier": 1}
    assert seen == {"auth": "Bearer tok123", "url": "https://laya.example/route"}


# ---- smoke test ----
class _Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def sleep(self, s): self.t += s


def _smoke(counts, post_status=200, unauth=401):
    st = _load("smoke_test")
    clk = _Clock()
    seq = iter(counts)
    def post(url, body, token, timeout=0):
        if token is None:
            return unauth, "denied"
        clk.t += 20.0 if not getattr(post, "warm", False) else 0.5
        post.warm = True
        return post_status, {}
    return st.run_smoke("https://x", "gwenlaya-gen", warm_n=3, max_tokens=8, prompt="p", scale_timeout_s=100,
                        poll_s=30, token_fn=lambda: "t", post=post, count_fn=lambda s: next(seq),
                        clock=clk, sleep=clk.sleep)


def test_smoke_cold_warm_and_scale_to_zero():
    r = _smoke([(0, "points"), (1, "points"), (0, "points")])
    assert r["cold_verified"] and r["unauth_rejected"] and r["ok"]
    assert r["cold_first_response_s"] == 20.0 and r["warm_median_s"] == 0.5 and len(r["warm_s"]) == 3
    assert r["scale_to_zero"]["reached"] and r["scale_to_zero"]["last_count"] == 0


def test_smoke_empty_window_is_not_zero_unless_positive_seen():
    r = _smoke([(None, "no_series")] * 10)
    assert not r["cold_verified"] and not r["scale_to_zero"]["reached"] and not r["ok"]
    r = _smoke([(None, "no_series"), (2, "points"), (None, "no_series")])
    assert r["scale_to_zero"]["reached"] and r["scale_to_zero"]["saw_positive_count"]


def test_smoke_fails_when_unauth_accepted_or_scale_not_reached():
    assert not _smoke([(0, "points"), (1, "points"), (1, "points"), (1, "points"), (1, "points"), (1, "points")])["ok"]
    assert not _smoke([(0, "points"), (0, "points")], unauth=200)["ok"]


def test_smoke_cli_gates(capsys, monkeypatch):
    st = _load("smoke_test")
    monkeypatch.delenv("GWAYA_CONFIRM_SPEND", raising=False)
    assert st.main([]) == 0 and "DRY-RUN" in capsys.readouterr().out
    assert st.main(["--yes-spend"]) == 3
    assert st.main(["--service", "anse-serverless-laya", "--yes-spend"]) == 2
