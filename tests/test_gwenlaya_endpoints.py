"""MCP tool and WebGWAYA endpoint for GwenLaya (offline, fake system injected)."""
import pytest
from fastapi.testclient import TestClient

import mcp_server
from gwaya import gwenlaya as g
from gwaya.domains.task import CheckResult
from webgwaya.app import WEBGWAYA_TOKEN, app


@pytest.fixture()
def fake_system():
    seen = {}

    def checker(task, response):
        seen["payload"] = task.checker_payload
        return CheckResult("VERIFIED" if "good" in response else "UNVERIFIED", {})

    g.set_system(g.GwenLaya([g.Tier("t", lambda p, d: "good")], checker=checker))
    yield seen
    g.set_system(None)


def test_mcp_gwaya_answer(fake_system):
    r = mcp_server.gwaya_answer("write f", "python", tests="assert f()")
    assert r["verdict"] == "VERIFIED" and r["answer"] == "good" and r["tier"] == "t"
    assert fake_system["payload"] == {"tests": "assert f()"}
    assert "error" in mcp_server.gwaya_answer("x", "cobol")
    assert "error" in mcp_server.gwaya_answer("  ", "python")


def test_api_requires_auth(fake_system):
    c = TestClient(app)
    assert c.post("/api/gwaya/answer", json={"prompt": "q"}).status_code == 401
    assert c.post("/api/gwaya/answer", json={"prompt": "q"},
                  headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_api_answer_and_validation(fake_system):
    c = TestClient(app)
    h = {"Authorization": f"Bearer {WEBGWAYA_TOKEN}"}
    r = c.post("/api/gwaya/answer", json={"prompt": "q", "domain": "math", "reference_answer": "4"}, headers=h)
    assert r.status_code == 200 and r.json()["verdict"] == "VERIFIED"
    assert fake_system["payload"] == {"answer": "4"}
    assert c.post("/api/gwaya/answer", json={"prompt": "q", "domain": "cobol"}, headers=h).status_code == 422
    assert c.post("/api/gwaya/answer", json={"prompt": ""}, headers=h).status_code == 422
