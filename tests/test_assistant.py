"""Offline tests for gwaya.assistant and the endpoint auth / cold-start handling of llama_tier (fake server, loopback)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gwaya.assistant import Assistant
from gwaya.llama_tier import LlamaServerClient, LlamaServerError
from gwaya.test_harness import HarnessResult


def body(text, lps=(-0.1, -0.3)):
    return {"choices": [{"text": text, "finish_reason": "stop",
                         "logprobs": {"token_logprobs": list(lps)}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": len(lps)}}


class Fake:
    def __init__(self, script):
        self.script, self.seen = list(script), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append({"auth": self.headers.get("Authorization"), "payload": payload})
                code, data = outer.script.pop(0) if outer.script else (200, body("x=1"))
                raw = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Retry-After", "0")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def hr(ok, passed=1, total=2, error="", isolation="bwrap", details=None):
    return HarnessResult(ok, passed, total, error, isolation, 1.0, details=details or {})


@pytest.fixture
def fake():
    made = []

    def make(script):
        f = Fake(script)
        made.append(f)
        return f
    yield make
    for f in made:
        f.close()


def test_bearer_key_sent_and_503_retried_within_cold_wait(fake):
    f = fake([(503, {"error": "starting"}), (503, {}), (200, body("ok"))])
    c = LlamaServerClient("m", f.url, api_key="gwk_x", cold_wait_s=30, timeout_s=5)
    text, lps = c.complete("p", "python")
    assert "ok" in text and len(f.seen) == 3 and all(s["auth"] == "Bearer gwk_x" for s in f.seen)


def test_503_without_cold_wait_raises(fake):
    f = fake([(503, {})])
    with pytest.raises(LlamaServerError):
        LlamaServerClient("m", f.url, timeout_s=5).complete("p", "python")
    assert f.seen[0]["auth"] is None


def test_repair_loop_verifies_on_second_round(fake):
    f = fake([(200, body("def f(a):\n    return a")), (200, body("def f(a):\n    return a + 1"))])
    c = LlamaServerClient("m", f.url, timeout_s=5)
    outcomes = [hr(False, 1, 2, "TEST_FAILED", details={"failed_idx": [0]}), hr(True, 2, 2)]
    a = Assistant(c, max_rounds=3, run_tests=lambda code, tests, timeout_s=10: outcomes.pop(0))
    r = a.solve("add one", "assert f(1)==2\nassert f(0)==1")
    assert r.verdict == "VERIFIED" and len(r.rounds) == 2 and "a + 1" in r.code
    assert "failing assert indices [0]" in f.seen[1]["payload"]["prompt"]
    assert r.signals["min_logprob"] == -0.3  # first-round signal only
    assert "logprobs" not in f.seen[1]["payload"]


def test_failed_after_max_rounds_and_no_tests_is_unverified(fake):
    f = fake([])
    c = LlamaServerClient("m", f.url, timeout_s=5)
    a = Assistant(c, max_rounds=2, run_tests=lambda *x, **k: hr(False, 0, 1, "TEST_FAILED"))
    assert a.solve("t", "assert 1 == 2").verdict == "FAILED" and len(f.seen) == 2
    assert Assistant(c).solve("t", "").verdict == "UNVERIFIED"


def test_no_sandbox_is_unverified_not_failed(fake):
    f = fake([])
    c = LlamaServerClient("m", f.url, timeout_s=5)
    a = Assistant(c, run_tests=lambda *x, **k: hr(False, 0, 1, "UNVERIFIED: bwrap", isolation="none",
                                                  details={"reason": "no_isolation"}))
    r = a.solve("t", "assert 1")
    assert r.verdict == "UNVERIFIED" and len(r.rounds) == 1 and "gate could not run" in r.note


def test_generator_down_is_unverified(fake):
    c = LlamaServerClient("m", "http://127.0.0.1:9", timeout_s=2)
    r = Assistant(c).solve("t", "assert 1")
    assert r.verdict == "UNVERIFIED" and "generator unavailable" in r.note
