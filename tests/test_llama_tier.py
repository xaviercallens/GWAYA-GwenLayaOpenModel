"""Offline tests for gwaya.llama_tier: a fake llama-server (python http.server in a thread) returns the llama.cpp
/v1/completions JSON shape. No model, no network beyond loopback."""
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gwaya import gwenlaya as g
from gwaya import llama_tier as lt
from gwaya.domains.task import CheckResult
from gwaya.generators import OllamaGenerator

LPS = [-0.1, -0.5, -2.0, -0.25]


def completion(text="x = 1", lps=LPS, finish="stop", shape="llamacpp"):
    ch: dict = {"text": text, "index": 0, "finish_reason": finish}
    if lps is not None:
        if shape == "llamacpp":
            ch["logprobs"] = {"content": [{"id": i, "token": "t", "logprob": v, "bytes": [], "top_logprobs": []}
                                          for i, v in enumerate(lps)]}
        else:
            ch["logprobs"] = {"tokens": ["t"] * len(lps), "token_logprobs": list(lps)}
    return {"choices": [ch], "usage": {"prompt_tokens": 12, "completion_tokens": len(lps or [])}}


class FakeServer:
    def __init__(self):
        self.requests: list[dict] = []
        self.responder = lambda payload: (200, completion())
        self.delay = 0.0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: D401
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "payload": body})
                if outer.delay:
                    time.sleep(outer.delay)
                code, data = outer.responder(body)
                raw = data if isinstance(data, bytes) else json.dumps(data).encode()
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except OSError:  # client timed out and went away
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def srv():
    s = FakeServer()
    yield s
    s.close()


# ---- prompt protocol ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("domain", ["python", "rust", "math", "math_pot"])
def test_prompt_and_stops_equal_the_study_protocol(srv, domain):
    name = "qwen3.5-2b-bf16"
    t = lt.llama_cpp_tier(name, srv.url)
    t.generate_with_signals("  Do the thing.  ", domain)
    req = srv.requests[0]
    gen = OllamaGenerator(model=name)
    p = req["payload"]
    assert req["path"] == "/v1/completions"
    assert p["prompt"] == gen.build_raw_prompt("  Do the thing.  ", domain)
    assert p["stop"] == gen.stop_sequences(domain)
    assert p["temperature"] == 0.0 and p["max_tokens"] == 1024 and p["seed"] == 0 and p["logprobs"] == 1
    if domain == "math":  # prose: empty think block selects non-thinking mode, no code fence
        assert p["prompt"].endswith("<think>\n\n</think>\n\n") and "<|im_end|>" in p["stop"] and "```" not in p["stop"]
    elif domain == "math_pot":
        assert p["prompt"].endswith("```python\n") and "```" in p["stop"]
    else:
        assert p["prompt"].endswith(f"```{'rust' if domain == 'rust' else 'python'}\n") and "```" in p["stop"]


def test_output_is_wrapped_like_ollama_generator(srv):
    srv.responder = lambda p: (200, completion("print(1)\n\n"))
    t = lt.llama_cpp_tier("qwen3.5-2b-bf16", srv.url)
    gen = OllamaGenerator(model="qwen3.5-2b-bf16")
    assert t.generate("p", "python") == gen.wrap_output("print(1)\n\n", "python") == "```python\nprint(1)\n```"
    srv.responder = lambda p: (200, completion("The answer is \\boxed{4}.\n"))
    assert t.generate("p", "math") == "The answer is \\boxed{4}."


def test_max_new_tokens_and_url_forms(srv):
    t = lt.llama_cpp_tier("m", srv.url + "/v1/", max_new_tokens=77)
    t.generate("p", "python")
    assert srv.requests[0]["payload"]["max_tokens"] == 77 and srv.requests[0]["path"] == "/v1/completions"


# ---- signals -----------------------------------------------------------------------------------------------------

def test_signals_present_llamacpp_shape(srv):
    t = lt.llama_cpp_tier("m", srv.url)
    text, sig = t.generate_with_signals("p", "python")
    assert text.startswith("```python")
    assert sig == {"mean_logprob": pytest.approx(sum(LPS) / 4), "min_logprob": -2.0}
    assert t.client.last["finish_reason"] == "stop" and t.client.last["n_logprobs"] == 4


def test_signals_present_openai_shape(srv):
    srv.responder = lambda p: (200, completion(shape="openai"))
    sig = lt.llama_cpp_tier("m", srv.url).generate_with_signals("p", "python")[1]
    assert sig["min_logprob"] == -2.0 and math.isclose(sig["mean_logprob"], sum(LPS) / 4)


@pytest.mark.parametrize("body", [completion(lps=None), completion(lps=[])])
def test_no_logprobs_gives_empty_signals_not_zeros(srv, body):
    srv.responder = lambda p: (200, body)
    text, sig = lt.llama_cpp_tier("m", srv.url).generate_with_signals("p", "python")
    assert sig == {} and text.startswith("```python")


def test_plain_generate_does_not_request_logprobs(srv):
    lt.llama_cpp_tier("m", srv.url).generate("p", "python")
    assert "logprobs" not in srv.requests[0]["payload"]


# ---- errors raise ------------------------------------------------------------------------------------------------

def test_http_500_raises(srv):
    srv.responder = lambda p: (500, {"error": "boom"})
    t = lt.llama_cpp_tier("m", srv.url)
    with pytest.raises(lt.LlamaServerError, match="HTTP 500"):
        t.generate_with_signals("p", "python")
    with pytest.raises(lt.LlamaServerError):
        t.generate("p", "python")


def test_timeout_raises(srv):
    srv.delay = 1.5
    t = lt.llama_cpp_tier("m", srv.url, timeout_s=0.3)
    with pytest.raises(lt.LlamaServerError):
        t.generate_with_signals("p", "python")


@pytest.mark.parametrize("body", [b"not json at all", json.dumps({"choices": []}).encode(),
                                  json.dumps({"error": "x"}).encode(), json.dumps({"choices": [{"text": 5}]}).encode()])
def test_malformed_response_raises(srv, body):
    srv.responder = lambda p: (200, body)
    with pytest.raises(lt.LlamaServerError):
        lt.llama_cpp_tier("m", srv.url).generate_with_signals("p", "python")


def test_connection_refused_raises():
    s = FakeServer()
    url = s.url
    s.close()
    with pytest.raises(lt.LlamaServerError):
        lt.llama_cpp_tier("m", url, timeout_s=2).generate("p", "python")


def test_generation_error_is_recorded_by_gwenlaya(srv):
    srv.responder = lambda p: (500, {"error": "boom"})
    s = g.GwenLaya([lt.llama_cpp_tier("m", srv.url)], checker=lambda task, r: CheckResult("VERIFIED", {}))
    out = s.answer("p", "python")
    assert out["verdict"] != "VERIFIED" and out["answer"] is None
    assert "LlamaServerError" in out["evidence"]["trajectory"][0]["generation_error"]


# ---- end to end through GwenLaya with a stub Laya ----------------------------------------------------------------

def test_end_to_end_signals_reach_the_scorer_features(srv):
    names = ["qwen3.5-2b-bf16", "qwen3.5-4b-bf16"]
    srv.responder = lambda p: (200, completion("def f():\n    return 1"))
    tiers = [lt.llama_cpp_tier(n, srv.url) for n in names]
    seen: list[dict] = []

    def router(prompt, domain):
        return {n: 0.2 if n == names[0] else 0.9 for n in names}  # starts at the 4B (start_at policy, tau 0.5)

    def scorer(features):
        seen.append({k: features.get(k) for k in ("tier", "gen_signals", "gate_status")})
        # what LayaRuntime.scorer does with them: incomplete signals are recorded, never zero-filled
        sig = features.get("gen_signals") or {}
        features["laya"] = {"applied": True, "signals_incomplete": any(k not in sig for k in ("mean_logprob", "min_logprob")),
                            "truncated_candidate": False, "model_sha": "stub"}
        return 0.6

    cal = g.CalibrationArtifact(calibrator=g.PlattCalibrator(1.0, 0.0).to_dict(), tau_hi=0.99, tau_lo=0.01)
    s = g.GwenLaya(tiers, router=router, scorer=scorer, calibrations={"python": cal}, tau_route=0.5,
                   route_policy="start_at", checker=lambda task, r: CheckResult("UNVERIFIED", {}))
    out = s.answer("write f", "python")
    assert seen and seen[0]["tier"] == names[1]
    for row in seen:
        assert set(row["gen_signals"]) == {"mean_logprob", "min_logprob"}
        assert row["gen_signals"]["min_logprob"] == -2.0
        assert row["gen_signals"]["mean_logprob"] == pytest.approx(sum(LPS) / 4)
    assert out["evidence"]["laya"]["signals_incomplete"] is False
    assert out["evidence"]["trajectory"][0]["gen_signals"]["min_logprob"] == -2.0
    assert out["p_correct"] == pytest.approx(0.6)


def test_end_to_end_server_without_logprobs_is_signals_incomplete(srv):
    srv.responder = lambda p: (200, completion(lps=None))
    seen = []

    def scorer(features):
        seen.append(dict(features.get("gen_signals") or {}))
        features["laya"] = {"signals_incomplete": "mean_logprob" not in seen[-1]}
        return 0.5

    cal = g.CalibrationArtifact(calibrator=g.PlattCalibrator(1.0, 0.0).to_dict(), tau_hi=0.99, tau_lo=0.01)
    s = g.GwenLaya([lt.llama_cpp_tier("m", srv.url)], scorer=scorer, calibrations={"python": cal},
                   checker=lambda task, r: CheckResult("UNVERIFIED", {}))
    out = s.answer("p", "python", allow_escalation=False)
    assert seen == [{}] and out["evidence"]["laya"]["signals_incomplete"] is True


# ---- wiring ------------------------------------------------------------------------------------------------------

def _clear(monkeypatch):
    for k in ("GWENLAYA_BACKEND", "GWENLAYA_LLAMA_URLS", "GWENLAYA_LAYA_DIR", "GWENLAYA_CALIBRATION",
              "GWENLAYA_TIERS", "GWENLAYA_SERVED_QUANT", "GWENLAYA_LAYA_TIER_MAP", "GWENLAYA_LAYA_THRESHOLDS"):
        monkeypatch.delenv(k, raising=False)


def test_default_backend_is_ollama_and_unchanged(monkeypatch):
    _clear(monkeypatch)
    s = g.system_from_env()
    assert [t.name for t in s.tiers] == ["qwen2.5-coder:1.5b", "qwen2.5-coder:3b", "qwen2.5-coder:7b"]
    assert all(t.generate_with_signals is None for t in s.tiers)
    monkeypatch.setenv("GWENLAYA_LLAMA_URLS", "a=http://127.0.0.1:1")  # ignored without the backend switch
    monkeypatch.setenv("GWENLAYA_TIERS", "x:1,y:2")
    assert [t.name for t in g.system_from_env().tiers] == ["x:1", "y:2"]


def test_llamacpp_backend_builds_named_tiers_in_order(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GWENLAYA_BACKEND", "llamacpp")
    monkeypatch.setenv("GWENLAYA_LLAMA_URLS",
                       "qwen3.5-2b-bf16=http://127.0.0.1:8091, qwen3.5-4b-bf16=http://127.0.0.1:8092")
    s = g.system_from_env()
    assert [t.name for t in s.tiers] == ["qwen3.5-2b-bf16", "qwen3.5-4b-bf16"]
    assert all(t.generate_with_signals is not None for t in s.tiers)
    assert s.tiers[1].client.url == "http://127.0.0.1:8092/v1"
    assert s.router is None and s.scorer is None


@pytest.mark.parametrize("env", [{"GWENLAYA_BACKEND": "llamacpp"},
                                 {"GWENLAYA_BACKEND": "llamacpp", "GWENLAYA_LLAMA_URLS": "no-equals"},
                                 {"GWENLAYA_BACKEND": "vllm"}])
def test_bad_backend_config_raises(monkeypatch, env):
    _clear(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError):
        g.system_from_env()


def test_laya_with_llamacpp_marks_quant_shift(monkeypatch):
    from gwaya import laya_runtime as L
    _clear(monkeypatch)

    class StubRT:
        def router(self, prompt, domain):
            return {}

        def scorer(self, features):
            features["laya"] = {"applied": True, "signals_incomplete": False, "truncated_candidate": False}
            return 0.5

        def calibration_artifacts(self, th):
            return {}

    seen = {}

    def fake_load(d, device="cpu", **kw):
        seen.update(kw)
        return StubRT()
    monkeypatch.setattr(L.LayaRuntime, "load", staticmethod(fake_load))
    monkeypatch.setenv("GWENLAYA_BACKEND", "llamacpp")
    monkeypatch.setenv("GWENLAYA_LLAMA_URLS", "qwen3.5-2b-bf16=http://127.0.0.1:8091")
    monkeypatch.setenv("GWENLAYA_LAYA_DIR", "/x")
    monkeypatch.setenv("GWENLAYA_SERVED_QUANT", "q4_k_m")
    s = g.system_from_env()
    assert seen["tier_map"] is None or seen["tier_map"] == {}  # names verbatim: no map invented
    assert [t.name for t in s.tiers] == ["qwen3.5-2b-bf16"]
    feats = {"domain": "python", "tier": "qwen3.5-2b-bf16"}
    assert s.scorer(feats) == 0.5
    assert feats["laya"]["quant_shift"] is True and feats["laya"]["served_quant"] == "q4_k_m"

    # same scorer is untouched for the ollama backend
    monkeypatch.setenv("GWENLAYA_BACKEND", "ollama")
    monkeypatch.setenv("GWENLAYA_TIERS", "a:1")
    f2 = {"domain": "python", "tier": "a:1"}
    g.system_from_env().scorer(f2)
    assert "quant_shift" not in f2["laya"]
