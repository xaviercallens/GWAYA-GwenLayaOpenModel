#!/usr/bin/env python3
"""Measure the endpoint end to end: cold start (warmup -> ready), first request, warm requests, then optionally stop.

  GWAYA_ENDPOINT=https://gwenlaya-endpoint-xxxx.a.run.app GWAYA_API_KEY=gwk_... python3 deploy/tpu_endpoint/smoke.py [--stop]

Writes nothing to disk; prints one JSON object. Costs one TPU VM start (about the list price of one v5e chip for the
cold-start time plus the idle window if you do not --stop).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def call(base, key, method, path, body=None, timeout=900, headers=None):
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(base + path, method=method, headers=h,
                                 data=json.dumps(body).encode() if body is not None else None)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}"), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", action="store_true", help="delete the TPU VM at the end")
    ap.add_argument("--max-wait-s", type=int, default=1200)
    a = ap.parse_args()
    base = os.environ.get("GWAYA_ENDPOINT", "").rstrip("/")
    key = os.environ.get("GWAYA_API_KEY", "")
    if not base or not key:
        print("set GWAYA_ENDPOINT and GWAYA_API_KEY", file=sys.stderr)
        return 2
    out: dict = {"endpoint": base}
    s, d, _ = call(base, key, "GET", "/health")
    out["gateway_health"] = s
    s, d, _ = call(base, key, "GET", "/v1/status")
    out["state_before"] = d.get("state")
    t0 = time.perf_counter()
    s, d, dt = call(base, key, "POST", "/v1/warmup")
    out["warmup_http"] = s
    while time.perf_counter() - t0 < a.max_wait_s:
        s, d, _ = call(base, key, "GET", "/v1/status")
        if d.get("state") == "ready":
            break
        time.sleep(10)
    out["cold_start_s"] = round(time.perf_counter() - t0, 1)
    out["state_after_warmup"] = d.get("state")
    body = {"messages": [{"role": "user", "content": "Reply with the single word: ok"}], "max_tokens": 16, "temperature": 0}
    lat = []
    for _ in range(3):
        s, d, dt = call(base, key, "POST", "/v1/chat/completions", body)
        lat.append({"http": s, "s": round(dt, 2), "usage": d.get("usage"),
                    "text": (d.get("choices") or [{}])[0].get("message", {}).get("content")})
    out["requests"] = lat
    if a.stop:
        s, d, _ = call(base, key, "POST", "/v1/stop")
        out["stop"] = d
    print(json.dumps(out, indent=1))
    return 0 if out["state_after_warmup"] == "ready" and all(r["http"] == 200 for r in lat) else 1


if __name__ == "__main__":
    sys.exit(main())
