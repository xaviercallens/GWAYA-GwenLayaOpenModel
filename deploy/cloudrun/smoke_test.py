#!/usr/bin/env python3
"""Smoke test for the GwenLaya Cloud Run endpoint (plan.json S7): auth, cold start, warm latency, scale to zero.

DRY-RUN BY DEFAULT (prints the plan, makes no call). A real run needs --yes-spend AND GWAYA_CONFIRM_SPEND=1,
because the requests start a billed L4 instance (about 1.0465 USD/h, plan.json planning rate) and the instance
then idles for up to ~10 min (prereg section 11).

Measured per run (written to --out as JSON; nothing is filled in by hand):
  unauth_status         status of a request WITHOUT a token (expect 401/403, release check R5)
  instances_before      active instance count before the cold call, or null when it could not be read
  cold_verified         true only if instances_before == 0 was read
  cold_first_response_s wall time of the first /generate call (includes cold start; this is NOT time-to-first-token)
  warm_s / warm_median_s / warm_min_s / warm_max_s   the next --warm-n sequential /generate calls
  scale_to_zero         {reached, seconds_after_last_request, last_count, evidence}
Instance counts come from Cloud Monitoring run.googleapis.com/container/instance_count (read-only; lags by
minutes). If the count cannot be read the field is null/"unknown": it is never reported as zero by assumption.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

PROJECT = os.environ.get("GCP_PROJECT", "gen-lang-client-0625573011")
REGION = "us-central1"
DEFAULT_PROMPT = "Write a Python function add(a, b) that returns a + b."


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def gcloud_token(kind: str = "identity") -> str:
    return _run(["gcloud", "auth", f"print-{kind}-token"])


def gcloud_service_url(service: str) -> str:
    return _run(["gcloud", "run", "services", "describe", service, "--project", PROJECT, "--region", REGION,
                 "--format", "value(status.url)"])


def post_json(url: str, body: dict, token: str | None, timeout: float = 900.0) -> tuple[int, Any]:
    """POST JSON; returns (status, parsed body or raw text). Never raises on HTTP errors."""
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode(errors="replace"), e.code
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


def instance_count(service: str) -> tuple[int | None, str]:
    """(count, evidence). Sums the latest point per series over the last 5 min. evidence is 'points',
    'no_series' (nothing reported in the window) or 'error'. Read-only Cloud Monitoring call."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=5)
    flt = ('metric.type="run.googleapis.com/container/instance_count" '
           f'AND resource.labels.service_name="{service}"')
    qs = urllib.request.quote(flt, safe="")
    url = (f"https://monitoring.googleapis.com/v3/projects/{PROJECT}/timeSeries?filter={qs}"
           f"&interval.startTime={start:%Y-%m-%dT%H:%M:%SZ}&interval.endTime={now:%Y-%m-%dT%H:%M:%SZ}&view=FULL")
    try:
        req = urllib.request.Request(url, headers={"authorization": f"Bearer {gcloud_token('access')}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
    except (urllib.error.URLError, subprocess.CalledProcessError, ValueError, OSError):
        return None, "error"
    series = data.get("timeSeries", [])
    if not series:
        return None, "no_series"
    total = 0
    for s in series:
        pts = s.get("points", [])
        if pts:  # newest first
            v = pts[0].get("value", {})
            total += int(v.get("int64Value", v.get("doubleValue", 0)))
    return total, "points"


def summarize(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"median": None, "min": None, "max": None}
    return {"median": round(statistics.median(xs), 4), "min": round(min(xs), 4), "max": round(max(xs), 4)}


def run_smoke(url: str, service: str, *, warm_n: int, max_tokens: int, prompt: str, scale_timeout_s: float,
              poll_s: float, token_fn: Callable[[], str] = gcloud_token,
              post: Callable[..., tuple[int, Any]] = post_json,
              count_fn: Callable[[str], tuple[int | None, str]] = instance_count,
              clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> dict:
    gen = url.rstrip("/") + "/generate"
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    out: dict[str, Any] = {"service": service, "url": url, "region": REGION, "project": PROJECT,
                           "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "max_tokens": max_tokens, "warm_n": warm_n}

    status, _ = post(gen, body, None, timeout=60.0)       # no token, so no instance is meant to start
    out["unauth_status"] = status
    out["unauth_rejected"] = status in (401, 403)

    before, ev = count_fn(service)
    out["instances_before"] = before
    out["instances_before_evidence"] = ev
    out["cold_verified"] = before == 0
    token = token_fn()

    t0 = clock()
    st, resp = post(gen, body, token)
    out["cold_first_response_s"] = round(clock() - t0, 3)
    out["cold_status"] = st
    ok = st == 200
    warm: list[float] = []
    if ok:
        for _ in range(warm_n):
            t = clock()
            s, _ = post(gen, body, token)
            dt = clock() - t
            if s != 200:
                out["warm_error_status"] = s
                break
            warm.append(round(dt, 4))
    out["warm_s"] = warm
    w = summarize(warm)
    out["warm_median_s"], out["warm_min_s"], out["warm_max_s"] = w["median"], w["min"], w["max"]

    t_last = clock()
    saw_positive = False
    reached, last, evidence = False, None, "not_polled"
    while clock() - t_last <= scale_timeout_s:
        last, evidence = count_fn(service)
        saw_positive = saw_positive or bool(last)
        # an empty Monitoring window counts as zero only after a positive count was seen in this run
        if last == 0 or (last is None and evidence == "no_series" and saw_positive):
            reached = True
            break
        sleep(poll_s)
    out["scale_to_zero"] = {"reached": reached, "seconds_after_last_request": round(clock() - t_last, 1),
                            "last_count": last, "evidence": evidence, "saw_positive_count": saw_positive,
                            "timeout_s": scale_timeout_s}
    out["ok"] = bool(ok and out["unauth_rejected"] and reached)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--service", default="gwenlaya-gen")
    ap.add_argument("--url", help="service URL (default: read from gcloud run services describe)")
    ap.add_argument("--warm-n", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--scale-timeout-s", type=float, default=1200.0, help="wait this long for 0 instances (idle may last ~10 min)")
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--out", default="smoke_result.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes-spend", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if not a.service.startswith("gwenlaya-"):
        print("REFUSED: service must start with gwenlaya-", file=sys.stderr)
        return 2
    if a.dry_run or not a.yes_spend:
        print("DRY-RUN: no request is sent. Plan:")
        print(f"  1. resolve URL of {a.service} ({REGION}) read-only; send one request WITHOUT a token (expect 401/403)")
        print("  2. read instance count (Cloud Monitoring, read-only); must be 0 for a verified cold sample")
        print(f"  3. cold call with identity token, then {a.warm_n} sequential warm calls (max_tokens={a.max_tokens})")
        print(f"  4. poll instance count up to {a.scale_timeout_s:.0f}s until it returns to 0")
        print(f"  5. write {a.out}")
        print("To run: GWAYA_CONFIRM_SPEND=1 smoke_test.py --yes-spend")
        return 0
    if os.environ.get("GWAYA_CONFIRM_SPEND") != "1":
        print("REFUSED: --yes-spend also requires GWAYA_CONFIRM_SPEND=1", file=sys.stderr)
        return 3
    url = a.url or gcloud_service_url(a.service)
    if not url:
        print("ABORT: could not resolve the service URL", file=sys.stderr)
        return 4
    res = run_smoke(url, a.service, warm_n=a.warm_n, max_tokens=a.max_tokens, prompt=a.prompt,
                    scale_timeout_s=a.scale_timeout_s, poll_s=a.poll_s)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
