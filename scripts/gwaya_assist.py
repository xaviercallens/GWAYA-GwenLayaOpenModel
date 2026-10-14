#!/usr/bin/env python3
"""Local coding assistant CLI over an OpenAI-format endpoint (the TPU gateway, llama-server, vLLM).

  export GWAYA_ENDPOINT=https://gwenlaya-endpoint-....run.app GWAYA_API_KEY=gwk_...
  scripts/gwaya_assist.py solve --task "Write add(a,b)" --tests tests.py [--warmup] [--rounds 3]
  scripts/gwaya_assist.py bench --tasks scripts/tpu/assist_smoke_tasks.jsonl --out results.json [--warmup] [--stop]

``--warmup`` POSTs /v1/warmup and polls /v1/status until ready (a cold TPU takes minutes and costs money while up).
``--stop`` deletes the TPU VM at the end. Generation is remote, the test gate runs locally in the bwrap sandbox.
bench rows are a smoke check (a handful of hand-written tasks), not a benchmark and not evidence for any registered claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gwaya.assistant import Assistant  # noqa: E402
from gwaya.llama_tier import LlamaServerClient  # noqa: E402


def _call(base: str, key: str, method: str, path: str, timeout: float = 60.0):
    req = urllib.request.Request(base + path, method=method, headers={"Authorization": f"Bearer {key}"},
                                 data=b"{}" if method == "POST" else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 - operator-supplied endpoint
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {}
    except (urllib.error.URLError, OSError, ValueError):
        return 0, {}


def warm(base: str, key: str, max_wait_s: float) -> float:
    t0 = time.perf_counter()
    _call(base, key, "POST", "/v1/warmup")
    while time.perf_counter() - t0 < max_wait_s:
        s, d = _call(base, key, "GET", "/v1/status")
        if d.get("state") == "ready":
            return time.perf_counter() - t0
        time.sleep(10)
    raise SystemExit(f"endpoint not ready after {max_wait_s:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["solve", "bench"])
    ap.add_argument("--task", help="task text (solve)")
    ap.add_argument("--tests", help="file of assert statements (solve)")
    ap.add_argument("--tasks", help="JSONL of {id, task, tests} (bench)")
    ap.add_argument("--out", help="write JSON here")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model", default="gwenlaya-4b")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--max-wait-s", type=float, default=1200)
    a = ap.parse_args()
    base = os.environ.get("GWAYA_ENDPOINT", "").rstrip("/")
    key = os.environ.get("GWAYA_API_KEY", "")
    if not base or not key:
        print("set GWAYA_ENDPOINT and GWAYA_API_KEY", file=sys.stderr)
        return 2
    out: dict = {"endpoint": base, "model": a.model}
    if a.warmup:
        out["cold_start_s"] = round(warm(base, key, a.max_wait_s), 1)
    client = LlamaServerClient(a.model, base, max_new_tokens=a.max_new_tokens, api_key=key, cold_wait_s=a.max_wait_s,
                               timeout_s=600)
    asst = Assistant(client, max_rounds=a.rounds)
    try:
        if a.cmd == "solve":
            if not a.task:
                ap.error("--task required")
            tests = Path(a.tests).read_text() if a.tests else ""
            r = asst.solve(a.task, tests)
            out["result"] = r.to_dict()
            print(r.code)
            print(f"\n# verdict: {r.verdict}  rounds: {len(r.rounds)}  {r.note}", file=sys.stderr)
        else:
            if not a.tasks:
                ap.error("--tasks required")
            rows = []
            for line in Path(a.tasks).read_text().splitlines():
                if not line.strip():
                    continue
                t = json.loads(line)
                t0 = time.perf_counter()
                r = asst.solve(t["task"], t.get("tests", ""))
                rows.append({"id": t["id"], "verdict": r.verdict, "rounds": len(r.rounds), "wall_s": round(time.perf_counter() - t0, 2),
                             "first_round_passed": bool(r.rounds and r.rounds[0].passed is not None
                                                        and r.rounds[0].passed == r.rounds[0].total),
                             "tokens": sum(x.completion_tokens for x in r.rounds), "signals": r.signals, "note": r.note})
                print(json.dumps(rows[-1]), file=sys.stderr)
            n = len(rows) or 1
            out["rows"] = rows
            out["summary"] = {"n": len(rows), "verified": sum(x["verdict"] == "VERIFIED" for x in rows),
                              "first_try": sum(x["first_round_passed"] for x in rows),
                              "mean_wall_s": round(sum(x["wall_s"] for x in rows) / n, 2),
                              "tok_per_s": round(sum(x["tokens"] for x in rows) / max(1e-9, sum(x["wall_s"] for x in rows)), 1)}
            print(json.dumps(out["summary"]))
    finally:
        if a.stop:
            out["stop_http"] = _call(base, key, "POST", "/v1/stop")[0]
        if a.out:
            Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
