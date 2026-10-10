#!/usr/bin/env python3
"""Live smoke test of GwenLaya + frozen Laya + llama.cpp tiers (NOT an evaluation; n = 3 by construction).

Needs a running llama-server (scripts/gwenlaya_serve.sh start) and the env of docs/LAYA_RUNTIME.md
("Local serving with log-probs"): GWENLAYA_BACKEND=llamacpp, GWENLAYA_LLAMA_URLS, GWENLAYA_LAYA_DIR, GWAYA_DATA_ROOT.
Runs three tiny tasks end to end (python with visible tests, python without tests, math without a gold answer),
records per-stage latency (route, generate, gate, calibrate), verdict, p_correct and the Laya evidence flags."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gwaya import gwenlaya as g  # noqa: E402

TASKS = [
    {"id": "smoke_python_tests", "domain": "python", "tests": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0\n",
     "prompt": "Write a Python function add(a, b) that returns the sum of two integers."},
    {"id": "smoke_python_no_tests", "domain": "python", "tests": "",
     "prompt": "Write a Python function is_palindrome(s) that returns True when the string s reads the same "
               "forwards and backwards, ignoring case."},
    {"id": "smoke_math", "domain": "math", "tests": "",
     "prompt": "What is the sum of the first 10 positive integers? Put the final answer in \\boxed{}."},
]


def timed(fn, bucket: list, key: str):
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            bucket.append((key, time.perf_counter() - t0))
    return wrapper


def load_avg() -> list[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("results/gwenlaya_v4/laya_runtime/live_smoke.json"))
    ap.add_argument("--thresholds", default=None, help="exploratory thresholds JSON (sets GWENLAYA_LAYA_THRESHOLDS)")
    args = ap.parse_args()
    if args.thresholds:
        os.environ["GWENLAYA_LAYA_THRESHOLDS"] = args.thresholds
    env_keys = ["GWENLAYA_BACKEND", "GWENLAYA_LLAMA_URLS", "GWENLAYA_LAYA_DIR", "GWENLAYA_SERVED_QUANT",
                "GWENLAYA_LAYA_THRESHOLDS", "GWENLAYA_LAYA_DOMAINS", "GWENLAYA_LAYA_TAU_ROUTE", "GWENLAYA_LAYA_THREADS"]
    env = {k: os.environ.get(k) for k in env_keys}
    load0 = load_avg()
    t0 = time.perf_counter()
    s = g.system_from_env()
    build_s = time.perf_counter() - t0
    rows: list[dict] = []
    buf: list[tuple[str, float]] = []  # (stage, seconds) of the current task
    s.router = timed(s.router, buf, "route")
    s.scorer = timed(s.scorer, buf, "calibrate")
    s._gate = timed(s._gate, buf, "gate")
    for t in TASKS:
        buf.clear()
        payload = g.payload_for(t["domain"], tests=t["tests"])
        load_before = load_avg()
        w0 = time.perf_counter()
        try:
            out = s.answer(t["prompt"], t["domain"], payload, task_id=t["id"])
            err = None
        except Exception as exc:  # noqa: BLE001 - record, do not hide
            out, err = {}, f"{type(exc).__name__}: {exc}"[:300]
        wall = time.perf_counter() - w0
        traj = (out.get("evidence") or {}).get("trajectory", [])
        pot_gen = sum(float(st.get("pot", {}).get("gen_s", 0.0)) for st in traj)
        gen_s = sum(float(st.get("gen_s", 0.0)) for st in traj)
        gate_s = sum(d for k, d in buf if k == "gate")
        lay = (out.get("evidence") or {}).get("laya", {})
        rows.append({
            "id": t["id"], "domain": t["domain"], "has_visible_tests": bool(t["tests"].strip()), "error": err,
            "verdict": out.get("verdict"), "answered": out.get("answered"), "p_correct": out.get("p_correct"),
            "tier": out.get("tier"), "wall_s": round(wall, 2),
            "latency_s": {"route": round(sum(d for k, d in buf if k == "route"), 2),
                          "generate": round(gen_s, 2),
                          "gate_total": round(gate_s, 2), "gate_of_which_pot_generation": round(pot_gen, 2),
                          "calibrate": round(sum(d for k, d in buf if k == "calibrate"), 2)},
            "gen_signals": [st.get("gen_signals") for st in traj],
            "gate": [{"status": st.get("gate", {}).get("status"), "reason": st.get("reason"),
                      "decision": st.get("decision"), "raw_score": st.get("raw_score"),
                      "scorer_error": st.get("scorer_error")} for st in traj],
            "routing": (out.get("evidence") or {}).get("routing"),
            "evidence_flags": {"signals_incomplete": lay.get("signals_incomplete"),
                               "truncated_candidate": lay.get("truncated_candidate"),
                               "quant_shift": lay.get("quant_shift"), "served_quant": lay.get("served_quant"),
                               "model_sha": lay.get("model_sha"),
                               "scorer_applied": (lay.get("scorer") or {}).get("applied"),
                               "scorer_reason": (lay.get("scorer") or {}).get("reason"),
                               "likely_correct_vetoed": (lay.get("scorer") or {}).get("likely_correct_vetoed")},
            "load_avg_before": load_before, "load_avg_after": load_avg(),
        })
        print(json.dumps({k: rows[-1][k] for k in ("id", "verdict", "p_correct", "wall_s", "latency_s", "error")}),
              flush=True)
    rec = {
        "note": "SMOKE TEST, n = 3: shows that the wiring runs end to end and what each stage costs. It is not an "
                "evaluation: no accuracy, calibration or LIKELY_CORRECT precision claim can be drawn from it.",
        "served_weights": "Qwen3.5-2B Q4_K_M GGUF via llama.cpp llama-server; the frozen Laya was trained on bf16 "
                          "generations (quant_shift, not measured here)",
        "machine": {"cpu_count": os.cpu_count(), "platform": platform.platform(), "load_avg_start": load0,
                    "load_avg_end": load_avg(),
                    "shared": "yes: another job's llama-server and python processes were running (see load)"},
        "env": env, "system_build_s": round(build_s, 2),
        "thresholds": "exploratory results/gwenlaya_v4/laya_runtime/thresholds_C.json (in-sample on C; never registered)"
                      if os.environ.get("GWENLAYA_LAYA_THRESHOLDS") else "none (Laya reports p, never decides)",
        "tasks": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
