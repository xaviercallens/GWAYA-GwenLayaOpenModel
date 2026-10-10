#!/usr/bin/env python3
"""Parity of gwaya.laya_runtime against the stored TPU predictions of the frozen Laya (n rows per head).

Rows come from $GWAYA_DATA_ROOT/laya_data/laya_calib/{router,calibrator}.jsonl (split C); the stored predictions are
results/gwenlaya_v4/laya/data/preds_C_{router,calibrator}.jsonl (bf16 on TPU v5e). Rows are picked deterministically
(evenly spaced). Writes a JSON with max/mean absolute difference and per-row latency; needs torch, transformers, peft.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STORED = ROOT / "results" / "gwenlaya_v4" / "laya" / "data"


def _jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def pick(rows: list, n: int) -> list:
    if n >= len(rows):
        return rows
    return [rows[i * len(rows) // n] for i in range(n)]


def parity(laya_dir: str, data_root: str, n: int = 20, device: str = "cpu", threads: int | None = None,
           verify: bool = True, backbone: str | None = None) -> dict[str, Any]:
    from gwaya.laya_runtime import LayaRuntime
    rt = LayaRuntime.load(laya_dir, device, verify=verify, threads=threads, backbone=backbone,
                          domains=("python", "rust", "math"))
    calib = Path(data_root) / "laya_data" / "laya_calib"
    out: dict[str, Any] = {}
    # router: stored rows have tier null and one prob per trained tier
    stored_r = {tuple(p["key"]): p["probs"] for p in _jsonl(STORED / "preds_C_router.jsonl")}
    rows = [r for r in _jsonl(calib / "router.jsonl") if (r["source"], r["id"]) in stored_r]
    diffs, lat = [], []
    for r in pick(rows, n):
        t0 = time.perf_counter()
        got = rt.router(r["prompt"], r["domain"])
        lat.append(time.perf_counter() - t0)
        want = stored_r[(r["source"], r["id"])]
        diffs += [abs(got[t] - w) for t, w in zip(rt.tiers, want)]
    out["router"] = {"rows": len(lat), "values": len(diffs), "max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / len(diffs),
                     "latency_s_mean": sum(lat) / len(lat), "latency_s_max": max(lat)}
    stored_c = {(p["key"][0], p["key"][1], p["tier"]): p["probs"][0] for p in _jsonl(STORED / "preds_C_calibrator.jsonl")}
    rows = [r for r in _jsonl(calib / "calibrator.jsonl") if (r["source"], r["id"], r["tier"]) in stored_c]
    diffs, lat, trunc, det = [], [], 0, []
    for r in pick(rows, n):
        feats = {"domain": r["domain"], "tier": r["tier"], "prompt": r["prompt"], "response": r["candidate"],
                 "gate_status": str(r["signals"].get("gate", "none")).upper(),
                 "gen_signals": {k: r["signals"].get(k) for k in ("mean_logprob", "min_logprob")}}
        t0 = time.perf_counter()
        raw = rt.scorer(feats)
        lat.append(time.perf_counter() - t0)
        want = stored_c[(r["source"], r["id"], r["tier"])]
        diffs.append(abs(raw - want))
        trunc += bool(feats["laya"]["truncated_candidate"])
        det.append({"id": f'{r["source"]}/{r["id"]}', "tier": r["tier"], "domain": r["domain"], "got": raw, "stored": want,
                    "n_tokens": feats["laya"]["n_tokens"]})
    out["calibrator"] = {"rows": len(lat), "max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / len(diffs),
                         "latency_s_mean": sum(lat) / len(lat), "latency_s_max": max(lat), "truncated_candidate_rows": trunc,
                         "rows_detail": det}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "gwenlaya_v4" / "laya_runtime" / "parity.json")
    a = ap.parse_args(argv)
    import peft
    import torch
    import transformers
    root = os.environ.get("GWAYA_DATA_ROOT", "/mnt/data/home/xavkal/gwaya-data")
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    res = parity(str(Path(root) / "laya_data" / "laya_v1"), root, a.n, a.device, a.threads)
    res["meta"] = {"command": "python scripts/laya_runtime_parity.py " + " ".join(argv if argv is not None else sys.argv[1:]),
                   "n_requested": a.n, "cpu": cpu, "threads": a.threads, "device": a.device, "precision": "fp32 (LoRA merged)",
                   "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                   "stored_predictions": "TPU v5e bf16 (torch 2.8.0, transformers 5.19.0, peft 0.21.2)",
                   "sha256_verified": True}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    brief = {k: v for k, v in res.items() if k != "calibrator"}
    brief["calibrator"] = {k: v for k, v in res["calibrator"].items() if k != "rows_detail"}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
