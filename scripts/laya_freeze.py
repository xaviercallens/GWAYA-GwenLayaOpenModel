#!/usr/bin/env python
"""Freeze Laya before E' is generated (docs/ANALYSIS_PLAN_E_TPU.md, A13b).

Fits the isotonic map raw score -> P(correct) on the calibration split C only, and writes the sha256 of every trained
artifact plus the map into a text file meant to be committed BEFORE any E' generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_laya as L  # noqa: E402


def sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", required=True, type=Path, help="dir holding router/ calibrator/ router_sh/ calibrator_sh/")
    ap.add_argument("--c-cal-rows", required=True, type=Path)
    ap.add_argument("--map-out", required=True, type=Path)
    ap.add_argument("--freeze-out", required=True, type=Path)
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    rows = L.read_jsonl(a.c_cal_rows)
    sc = L.raw_scores(L.read_jsonl(a.runs_dir / "calibrator" / "preds_calib.jsonl"))
    raw = np.array([sc[(r["source"], r["id"], r["tier"])] for r in rows])
    y = np.array([r["correct"] for r in rows])
    m = L.fit_isotonic(raw, y)
    a.map_out.write_text(json.dumps({"isotonic_fitted_on": "C", "n": len(rows), **m}, indent=1))
    lines = ["# Laya freeze file (A13b). Committed before any E' generation.", a.note, ""]
    for run in ("router", "calibrator", "router_sh", "calibrator_sh"):
        for f in ("adapter/adapter_model.safetensors", "heads.safetensors", "head_config.json", "preds_calib.jsonl"):
            p = a.runs_dir / run / f
            lines.append(f"{sha(p)}  {run}/{f}")
    lines.append(f"{sha(a.map_out)}  {a.map_out.name}  (isotonic on C, n={len(rows)})")
    a.freeze_out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
