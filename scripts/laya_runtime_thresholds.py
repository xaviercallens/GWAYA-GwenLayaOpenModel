#!/usr/bin/env python3
"""Per-domain tau_hi / tau_lo for the runtime, by the preregistered recipe, EXPLORATORY and in-sample on C.

No Laya tau_hi / tau_lo was registered (only tau_route = 0.5, docs/ANALYSIS_PLAN_E_TPU.md A13b). This applies the
preregistered recipe (docs/GWENLAYA_PREREGISTRATION.md: tau_hi by split-conformal risk control at alpha = 0.10,
tau_lo = largest threshold with LIKELY_WRONG precision >= 0.90, per domain) to the stored raw calibrator scores on C
mapped through the registered isotonic map (itself fitted on C, so p is in-sample and optimistic). It also reports,
descriptively, what those thresholds do on E' (already used by the registered evaluation; not a new test).
Stored predictions only; no model is run. The output is opt-in at runtime via GWENLAYA_LAYA_THRESHOLDS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gwaya.gwenlaya import IsotonicCalibrator, choose_tau_hi, choose_tau_lo  # noqa: E402

STORED = ROOT / "results" / "gwenlaya_v4" / "laya" / "data"
DOMAINS = ("python", "math")  # the runtime default; rust is at chance on fresh data and gets no thresholds


def _jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def joined(rows_path: Path, preds_path: Path) -> list[tuple[str, float, int]]:
    raw = {(p["key"][0], p["key"][1], p["tier"]): p["probs"][0] for p in _jsonl(preds_path)}
    return [(r["domain"], raw[(r["source"], r["id"], r["tier"])], int(r["correct"]))
            for r in _jsonl(rows_path) if (r["source"], r["id"], r["tier"]) in raw]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "gwenlaya_v4" / "laya_runtime" / "thresholds_C.json")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--lo-precision", type=float, default=0.90)
    a = ap.parse_args(argv)
    lake = Path(os.environ.get("GWAYA_DATA_ROOT", "/mnt/data/home/xavkal/gwaya-data")) / "laya_data"
    iso_d = json.loads((ROOT / "docs" / "laya_isotonic_C.json").read_text())
    iso = IsotonicCalibrator(iso_d["x"], iso_d["y"])
    c = joined(lake / "laya_calib" / "calibrator.jsonl", STORED / "preds_C_calibrator.jsonl")
    e = joined(lake / "laya_eprime" / "calibrator.jsonl", STORED / "preds_Eprime_calibrator.jsonl")
    out: dict[str, Any] = {}
    for d in DOMAINS:
        pc = [iso.predict(s)[0] for dom, s, _ in c if dom == d]
        yc = [y for dom, _, y in c if dom == d]
        hi, lo = choose_tau_hi(pc, yc, a.alpha), choose_tau_lo(pc, yc, a.lo_precision)
        pe = [iso.predict(s)[0] for dom, s, _ in e if dom == d]
        ye = [y for dom, _, y in e if dom == d]
        sel_hi = [y for p, y in zip(pe, ye) if p >= hi]
        sel_lo = [y for p, y in zip(pe, ye) if p <= lo]
        out[d] = {"tau_hi": hi, "tau_lo": lo, "n_C": len(yc),
                  "E_prime_descriptive": {"n": len(ye), "base_accuracy": sum(ye) / len(ye) if ye else None,
                                          "likely_correct_n": len(sel_hi),
                                          "likely_correct_error": (1 - sum(sel_hi) / len(sel_hi)) if sel_hi else None,
                                          "likely_wrong_n": len(sel_lo),
                                          "likely_wrong_precision": (1 - sum(sel_lo) / len(sel_lo)) if sel_lo else None}}
    res = {"thresholds": {d: {"tau_hi": v["tau_hi"], "tau_lo": v["tau_lo"]} for d, v in out.items()},
           "detail": out,
           "meta": {"status": "EXPLORATORY, in-sample on C (isotonic fitted on C); no Laya threshold was registered",
                    "alpha": a.alpha, "lo_precision": a.lo_precision, "recipe": "gwaya.gwenlaya.choose_tau_hi / choose_tau_lo",
                    "inputs": ["preds_C_calibrator.jsonl", "preds_Eprime_calibrator.jsonl", "docs/laya_isotonic_C.json"],
                    "caveat": "E' numbers assume the full-signal setting (log-probs present, candidate not truncated)"}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=lambda x: str(x)))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
