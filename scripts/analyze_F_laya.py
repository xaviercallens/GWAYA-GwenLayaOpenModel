#!/usr/bin/env python
"""A21: frozen-Laya evaluation on the fresh sets F (python + math), registered analyses H1 and H3, Holm over {H1, H3}.

Runs on the prediction files written by the TPU stage (scripts/tpu/launch_f_laya_predict.sh ->
out/{calibrator,router}/preds_F.jsonl). Definitions are those of A13b (scripts/analyze_laya.py, re-used):

  H1 = AURC difference, Laya calibrator P(correct) (through the FROZEN isotonic map of C, docs/laya_isotonic_C.json,
       its sha256 is checked against docs/LAYA_FREEZE.txt) minus raw mean log-prob (B3); lower AURC is better.
  H3 = selective-risk difference at the gate's own coverage: risk of the k answers Laya ranks highest minus the risk of
       the gate-VERIFIED answers, k = the number of VERIFIED answers (B5); lower is better.
  Primary tier Qwen3.5-9B, python and math pooled (A21: 908 tasks). Paired cluster bootstrap, 10,000 resamples,
  numpy default_rng(0), percentile 95 percent CI, cluster = the task's `cluster` field, resampled within domain.
  Supported only if the CI excludes zero in Laya's favour (upper bound < 0) after Holm over {H1, H3}.

Secondary (descriptive, same estimators, no Holm): per tier, per domain and per stratum; exploratory: Laya vs the
hand-feature logistic regression trained on the same rows as the Laya (--train-cal-rows), and the router policy with the
already-chosen tau = 0.5 (not re-chosen) vs the static ladder and always-9B (--study-dirs and --router-preds).

  python scripts/analyze_F_laya.py --cal-rows .../calibrator_F.jsonl --cal-preds out/calibrator/preds_F.jsonl \
      --tasks tasks_F_python.jsonl tasks_F_math.jsonl --out numbers_F_laya.json [--router-preds ... --study-dirs ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_cascades as AC  # noqa: E402
from scripts import analyze_laya as AL  # noqa: E402

TIERS = ("qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16")
STRATA = ("python", "math", "python/LeetCode", "python/FSB", "python/BCB", "python/gate_visible", "python/gate_not_visible",
          "math/MATH-test", "math/GSM8K", "math/competition")
_STRAT = {"python/LeetCode": "python/LeetCodeDataset", "python/FSB": "python/FullStackBench", "python/BCB": "python/BigCodeBench",
          "math/MATH-test": "math/MATH-test", "math/GSM8K": "math/GSM8K", "math/competition": "math/competition"}
PRIMARY_NAMES = ("H1_dAURC_laya_minus_logprob", "H3_dRisk_at_gate_coverage_laya_minus_gate")


def clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)) and not math.isfinite(float(o)):
        return None
    if isinstance(o, np.generic):
        return o.item()
    return o


def frozen_map(path: Path, freeze: Path | None) -> dict[str, list[float]]:
    if freeze is not None:
        want = {line.split()[1]: line.split()[0] for line in freeze.read_text().splitlines()
                if len(line.split()) >= 2 and len(line.split()[0]) == 64}
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if want.get(path.name) != got:
            raise SystemExit(f"isotonic map {path.name}: sha256 {got} != frozen {want.get(path.name)}")
    m = json.loads(path.read_text())
    return {"x": m["x"], "y": m["y"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cal-rows", required=True, type=Path, help="calibrator_F.jsonl (python + math rows with labels)")
    ap.add_argument("--cal-preds", required=True, type=Path, help="calibrator preds_F.jsonl (raw scores)")
    ap.add_argument("--tasks", nargs="+", required=True, type=Path)
    ap.add_argument("--isotonic", type=Path, default=ROOT / "docs" / "laya_isotonic_C.json")
    ap.add_argument("--freeze", type=Path, default=ROOT / "docs" / "LAYA_FREEZE.txt")
    ap.add_argument("--train-cal-rows", type=Path, help="laya_train/calibrator.jsonl (hand-feature LR, exploratory)")
    ap.add_argument("--router-preds", type=Path)
    ap.add_argument("--study-dirs", nargs="*", default=[], type=Path)
    ap.add_argument("--overlay", nargs="*", default=[], type=Path)
    ap.add_argument("--tiers", nargs=3, default=list(TIERS))
    ap.add_argument("--primary", default=TIERS[2])
    ap.add_argument("--tau", type=float, default=0.5, help="the already chosen router threshold (A13b); not re-chosen")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    tiers = list(a.tiers)

    tinfo = {(t["domain"], t["task_id"]): t for p in a.tasks for t in AL.read_jsonl(p)}
    rows = AL.read_jsonl(a.cal_rows)
    preds = {(p["key"][0], p["key"][1], p.get("tier")): p["probs"][0] for p in AL.read_jsonl(a.cal_preds)}
    miss = [r for r in rows if (r["source"], r["id"], r["tier"]) not in preds]
    if miss:
        raise SystemExit(f"{len(miss)} calibrator rows without a prediction, e.g. {(miss[0]['source'], miss[0]['id'], miss[0]['tier'])}")
    iso = frozen_map(a.isotonic, a.freeze if a.freeze and a.freeze.exists() else None)
    raw = np.array([preds[(r["source"], r["id"], r["tier"])] for r in rows])
    laya = AL.apply_isotonic(iso, raw)
    lr = None
    if a.train_cal_rows:
        lr = AL.fit_lr(AL.read_jsonl(a.train_cal_rows), tiers)(rows)

    task_of = [tinfo[(r["domain"], f'{r["source"]}/{r["id"]}')] for r in rows]
    tier_a = np.array([r["tier"] for r in rows])
    dom = np.array([r["domain"] for r in rows])
    clu = np.array([t["cluster"] for t in task_of])
    stratum = np.array([t.get("stratum", "") for t in task_of])
    vis = np.array([str(t.get("gate_visible_in_prompt")) for t in task_of])
    y = np.array([bool(r["correct"]) for r in rows])
    mlp = np.array([r["signals"].get("mean_logprob") if r["signals"].get("mean_logprob") is not None else -20.0 for r in rows])
    gate = np.array([str(r["signals"].get("gate")).lower() == "verified" for r in rows])

    def mask_of(name: str) -> np.ndarray:
        if name == "python":
            return dom == "python"
        if name == "math":
            return dom == "math"
        if name == "python/gate_visible":
            return (dom == "python") & (vis == "True")
        if name == "python/gate_not_visible":
            return (dom == "python") & (vis == "False")
        return stratum == _STRAT[name]

    res: dict[str, Any] = {"meta": {"plan": "A21 / A13b", "tiers": tiers, "primary": a.primary, "n_boot": a.n_boot, "seed": a.seed,
                                    "n_rows": len(rows), "isotonic": str(a.isotonic.name), "tau": a.tau,
                                    "cluster_unit": "task cluster field (one per source problem)"},
                           "primary": {}, "per_tier": {}, "secondary": {}}

    def block(m: np.ndarray, with_lr: bool) -> dict[str, Any]:
        idx = np.where(m)[0]
        if len(idx) == 0:
            return {"n": 0}
        b: dict[str, Any] = {"n": int(len(idx)), "accuracy": float(y[idx].mean()), "gate_coverage": float(gate[idx].mean()),
                             "metrics": {"laya": AL.metrics_block(laya[idx], y[idx]),
                                         "raw_logprob": AL.metrics_block(np.exp(np.minimum(mlp[idx], 0.0)), y[idx]),
                                         "gate": AL.metrics_block(gate[idx].astype(float), y[idx])}}
        if lr is not None:
            b["metrics"]["hand_lr"] = AL.metrics_block(lr[idx], y[idx])
        lr_i = lr[idx] if lr is not None else laya[idx]
        b["inference"] = AL.confirmatory(dom[idx], clu[idx], y[idx], laya[idx], mlp[idx], gate[idx], lr_i, a.n_boot, a.seed)
        if lr is None:
            for k in [k for k in b["inference"] if k.startswith("X_")]:
                b["inference"].pop(k)
        return b

    for tier in tiers:
        res["per_tier"][tier] = block(tier_a == tier, True)
    prim = res["per_tier"][a.primary]
    res["primary"] = {"definition": "tier " + a.primary + ", python + math pooled, Holm over {H1, H3}",
                      "n": prim["n"], "inference": prim["inference"]}
    for tier in tiers:  # p-values and 'supported' are meaningful for the primary tier only
        if tier != a.primary:
            for k in [k for k in res["per_tier"][tier]["inference"] if k.endswith(("_p_one_sided", "_p_holm", "_supported"))]:
                res["per_tier"][tier]["inference"].pop(k)
    for tier in tiers:
        for s in STRATA:
            b = block((tier_a == tier) & mask_of(s), True)
            if b.get("n"):
                for k in [k for k in b["inference"] if k.endswith(("_p_one_sided", "_p_holm", "_supported"))]:
                    b["inference"].pop(k)
            res["secondary"].setdefault(s, {})[tier] = b

    # router policy (exploratory; tau = 0.5 as registered in A13b, not re-chosen)
    if a.router_preds and a.study_dirs:
        srows = [r for d in a.study_dirs for r in AL.read_jsonl(d / "rows.jsonl")] + [r for p in a.overlay for r in AL.read_jsonl(p)]
        sgens = [g for d in a.study_dirs
                 for g in AL.read_jsonl(d / "gens.jsonl" if (d / "gens.jsonl").exists() else d / "gens_summary.jsonl")]
        tl = sorted(({"domain": d, "task_id": tid, "cluster": t["cluster"]} for (d, tid), t in tinfo.items()
                     if d in ("python", "math")), key=lambda t: (t["domain"], t["task_id"]))
        keys = [(t["domain"], t["task_id"]) for t in tl]
        arrays = [AC.tier_arrays(m, keys, srows, sgens) for m in tiers]
        pr = {(p["key"][0], p["key"][1]): p["probs"] for p in AL.read_jsonl(a.router_preds)}
        rp = np.array([pr[(t["task_id"].partition("/")[0], t["task_id"].partition("/")[2])] for t in tl])
        d_ = np.array([t["domain"] for t in tl])
        c_ = np.array([t["cluster"] for t in tl])
        res["router_policy"] = AL.router_eval(d_, c_, arrays, rp, a.tau, a.n_boot, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(clean(res), indent=1))
    inf = prim["inference"]
    print(json.dumps({n: {k: inf[n][k] for k in ("point", "lo", "hi")} for n in PRIMARY_NAMES}
                     | {k: inf[k] for k in ("H1_p_holm", "H3_p_holm", "H1_supported", "H3_supported")}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
