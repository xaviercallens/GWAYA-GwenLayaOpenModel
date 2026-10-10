#!/usr/bin/env python
"""A22: paired layout ablation of the Laya calibrator (docs/ANALYSIS_PLAN_E_TPU.md, Addendum A22). Exploratory.

v1 = frozen calibrator (candidate last, cut at 512 tokens); v2 = the SAME recipe retrained with `--layout candidate_first`.
Both scored the SAME stored rows (E' python+math, R' fresh Rust; optional C for the isotonic maps).

Per set (and per tier): AUROC, AURC, ECE (15 equal-mass bins), Brier for v1, v2, the hand-feature logistic regression
(scripts/analyze_laya.fit_lr, trained on the training rows), raw mean log-prob (exp of it) and the gate; paired cluster
bootstrap (10,000, seed 0, cluster = problem, stratified by domain as scripts/analyze_study.cluster_boot) of AUROC and
AURC differences v2 - v1, v2 - gate, v2 - hand LR; and the fraction of rows whose candidate is inside the 512-token
window under each layout (needs the ModernBERT tokenizer; skipped when not given).

Decision rule (fixed in A22 before the run): v2 is "better" only if the AUROC difference v2 - v1 has a 95% CI above
zero on E' AND on R'. Nothing else enters the rule.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gwaya import laya_features as F  # noqa: E402
from scripts import analyze_laya as AL  # noqa: E402
from scripts import analyze_study as A  # noqa: E402

N_BOOT = 10000
SEED = 0


def row_id(r: dict[str, Any]) -> tuple[str, str, str]:
    return (str(r["source"]), str(r["id"]), str(r.get("tier")))


def load_preds(path: Path) -> dict[tuple[str, str, str], float]:
    out = {}
    for p in AL.read_jsonl(path):
        out[(str(p["key"][0]), str(p["key"][1]), str(p.get("tier")))] = float(p["probs"][0])
    return out


def aligned(rows: list[dict], preds: dict, name: str) -> np.ndarray:
    miss = [row_id(r) for r in rows if row_id(r) not in preds]
    if miss:
        raise KeyError(f"{name}: {len(miss)} row keys have no prediction, e.g. {miss[:3]}")
    return np.array([preds[row_id(r)] for r in rows], dtype=float)


def clusters(rows: list[dict], cluster_of: dict[tuple[str, str], str] | None) -> np.ndarray:
    """Cluster = problem: the tasks-file cluster when given, else source/id (a problem's tiers share it)."""
    out = []
    for r in rows:
        pid = f'{r["source"]}/{r["id"]}'
        out.append((cluster_of or {}).get((r["domain"], pid), pid))
    return np.array(out)


def calibrated(raw: np.ndarray, c_raw: np.ndarray | None, c_y: np.ndarray | None) -> np.ndarray:
    """Isotonic map fitted on C only (pooled), as A13b; identity when C is not supplied (flagged in the output)."""
    if c_raw is None or c_y is None:
        return raw
    return AL.apply_isotonic(AL.fit_isotonic(c_raw, c_y), raw)


def _logprob_score(rows: list[dict]) -> np.ndarray:
    mlp = np.array([(r["signals"].get("mean_logprob") if r["signals"].get("mean_logprob") is not None else -20.0) for r in rows])
    return np.exp(np.minimum(mlp, 0.0))


def _gate_score(rows: list[dict]) -> np.ndarray:
    return np.array([str(r["signals"].get("gate")).lower() == "verified" for r in rows], dtype=float)


PAIRS = (("v2", "v1"), ("v2", "gate"), ("v2", "hand_lr"), ("v2", "raw_logprob"))


def compare(y: np.ndarray, dom: np.ndarray, clu: np.ndarray, raw: dict[str, np.ndarray], cal: dict[str, np.ndarray],
            n_boot: int, seed: int) -> dict[str, Any]:
    """Metrics per arm (AUROC on raw scores, AURC/ECE/Brier on calibrated ones) and paired bootstrap differences."""
    arms = list(raw)
    out: dict[str, Any] = {"n": int(len(y)), "accuracy": float(y.mean()) if len(y) else float("nan"),
                           "n_clusters": int(len(set(clu.tolist()))), "metrics": {}, "paired": {}}
    for a in arms:
        out["metrics"][a] = {"auroc": A.auroc(raw[a], y), "aurc": A.aurc(cal[a], y),
                             "ece15": A.ece_equal_mass(np.clip(cal[a], 0, 1), y), "brier": A.brier(np.clip(cal[a], 0, 1), y)}
    pairs = [(a, b) for a, b in PAIRS if a in raw and b in raw]

    def stats(idx):
        yi = y[idx]
        v = []
        for a, b in pairs:
            v += [A.auroc(raw[a][idx], yi) - A.auroc(raw[b][idx], yi), A.aurc(cal[a][idx], yi) - A.aurc(cal[b][idx], yi)]
        return tuple(v)

    if len(y):
        point = stats(np.arange(len(y)))
        boot = A.cluster_boot(stats, dom, clu, n_boot=n_boot, seed=seed)
        for i, (a, b) in enumerate(pairs):
            out["paired"][f"{a}_minus_{b}"] = {"dAUROC": A.pct_ci(point[2 * i], boot[:, 2 * i]),
                                               "dAURC": A.pct_ci(point[2 * i + 1], boot[:, 2 * i + 1])}
    return out


def analyze_set(rows: list[dict], v1: dict, v2: dict, lr_scores: np.ndarray, cal_maps: dict[str, tuple[Any, Any]],
                cluster_of: dict | None, tiers: list[str], n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, Any]:
    """cal_maps: {"v1": (c_raw, c_y), "v2": (c_raw, c_y)} for the isotonic maps (None entries = identity)."""
    raw = {"v1": aligned(rows, v1, "v1"), "v2": aligned(rows, v2, "v2"), "hand_lr": np.asarray(lr_scores, float),
           "raw_logprob": _logprob_score(rows), "gate": _gate_score(rows)}
    cal = {"v1": calibrated(raw["v1"], *cal_maps.get("v1", (None, None))), "v2": calibrated(raw["v2"], *cal_maps.get("v2", (None, None))),
           "hand_lr": raw["hand_lr"], "raw_logprob": raw["raw_logprob"], "gate": raw["gate"]}
    y = np.array([bool(r["correct"]) for r in rows])
    dom = np.array([r["domain"] for r in rows])
    clu = clusters(rows, cluster_of)
    res = {"pooled": compare(y, dom, clu, raw, cal, n_boot, seed), "per_tier": {}}
    for t in tiers:
        idx = np.array([i for i, r in enumerate(rows) if r.get("tier") == t], dtype=int)
        if len(idx):
            res["per_tier"][t] = compare(y[idx], dom[idx], clu[idx], {k: v[idx] for k, v in raw.items()},
                                         {k: v[idx] for k, v in cal.items()}, n_boot, seed)
    return res


def window_fractions(rows: list[dict], tokenizer: Any, layouts: tuple[str, ...] = ("prompt_first", "candidate_first"),
                     max_len: int = F.MAX_LEN) -> dict[str, Any]:
    """Share of rows whose candidate is at least partly / fully inside the window, per layout (and v1 per domain)."""
    out: dict[str, Any] = {"n": len(rows)}
    for lay in layouts:
        w = [F.candidate_window(r, lay, tokenizer, max_len) for r in rows]
        n = max(1, len(w))
        out[lay] = {"any_visible": sum(x["any_visible"] for x in w) / n, "fully_visible": sum(x["fully_visible"] for x in w) / n,
                    "entirely_outside": sum(not x["any_visible"] for x in w) / n,
                    "mean_visible_token_share": float(np.mean([x["visible_tokens"] / max(1, x["candidate_tokens"]) for x in w])) if w else float("nan")}
    return out


def decision(eprime: dict[str, Any], rfresh: dict[str, Any]) -> dict[str, Any]:
    """The A22 rule: dAUROC(v2 - v1) CI lower bound > 0 on BOTH E' (pooled) and R' (pooled)."""
    def lo(s):
        v = s["pooled"]["paired"].get("v2_minus_v1", {}).get("dAUROC", {}).get("lo", float("nan"))
        return v
    le, lr = lo(eprime), lo(rfresh)
    better = bool(le > 0 and lr > 0)
    return {"rule": "v2 better only if CI(dAUROC v2-v1) > 0 on E' AND on R'", "eprime_lo": le, "rfresh_lo": lr,
            "v2_better": better, "window_explanation_supported": better}


def read_rows(p: Path | None) -> list[dict]:
    return AL.read_jsonl(p) if p else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tiers", nargs="+", default=["qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16"])
    ap.add_argument("--train-rows", type=Path, required=True)
    ap.add_argument("--eprime-rows", type=Path, required=True)
    ap.add_argument("--rfresh-rows", type=Path, required=True)
    ap.add_argument("--c-rows", type=Path)
    ap.add_argument("--v1-eprime", type=Path, required=True)
    ap.add_argument("--v2-eprime", type=Path, required=True)
    ap.add_argument("--v1-rfresh", type=Path, required=True)
    ap.add_argument("--v2-rfresh", type=Path, required=True)
    ap.add_argument("--v1-c", type=Path)
    ap.add_argument("--v2-c", type=Path)
    ap.add_argument("--tasks", nargs="*", type=Path, default=[], help="tasks files with {domain, task_id, cluster}")
    ap.add_argument("--backbone", help="local ModernBERT dir or name (HF cache) for the window fractions")
    ap.add_argument("--v2-layout", default="candidate_first", choices=list(F.LAYOUTS))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    train, e_rows, r_rows = read_rows(a.train_rows), read_rows(a.eprime_rows), read_rows(a.rfresh_rows)
    cluster_of = {(t["domain"], t["task_id"]): t["cluster"] for f in a.tasks for t in AL.read_jsonl(f)} or None
    lr_fn = AL.fit_lr(train, list(a.tiers))
    maps: dict[str, tuple[Any, Any]] = {}
    if a.c_rows and a.v1_c and a.v2_c:
        c_rows = read_rows(a.c_rows)
        y_c = np.array([r["correct"] for r in c_rows], dtype=int)
        maps = {"v1": (aligned(c_rows, load_preds(a.v1_c), "v1 C"), y_c), "v2": (aligned(c_rows, load_preds(a.v2_c), "v2 C"), y_c)}
    res: dict[str, Any] = {"meta": {"plan": "A22", "n_boot": a.n_boot, "seed": a.seed, "tiers": list(a.tiers), "v2_layout": a.v2_layout,
                                    "isotonic_fitted_on_C": bool(maps), "n_train_rows": len(train),
                                    "caveat": "exploratory; E' and R' already seen; single seed; no shuffled-label control"}}
    res["eprime"] = analyze_set(e_rows, load_preds(a.v1_eprime), load_preds(a.v2_eprime), lr_fn(e_rows), maps, cluster_of, list(a.tiers),
                                a.n_boot, a.seed)
    res["rfresh"] = analyze_set(r_rows, load_preds(a.v1_rfresh), load_preds(a.v2_rfresh), lr_fn(r_rows), maps, cluster_of, list(a.tiers),
                                a.n_boot, a.seed)
    res["decision"] = decision(res["eprime"], res["rfresh"])
    if a.backbone:
        from transformers import AutoTokenizer
        from gwaya.laya_runtime import resolve_backbone
        tok = AutoTokenizer.from_pretrained(resolve_backbone(a.backbone), local_files_only=True)
        lays = ("prompt_first",) + (() if a.v2_layout == "prompt_first" else (a.v2_layout,))
        res["window"] = {"eprime": window_fractions(e_rows, tok, lays), "rfresh": window_fractions(r_rows, tok, lays)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else o))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
