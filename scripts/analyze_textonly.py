#!/usr/bin/env python
"""A16: text-only Laya ablation vs raw log-prob, the hand-feature logistic regression and the full Laya (descriptive).

Per evaluation set (E' = math/Python, R' = fresh Rust) and tier: AUROC/AURC/ECE/Brier of the text-only score (isotonic map
fitted on C only), AUROC within each domain, and paired cluster-bootstrap differences in AUROC:
text-only minus raw log-prob, text-only minus the hand-feature LR, full Laya minus text-only. Nothing here is a hypothesis test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_laya as L  # noqa: E402
from scripts import analyze_study as A  # noqa: E402


def score_set(rows, preds_text, preds_full, iso_text, iso_full, lr_fn, tier, cluster_of):
    sub = [r for r in rows if r["tier"] == tier]
    key = lambda r: (r["source"], r["id"], r["tier"])  # noqa: E731
    st = L.raw_scores(preds_text)
    sf = L.raw_scores(preds_full)
    y = np.array([r["correct"] for r in sub], bool)
    dom = np.array([r["domain"] for r in sub])
    clu = np.array([cluster_of[(r["domain"], f'{r["source"]}/{r["id"]}')] for r in sub])
    text = L.apply_isotonic(iso_text, np.array([st[key(r)] for r in sub]))
    full = L.apply_isotonic(iso_full, np.array([sf[key(r)] for r in sub]))
    mlp = np.array([r["signals"].get("mean_logprob") if r["signals"].get("mean_logprob") is not None else -20.0 for r in sub])
    lp = np.exp(np.minimum(mlp, 0.0))
    lr = lr_fn(sub)
    return y, dom, clu, text, full, lp, lr


def analyze_one(y, dom, clu, text, full, lp, lr, n_boot, seed):
    domains = sorted(set(dom.tolist()))

    def stats(idx):
        yi = y[idx]
        out = [A.auroc(text[idx], yi), A.auroc(text[idx], yi) - A.auroc(lp[idx], yi),
               A.auroc(text[idx], yi) - A.auroc(lr[idx], yi), A.auroc(full[idx], yi) - A.auroc(text[idx], yi)]
        for d in domains:
            m = dom[idx] == d
            out.append(A.auroc(text[idx][m], yi[m]) if m.any() else float("nan"))
        return tuple(out)
    full_idx = np.arange(len(y))
    pt = stats(full_idx)
    boot = A.cluster_boot(stats, dom, clu, n_boot=n_boot, seed=seed)
    names = ["auroc_text", "d_auroc_text_minus_logprob", "d_auroc_text_minus_handlr", "d_auroc_full_minus_text"] + [f"auroc_text.{d}" for d in domains]
    res = {n: A.pct_ci(pt[i], boot[:, i]) for i, n in enumerate(names)}
    res["metrics_text"] = L.metrics_block(text, y)
    res["metrics_full"] = L.metrics_block(full, y)
    res["n"] = int(len(y))
    res["accuracy"] = float(y.mean())
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tiers", nargs=3, required=True)
    ap.add_argument("--train-cal-rows", required=True, type=Path)
    ap.add_argument("--c-cal-rows", required=True, type=Path)
    ap.add_argument("--c-text-preds", required=True, type=Path)
    ap.add_argument("--c-full-preds", required=True, type=Path)
    ap.add_argument("--set", action="append", nargs=5, required=True,
                    metavar=("NAME", "ROWS", "TEXT_PREDS", "FULL_PREDS", "TASKS"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    tiers = list(a.tiers)
    c_rows = L.read_jsonl(a.c_cal_rows)
    y_c = np.array([r["correct"] for r in c_rows])

    def raw_for(rows, path):
        sc = L.raw_scores(L.read_jsonl(path))
        return np.array([sc[(r["source"], r["id"], r["tier"])] for r in rows])
    iso_text = L.fit_isotonic(raw_for(c_rows, a.c_text_preds), y_c)
    iso_full = L.fit_isotonic(raw_for(c_rows, a.c_full_preds), y_c)
    lr_fn = L.fit_lr(L.read_jsonl(a.train_cal_rows), tiers)
    res: dict = {"meta": {"plan": "A16", "tiers": tiers, "n_boot": a.n_boot, "seed": a.seed, "n_C_rows": len(c_rows)}, "sets": {}}
    for name, rows_p, tp, fp, tasks_p in a.set:
        cluster_of = {(t["domain"], t["task_id"]): t["cluster"] for t in L.read_jsonl(Path(tasks_p))}
        rows = L.read_jsonl(Path(rows_p))
        for tier in tiers:
            arr = score_set(rows, L.read_jsonl(Path(tp)), L.read_jsonl(Path(fp)), iso_text, iso_full, lr_fn, tier, cluster_of)
            res["sets"].setdefault(name, {})[tier] = analyze_one(*arr, a.n_boot, a.seed)
            print(name, tier, round(res["sets"][name][tier]["auroc_text"]["point"], 3), flush=True)
    a.out.write_text(json.dumps(res, indent=1, default=lambda o: None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
