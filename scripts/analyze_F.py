#!/usr/bin/env python
"""A21 descriptive analyses on the fresh sets F (python, math, Lean 4): accuracy, gate, log-prob signals, cascade, Lean.

Nothing here uses the Laya (that is scripts/analyze_F_laya.py, run on the TPU predictions). Definitions are those of
A1-A5/A12 (docs/ANALYSIS_PLAN_E_TPU.md) re-used from scripts/analyze_study.py and scripts/analyze_cascades.py:
cluster = the task file's `cluster` field (one per source problem), resampled with replacement within each domain,
10,000 resamples, numpy default_rng(0), percentile 95 percent CIs.

  python scripts/analyze_F.py --study-dirs D/qwen3.5-2b-bf16 D/qwen3.5-4b-bf16 D/qwen3.5-9b-bf16 \
      --tiers qwen3.5-2b-bf16 qwen3.5-4b-bf16 qwen3.5-9b-bf16 --lean-scored results/.../lean_scored --out numbers_F.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gwaya.metrics_selective import holm, mcnemar_exact  # noqa: E402
from scripts import analyze_cascades as AC  # noqa: E402
from scripts import analyze_study as A  # noqa: E402

PRICE_CHIP_H = 1.20


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _mean(a: np.ndarray) -> float:
    return float(a.mean()) if len(a) else float("nan")


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def ci_clean(ci: dict[str, float]) -> dict[str, Any]:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in ci.items()}


def boot_all(specs: list[tuple[str, Callable[[np.ndarray], float]]], dom: np.ndarray, clu: np.ndarray,
             n_boot: int, seed: int) -> dict[str, dict[str, Any]]:
    full = np.arange(len(dom))
    point = [float(fn(full)) for _, fn in specs]
    boot = A.cluster_boot(lambda idx: tuple(fn(idx) for _, fn in specs), dom, clu, n_boot=n_boot, seed=seed)
    return {name: ci_clean(A.pct_ci(point[j], boot[:, j])) for j, (name, _) in enumerate(specs)}


def slices(tasks: list[dict]) -> dict[str, np.ndarray]:
    dom = np.array([t["domain"] for t in tasks])
    strat = np.array([t.get("stratum", "") for t in tasks])
    vis = np.array([str(t.get("gate_visible_in_prompt")) for t in tasks])
    py, ma = dom == "python", dom == "math"
    out = {
        "all_python_math": py | ma,
        "python": py, "python/LeetCode": py & (strat == "python/LeetCodeDataset"), "python/FSB": py & (strat == "python/FullStackBench"),
        "python/BCB": py & (strat == "python/BigCodeBench"),
        "python/gate_visible": py & (vis == "True"), "python/gate_not_visible": py & (vis == "False"),
        "python/gate_undecidable": py & (vis == "None"),
        "math": ma, "math/MATH-test": ma & (strat == "math/MATH-test"), "math/GSM8K": ma & (strat == "math/GSM8K"),
        "math/competition": ma & (strat == "math/competition"),
    }
    out["math/non_competition"] = out["math/MATH-test"] | out["math/GSM8K"]
    return out


def tier_registry(arr: dict[str, np.ndarray], minlp: np.ndarray, masks: dict[str, np.ndarray], tier: str,
                  ) -> list[tuple[str, Callable[[np.ndarray], float]]]:
    cor, gate, conf, trunc = arr["cor"], arr["gate_ok"], arr["conf"], arr["trunc"]
    specs: list[tuple[str, Callable[[np.ndarray], float]]] = []
    for sname, m in masks.items():
        def sel(i, m=m):
            return i[m[i]]
        P = f"{tier}|{sname}|"
        specs.append((P + "n", lambda i, sel=sel: float(len(sel(i)))))
        specs.append((P + "acc", lambda i, sel=sel: _mean(cor[sel(i)])))
        specs.append((P + "truncated", lambda i, sel=sel: _mean(trunc[sel(i)])))
        specs.append((P + "gate_coverage", lambda i, sel=sel: _mean(gate[sel(i)])))
        specs.append((P + "gate_precision", lambda i, sel=sel: A.answered_acc(gate[sel(i)], cor[sel(i)])))
        specs.append((P + "gate_recall", lambda i, sel=sel: _mean(gate[sel(i)][cor[sel(i)]]) if cor[sel(i)].any() else float("nan")))
        specs.append((P + "gate_false_accept", lambda i, sel=sel: _mean(gate[sel(i)][~cor[sel(i)]]) if (~cor[sel(i)]).any() else float("nan")))
        specs.append((P + "gate_confident_wrong_rate", lambda i, sel=sel: A.cwr(gate[sel(i)], cor[sel(i)])))
        specs.append((P + "logprob_matched_confident_wrong_rate",
                      lambda i, sel=sel: _matched(conf[sel(i)], cor[sel(i)], int(gate[sel(i)].sum()))))
        specs.append((P + "auroc_mean_logprob", lambda i, sel=sel: A.auroc(conf[sel(i)], cor[sel(i)])))
        specs.append((P + "aurc_mean_logprob", lambda i, sel=sel: A.aurc(conf[sel(i)], cor[sel(i)])))
        specs.append((P + "auroc_min_logprob", lambda i, sel=sel: A.auroc(minlp[sel(i)], cor[sel(i)])))
        specs.append((P + "aurc_min_logprob", lambda i, sel=sel: A.aurc(minlp[sel(i)], cor[sel(i)])))
        specs.append((P + "auroc_gate_binary", lambda i, sel=sel: A.auroc(gate[sel(i)].astype(float), cor[sel(i)])))
    return specs


def _matched(conf: np.ndarray, cor: np.ndarray, k: int) -> float:
    """CWR of 'answer the k most confident tasks' over all n tasks (ties wrong first), as analyze_e_tpu.matched_cwr."""
    n = len(conf)
    if n == 0:
        return float("nan")
    order = np.lexsort((cor.astype(int), -np.where(np.isnan(conf), -np.inf, conf)))
    top = order[:max(0, min(k, n))]
    return float((~cor[top]).sum() / n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="+", required=True, type=Path, help="tasks_F_python.jsonl tasks_F_math.jsonl")
    ap.add_argument("--study-dirs", nargs="+", required=True, type=Path, help="scored dirs (rows.jsonl, gens.jsonl)")
    ap.add_argument("--overlay", nargs="*", default=[], type=Path, help="serial re-check rows files; later rows win")
    ap.add_argument("--tiers", nargs=3, required=True)
    ap.add_argument("--raw-dirs", nargs="*", default=[], type=Path, help="TPU raw dirs (for the compile warm-up chunk)")
    ap.add_argument("--lean-scored", type=Path)
    ap.add_argument("--lean-tasks", type=Path)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    tiers = list(a.tiers)

    tasks = sorted((t for p in a.tasks for t in read_jsonl(p)), key=lambda t: (t["domain"], t["task_id"]))
    keys = [(t["domain"], t["task_id"]) for t in tasks]
    rows = [r for d in a.study_dirs for r in read_jsonl(d / "rows.jsonl")] + [r for p in a.overlay for r in read_jsonl(p)]
    gens = [g for d in a.study_dirs
            for g in read_jsonl(d / "gens.jsonl" if (d / "gens.jsonl").exists() else d / "gens_summary.jsonl")]
    arrays = {m: AC.tier_arrays(m, keys, rows, gens) for m in tiers}
    ans = {m: {(g["domain"], g["task_id"]): g for g in gens if g["model"] == m and g.get("temperature") == 0.0
               and g.get("kind") != "pot"} for m in tiers}
    def _min(g: dict) -> float:
        if g.get("min_logprob") is not None:
            return float(g["min_logprob"])
        return float(min(g["token_logprobs"])) if g.get("token_logprobs") else float("nan")
    minlp = {m: np.array([_min(ans[m][k]) for k in keys]) for m in tiers}
    dom = np.array([k[0] for k in keys])
    clu = np.array([t["cluster"] for t in tasks])
    masks = slices(tasks)
    res: dict[str, Any] = {"meta": {"plan": "A21 (descriptive part)", "n_boot": a.n_boot, "seed": a.seed, "tiers": tiers,
                                    "n_tasks": len(keys), "n_clusters": int(len(set(clu.tolist()))),
                                    "slice_n": {k: int(v.sum()) for k, v in masks.items()},
                                    "cost_unit": "accelerator chip-seconds (9B x 4 chips); USD at 1.20 per chip-hour"}}

    # ---- accuracy, gate, signals per tier x slice ------------------------------------------------------------
    specs = []
    for m in tiers:
        specs += tier_registry(arrays[m], minlp[m], masks, m)
    flat = boot_all(specs, dom, clu, a.n_boot, a.seed)
    per: dict[str, dict[str, dict[str, Any]]] = {}
    for name, ci in flat.items():
        tier, sname, metric = name.split("|")
        per.setdefault(sname, {}).setdefault(tier, {})[metric] = ci
    res["per_slice"] = per

    # ---- cascade 2B -> 4B -> 9B vs always-9B at the registered cost model -------------------------------------
    warm = np.zeros(len(keys), bool)
    for rd in a.raw_dirs:
        for f in list(rd.glob("raw_qwen*.jsonl")) + list(rd.glob("raw_pot_qwen*.jsonl")):
            ids = {(r["domain"], r["task_id"]) for r in map(json.loads, f.read_text().splitlines()) if r.get("chunk") == 0}
            warm |= np.array([k in ids for k in keys])
    arr = [arrays[m] for m in tiers]
    casc = AC.offline_cascade(arr)
    casc12 = AC.offline_cascade(arr[:2])
    big, mid, small = arr[-1], arr[1], arr[0]
    cor_any = np.column_stack([x["cor"] for x in arr]).any(axis=1)
    first = np.where(cor_any, np.column_stack([x["cor"] for x in arr]).argmax(axis=1), 2)
    ocost = np.column_stack([x["cost"] for x in arr])[np.arange(len(keys)), first]
    cspecs = []
    for sname in ("all_python_math", "python", "math", "math/non_competition", "math/competition",
                  "python/gate_not_visible", "python/gate_visible"):
        m = masks[sname]
        for nowarm in (False, True):
            def sel(i, m=m, nowarm=nowarm):
                j = i[m[i]]
                return j[~warm[j]] if nowarm else j
            sfx = f"{sname}|{'nowarm' if nowarm else 'warm_included'}"
            acc = lambda c, sel=sel: (lambda i: _mean(c[sel(i)]))
            cspecs += [
                (f"{sfx}|acc_cascade3", lambda i, sel=sel: _mean(casc["cor"][sel(i)])),
                (f"{sfx}|acc_always_9b", lambda i, sel=sel: _mean(big["cor"][sel(i)])),
                (f"{sfx}|acc_always_4b", lambda i, sel=sel: _mean(mid["cor"][sel(i)])),
                (f"{sfx}|acc_always_2b", lambda i, sel=sel: _mean(small["cor"][sel(i)])),
                (f"{sfx}|d_acc_cascade3_minus_9b", lambda i, sel=sel: _mean(casc["cor"][sel(i)]) - _mean(big["cor"][sel(i)])),
                (f"{sfx}|oracle_acc", lambda i, sel=sel: _mean(cor_any[sel(i)])),
                (f"{sfx}|cost_per_task_cascade3", lambda i, sel=sel: _mean(casc["cost"][sel(i)])),
                (f"{sfx}|cost_per_task_always_9b", lambda i, sel=sel: _mean(big["cost"][sel(i)])),
                (f"{sfx}|cost_per_task_always_9b_with_program", lambda i, sel=sel: _mean((big["cost"] + big["pot"])[sel(i)])),
                (f"{sfx}|cost_ratio_cascade3_over_9b", lambda i, sel=sel: _ratio(casc["cost"][sel(i)].sum(), big["cost"][sel(i)].sum())),
                (f"{sfx}|cost_ratio_cascade3_over_9b_with_program",
                 lambda i, sel=sel: _ratio(casc["cost"][sel(i)].sum(), (big["cost"] + big["pot"])[sel(i)].sum())),
                (f"{sfx}|cost_per_correct_cascade3", lambda i, sel=sel: _ratio(casc["cost"][sel(i)].sum(), casc["cor"][sel(i)].sum())),
                (f"{sfx}|cost_per_correct_always_9b", lambda i, sel=sel: _ratio(big["cost"][sel(i)].sum(), big["cor"][sel(i)].sum())),
                (f"{sfx}|escalated_share", lambda i, sel=sel: _mean(casc["escalated"][sel(i)])),
                (f"{sfx}|answered_share", lambda i, sel=sel: _mean(casc["answered"][sel(i)])),
                (f"{sfx}|confident_wrong_rate_cascade3", lambda i, sel=sel: A.cwr(casc["answered"][sel(i)], casc["cor"][sel(i)])),
                (f"{sfx}|cost_per_task_oracle", lambda i, sel=sel: _mean(ocost[sel(i)])),
                (f"{sfx}|acc_cascade2_2b_4b", lambda i, sel=sel: _mean(casc12["cor"][sel(i)])),
                (f"{sfx}|cost_ratio_cascade2_2b_4b_over_9b", lambda i, sel=sel: _ratio(casc12["cost"][sel(i)].sum(), big["cost"][sel(i)].sum())),
            ]
    casc_ci = boot_all(cspecs, dom, clu, a.n_boot, a.seed)
    cb, cc = casc["cor"], big["cor"]
    mc = {"cascade_only_correct": int((cb & ~cc).sum()), "9b_only_correct": int((~cb & cc).sum())}
    mc["p_mcnemar_two_sided"] = mcnemar_exact(mc["cascade_only_correct"], mc["9b_only_correct"], "two-sided")
    res["cascade"] = {"definition": "2B -> gate -> 4B -> gate -> 9B; stop at first VERIFIED, else last candidate unanswered; "
                                    "cost = consulted tiers' chip-seconds + math program seconds of every consulted tier "
                                    "(A10/A12 charge); always-9B plain = its own answer seconds (B2), also shown with its program",
                      "n_warm_tasks_excluded_in_nowarm": int(warm.sum()), "ci": casc_ci, "mcnemar_vs_9b_pooled": mc,
                      "total_cost_chip_s": {"cascade3": float(casc["cost"].sum()), "always_9b": float(big["cost"].sum()),
                                             "always_9b_with_program": float((big["cost"] + big["pot"]).sum()),
                                             "always_4b": float(mid["cost"].sum()), "always_2b": float(small["cost"].sum())}}

    # ---- Lean (descriptive; gate == scorer) ------------------------------------------------------------------
    if a.lean_scored and a.lean_tasks:
        lt = read_jsonl(a.lean_tasks)
        ids = sorted(t["task_id"] for t in lt)
        lcl = {t["task_id"]: t["cluster"] for t in lt}
        proved, trunc, secs, match = {}, {}, {}, {}
        for m in tiers:
            sc = {r["task_id"]: r for r in read_jsonl(a.lean_scored / f"scored_raw_{m}.jsonl")}
            proved[m] = np.array([sc[i]["status"] == "VERIFIED" for i in ids])
            trunc[m] = np.array([sc[i].get("finish_reason") == "length" for i in ids])
            chips = 4 if "9b" in m else 1
            secs[m] = float(sum(sc[i].get("accel_s", 0.0) for i in ids) * chips)
            match[m] = {k: sum(1 for i in ids if sc[i]["status"] == "VERIFIED" and (sc[i].get("match") or "unknown") == k)
                        for k in ("exact", "defeq", "unknown")}
        ld = np.array(["lean4"] * len(ids))
        lc = np.array([lcl[i] for i in ids])
        lspecs = []
        for m in tiers:
            lspecs += [(f"{m}|proof_rate", lambda i, m=m: _mean(proved[m][i])),
                       (f"{m}|truncated", lambda i, m=m: _mean(trunc[m][i]))]
        lspecs += [("any_tier_proof_rate", lambda i: _mean(np.column_stack([proved[m] for m in tiers]).any(axis=1)[i]))]
        lci = boot_all(lspecs, ld, lc, a.n_boot, a.seed)
        pv = {m: set(np.array(ids)[proved[m]].tolist()) for m in tiers}
        res["lean"] = {"n_tasks": len(ids), "gate_equals_scorer": True, "ci": lci,
                       "proved_n": {m: int(proved[m].sum()) for m in tiers},
                       "verified_match_type": match,
                       "chip_seconds_total": secs,
                       "chip_seconds_per_proof": {m: (secs[m] / proved[m].sum() if proved[m].sum() else None) for m in tiers},
                       "usd_total": {m: secs[m] * PRICE_CHIP_H / 3600 for m in tiers},
                       "overlap": {f"{x}&{y}": len(pv[x] & pv[y]) for k, x in enumerate(tiers) for y in tiers[k + 1:]},
                       "union_n": len(set().union(*pv.values())),
                       "only_largest_n": len(pv[tiers[2]] - pv[tiers[0]] - pv[tiers[1]]),
                       "smaller_tier_proves_n_not_proved_by_9b": len((pv[tiers[0]] | pv[tiers[1]]) - pv[tiers[2]])}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
