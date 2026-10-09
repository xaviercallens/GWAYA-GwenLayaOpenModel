#!/usr/bin/env python
"""A12: gate-driven cascades over 2-3 tiers, replayed offline from per-tier rows (docs/ANALYSIS_PLAN_E_TPU.md, A12).

The ladder rule is the one gwaya.gwenlaya.GwenLaya applies with no router, scorer or calibrator: consult the tiers in
order; stop at the first tier whose executed gate VERIFIES (answered); if none does, the last tier's candidate is the
final one (not answered). Cost = sum over consulted tiers of the tier's attributed generation seconds, plus its
program-of-thought seconds on math tasks (the gate's program is a second accelerator generation). Code gates run on
CPU and cost no accelerator time. `validate_against_study` shows the reconstruction reproduces the Study's own
`gwenlaya` rows exactly, so no cascade is re-scored in the sandbox.

Each configuration is analysed with scripts/analyze_e_tpu.py (same metrics, bootstrap and S1/S2 sensitivities);
`large` is the always-answer comparator (B2) and the largest tier of the ladder.
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
from gwaya.metrics_selective import holm, mcnemar_exact  # noqa: E402
from scripts import analyze_e_tpu as E  # noqa: E402
from scripts import analyze_study as A  # noqa: E402

DOMAINS = E.DOMAINS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def tier_arrays(model: str, keys: list[tuple[str, str]], rows: list[dict], gens: list[dict]) -> dict[str, np.ndarray]:
    base, gate = {}, {}
    for r in rows:  # later rows win (overlays)
        if r.get("model") != model:
            continue
        k = (r["domain"], r["task_id"])
        if r["arm"] == "base":
            base[k] = r
        elif r["arm"] == "gate_only":
            gate[k] = r
    ans = {(g["domain"], g["task_id"]): g for g in gens
           if g["model"] == model and g.get("temperature") == 0.0 and g.get("kind") != "pot"}
    pot = {(g["domain"], g["task_id"]): float(g.get("gpu_s") or 0.0) for g in gens
           if g["model"] == model and g.get("kind") == "pot"}
    miss = [k for k in keys if k not in base or k not in gate or k not in ans]
    if miss:
        raise SystemExit(f"{model}: {len(miss)} tasks without base/gate_only rows or a generation, e.g. {miss[0]}")
    conf = np.array([math.exp(min(0.0, ans[k]["mean_logprob"])) if ans[k].get("mean_logprob") is not None else float("nan")
                     for k in keys])
    return {
        "cor": np.array([base[k]["score"] == "VERIFIED" for k in keys]),
        "gate_ok": np.array([gate[k]["gate"] == "VERIFIED" for k in keys]),
        "conf": conf,
        "trunc": np.array([ans[k].get("done_reason") == "length" for k in keys]),
        "cost": np.array([float(ans[k].get("gpu_s") or 0.0) for k in keys]),
        "toks": np.array([float(ans[k].get("completion_tokens") or 0.0) for k in keys]),
        # program-of-thought seconds are charged only where the gate needs them: math tasks
        "pot": np.array([pot.get(k, 0.0) if k[0] == "math" else 0.0 for k in keys]),
    }


def offline_cascade(tiers: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    n = len(tiers[0]["cor"])
    cor = np.zeros(n, dtype=bool)
    answered = np.zeros(n, dtype=bool)
    cost = np.zeros(n)
    consulted = np.zeros(n, dtype=int)
    done = np.zeros(n, dtype=bool)
    for i, t in enumerate(tiers):
        active = ~done
        cost += np.where(active, t["cost"] + t["pot"], 0.0)
        consulted += active.astype(int)
        ok = active & t["gate_ok"]
        cor = np.where(ok, t["cor"], cor)
        answered |= ok
        done |= ok
        if i == len(tiers) - 1:
            last = active & ~ok  # nobody verified: the last tier's candidate is final, not answered
            cor = np.where(last, t["cor"], cor)
    return {"cor": cor, "answered": answered, "cost": cost, "escalated": consulted > 1}


def make_data(tasks: list[dict], rows: list[dict], gens: list[dict], ladder: list[str], small: str, large: str,
              warmup_chunk: int = 192) -> tuple[E.Data, dict[str, dict[str, np.ndarray]]]:
    warm_ids = {(t["domain"], t["task_id"]) for t in tasks[:warmup_chunk]}
    tasks = sorted(tasks, key=lambda t: (t["domain"], t["task_id"]))
    keys = [(t["domain"], t["task_id"]) for t in tasks]
    arrays = {m: tier_arrays(m, keys, rows, gens) for m in dict.fromkeys([*ladder, small, large])}
    models = {m: {k: arrays[m][k] for k in ("cor", "conf", "trunc", "cost", "toks")} for m in (small, large)}
    gate = {m: arrays[m]["gate_ok"] for m in (small, large)}
    cascade = offline_cascade([arrays[m] for m in ladder])
    D = E.Data(tasks, models, gate, cascade, small, large)
    D.warm = np.array([k in warm_ids for k in keys])
    return D, arrays


def validate_against_study(D: E.Data, rows: list[dict]) -> dict[str, Any]:
    """The offline 2-tier cascade must equal the Study's own `gwenlaya` rows (later rows win)."""
    st = {}
    for r in rows:
        if r["arm"] == "gwenlaya":
            st[(r["domain"], r["task_id"])] = r
    bad = {"cor": 0, "answered": 0, "escalated": 0, "cost": 0}
    for i, k in enumerate(D.keys):
        r = st[k]
        bad["cor"] += (r["score"] == "VERIFIED") != bool(D.cascade["cor"][i])
        bad["answered"] += bool(r["answered"]) != bool(D.cascade["answered"][i])
        bad["escalated"] += (len(r.get("tiers_invoked") or []) > 1) != bool(D.cascade["escalated"][i])
        bad["cost"] += abs(float(r.get("gpu_s") or 0.0) - float(D.cascade["cost"][i])) > 1e-6
    return {"n": len(D.keys), "mismatches": bad, "exact": not any(bad.values())}


def oracle_ceiling(arrays: dict[str, dict[str, np.ndarray]], order: list[str], D: E.Data, n_boot: int, seed: int) -> dict[str, Any]:
    """Cheapest-correct-tier oracle over `order` (a ceiling only): accuracy and cost, with the bootstrap."""
    cor = np.column_stack([arrays[m]["cor"] for m in order])
    cost = np.column_stack([arrays[m]["cost"] for m in order])
    any_c = cor.any(axis=1)
    first = np.where(any_c, cor.argmax(axis=1), len(order) - 1)
    ocost = cost[np.arange(len(first)), first]
    acc = lambda idx, d: float(any_c[D.sel(idx, d)].mean())
    cst = lambda idx, d: float(ocost[D.sel(idx, d)].mean())
    full = np.arange(D.n)
    names = [(f"A6.oracle_acc.{d}", lambda i, d=d: acc(i, d)) for d in (*DOMAINS, E.SEL)] + \
            [(f"A6.oracle_cost.{d}", lambda i, d=d: cst(i, d)) for d in (*DOMAINS, E.SEL)]
    boot = A.cluster_boot(lambda idx: tuple(fn(idx) for _, fn in names), D.dom, D.clu, n_boot=n_boot, seed=seed)
    out = {}
    for j, (name, fn) in enumerate(names):
        ci = A.pct_ci(float(fn(full)), boot[:, j])
        out[name] = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in ci.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--rows", nargs="+", required=True, type=Path, help="rows.jsonl files; later files win")
    ap.add_argument("--gens", nargs="+", required=True, type=Path, help="gens.jsonl files (answers and kind=pot programs)")
    ap.add_argument("--tiers", nargs=3, default=["qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16"], metavar=("S", "M", "L"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    S, M, L = a.tiers
    tasks = read_jsonl(a.tasks)
    rows = [r for p in a.rows for r in read_jsonl(p)]
    gens = [g for p in a.gens for g in read_jsonl(p)]
    # validation of the offline reconstruction on the Study's own 2B -> 4B cascade
    Dv, _ = make_data(tasks, rows, gens, [S, M], S, M)
    val = validate_against_study(Dv, rows)
    print("validation vs Study gwenlaya rows:", json.dumps(val))
    if not val["exact"]:
        raise SystemExit("offline cascade does not reproduce the Study's gwenlaya rows; refusing to analyse")
    configs = {"2B_to_9B": ([S, L], S, L), "4B_to_9B": ([M, L], M, L), "ladder_2B_4B_9B": ([S, M, L], S, L)}
    res: dict[str, Any] = {"meta": {"plan": "docs/ANALYSIS_PLAN_E_TPU.md#A12", "tiers": [S, M, L], "n_boot": a.n_boot, "seed": a.seed,
                                    "cost_unit": "accelerator chip-seconds (slice wall time split by token share x chips)",
                                    "validation": val, "ladder_rule": "first tier whose gate VERIFIES, else last tier unanswered"}}
    for name, (ladder, s, l) in configs.items():
        D, arrays = make_data(tasks, rows, gens, ladder, s, l)
        out = E.analyze(D, a.n_boot, a.seed)
        res[name] = {"ladder": ladder, **out}
        print(f"{name}: {len(out['metrics'])} metrics")
    ps = [res[c]["paired_tests"]["cascade_vs_large"]["p_two_sided"] for c in configs]
    adj = holm(ps)
    for c, p_adj in zip(configs, adj):
        res[c]["paired_tests"]["cascade_vs_large"]["p_holm_across_cascades"] = p_adj
    D3, arrays3 = make_data(tasks, rows, gens, [S, M, L], S, L)
    res["oracle_router_ceiling_3tier"] = oracle_ceiling(arrays3, [S, M, L], D3, a.n_boot, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
