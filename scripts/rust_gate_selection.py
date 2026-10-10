#!/usr/bin/env python
"""A15.3: which visible tests should the Rust gate see? Selection rules at a fixed budget k (docs/ANALYSIS_PLAN_E_TPU.md).

For every stored Rust answer of each set and tier (tasks with at least --max-k visible units), re-runs the gate with
the payload opt-in `visible_test_selection` in {first, first_last, longest, spread} and `visible_tests_k` in 1..K,
plus the full hidden suite, all under one edition (2021 by default, D58). Identical gate texts are compiled once per
(answer, edition); verdicts are cached on disk by sha256(edition, tests, answer) so an interrupted run resumes.
Sanity rule: an answer passing the full suite must pass every subset; violating (tier, task) items are listed and
excluded from every rule and k of that tier. Descriptive only (cluster bootstrap CIs, no tests).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gwaya.domains.rust_tests import RULES, select_visible_tests, suite_units  # noqa: E402
from scripts.rust_gate_depth import norm  # noqa: E402

MEASURES = ("coverage", "precision", "confident_wrong")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _init(edition: str) -> None:
    os.environ["GWAYA_RUST_EDITION"] = edition


def _check(job: tuple[str, str, dict, str]) -> tuple[str, str, str]:
    from gwaya.domains.checkers import check
    from gwaya.domains.task import Task
    key, prompt, payload, text = job
    r = check(Task("rust", key, prompt, dict(payload)), text)
    return key, r.status, str(r.evidence.get("error") or r.evidence.get("reason") or "")[:300]


def cache_key(edition: str, tests: str, text: str) -> str:
    return hashlib.sha256(f"{edition}\0{norm(tests)}\0{text}".encode()).hexdigest()


def plan_set(tasks: dict[str, dict], max_k: int) -> tuple[dict[str, dict], dict[str, Any]]:
    """Validate the split on every task and return the usable ones (>= max_k units) with their gate texts."""
    usable, mismatch, too_few = {}, [], []
    for tid, t in tasks.items():
        full = t["checker_payload"]["tests"]
        try:
            _, units = suite_units(full)
            ok1 = norm(select_visible_tests(full, "first", 1)) == norm(t["gate_payload"]["tests"])
            okn = norm(select_visible_tests(full, "first", len(units))) == norm(full)
        except ValueError:
            mismatch.append(tid)
            continue
        if not (ok1 and okn):
            mismatch.append(tid)
        elif len(units) < max_k:
            too_few.append(tid)
        else:
            gates = {(r, k): select_visible_tests(full, r, k) for r in RULES for k in range(1, max_k + 1)}
            usable[tid] = {"n_units": len(units), "gates": gates, "full": full}
    info = {"tasks": len(tasks), "split_mismatch": mismatch, "fewer_than_K_units": too_few, "usable": len(usable),
            "n_units_hist": {str(n): sum(1 for u in usable.values() if u["n_units"] == n)
                             for n in sorted({u["n_units"] for u in usable.values()})}}
    return usable, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", action="append", required=True, nargs=3, metavar=("NAME", "TASKS", "STUDY_DIRS_COMMA"))
    ap.add_argument("--tiers", nargs="+", required=True)
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--edition", default="2021")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--cache", type=Path, required=True, help="jsonl verdict cache (appended; resumes a run)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    from scripts import analyze_study as A

    cache: dict[str, tuple[str, str]] = {}
    if a.cache.exists():
        for c in read_jsonl(a.cache):
            cache[c["key"]] = (c["status"], c.get("err", ""))
    res: dict[str, Any] = {"meta": {"plan": "docs/ANALYSIS_PLAN_E_TPU.md A15.3", "edition": a.edition, "max_k": a.max_k,
                                    "rules": list(RULES), "n_boot": a.n_boot, "seed": 0,
                                    "note": "first_last at k=1 is the last unit alone; spread at k=1 is the middle unit"},
                           "sets": {}}
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.edition,)) as pool, a.cache.open("a") as cfh:
        for name, tasks_path, dirs in a.set:
            tasks = {t["task_id"]: t for t in read_jsonl(Path(tasks_path)) if t["domain"] == "rust"}
            usable, info = plan_set(tasks, a.max_k)
            print(name, json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in info.items()}), flush=True)
            dirs_ = [Path(d) for d in dirs.split(",")]
            rows = [r for d in dirs_ for r in read_jsonl(d / "rows.jsonl")]
            gens = [g for d in dirs_ for g in read_jsonl(d / "gens.jsonl")]
            sres: dict[str, Any] = {"split": info, "tiers": {}}
            for tier in a.tiers:
                text = {g["task_id"]: g["text"] for g in gens if g["model"] == tier and g.get("kind") != "pot"}
                stored_hidden = {r["task_id"]: r["score"] == "VERIFIED" for r in rows
                                 if r["model"] == tier and r["arm"] == "base"}
                stored_gate = {r["task_id"]: r["gate"] == "VERIFIED" for r in rows
                               if r["model"] == tier and r["arm"] == "gate_only"}
                ids = sorted(i for i in usable if i in text and i in stored_hidden)
                if not ids:
                    continue
                # one job per distinct (answer, gate text); the payload uses the opt-in keys so the gate code is exercised
                need: dict[str, tuple[str, str, dict, str]] = {}
                keymap: dict[tuple[str, Any], str] = {}
                for i in ids:
                    t, u = tasks[i], usable[i]
                    ck = cache_key(a.edition, u["full"], text[i])
                    keymap[(i, "full")] = ck
                    if ck not in cache:
                        need[ck] = (ck, t["prompt"], dict(t["checker_payload"]), text[i])
                    for (r, k), g in u["gates"].items():
                        ck = cache_key(a.edition, g, text[i])
                        keymap[(i, (r, k))] = ck
                        if ck not in cache and ck not in need:
                            need[ck] = (ck, t["prompt"], {**t["gate_payload"], "tests": u["full"],
                                                          "visible_test_selection": r, "visible_tests_k": k}, text[i])
                print(f"{name} {tier}: {len(ids)} answers, {len(need)} checks to run ({len(cache)} cached)", flush=True)
                for j, (ck, status, err) in enumerate(pool.imap_unordered(_check, list(need.values()), chunksize=1)):
                    cache[ck] = (status, err)
                    cfh.write(json.dumps({"key": ck, "status": status, "err": err}) + "\n")
                    cfh.flush()
                    if (j + 1) % 500 == 0:
                        print(f"  {j + 1}/{len(need)}", flush=True)
                hidden = {i: cache[keymap[(i, "full")]][0] == "VERIFIED" for i in ids}
                ver = {(i, rk): cache[keymap[(i, rk)]][0] == "VERIFIED" for i in ids for rk in usable[i]["gates"]}
                # sanity rule
                viol = [{"task": i, "rule": r, "k": k, "status": cache[keymap[(i, (r, k))]][0],
                         "error": cache[keymap[(i, (r, k))]][1]}
                        for i in ids for (r, k) in usable[i]["gates"] if hidden[i] and not ver[(i, (r, k))]]
                bad = {v["task"] for v in viol}
                keep = [i for i in ids if i not in bad]
                stored_diff = {
                    "hidden": [{"task": i, "stored_2015": stored_hidden[i], "now": hidden[i]} for i in ids
                               if stored_hidden[i] != hidden[i]],
                    "gate_first_k1": [{"task": i, "stored_2015": stored_gate.get(i), "now": ver[(i, ("first", 1))]}
                                      for i in ids if stored_gate.get(i) != ver[(i, ("first", 1))]]}
                ok = np.array([hidden[i] for i in keep])
                V = {rk: np.array([ver[(i, rk)] for i in keep]) for rk in usable[keep[0]]["gates"]}
                order = [(r, k) for r in RULES for k in range(1, a.max_k + 1)]
                clu = np.array([tasks[i].get("cluster") or i for i in keep])
                dom = np.array(["rust"] * len(keep))

                def stats(idx, V=V, ok=ok):
                    out = []
                    o = ok[idx]
                    for r, k in order:
                        v = V[(r, k)][idx]
                        out += [v.mean(), (v & o).sum() / v.sum() if v.sum() else float("nan"), (v & ~o).mean(),
                                (v & ~o).mean() - (V[("first", k)][idx] & ~o).mean()]
                    return tuple(out)
                pt = stats(np.arange(len(keep)))
                boot = A.cluster_boot(stats, dom, clu, n_boot=a.n_boot, seed=0)
                tres: dict[str, Any] = {"n_answers": len(ids), "n": len(keep), "n_clusters": int(len(set(clu.tolist()))),
                                        "accuracy": float(ok.mean()), "sanity_violations": viol,
                                        "excluded_by_sanity": sorted(bad), "stored_2015_differences": stored_diff,
                                        "missing_answer": sorted(set(usable) - set(ids))}
                for j, (r, k) in enumerate(order):
                    v = V[(r, k)]
                    d = {nm: A.pct_ci(pt[4 * j + m], boot[:, 4 * j + m]) for m, nm in enumerate(MEASURES)}
                    d["confident_wrong_minus_first"] = A.pct_ci(pt[4 * j + 3], boot[:, 4 * j + 3])
                    d["verified"] = int(v.sum())
                    d["confident_wrong_count"] = int((v & ~ok).sum())
                    tres[f"{r}.k{k}"] = d
                sres["tiers"][tier] = tres
                print(name, tier, "n", len(keep), "viol", len(viol), json.dumps(
                    {f"{r}.k{k}": [round(tres[f'{r}.k{k}'][m]["point"], 3) for m in MEASURES] for r, k in order}),
                    flush=True)
            res["sets"][name] = sres
    # the rule "chosen on E" (A15.3 d): lowest confident-wrong averaged over tiers and k = 2..K, ties -> higher coverage
    if "E" in res["sets"] and res["sets"]["E"]["tiers"]:
        T = res["sets"]["E"]["tiers"]
        score = {r: (float(np.mean([T[t][f"{r}.k{k}"]["confident_wrong"]["point"] for t in T for k in range(2, a.max_k + 1)])),
                     -float(np.mean([T[t][f"{r}.k{k}"]["coverage"]["point"] for t in T for k in range(2, a.max_k + 1)])))
                 for r in RULES}
        res["chosen_on_E"] = {"rule": min(score, key=lambda r: score[r]), "mean_cw_k2_to_K": {r: s[0] for r, s in score.items()},
                              "mean_coverage_k2_to_K": {r: -s[1] for r, s in score.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=lambda o: None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
