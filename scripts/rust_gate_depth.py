#!/usr/bin/env python
"""A15.2: how much does one more visible test buy the gate? Rust only (docs/ANALYSIS_PLAN_E_TPU.md).

The gate currently sees exactly one assertion. Here it sees the first k (k = 1..K) of the same hidden suite, on stored
answers, for tasks with at least K assertions (so every k uses the same tasks). k = all is the hidden verdict.
`split_main` is validated by rebuilding the stored one-assertion gate payloads of E exactly.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from gwaya.domains.rust_tests import _scan, split_main  # noqa: E402,F401  (moved there for A15.3; unchanged)


def build_gate(tests: str, k: int) -> str:
    head, pre, asserts, tail = split_main(tests)
    return head + "".join(pre) + "".join(asserts[:k]) + tail


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _init() -> None:
    os.environ["GWAYA_RUST_EDITION"] = "2015"  # the stored k=1 gate and hidden verdicts were compiled with edition 2015


def _check(job: tuple[str, str, dict, str]) -> tuple[str, str]:
    from gwaya.domains.checkers import check
    from gwaya.domains.task import Task
    key, prompt, payload, text = job
    return key, check(Task("rust", key, prompt, dict(payload)), text).status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True, type=Path, help="E task file (E Rust assertion-style tasks)")
    ap.add_argument("--study-dirs", nargs="+", required=True, type=Path)
    ap.add_argument("--tiers", nargs="+", required=True)
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    tasks = {t["task_id"]: t for t in map(json.loads, a.tasks.read_text().splitlines()) if t["domain"] == "rust"}
    # validation: rebuilding k=1 must equal the stored gate payload for every task (whitespace-insensitive)
    bad, usable = [], {}
    for tid, t in tasks.items():
        try:
            _, _, asserts, _ = split_main(t["checker_payload"]["tests"])
            if norm(build_gate(t["checker_payload"]["tests"], 1)) != norm(t["gate_payload"]["tests"]):
                bad.append(tid)
            elif len(asserts) >= a.max_k:
                usable[tid] = t
        except ValueError:
            bad.append(tid)
    res: dict[str, Any] = {"validation": {"tasks": len(tasks), "k1_rebuild_mismatches": len(bad), "mismatch_ids": bad[:10],
                                          "usable_with_at_least_K_assertions": len(usable), "K": a.max_k}, "tiers": {}}
    if bad:
        print(f"WARNING: {len(bad)} tasks do not rebuild their stored 1-assertion gate; excluded")
    rows = [r for d in a.study_dirs for r in map(json.loads, (d / "rows.jsonl").read_text().splitlines()) if r.strip()] if False else \
        [json.loads(l) for d in a.study_dirs for l in (d / "rows.jsonl").read_text().splitlines() if l.strip()]
    gens = [json.loads(l) for d in a.study_dirs for l in (d / "gens.jsonl").read_text().splitlines() if l.strip()]
    ctx = mp.get_context("spawn")
    from scripts import analyze_study as A
    with ctx.Pool(a.workers, initializer=_init) as pool:
        for tier in a.tiers:
            hidden = {r["task_id"]: r["score"] == "VERIFIED" for r in rows if r["model"] == tier and r["arm"] == "base"}
            text = {g["task_id"]: g["text"] for g in gens if g["model"] == tier and g.get("kind") != "pot"}
            ids = sorted(i for i in usable if i in hidden and i in text)
            jobs = [(f"{k}|{i}", usable[i]["prompt"], {**usable[i]["gate_payload"], "tests": build_gate(usable[i]["checker_payload"]["tests"], k)}, text[i])
                    for i in ids for k in range(2, a.max_k + 1)]
            stored = {r["task_id"]: r["gate"] == "VERIFIED" for r in rows if r["model"] == tier and r["arm"] == "gate_only"}
            got = dict(pool.imap_unordered(_check, jobs, chunksize=2))
            ver = {1: np.array([stored[i] for i in ids])}
            for k in range(2, a.max_k + 1):
                ver[k] = np.array([got[f"{k}|{i}"] == "VERIFIED" for i in ids])
            ok = np.array([hidden[i] for i in ids])
            dom = np.array(["rust"] * len(ids)); clu = np.array([i.split("_")[0] + i.split("_")[1] if "_" in i else i for i in ids])

            def stats(idx):
                out = []
                for k in range(1, a.max_k + 1):
                    v = ver[k][idx]
                    out += [v.mean(), (v & ok[idx]).sum() / v.sum() if v.sum() else float("nan"), (v & ~ok[idx]).mean()]
                return tuple(out)
            pt = stats(np.arange(len(ids)))
            boot = A.cluster_boot(stats, dom, clu, n_boot=a.n_boot, seed=0)
            t_res = {"n": len(ids), "accuracy": float(ok.mean())}
            for k in range(1, a.max_k + 1):
                for j, nm in enumerate(("coverage", "precision", "confident_wrong")):
                    t_res[f"k{k}.{nm}"] = A.pct_ci(pt[3 * (k - 1) + j], boot[:, 3 * (k - 1) + j])
            res["tiers"][tier] = t_res
            print(tier, json.dumps({k: (round(v["point"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in t_res.items() if k != "n"}), flush=True)
    a.out.write_text(json.dumps(res, indent=1, default=lambda o: None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
