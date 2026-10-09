#!/usr/bin/env python
"""A15: how much do Rust verdicts depend on the compiler edition? (docs/ANALYSIS_PLAN_E_TPU.md)

Re-checks every stored Rust answer under edition 2021 (hidden tests and gate tests), compares with the stored verdicts
(compiled with rustc's default, edition 2015), and CONFIRMS each difference by re-running both editions on that item,
so a timeout that depends on machine load is not counted as an edition effect. Generations are never regenerated.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _init(edition: str) -> None:
    os.environ["GWAYA_RUST_EDITION"] = edition


def _check_one(job: tuple[str, str, dict, str]) -> tuple[str, str]:
    from gwaya.domains.checkers import check
    from gwaya.domains.task import Task
    key, prompt, payload, text = job
    r = check(Task("rust", key, prompt, dict(payload)), text)
    return key, r.status


def run(jobs: list[tuple[str, str, dict, str]], edition: str, workers: int) -> dict[str, str]:
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(edition,)) as pool:
        return dict(pool.imap_unordered(_check_one, jobs, chunksize=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", action="append", required=True, nargs=3, metavar=("NAME", "TASKS", "STUDY_DIRS_COMMA"),
                    help="a task file and the comma-separated scored study dirs holding its rows.jsonl + gens.jsonl")
    ap.add_argument("--tiers", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    result: dict[str, Any] = {}
    for name, tasks_path, dirs in a.set:
        tasks = {t["task_id"]: t for t in read_jsonl(Path(tasks_path)) if t["domain"] == "rust"}
        dirs_ = [Path(d) for d in dirs.split(",")]
        rows = [r for d in dirs_ for r in read_jsonl(d / "rows.jsonl")]
        gens = [g for d in dirs_ for g in read_jsonl(d / "gens.jsonl")]
        for tier in a.tiers:
            base = {r["task_id"]: r["score"] for r in rows if r["model"] == tier and r["arm"] == "base" and r["task_id"] in tasks}
            gate = {r["task_id"]: r["gate"] for r in rows if r["model"] == tier and r["arm"] == "gate_only" and r["task_id"] in tasks}
            text = {g["task_id"]: g["text"] for g in gens if g["model"] == tier and g.get("kind") != "pot" and g["task_id"] in tasks}
            ids = sorted(set(base) & set(gate) & set(text))
            jobs = []
            for i in ids:
                t = tasks[i]
                jobs.append((f"H|{i}", t["prompt"], t["checker_payload"], text[i]))
                jobs.append((f"G|{i}", t["prompt"], t["gate_payload"], text[i]))
            new = run(jobs, "2021", a.workers)
            old_status = {**{f"H|{i}": base[i] for i in ids}, **{f"G|{i}": gate[i] for i in ids}}
            diff = [j for j in old_status if (new[j] == "VERIFIED") != (old_status[j] == "VERIFIED")]
            confirmed = []
            if diff:
                redo = [j for j in jobs if j[0] in set(diff)]
                again_old = run(redo, "", max(1, a.workers // 2))
                again_new = run(redo, "2021", max(1, a.workers // 2))
                confirmed = [j for j in diff if (again_old[j] == "VERIFIED") == (old_status[j] == "VERIFIED")
                             and (again_new[j] == "VERIFIED") == (new[j] == "VERIFIED")]
            res = {"n": len(ids), "differences_first_pass": len(diff), "differences_confirmed": len(confirmed),
                   "hidden_fail_to_pass": sorted(j[2:] for j in confirmed if j[0] == "H" and new[j] == "VERIFIED"),
                   "hidden_pass_to_fail": sorted(j[2:] for j in confirmed if j[0] == "H" and new[j] != "VERIFIED"),
                   "gate_fail_to_verified": sorted(j[2:] for j in confirmed if j[0] == "G" and new[j] == "VERIFIED"),
                   "gate_verified_to_fail": sorted(j[2:] for j in confirmed if j[0] == "G" and new[j] != "VERIFIED"),
                   "pass_2015": sum(base[i] == "VERIFIED" for i in ids),
                   # unconfirmed differences keep the stored verdict
                   "pass_2021": sum((new[f"H|{i}"] == "VERIFIED") if f"H|{i}" in confirmed else (base[i] == "VERIFIED") for i in ids),
                   "gate_verified_2015": sum(gate[i] == "VERIFIED" for i in ids),
                   "gate_verified_2021": sum((new[f"G|{i}"] == "VERIFIED") if f"G|{i}" in confirmed else (gate[i] == "VERIFIED") for i in ids)}
            result[f"{name}/{tier}"] = res
            print(name, tier, json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}), flush=True)
    a.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
