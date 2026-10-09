#!/usr/bin/env python
"""A17: how much does one more visible test buy the gate? Python (docs/ANALYSIS_PLAN_E_TPU.md).

The hidden tests of E's Python tasks are generated suites with one `inputs` list and one `results` list. The k-test gate is
the same test text with both lists cut to their first k elements. Stored answers only; every k uses the same tasks.
"""
from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _offsets(src: str) -> list[int]:
    """Byte offset of the start of each line (ast columns are utf-8 byte offsets)."""
    out, pos = [0], 0
    for line in src.encode("utf-8").splitlines(keepends=True):
        pos += len(line)
        out.append(pos)
    return out


def find_pairs(src: str) -> tuple[ast.List, ast.List] | None:
    """The `inputs` and `results` list literals, each assigned exactly once, same length; else None."""
    try:
        mod = ast.parse(src)
    except SyntaxError:
        return None
    found: dict[str, list[ast.Assign]] = {"inputs": [], "results": []}
    for node in ast.walk(mod):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in found:
                    found[t.id].append(node)
    if any(len(v) != 1 for v in found.values()):
        return None
    lists = [found["inputs"][0].value, found["results"][0].value]
    if not all(isinstance(v, ast.List) for v in lists) or len(lists[0].elts) != len(lists[1].elts):
        return None
    return lists[0], lists[1]


def n_pairs(src: str) -> int:
    p = find_pairs(src)
    return len(p[0].elts) if p else 0


def truncate_pairs(src: str, k: int) -> str:
    """Hidden test text with `inputs` and `results` cut to their first k elements."""
    pairs = find_pairs(src)
    if pairs is None:
        raise ValueError("no_pairs")
    if len(pairs[0].elts) < k:
        raise ValueError("fewer_than_k_pairs")
    raw, off = src.encode("utf-8"), _offsets(src)
    edits = []
    for lst in pairs:
        a = off[lst.lineno - 1] + lst.col_offset
        b = off[lst.end_lineno - 1] + lst.end_col_offset
        elts = [raw[off[e.lineno - 1] + e.col_offset: off[e.end_lineno - 1] + e.end_col_offset].decode("utf-8") for e in lst.elts[:k]]
        edits.append((a, b, "[" + ", ".join(elts) + "]"))
    for a, b, rep in sorted(edits, reverse=True):
        raw = raw[:a] + rep.encode("utf-8") + raw[b:]
    out = raw.decode("utf-8")
    again = find_pairs(out)
    if again is None or len(again[0].elts) != k:
        raise ValueError("rebuild_failed")
    return out


def _check(job: tuple[str, str, dict, str]) -> tuple[str, str]:
    from gwaya.domains.checkers import check
    from gwaya.domains.task import Task
    key, prompt, payload, text = job
    return key, check(Task("python", key, prompt, dict(payload)), text).status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--study-dirs", nargs="+", required=True, type=Path)
    ap.add_argument("--tiers", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    K = max(a.ks)
    tasks = {t["task_id"]: t for t in map(json.loads, a.tasks.read_text().splitlines()) if t["domain"] == "python"}
    usable, excluded = {}, {}
    for tid, t in tasks.items():
        src = t["checker_payload"]["tests"]
        try:
            truncate_pairs(src, K)
            usable[tid] = t
        except ValueError as e:
            excluded[tid] = str(e)
    res: dict[str, Any] = {"validation": {"tasks": len(tasks), "usable_with_at_least_K_pairs": len(usable), "K": K, "ks": a.ks,
                                          "excluded_reasons": {r: sum(v == r for v in excluded.values()) for r in set(excluded.values())}},
                           "tiers": {}}
    rows = [json.loads(l) for d in a.study_dirs for l in (d / "rows.jsonl").read_text().splitlines() if l.strip()]
    gens = [json.loads(l) for d in a.study_dirs for l in (d / "gens.jsonl").read_text().splitlines() if l.strip()]
    from scripts import analyze_study as A
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for tier in a.tiers:
            hidden = {r["task_id"]: r["score"] == "VERIFIED" for r in rows if r["model"] == tier and r["arm"] == "base"}
            stored = {r["task_id"]: r["gate"] == "VERIFIED" for r in rows if r["model"] == tier and r["arm"] == "gate_only"}
            text = {g["task_id"]: g["text"] for g in gens if g["model"] == tier and g.get("kind") != "pot"}
            ids = sorted(i for i in usable if i in hidden and i in stored and i in text)
            jobs = [(f"{k}|{i}", usable[i]["prompt"], {**usable[i]["gate_payload"], "tests": truncate_pairs(usable[i]["checker_payload"]["tests"], k)}, text[i])
                    for i in ids for k in a.ks]
            got = dict(pool.imap_unordered(_check, jobs, chunksize=2))
            ver = {k: np.array([got[f"{k}|{i}"] == "VERIFIED" for i in ids]) for k in a.ks}
            ok = np.array([hidden[i] for i in ids])
            # logical sanity: an answer that passes the full hidden suite must pass every subset gate
            viol = sorted({i for k in a.ks for i, v, h in zip(ids, ver[k], ok) if h and not v})
            keep = np.array([i not in viol for i in ids])
            ids_k = [i for i, kp in zip(ids, keep) if kp]
            clu = np.array([usable[i]["cluster"] for i in ids_k]); dom = np.array(["python"] * len(ids_k))
            okk = ok[keep]; verk = {k: ver[k][keep] for k in a.ks}; stk = np.array([stored[i] for i in ids_k])

            def stats(idx):
                out = []
                for v in [stk] + [verk[k] for k in a.ks]:
                    vv = v[idx]
                    out += [vv.mean(), (vv & okk[idx]).sum() / vv.sum() if vv.sum() else float("nan"), (vv & ~okk[idx]).mean()]
                return tuple(out)
            pt = stats(np.arange(len(ids_k)))
            boot = A.cluster_boot(stats, dom, clu, n_boot=a.n_boot, seed=0)
            t_res: dict[str, Any] = {"n": len(ids_k), "accuracy": float(okk.mean()), "sanity_violations": viol}
            for j, name in enumerate(["stored"] + [f"k{k}" for k in a.ks]):
                for m, nm in enumerate(("coverage", "precision", "confident_wrong")):
                    t_res[f"{name}.{nm}"] = A.pct_ci(pt[3 * j + m], boot[:, 3 * j + m])
            res["tiers"][tier] = t_res
            print(tier, "n", len(ids_k), "violations", len(viol),
                  json.dumps({k: round(v["point"], 3) for k, v in t_res.items() if isinstance(v, dict) and k.endswith("confident_wrong")}), flush=True)
    a.out.write_text(json.dumps(res, indent=1, default=lambda o: None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
