#!/usr/bin/env python3
"""A33 Verifier V2, step 1: run every candidate of a tier on the inputs of the synthesized asserts, in the sandbox.

Asserts come from raw_tests_<writer>.jsonl (kind `tests`: written from the prompt alone, never from a candidate).
For each task: parse `assert <call> == <expected>` lines, keep up to --max-calls distinct calls, then for every
candidate run ONE sandboxed process that evaluates each call and each expected value and prints their reprs.
Hidden tests are not read here. Output: one JSONL row per (tier, task, candidate) with the output vector.

  exec_tests.py --tasks tasks.jsonl --tests raw_tests_X.jsonl --pool pool.jsonl --out exec.jsonl [--workers 8]
pool.jsonl rows: {"tier", "task_id", "cand", "text"}  (cand = "g" for greedy or the sample index)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.pop("GWAYA_ALLOW_UNISOLATED", None)
from gwaya.sandbox import run_in_sandbox  # noqa: E402

LC_PRELUDE = ("from typing import *\nfrom collections import *\nfrom itertools import *\nfrom functools import *\n"
              "from heapq import *\nfrom bisect import *\nfrom math import *\nimport math, collections, itertools, "
              "functools, heapq, bisect, string, re\ninf = float('inf')\n"
              "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n"
              "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n"
              "        self.left = left\n        self.right = right\n")
MARK = "@@GWAYA_OUT@@"
DRIVER = '''
import json as _gj
_gcalls = {calls!r}
_gexp = {exps!r}
def _gev(src):
    try:
        v = eval(src)
        try:
            r = repr(v)
        except BaseException:
            r = "<unrepr>"
        return r[:400]
    except BaseException as e:
        return "EXC:" + type(e).__name__
_go = [_gev(c) for c in _gcalls]
_ge = [_gev(e) for e in _gexp]
print({mark!r} + _gj.dumps({{"out": _go, "exp": _ge}}))
'''
UNSAFE = ("open(", "random", "requests", "urllib", "socket", "os.", "subprocess", "time.sleep", "input(")


def parse_asserts(texts: list[str], max_calls: int) -> list[tuple[str, str]]:
    """Distinct (call, expected) pairs from `assert <call> == <expected>` lines; calls with I/O or randomness dropped."""
    seen, out = set(), []
    for t in texts:
        body = t.split("```")[0]
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("assert "):
                continue
            try:
                node = ast.parse(line).body[0]
            except SyntaxError:
                continue
            tst = getattr(node, "test", None)
            if not (isinstance(tst, ast.Compare) and len(tst.ops) == 1 and isinstance(tst.ops[0], ast.Eq)):
                continue
            left, right = tst.left, tst.comparators[0]
            if not any(isinstance(n, ast.Call) for n in ast.walk(left)):
                continue
            src = ast.unparse(left)
            if src in seen or any(k in src for k in UNSAFE):
                continue
            seen.add(src)
            out.append((src, ast.unparse(right)))
            if len(out) >= max_calls:
                return out
    return out


def code_of(text: str) -> str:
    """Candidate code: fenced block if the text starts with a fence, else the raw text up to a closing fence."""
    t = text or ""
    if t.lstrip().startswith("```"):
        m = re.search(r"```(?:python|py)?\s*\n(.*?)(?:\n```|$)", t, flags=re.S)
        if m:
            t = m.group(1)
    return t.split("\n```")[0]


TASKS: dict = {}
CALLS: dict = {}


def init(tasks_path: str, calls: dict) -> None:
    for line in open(tasks_path):
        d = json.loads(line)
        TASKS[d["task_id"]] = d
    CALLS.update(calls)


def work(row: dict) -> dict:
    key = {k: row[k] for k in ("tier", "task_id", "cand")}
    tid = row["task_id"]
    t = TASKS[tid]
    calls = CALLS.get(tid) or []
    if not calls:
        return {**key, "n_calls": 0, "out": [], "exp": [], "status": "no_calls"}
    pre = LC_PRELUDE if "LeetCode" in t.get("source", "") else ""
    prog = pre + code_of(row["text"]) + "\n" + DRIVER.format(
        calls=[c for c, _ in calls], exps=[e for _, e in calls], mark=MARK)
    _ok, so, _se, to = run_in_sandbox([sys.executable, "main.py"], files={"main.py": prog},
                                      timeout_s=20, cpu_s=20, mem_mb=1536)
    res = {"out": [], "exp": []}
    st = "timeout" if to else ("ok" if MARK in so else "crash")
    if MARK in so:
        try:
            res = json.loads(so.split(MARK, 1)[1].strip().splitlines()[0])
        except (ValueError, IndexError):
            st = "bad_output"
    return {**key, "n_calls": len(calls), **res, "status": st}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--tests", required=True, nargs="+")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-calls", type=int, default=16)
    a = ap.parse_args()
    by_task: dict = {}
    for f in a.tests:
        for line in open(f):
            d = json.loads(line)
            by_task.setdefault(d["task_id"], []).append(d.get("text") or "")
    calls = {tid: parse_asserts(ts, a.max_calls) for tid, ts in by_task.items()}
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            d = json.loads(line)
            done.add((d["tier"], d["task_id"], str(d["cand"])))
    rows = [json.loads(line) for line in open(a.pool)]
    rows = [r for r in rows if (r["tier"], r["task_id"], str(r["cand"])) not in done]
    print(f"tasks with calls {sum(1 for v in calls.values() if v)}/{len(calls)}; mean calls "
          f"{sum(len(v) for v in calls.values()) / max(len(calls), 1):.1f}; todo {len(rows)}", flush=True)
    with Pool(a.workers, initializer=init, initargs=(a.tasks, calls)) as p, open(a.out, "a") as fh:
        for i, r in enumerate(p.imap_unordered(work, rows, chunksize=4)):
            fh.write(json.dumps(r) + "\n")
            if i % 500 == 0:
                fh.flush()
                print(i, flush=True)


if __name__ == "__main__":
    main()
