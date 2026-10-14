#!/usr/bin/env python3
"""A29: score sampled candidates locally: gate (visible payload) and hidden check for every F python sample.
Resumable JSONL output; uses the project's own checker (same sandbox path as the greedy scoring)."""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
os.environ.pop("GWAYA_ALLOW_UNISOLATED", None)
from gwaya.domains.task import Task
from gwaya.domains.checkers import check

TASKS = {}
def init(tasks_path):
    for l in open(tasks_path):
        d = json.loads(l); TASKS[d["task_id"]] = d

def work(a):
    model, row = a
    d = TASKS[row["task_id"]]
    hid = check(Task("python", d["task_id"], d["prompt"], dict(d["checker_payload"])), row["text"]).status
    gp = d.get("gate_payload")
    gate = check(Task("python", d["task_id"], d["prompt"], dict(gp)), row["text"]).status if gp else "none"
    return dict(model=model, task_id=row["task_id"], sample=row["sample"], hidden=hid, gate=gate,
                mean_logprob=row.get("mean_logprob"), min_logprob=row.get("min_logprob"),
                completion_tokens=row.get("completion_tokens"), finish=row.get("finish_reason"), text=row["text"])

if __name__ == "__main__":
    tasks, rawdir, out, nproc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    done = set()
    if os.path.exists(out):
        for l in open(out): r = json.loads(l); done.add((r["model"], r["task_id"], r["sample"]))
    jobs = []
    for m in (sys.argv[5].split(",") if len(sys.argv) > 5 else ["qwen3.5-2b-bf16", "qwen3.5-4b-bf16"]):
        for l in open(f"{rawdir}/raw_sample_{m}.jsonl"):
            r = json.loads(l)
            if r.get("domain") == "python" and (m, r["task_id"], r["sample"]) not in done: jobs.append((m, r))
    print("todo", len(jobs), flush=True)
    with Pool(nproc, initializer=init, initargs=(tasks,)) as p, open(out, "a") as fh:
        for i, res in enumerate(p.imap_unordered(work, jobs, chunksize=4)):
            fh.write(json.dumps(res) + "\n"); fh.flush()
            if i % 200 == 0: print(i, flush=True)
