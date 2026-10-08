"""Re-score the stored night Rust generations with the current checker (usage: rescore_rust.py out.json)."""
import json, os, sys
from collections import Counter
from gwaya.domains.checkers import check_rust
from gwaya.domains.task import Task
ROOT = os.environ.get("GWAYA_DATA_ROOT", os.path.expanduser("~/gwaya-data"))
C = f"{ROOT}/night/cache/L2score"
T = f"{ROOT}/night/results/tasks_E_night.jsonl"
tasks = {d["task_id"]: d for d in map(json.loads, open(T)) if d["domain"] == "rust"}
old = {d["task_id"]: d["score"] for d in map(json.loads, open(f"{C}/rows.jsonl")) if d["domain"] == "rust"}
out = []
for g in map(json.loads, open(f"{C}/gens.jsonl")):
    if g["domain"] != "rust":
        continue
    d = tasks[g["task_id"]]
    r = check_rust(Task("rust", d["task_id"], d["prompt"], d["checker_payload"]), g["text"])
    reason = (r.evidence.get("details") or {}).get("reason", r.evidence.get("reason"))
    out.append({"task_id": g["task_id"], "old": old[g["task_id"]], "new": r.status, "reason": reason, "error": (r.evidence.get("error") or "")[:200]})
    print(f"{g['task_id']:42s} {old[g['task_id']]:10s} -> {r.status:10s} {reason}")
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("old", Counter(o["old"] for o in out), "| new", Counter(o["new"] for o in out), "| reasons", Counter(o["reason"] for o in out))
