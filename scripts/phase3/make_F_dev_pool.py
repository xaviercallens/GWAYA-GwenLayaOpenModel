#!/usr/bin/env python3
"""A33: assemble the existing F python candidates (2B/4B/9B greedy + 8 samples) into the phase-3 pool format.

Greedy rows: results/gwenlaya_v4/trust_loop/dev_table.jsonl (set F). Samples: the A29/A30 raw files on the data disk
(text and token log-probs) with their local scores (hidden, gate). Writes pool and scored files; no model is run.
  make_F_dev_pool.py --out-pool F_pool.jsonl --out-scored F_scored.jsonl
"""
import argparse
import json
from pathlib import Path

D = Path("/mnt/data/home/xavkal/gwaya-data")
R = Path(__file__).resolve().parents[2] / "results/gwenlaya_v4"
RAW = {"qwen3.5-2b-bf16": D / "eval_sets_F_sample/out/raw_sample_qwen3.5-2b-bf16.jsonl",
       "qwen3.5-4b-bf16": D / "eval_sets_F_sample/out/raw_sample_qwen3.5-4b-bf16.jsonl",
       "qwen3.5-9b-bf16": D / "eval_sets_F_sample_9b/out/raw_sample_qwen3.5-9b-bf16.jsonl"}
SC = [D / "eval_sets_F_sample/scored_samples.jsonl", D / "eval_sets_F_sample_9b/scored_samples.jsonl"]


def tid(t):
    return t if t.startswith("py/") else "py/" + t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-pool", required=True)
    ap.add_argument("--out-scored", required=True)
    a = ap.parse_args()
    pool, scored = [], []
    for line in open(R / "trust_loop/dev_table.jsonl"):
        r = json.loads(line)
        if r["set"] != "F":
            continue
        pool.append({"tier": r["tier"], "task_id": tid(r["task_id"]), "cand": "g", "text": r["candidate"],
                     "mean_logprob": r["mean_logprob"], "min_logprob": r["min_logprob"],
                     "completion_tokens": r["completion_tokens"], "finish": r["finish_reason"]})
        scored.append({"tier": r["tier"], "task_id": tid(r["task_id"]), "cand": "g",
                       "hidden": "VERIFIED" if int(r["correct"]) else "FAILED",
                       "gate": "VERIFIED" if r["gate"] == "verified" else "FAILED"})
    for tier, p in RAW.items():
        for line in open(p):
            d = json.loads(line)
            if d.get("domain") != "python":
                continue
            lps = d.get("token_logprobs") or []
            pool.append({"tier": tier, "task_id": tid(d["task_id"]), "cand": int(d["sample"]), "text": d["text"],
                         "mean_logprob": d.get("mean_logprob"), "min_logprob": min(lps) if lps else None,
                         "completion_tokens": d.get("completion_tokens"), "finish": d.get("finish_reason")})
    for p in SC:
        for line in open(p):
            r = json.loads(line)
            scored.append({"tier": r["model"], "task_id": tid(r["task_id"]), "cand": int(r["sample"]),
                           "hidden": r["hidden"], "gate": r["gate"]})
    Path(a.out_pool).write_text("".join(json.dumps(x) + "\n" for x in pool))
    Path(a.out_scored).write_text("".join(json.dumps(x) + "\n" for x in scored))
    print("pool", len(pool), "scored", len(scored))


if __name__ == "__main__":
    main()
