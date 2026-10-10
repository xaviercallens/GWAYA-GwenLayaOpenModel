#!/usr/bin/env python
"""Consistency check (not a new result): replay stored E math answers and stored PoT programs through the runtime
path (GwenLaya.answer -> math_pot_runtime) with a fake tier that returns the stored text, and compare each runtime
verdict with the stored registered-gate verdict (rows.jsonl arm gate_only).

The runtime calls the same check_math_gate, but the working tree's answer scorer is the post hoc D62 one, while the
stored verdicts used the registered scorer; the few rows where only the boxed-vs-program comparison changed are
listed in rescore_exploratory.json (gate.comparison_flips) and are reported as `expected_scorer_flips`, not hidden.

Input: $GWAYA_DATA_ROOT/etpu/{mathgate,nine}/{tasks_math,gens,rows}.jsonl. Needs the bwrap sandbox and sympy.
Output: results/gwenlaya_v4/math_pot_runtime/consistency.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwaya import gwenlaya as g  # noqa: E402
from gwaya import math_pot_runtime as mpr  # noqa: E402

TIERS = {"mathgate": ["qwen3.5-2b-bf16", "qwen3.5-4b-bf16"], "nine": ["qwen3.5-9b-bf16"]}
SHORT = {"qwen3.5-2b-bf16": "2B", "qwen3.5-4b-bf16": "4B", "qwen3.5-9b-bf16": "9B"}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def main(argv: list[str] | None = None) -> int:
    root = Path(os.environ.get("GWAYA_DATA_ROOT", "/mnt/data/home/xavkal/gwaya-data")) / "etpu"
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=root)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rescore", type=Path, default=Path("results/gwenlaya_v4/e_tpu_mathgate/rescore_exploratory.json"))
    ap.add_argument("--out", type=Path, default=Path("results/gwenlaya_v4/math_pot_runtime/consistency.json"))
    a = ap.parse_args(argv)
    pool = []  # (tier, task_id, prompt, answer_text, pot_text, stored_verdict)
    for sub, models in TIERS.items():
        tasks = {t["task_id"]: t for t in read_jsonl(a.data / sub / "tasks_math.jsonl")}
        gens = read_jsonl(a.data / sub / "gens.jsonl")
        rows = read_jsonl(a.data / sub / "rows.jsonl")
        for m in models:
            ans = {x["task_id"]: x["text"] for x in gens if x["model"] == m and x.get("kind") != "pot"
                   and x.get("temperature") == 0.0}
            pot = {x["task_id"]: x["text"] for x in gens if x["model"] == m and x.get("kind") == "pot"}
            gate = {r["task_id"]: r["gate"] for r in rows if r["model"] == m and r["arm"] == "gate_only"}
            for tid in sorted(tasks):
                if tid in ans and tid in pot and tid in gate:
                    pool.append((m, tid, tasks[tid]["prompt"], ans[tid], pot[tid], gate[tid]))
    sample = random.Random(a.seed).sample(pool, a.n)
    flips = json.loads(a.rescore.read_text()) if a.rescore.is_file() else {"tiers": {}}
    known = {(m, f["task_id"]) for t, v in flips["tiers"].items() for f in v["gate"]["comparison_flips"]
             for m in [v["model"]]}
    known_rerun = {(v["model"], f["task_id"]) for v in flips["tiers"].values()
                   for f in v["gate"]["stored_verdict_not_reproduced_on_rerun"]}
    items, agree = [], 0
    for m, tid, prompt, answer, pot_text, stored in sample:
        seen: list[tuple[str, str]] = []

        def gen(p: str, d: str, answer=answer, pot_text=pot_text, seen=seen) -> str:
            seen.append((p, d))
            return pot_text if d == "math_pot" else answer
        gl = g.GwenLaya([g.Tier(m, gen)])
        out = gl.answer(prompt, "math", task_id=tid)  # no gold answer in the payload
        step = out["evidence"]["trajectory"][0]
        runtime = step["gate"]["status"]
        same = runtime == stored
        agree += same
        kind = None if same else ("scorer_comparison_flip" if (m, tid) in known else
                                   "rerun_timeout" if (m, tid) in known_rerun else "UNEXPLAINED")
        items.append({"tier": SHORT[m], "task_id": tid, "stored_verdict": stored, "runtime_verdict": runtime,
                      "agree": same, "disagreement": kind, "pot_called": any(d == "math_pot" for _, d in seen),
                      "runtime_reason": step["gate"]["evidence"].get("reason"),
                      "pot_prompt_ok": all(p == mpr.pot_prompt(prompt) for p, d in seen if d == "math_pot")})
    unexplained = sum(1 for i in items if i["disagreement"] == "UNEXPLAINED")
    res = {"meta": {"status": "consistency check of the runtime wiring against stored registered-gate verdicts; not a new "
                              "result", "sample_n": len(items), "pool_n": len(pool), "seed": a.seed,
                    "data": str(a.data), "script": "scripts/check_math_pot_runtime_consistency.py",
                    "note": "runtime uses the working-tree (D62) scorer; stored verdicts used the registered scorer; "
                            "comparison flips known from rescore_exploratory.json are classified, not excluded"},
           "n": len(items), "agree": agree, "agreement": round(agree / len(items), 4),
           "disagreements_by_kind": {k: sum(1 for i in items if i["disagreement"] == k)
                                     for k in sorted({i["disagreement"] for i in items if i["disagreement"]})},
           "unexplained_disagreements": unexplained,
           "stored_verdict_counts": {v: sum(1 for i in items if i["stored_verdict"] == v)
                                     for v in sorted({i["stored_verdict"] for i in items})},
           "pot_called_when_no_boxed_answer": sum(1 for i in items if i["runtime_reason"] == "no_boxed_answer"
                                                  and i["pot_called"]),
           "items": items}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("items", "meta")}))
    return 1 if unexplained else 0


if __name__ == "__main__":
    raise SystemExit(main())
