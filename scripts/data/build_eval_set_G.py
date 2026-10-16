#!/usr/bin/env python3
"""A33 S0: build the sealed Python set G (phase 3), deterministic and seeded. No model is run.

G = FullStackBench (en, python/python) and BigCodeBench (v0.1.4) tasks that are NOT in F and pass every screen F
passed, screened additionally against F python itself (same gwaya.decontaminate methods, any hit removes).
LeetCodeDataset is not used: F already took its whole test split. Conversion, sanity (reference passes hidden and
gate, stub fails both) and decontamination are the F builder's own functions (scripts/data/build_eval_sets_F.py).

  GWAYA_DATA_ROOT=... python scripts/data/build_eval_set_G.py --out results/gwenlaya_v4/eval_set_G
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "data"))
import build_eval_sets_F as F

SEED = 20261016
N_FSB_G, N_BCB_G = 150, 250

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = Path(os.environ.get("GWAYA_DATA_ROOT", "/mnt/data/home/xavkal/gwaya-data"))
    a.out.mkdir(parents=True, exist_ok=True)
    lean_a18 = ROOT / "results/gwenlaya_v4/lean_a18/tasks_lean.jsonl"
    if not lean_a18.exists():
        lean_a18 = Path("/home/xavkal/xdev/GWAYA-GwenLayaOpenModel/.claude/worktrees/agent-a261411ebec1b591e/results/gwenlaya_v4/lean_a18/tasks_lean.jsonl")
    paths = F.existing_sets(root, lean_a18)
    paths["F_python"] = ROOT / "results/gwenlaya_v4/eval_sets_F/tasks_F_python.jsonl"
    dec, sizes = F.build_decontaminator(paths)
    src = F.load_python_sources(root)
    src["leetcode"] = []          # fully used by F
    f_ids = {json.loads(l)["task_id"] for l in open(paths["F_python"])}
    rows, rep = F.build_python(src, dec, a.workers, N_FSB_G, N_BCB_G, SEED)
    overlap = sorted(r["task_id"] for r in rows if r["task_id"] in f_ids)
    assert not overlap, f"G overlaps F by task id: {overlap[:5]}"
    for r in rows:
        r["source"] = r["source"]; r["set"] = "G"
    p = a.out / "tasks_G_python.jsonl"
    sha = F.dump(p, [F.clean(r) for r in rows])
    man = {"addendum": "A33", "seed": SEED, "script": "scripts/data/build_eval_set_G.py",
           "purpose": "sealed set G for phase 3; read once by the final certificate stage",
           "screened_against": {n: {"path": str(q), "n_records": sizes[n], "sha256": F.sha256_file(q)} for n, q in paths.items()},
           "laya_rows_screened": {k: v for k, v in sizes.items() if k.startswith("laya:")},
           "source_revisions": src["revisions"], "source_files_sha256": src["files"],
           "caps": {"FullStackBench": N_FSB_G, "BigCodeBench": N_BCB_G, "LeetCodeDataset": "not used (F took the whole test split)"},
           "files": {p.name: {"rows": len(rows), "sha256": sha}},
           "counts": {"per_source": dict(Counter(r["source"] for r in rows)),
                      "gate_visible_in_prompt": dict(Counter(str(r["gate_visible_in_prompt"]) for r in rows))},
           "report": json.loads(json.dumps({k: v for k, v in rep.items() if k not in ("decontam_dropped",)}, default=str))}
    for s in man["report"]["sources"].values():
        if "sanity_dropped" in s: s["sanity_dropped"] = f"{len(s['sanity_dropped'])} items"
    man["report"]["n_decontam_dropped"] = len(rep.get("decontam_dropped", []))
    (a.out / "manifest_G.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({"rows": len(rows), "per_source": man["counts"]["per_source"], "sha256": sha}, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
