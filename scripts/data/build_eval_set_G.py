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

def make_dec(root, lean_a18):
    paths = F.existing_sets(root, lean_a18)
    f_path = ROOT / "results/gwenlaya_v4/eval_sets_F/tasks_F_python.jsonl"
    # Same reference items as F's builder (E, P_all, Rust_fresh, A18, Laya rows), plus F python screened by PROMPT
    # only: every converted BigCodeBench suite ends with the same unittest-runner harness, so test-field n-grams match
    # all BCB tasks with each other (diagnosed 2026-10-16: 512 of 723 candidates dropped against F, ~207 of them F's
    # own tasks, the rest harness matches). F's own tasks are still removed (identical prompts) and also by task id.
    import glob
    from gwaya import decontaminate as dc
    items, sizes = [], {}
    for name, p in paths.items():
        recs = F.read_jsonl(p)
        sizes[name] = len(recs)
        items.extend(F.record_to_item(name, r) for r in recs)
    for f in sorted(glob.glob(str(paths["E"].parents[2] / "laya_data/laya_*/router.jsonl"))):
        name, n = "laya:" + Path(f).parent.name, 0
        for line in open(f):
            r = json.loads(line)
            items.append(dc.Item(dataset=name, item_id=str(r.get("id")), domain=r.get("domain", "unknown"),
                                 fields=[("prompt", F._strip_suffix(r.get("prompt") or ""))]))
            n += 1
        sizes[name] = n
    dec = dc.Decontaminator(items)
    # F is screened by a SEPARATE prompt-only decontaminator whose hits count only for same-task methods (exact,
    # jaccard >= 0.8, py_ast, id): BigCodeBench/FullStackBench prompts share docstring templates ("Requirements:",
    # "Example: >>>"), so 13-gram and math-normalisation hits against F's prompts flag distinct tasks (diagnosed
    # 2026-10-16: 309 of 511 F hits were template-only; the other 202 were F's own tasks, matched exactly).
    f_rows = F.read_jsonl(f_path)
    dec_f = dc.Decontaminator([dc.Item(dataset="F_python", item_id=r["task_id"], domain="python",
                                       fields=[("prompt", r["prompt"])]) for r in f_rows])
    sizes["F_python"] = len(f_rows)
    paths["F_python"] = f_path
    return dec, dec_f, sizes, paths


SAME_TASK = ("exact", "jaccard", "py_ast", "id")


def screen_F(dec_f, rows, f_ids):
    from gwaya import decontaminate as dc
    kept, drops = [], []
    for r in rows:
        hits = dec_f.check(dc.Item(dataset="G", item_id=r["task_id"], domain="python", fields=[("prompt", r["prompt"])]))
        same = [h for h in hits if h.method.split(":")[0] in SAME_TASK]
        if same or r["task_id"] in f_ids:
            drops.append({"task_id": r["task_id"], "methods": sorted({h.method for h in same}) or ["task_id"]})
        else:
            kept.append(r)
    return kept, drops


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
    dec, dec_f, sizes, paths = make_dec(root, lean_a18)
    src = F.load_python_sources(root)
    src["leetcode"] = []          # fully used by F
    f_ids = {json.loads(l)["task_id"] for l in open(paths["F_python"])}
    rows, rep = F.build_python(src, dec, a.workers, 10**6, 10**6, SEED)   # no cap yet: F screen first
    rows, f_drops = screen_F(dec_f, rows, f_ids)
    rep["F_screen_drops"] = len(f_drops)
    capped = []
    for name, cap in (("ByteDance/FullStackBench:en", N_FSB_G), ("bigcode/bigcodebench:v0.1.4", N_BCB_G)):
        part = [r for r in rows if r["source"] == name]
        capped.extend(F.seeded_sample(part, cap, SEED, "G:" + name) if len(part) > cap else part)
    rows = sorted(capped, key=lambda r: r["task_id"])
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
    (a.out / "decontam_dropped_G.json").write_text(json.dumps({"base": rep.get("decontam_dropped", []), "F_same_task": f_drops}, indent=0, default=str))
    man["F_screen"] = {"methods_counted": list(SAME_TASK), "dropped": len(f_drops),
                       "note": "13-gram and math-normalisation hits against F prompts ignored (shared docstring templates)"}
    (a.out / "manifest_G.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({"rows": len(rows), "per_source": man["counts"]["per_source"], "sha256": sha}, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
