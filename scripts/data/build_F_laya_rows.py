#!/usr/bin/env python3
"""A21: build the row files the FROZEN Laya needs to score the fresh sets F (same schema as router_eprime / calibrator_eprime).

Rows come from scripts/data/build_laya_data.py (the E' builder, unchanged): one router row per task ({"source","id",
"domain","prompt","tiers":{tier: 0|1}}) and one calibrator row per (task, tier) ({"source","id","domain","prompt","tier",
"candidate","correct","signals":{"gate","mean_logprob","min_logprob"}}). `correct` and `tiers` are labels for the later
analysis; train_laya.py never renders them (render_router_text / render_calibrator_text read domain, tier, signals, prompt
and candidate only), and this script CHECKS that: the rendered text of every row is rebuilt after blanking the labels and
must be identical, and no checker/gate payload (hidden tests, gold answer) is part of any row field.

Python + math go to router_F.jsonl / calibrator_F.jsonl (the launcher scores only these: the frozen Laya never saw Lean
and A21 registers no Laya score on Lean). Lean rows go to router_F_lean.jsonl / calibrator_F_lean.jsonl for completeness;
the gate signal there is the kernel verdict (gate == scorer).

  python scripts/data/build_F_laya_rows.py --scored-root $GWAYA_DATA_ROOT/eval_sets_F_scored --raw-root \
      $GWAYA_DATA_ROOT/eval_sets_F_tpu --lean-scored results/gwenlaya_v4/eval_sets_F/lean_scored --out-dir \
      results/gwenlaya_v4/eval_sets_F/laya_rows
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data import build_laya_data as B  # noqa: E402
from gwaya.laya_features import render_calibrator_text, render_router_text  # noqa: E402

TIERS = ["qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16"]
S = "results/gwenlaya_v4/eval_sets_F"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def lean_rows(tasks: list[dict], scored: Path, raw_root: Path) -> tuple[list[dict], list[dict]]:
    router: dict[str, dict] = {}
    calib: list[dict] = []
    by_id = {t["task_id"]: t for t in tasks}
    for tier in TIERS:
        if not (scored / f"scored_raw_{tier}.jsonl").exists():
            print(f"WARNING: no Lean scored file for {tier}; its Lean rows are skipped", file=sys.stderr)
            continue
        sc = {r["task_id"]: r for r in B.read_jsonl(scored / f"scored_raw_{tier}.jsonl")}
        raw_f = next(p for p in (raw_root / "out_small" / f"raw_{tier}.jsonl", raw_root / "out_9b" / f"raw_{tier}.jsonl") if p.exists())
        lp = {}
        for line in raw_f.read_text().splitlines():
            r = json.loads(line)
            if r["domain"] == "lean4" and r.get("kind", "answer") == "answer":
                lp[r["task_id"]] = r
        for tid, t in by_id.items():
            if tid not in sc or tid not in lp:
                continue
            src, rid = B.split_id(tid)
            toks = lp[tid].get("token_logprobs") or []
            ok = int(sc[tid]["status"] == "VERIFIED")
            calib.append({"source": src, "id": rid, "domain": "lean4", "prompt": t["prompt"], "tier": tier,
                          "candidate": lp[tid].get("text", ""), "correct": ok,
                          "signals": {"gate": str(sc[tid]["status"]).lower(), "mean_logprob": lp[tid].get("mean_logprob"),
                                      "min_logprob": min(toks) if toks else None}})
            router.setdefault(tid, {"source": src, "id": rid, "domain": "lean4", "prompt": t["prompt"], "tiers": {}})["tiers"][tier] = ok
    return list(router.values()), calib


def leak_check(tasks: list[dict], router: list[dict], calib: list[dict]) -> dict:
    """Labels and payloads must not reach the rendered text; row prompts must be the task prompts verbatim."""
    prompt_of = {B.split_id(t["task_id"]): t["prompt"] for t in tasks}
    n_bad_text = n_bad_prompt = 0
    for r in calib:
        blank = copy.deepcopy(r)
        blank["correct"] = 1 - int(r["correct"])
        if render_calibrator_text(r) != render_calibrator_text(blank):
            n_bad_text += 1
        n_bad_prompt += prompt_of[(r["source"], r["id"])] != r["prompt"]
    for r in router:
        blank = copy.deepcopy(r)
        blank["tiers"] = {k: 1 - int(v) for k, v in r["tiers"].items()}
        if render_router_text(r) != render_router_text(blank):
            n_bad_text += 1
        n_bad_prompt += prompt_of[(r["source"], r["id"])] != r["prompt"]
    keys = {k for r in calib for k in r} | {k for r in router for k in r}
    allowed = {"source", "id", "domain", "prompt", "tier", "candidate", "correct", "signals", "tiers"}
    return {"label_changes_rendered_text": n_bad_text, "prompt_differs_from_task_prompt": n_bad_prompt,
            "unexpected_fields": sorted(keys - allowed)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scored-root", required=True, type=Path)
    ap.add_argument("--raw-root", required=True, type=Path)
    ap.add_argument("--lean-scored", required=True, type=Path)
    ap.add_argument("--overlay", nargs="*", default=[], type=Path, help="serial re-check rows files (later rows win)")
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    tasks_py = B.read_jsonl(ROOT / S / "tasks_F_python.jsonl")
    tasks_ma = B.read_jsonl(ROOT / S / "tasks_F_math.jsonl")
    tasks_le = B.read_jsonl(ROOT / S / "tasks_F_lean.jsonl")
    study_dirs = [a.scored_root / t for t in TIERS]
    study = B.index_study(study_dirs)
    for p in a.overlay:  # overlay rows (serial re-check) replace base/gate_only verdicts
        for r in B.read_jsonl(p):
            t = study.setdefault(r["model"], {"correct": {}, "gate": {}, "gen": {}})
            k = (r["domain"], r["task_id"])
            if r["arm"] == "base":
                t["correct"][k] = r["score"] == "VERIFIED"
            elif r["arm"] == "gate_only":
                t["gate"][k] = str(r["gate"]).lower()
    router, calib, counts = B.build(tasks_py + tasks_ma, study, TIERS)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    dump(a.out_dir / "router_F.jsonl", router)
    dump(a.out_dir / "calibrator_F.jsonl", calib)
    lr, lc = lean_rows(tasks_le, a.lean_scored, a.raw_root)
    dump(a.out_dir / "router_F_lean.jsonl", lr)
    dump(a.out_dir / "calibrator_F_lean.jsonl", lc)
    leak = leak_check(tasks_py + tasks_ma, router, calib)
    leak_lean = leak_check(tasks_le, lr, lc)
    names = ["router_F.jsonl", "calibrator_F.jsonl", "router_F_lean.jsonl", "calibrator_F_lean.jsonl"]
    (a.out_dir / "SHA256SUMS_F_rows.txt").write_text("".join(f"{sha(a.out_dir / n)}  {n}\n" for n in names))
    manifest = {"counts_python_math": counts, "n_router_lean": len(lr), "n_calibrator_lean": len(lc), "tiers": TIERS,
                "leak_check_python_math": leak, "leak_check_lean": leak_lean,
                "task_file_sha256": {n: sha(ROOT / S / n) for n in ("tasks_F_python.jsonl", "tasks_F_math.jsonl", "tasks_F_lean.jsonl")},
                "row_file_sha256": {n: sha(a.out_dir / n) for n in names},
                "calibrator_rows_by_domain_tier": {
                    f"{d}|{t}": sum(1 for r in calib if r["domain"] == d and r["tier"] == t)
                    for d in ("python", "math") for t in TIERS},
                "label_mean_by_domain_tier": {
                    f"{d}|{t}": (lambda xs: round(sum(xs) / len(xs), 4))([r["correct"] for r in calib if r["domain"] == d and r["tier"] == t])
                    for d in ("python", "math") for t in TIERS}}
    (a.out_dir / "manifest_F_laya_rows.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("counts_python_math", "n_router_lean", "n_calibrator_lean", "leak_check_python_math")}))
    bad = leak["label_changes_rendered_text"] or leak["prompt_differs_from_task_prompt"] or leak["unexpected_fields"]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
