#!/usr/bin/env python
"""Per-tier descriptive summary of a scored task set: accuracy, gate coverage/precision, confident-wrong rates, truncation.

  python scripts/rust_fresh_summary.py --tasks tasks.jsonl --study-dirs d2b d4b d9b --tiers a b c --out summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def summarize(rows: list[dict], gens: list[dict], model: str) -> dict:
    base = {r["task_id"]: r for r in rows if r["model"] == model and r["arm"] == "base"}
    gate = {r["task_id"]: r for r in rows if r["model"] == model and r["arm"] == "gate_only"}
    gen = {g["task_id"]: g for g in gens if g["model"] == model and g.get("kind") != "pot"}
    ids = sorted(base)
    n = len(ids)
    correct = {i: base[i]["score"] == "VERIFIED" for i in ids}
    ver = [i for i in ids if gate[i]["gate"] == "VERIFIED"]
    ok = sum(correct[i] for i in ver)
    return {"n": n, "pass": sum(correct.values()), "accuracy": sum(correct.values()) / n,
            "gate_verified": len(ver), "gate_coverage": len(ver) / n,
            "gate_precision": ok / len(ver) if ver else None,
            "confident_wrong_gate": (len(ver) - ok) / n, "confident_wrong_answer_all": 1 - sum(correct.values()) / n,
            "truncated_share": sum(gen[i].get("done_reason") == "length" for i in ids) / n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--study-dirs", nargs="+", required=True, type=Path)
    ap.add_argument("--tiers", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    rows = [r for d in a.study_dirs for r in read_jsonl(d / "rows.jsonl")]
    gens = [g for d in a.study_dirs for g in read_jsonl(d / "gens.jsonl")]
    out = {t: summarize(rows, gens, t) for t in a.tiers}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({t: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()} for t, s in out.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
