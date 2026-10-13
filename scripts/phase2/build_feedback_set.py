#!/usr/bin/env python3
"""Phase 2 (A28): retrofit stored outcomes as feedback and build the Laya v2 training sets. CPU only, no spend.

"Feedback" here is what the project actually has: executed-test outcomes of earlier runs (hidden-check `correct`, gate
verdict). Rows where the gate said `verified` but the hidden check failed (confident-wrong) are the corrections the v1
Laya never learned from; they are duplicated `--hard-neg-copies` times in TRAIN only. Real human judgments (thumbs,
edits) enter only through `--human-feedback` and are matched to an existing row by (source, id, tier); unmatched ones
are counted and dropped. There are none in the repo today, and the report says so.

Splits are grouped by source problem so the tiers of one task never straddle folds:
  fold = sha256("p2|source|id") mod 5; fold 0 = held-out test (never trained, never used for early stopping),
  folds 1-4 = F train. The v1 training rows are added unchanged. Overlap of base rows with F (by key or prompt hash)
  aborts the build: F is only usable as training data for v2 if it was disjoint from v1 training.

Outputs (out-dir): calibrator_v2_train.jsonl, router_v2_train.jsonl, calibrator_F_heldout.jsonl, router_F_heldout.jsonl,
feedback_report.json. Nothing here is a measured result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HELDOUT_FOLD = 0
N_FOLDS = 5


def read_jsonl(p: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def key(r: dict) -> tuple[str, str]:
    return str(r["source"]), str(r["id"])


def fold_of(r: dict) -> int:
    return int(hashlib.sha256(("p2|%s|%s" % key(r)).encode()).hexdigest(), 16) % N_FOLDS


def prompt_hash(r: dict) -> str:
    return hashlib.sha256(r.get("prompt", "").encode()).hexdigest()


def is_hard_negative(r: dict) -> bool:
    """Gate said verified, hidden check failed: the confident-wrong case."""
    sig = r.get("signals") or {}
    return str(sig.get("gate")) == "verified" and not bool(int(r.get("correct", 1)))


def apply_human(rows: list[dict], human: list[dict]) -> tuple[list[dict], dict]:
    """Human accept/reject becomes an extra labelled copy of the matched row (flagged `human`); never overwrites."""
    idx = {(*key(r), r.get("tier")): r for r in rows}
    out, stats = [], Counter()
    for h in human:
        base = idx.get((*key(h), h.get("tier")))
        if base is None or h.get("verdict") not in ("accept", "reject"):
            stats["dropped"] += 1
            continue
        r = dict(base)
        r["correct"] = 1 if h["verdict"] == "accept" else 0
        r["human"] = True
        out.append(r)
        stats["matched"] += 1
        stats["disagrees_with_stored_label"] += int(int(base["correct"]) != r["correct"])
    return out, dict(stats)


def build(base_cal, base_rou, f_cal, f_rou, human, copies: int) -> tuple[dict, dict]:
    base_keys = {key(r) for r in base_cal + base_rou}
    base_ph = {prompt_hash(r) for r in base_cal + base_rou}
    overlap_k = sorted({key(r) for r in f_cal + f_rou} & base_keys)
    overlap_p = len({prompt_hash(r) for r in f_cal + f_rou} & base_ph)
    if overlap_k or overlap_p:
        raise SystemExit(f"ABORT: F overlaps v1 training data (keys={len(overlap_k)}, prompts={overlap_p}); "
                         "F cannot be used as v2 training data")
    f_tr_cal = [r for r in f_cal if fold_of(r) != HELDOUT_FOLD]
    f_ho_cal = [r for r in f_cal if fold_of(r) == HELDOUT_FOLD]
    f_tr_rou = [r for r in f_rou if fold_of(r) != HELDOUT_FOLD]
    f_ho_rou = [r for r in f_rou if fold_of(r) == HELDOUT_FOLD]
    hard = [r for r in f_tr_cal if is_hard_negative(r)]
    hum, hstats = apply_human([r for r in f_tr_cal], human)
    sets = {
        "calibrator_v2_train": base_cal + f_tr_cal + hard * max(0, copies - 1) + hum,
        "router_v2_train": base_rou + f_tr_rou,
        "calibrator_F_heldout": f_ho_cal,
        "router_F_heldout": f_ho_rou,
    }
    rep = {
        "base_calibrator_rows": len(base_cal), "base_router_rows": len(base_rou),
        "F_calibrator_rows": len(f_cal), "F_router_rows": len(f_rou),
        "heldout_fold": HELDOUT_FOLD, "n_folds": N_FOLDS,
        "heldout_tasks": len({key(r) for r in f_ho_cal + f_ho_rou}),
        "F_train_calibrator_rows": len(f_tr_cal),
        "hard_negatives_in_train": len(hard), "hard_negative_copies": copies,
        "hard_negatives_in_heldout": sum(is_hard_negative(r) for r in f_ho_cal),
        "human_feedback_rows_matched": hstats.get("matched", 0), "human_feedback_stats": hstats,
        "out_rows": {k: len(v) for k, v in sets.items()},
        "label_balance_calibrator_train": dict(Counter(int(r["correct"]) for r in sets["calibrator_v2_train"])),
    }
    return sets, rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-calibrator", required=True)
    ap.add_argument("--base-router", required=True)
    ap.add_argument("--f-calibrator", required=True)
    ap.add_argument("--f-router", required=True)
    ap.add_argument("--human-feedback", default=None, help="JSONL {source,id,tier,verdict: accept|reject}")
    ap.add_argument("--hard-neg-copies", type=int, default=3, help="total copies of each confident-wrong train row")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = ap.parse_args(argv)
    human = read_jsonl(a.human_feedback) if a.human_feedback else []
    sets, rep = build(read_jsonl(a.base_calibrator), read_jsonl(a.base_router),
                      read_jsonl(a.f_calibrator), read_jsonl(a.f_router), human, a.hard_neg_copies)
    print(json.dumps(rep, indent=1))
    if a.dry_run:
        return 0
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in sets.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "feedback_report.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
