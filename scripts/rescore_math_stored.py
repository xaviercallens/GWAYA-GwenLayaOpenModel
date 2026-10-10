#!/usr/bin/env python
"""EXPLORATORY re-score of the stored E math generations with the post hoc scorer normalization (D62).

Read-only on inputs. For every (tier, task) the stored answer is scored with the REGISTERED scorer (gwaya/domains/
math_check.py at --old-rev, imported from `git show` into a temp module) and with the NEW scorer (working tree).
The math gate (program-of-thought) is re-run in the bwrap sandbox once per program so the boxed-vs-program
comparison can be evaluated under both scorers; the stored gate verdict stays the record unless the comparison
itself changes for a program whose re-run reproduces the stored verdict.

Inputs ($GWAYA_DATA_ROOT/etpu, as produced by run_mathgate.sh / run_nine.sh and read by scripts/analyze_e_tpu.py):
  mathgate/{tasks_math,gens,rows}.jsonl   2B/4B answers + programs (imported from mathgate/raw_ans_*, pot_raw/)
  nine/{gens,rows}.jsonl                   9B answers + programs (imported from nine_raw/)
Output: results/gwenlaya_v4/e_tpu_mathgate/rescore_exploratory.json. The registered numbers
(numbers_e_tpu_mathgate.json, rows_*.jsonl) are not touched and remain the primary record.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess  # nosec B404
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gwaya.domains import math_check as NEW  # noqa: E402
from gwaya.domains import math_gate  # noqa: E402
from scripts.analyze_e_tpu import read_jsonl  # noqa: E402

REGISTERED_REV = "78cf88b"  # last commit before D62; math_check.py unchanged there since 2c1f6b1 (before E was scored)
TIERS = {"2B": ("qwen3.5-2b-bf16", "mathgate"), "4B": ("qwen3.5-4b-bf16", "mathgate"), "9B": ("qwen3.5-9b-bf16", "nine")}


def load_registered(rev: str, fallback: Path | None) -> tuple[ModuleType, str, str]:
    if fallback is not None:
        src, origin = fallback.read_text(), str(fallback)
    else:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:gwaya/domains/math_check.py"],  # nosec B603 B607
                             check=True, capture_output=True, text=True).stdout
        origin = f"git show {rev}:gwaya/domains/math_check.py"
    path = Path(tempfile.mkdtemp(prefix="mc_reg_")) / "math_check_registered.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("math_check_registered", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, origin, hashlib.sha256(src.encode()).hexdigest()


def gold_of(task: dict) -> str:
    p = task["checker_payload"]
    return str((p if isinstance(p, dict) else json.loads(p))["answer"])


class CachingRunner:
    """Runs each distinct program once in the real sandbox; both scorers then read the same stdout."""

    def __init__(self) -> None:
        from gwaya.sandbox import run_in_sandbox
        self.run, self.cache = run_in_sandbox, {}

    def __call__(self, cmd, **kw):
        key = hashlib.sha256(kw["files"]["pot.py"].encode()).hexdigest()
        if key not in self.cache:
            self.cache[key] = self.run(cmd, **kw)
        return self.cache[key]


def gate_verdict(mod: ModuleType, response: str, pot: str, runner: CachingRunner) -> tuple[str, dict]:
    saved = math_gate.answers_equivalent, math_gate.extract_final_answer
    math_gate.answers_equivalent, math_gate.extract_final_answer = mod.answers_equivalent, mod.extract_final_answer
    try:
        r = math_gate.check_math_gate(response, pot, runner=runner)
    finally:
        math_gate.answers_equivalent, math_gate.extract_final_answer = saved
    return r.status, r.evidence


def r3(k: int, n: int) -> float | None:
    """Rule-of-three 95% upper bound when nothing was observed."""
    return round(3.0 / n, 4) if k == 0 and n else None


def rescore_tier(label: str, model: str, sub: str, root: Path, tasks: dict[str, dict], OLD: ModuleType,
                 workers: int, rerun: bool) -> dict[str, Any]:
    gens = read_jsonl(root / sub / "gens.jsonl")
    rows = read_jsonl(root / sub / "rows.jsonl")
    ans = {g["task_id"]: g["text"] for g in gens if g["model"] == model and g["domain"] == "math"
           and g.get("kind") != "pot" and g.get("temperature") == 0.0}
    pot = {g["task_id"]: g["text"] for g in gens if g["model"] == model and g["domain"] == "math" and g.get("kind") == "pot"}
    base = {r["task_id"]: r["score"] for r in rows if r["model"] == model and r["domain"] == "math" and r["arm"] == "base"}
    gate = {r["task_id"]: r["gate"] for r in rows if r["model"] == model and r["domain"] == "math" and r["arm"] == "gate_only"}
    ids = sorted(tasks)
    miss = [t for t in ids if t not in ans or t not in base or t not in gate]
    if miss:
        raise SystemExit(f"{label}: {len(miss)} math tasks without answer/row/gate, e.g. {miss[0]}")
    old = {t: OLD.check_math(ans[t], gold_of(tasks[t])) for t in ids}
    new = {t: NEW.check_math(ans[t], gold_of(tasks[t])) for t in ids}
    repro = sum(old[t].status == base[t] for t in ids)
    cor_o = {t: old[t].status == "VERIFIED" for t in ids}
    cor_n = {t: new[t].status == "VERIFIED" for t in ids}

    def item(t: str) -> dict:
        return {"task_id": t, "gold": gold_of(tasks[t]), "extracted": new[t].evidence.get("extracted"),
                "old": [old[t].status, old[t].evidence.get("method")], "new": [new[t].status, new[t].evidence.get("method")]}

    w2r = [item(t) for t in ids if cor_n[t] and not cor_o[t]]
    r2w = [item(t) for t in ids if cor_o[t] and not cor_n[t]]

    # gate: re-run each program once; compare the boxed-vs-program decision under both scorers
    g_stored = {t: gate[t] == "VERIFIED" for t in ids}
    g_new = dict(g_stored)
    gate_flips, nonrepro = [], []
    if rerun:
        runner = CachingRunner()
        with ThreadPoolExecutor(workers) as ex:  # phase 1: run every program once (fills the cache; no patching)
            list(ex.map(lambda t: math_gate.check_math_gate(ans[t], pot.get(t, ""), runner=runner), ids))
        # phase 2, sequential (gate_verdict swaps module globals): both scorers on the cached outputs
        res = {t: (gate_verdict(OLD, ans[t], pot.get(t, ""), runner), gate_verdict(NEW, ans[t], pot.get(t, ""), runner))
               for t in ids}
        for t in ids:
            (so, eo), (sn, en) = res[t]
            if (so == "VERIFIED") != g_stored[t]:
                nonrepro.append({"task_id": t, "stored": gate[t], "rerun_old": so, "reason": eo.get("reason")})
                continue  # the stored verdict stays: its program output was not reproduced
            if so != sn:
                g_new[t] = sn == "VERIFIED"
                gate_flips.append({"task_id": t, "answer": en.get("answer"), "program_output": en.get("program_output"),
                                   "old": [so, eo.get("method"), eo.get("reason")], "new": [sn, en.get("method"), en.get("reason")],
                                   "correct_new": cor_n[t]})
    n = len(ids)
    boxed = {t: NEW.extract_final_answer(ans[t], boxed_only=True) is not None for t in ids}
    vbw_o = sorted(t for t in ids if g_stored[t] and not cor_o[t])
    vbw_n = sorted(t for t in ids if g_new[t] and not cor_n[t])
    vbw_n_stored_gate = sorted(t for t in ids if g_stored[t] and not cor_n[t])
    nv_o, nv_n = sum(g_stored.values()), sum(g_new.values())
    aib_o = sum(boxed[t] and not cor_o[t] for t in ids)
    aib_n = sum(boxed[t] and not cor_n[t] for t in ids)
    return {
        "model": model, "n": n,
        "registered_scorer_reproduces_stored_base_rows": f"{repro}/{n}",
        "accuracy": {"old": round(sum(cor_o.values()) / n, 4), "new": round(sum(cor_n.values()) / n, 4)},
        "label_flips": {"wrong_to_right": len(w2r), "right_to_wrong": len(r2w),
                        "wrong_to_right_items": w2r, "right_to_wrong_items": r2w,
                        "status_changes_without_label_change": sum(old[t].status != new[t].status and cor_o[t] == cor_n[t] for t in ids)},
        "gate": {
            "programs_rerun": rerun, "stored_verdict_not_reproduced_on_rerun": nonrepro,
            "comparison_flips": gate_flips,
            "verified": {"old": nv_o, "new": nv_n},
            "verified_but_wrong": {"old": len(vbw_o), "new": len(vbw_n), "old_items": vbw_o, "new_items": vbw_n,
                                   "new_labels_with_stored_gate": len(vbw_n_stored_gate)},
            "cwr_rate_over_n": {"old": round(len(vbw_o) / n, 4), "new": round(len(vbw_n) / n, 4),
                                "new_rule_of_three_upper95": r3(len(vbw_n), n)},
            "wrong_among_verified": {"old": round(len(vbw_o) / nv_o, 4) if nv_o else None,
                                     "new": round(len(vbw_n) / nv_n, 4) if nv_n else None,
                                     "new_rule_of_three_upper95": r3(len(vbw_n), nv_n)},
        },
        "answer_if_boxed_confident_wrong": {
            "n_boxed": sum(boxed.values()), "old": round(aib_o / n, 4), "new": round(aib_n / n, 4),
            "count_old": aib_o, "count_new": aib_n,
            "note": "answer whenever a final \\boxed{} exists; wrong = not VERIFIED (FAILED or unparseable), rate over all n tasks"},
        "cwr_all": {"old": round(1 - sum(cor_o.values()) / n, 4), "new": round(1 - sum(cor_n.values()) / n, 4),
                    "n_no_boxed": n - sum(boxed.values()),
                    "note": "answer everything (A3.cwr_all): counts tasks with no boxed answer (mostly truncated at 1024 tokens) as wrong"},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("GWAYA_DATA_ROOT", "/mnt/data/home/xavkal/gwaya-data")) / "etpu")
    ap.add_argument("--old-rev", default=REGISTERED_REV)
    ap.add_argument("--old-module", type=Path, help="a saved copy of the registered math_check.py instead of git show")
    ap.add_argument("--registered-numbers", type=Path, default=ROOT / "results/gwenlaya_v4/e_tpu_mathgate/numbers_e_tpu_mathgate.json")
    ap.add_argument("--tiers", nargs="*", default=list(TIERS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-rerun", action="store_true", help="skip the program re-run (gate verdicts stay as stored)")
    ap.add_argument("--out", type=Path, default=ROOT / "results/gwenlaya_v4/e_tpu_mathgate/rescore_exploratory.json")
    a = ap.parse_args(argv)
    if a.out.name.startswith(("numbers_", "rows_", "tables_")):
        raise SystemExit("refusing to overwrite a registered output")
    OLD, origin, old_sha = load_registered(a.old_rev, a.old_module)
    tasks = {t["task_id"]: t for t in read_jsonl(a.data / "mathgate" / "tasks_math.jsonl")}
    reg = json.loads(a.registered_numbers.read_text())["metrics"] if a.registered_numbers.exists() else {}
    out: dict[str, Any] = {"meta": {
        "status": "EXPLORATORY post hoc re-score (docs/DEVIATIONS.md D62); the registered numbers remain the primary record",
        "old_scorer": origin, "old_scorer_sha256": old_sha,
        "new_scorer_sha256": hashlib.sha256((ROOT / "gwaya/domains/math_check.py").read_bytes()).hexdigest(),
        "data": str(a.data), "python": sys.executable, "n_tasks": len(tasks),
        "gate_rule": "stored gate verdict, replaced only where the re-run program reproduces the stored verdict under the "
                     "old scorer and the new scorer changes the boxed-vs-program decision",
        "caveat": "E is easy for these tiers; a zero count is bounded by the rule of three, not a claim of zero error. "
                  "No low-tier model changed: only the scorer did."}, "tiers": {}}
    for label in a.tiers:
        model, sub = TIERS[label]
        out["tiers"][label] = res = rescore_tier(label, model, sub, a.data, tasks, OLD, a.workers, not a.no_rerun)
        res["registered_A3"] = {k.split(".")[1]: reg[k]["point"] for k in
                                (f"A3.cov.{model}.math", f"A3.prec.{model}.math", f"A3.cwr.{model}.math", f"A3.cwr_all.{model}.math")
                                if k in reg}
        g = res["gate"]
        print(f"{label}: acc {res['accuracy']['old']} -> {res['accuracy']['new']}  flips w->r {res['label_flips']['wrong_to_right']}"
              f" r->w {res['label_flips']['right_to_wrong']}  gate VbW {g['verified_but_wrong']['old']} -> {g['verified_but_wrong']['new']}"
              f"  gate flips {len(g['comparison_flips'])} nonrepro {len(g['stored_verdict_not_reproduced_on_rerun'])}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
