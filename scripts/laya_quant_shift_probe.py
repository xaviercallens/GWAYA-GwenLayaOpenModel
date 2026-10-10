#!/usr/bin/env python3
"""A19: how far do the frozen Laya's outputs move between bf16 (TPU) and Q4_K_M (llama.cpp) answers to the same tasks?

Descriptive and exploratory (docs/ANALYSIS_PLAN_E_TPU.md, Addendum A19). The 80 night tasks are part of E, which was in
Laya's training data, so absolute scores here are in-sample and are not performance; the probe is the paired DIFFERENCE.

Two stages (the order matters: the plan was committed before `score` was first run):
  score    build the rows, recompute gate verdicts, run the frozen Laya through gwaya.laya_runtime (fp32, CPU), append
           one JSON line per (task, arm) to --rows (resumable; torch/transformers/peft needed, 3 threads by default).
  analyze  read --rows and write --out (deterministic; bootstrap 10,000 resamples, seed 0, cluster = task).

Arms: bf16 = the stored training row of the 2B tier (vLLM-TPU, gate and log-prob signals as trained);
      q4   = night run Qwen3.5-2B Q4_K_M (llama.cpp), gate recomputed with the same code, log-probs from token_logprobs;
      bf16_regate = the bf16 answer with the gate recomputed under the D25 payload (only scored where it differs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results" / "gwenlaya_v4"
TIER = "qwen3.5-2b-bf16"
NBOOT, SEED, TAU = 10_000, 0, 0.5


def _jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- score stage
def build_pairs(data_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    night = data_root / "night"
    tasks = {r["task_id"]: r for r in _jsonl(night / "results" / "tasks_E_night.d25.jsonl")}
    cands = {r["task_id"]: r for r in _jsonl(night / "cache" / "candidates_2b.jsonl")}
    q4rows = {r["task_id"]: r for r in _jsonl(RES / "night_L2fix" / "rows.jsonl")}
    bf = {}
    for r in _jsonl(data_root / "laya_data" / "laya_train" / "calibrator.jsonl"):
        if r["tier"] == TIER:
            bf[f'{r["source"]}/{r["id"]}'] = r
    pairs, notes = [], []
    for tid in sorted(tasks):
        t = tasks[tid]
        if tid not in bf or tid not in cands:
            notes.append(f"unmatched:{tid}")
            continue
        b, c = bf[tid], cands[tid]
        if b["prompt"] != t["prompt"] or b["domain"] != t["domain"]:
            notes.append(f"prompt_mismatch:{tid}")
            continue
        lps = c.get("token_logprobs") or []
        pairs.append({
            "task_id": tid, "domain": t["domain"], "prompt": t["prompt"], "gate_payload": t.get("gate_payload") or {},
            "bf16": {"text": b["candidate"], "gate": b["signals"].get("gate"), "mean_logprob": b["signals"].get("mean_logprob"),
                     "min_logprob": b["signals"].get("min_logprob"), "correct": b.get("correct")},
            "q4": {"text": c["text"], "mean_logprob": c.get("mean_logprob"), "min_logprob": min(lps) if lps else None,
                   "n_tokens_gen": c.get("completion_tokens"), "done_reason": c.get("done_reason"),
                   "correct": (None if q4rows.get(tid, {}).get("score") in (None, "UNDECIDED", "UNVERIFIED")
                               else int(q4rows[tid]["score"] == "VERIFIED")), "score": q4rows.get(tid, {}).get("score")},
        })
    return pairs, notes


def gate_status(domain: str, tid: str, prompt: str, payload: dict[str, Any], text: str) -> tuple[str, str | None]:
    """Same gate code and edition as the stored runs (Rust edition 2015)."""
    from gwaya.domains.checkers import check
    from gwaya.domains.task import Task
    r = check(Task(domain, tid, prompt, dict(payload)), text)
    ev = r.evidence if isinstance(r.evidence, dict) else {}
    return str(r.status).lower(), ev.get("reason") or ev.get("error") or None


def score(a: argparse.Namespace) -> int:
    os.environ["GWAYA_RUST_EDITION"] = "2015"
    from gwaya.laya_runtime import LayaRuntime
    root = Path(os.environ.get("GWAYA_DATA_ROOT", "/mnt/data/home/xavkal/gwaya-data"))
    pairs, notes = build_pairs(root)
    print(f"pairs {len(pairs)} notes {notes}", flush=True)
    rt = LayaRuntime.load(str(root / "laya_data" / "laya_v1"), "cpu", verify=True, threads=a.threads,
                          domains=("python", "rust", "math"))
    done = {(r["task_id"], r["arm"]) for r in _jsonl(a.rows)} if a.rows.exists() else set()
    a.rows.parent.mkdir(parents=True, exist_ok=True)
    out = a.rows.open("a")

    def emit(p: dict[str, Any], arm: str, gate: str, sig: dict[str, Any], text: str, extra: dict[str, Any]) -> None:
        if (p["task_id"], arm) in done:
            return
        feats = {"domain": p["domain"], "tier": TIER, "prompt": p["prompt"], "response": text,
                 "gate_status": gate.upper(), "gen_signals": {"mean_logprob": sig["mean_logprob"], "min_logprob": sig["min_logprob"]}}
        t0 = time.perf_counter()
        raw = rt.scorer(feats)
        lay = feats["laya"]
        row = {"task_id": p["task_id"], "domain": p["domain"], "arm": arm, "gate": gate, "mean_logprob": sig["mean_logprob"],
               "min_logprob": sig["min_logprob"], "raw": raw, "p_iso": lay.get("p_isotonic"), "applied": lay.get("applied"),
               "truncated_candidate": lay.get("truncated_candidate"), "n_tokens": lay.get("n_tokens"),
               "candidate_tokens_visible": lay.get("candidate_tokens_visible"), "text_sha": _sha(text), "text_chars": len(text),
               "wall_s": round(time.perf_counter() - t0, 2), **extra}
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(p["task_id"], arm, gate, None if raw is None else round(raw, 4), flush=True)

    for p in pairs:
        # router: prompt only, so identical for both precisions; written once
        if (p["task_id"], "router") not in done:
            rr = rt.router(p["prompt"], p["domain"])
            out.write(json.dumps({"task_id": p["task_id"], "domain": p["domain"], "arm": "router", "p_tier": dict(rr),
                                  "applied": rr.evidence.get("applied"), "n_tokens": rr.evidence.get("n_tokens")}) + "\n")
            out.flush()
        g_bf_stored = p["bf16"]["gate"]
        g_bf_new, r_bf = gate_status(p["domain"], p["task_id"], p["prompt"], p["gate_payload"], p["bf16"]["text"])
        g_q4, r_q4 = gate_status(p["domain"], p["task_id"], p["prompt"], p["gate_payload"], p["q4"]["text"])
        emit(p, "bf16", g_bf_stored, p["bf16"], p["bf16"]["text"],
             {"gate_regate": g_bf_new, "gate_regate_reason": r_bf, "correct": p["bf16"]["correct"],
              "done_reason": None})
        emit(p, "q4", g_q4, p["q4"], p["q4"]["text"],
             {"gate_reason": r_q4, "correct": p["q4"]["correct"], "score": p["q4"]["score"], "done_reason": p["q4"]["done_reason"],
              "text_identical_to_bf16": p["q4"]["text"] == p["bf16"]["text"]})
        if g_bf_new != g_bf_stored:
            emit(p, "bf16_regate", g_bf_new, p["bf16"], p["bf16"]["text"], {"gate_stored": g_bf_stored})
    return 0


# ---------------------------------------------------------------- analyze stage
def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def boot(fn, n: int, rng: np.random.Generator) -> list[float | None]:
    """95% percentile CI of fn(index array) over task resamples (cluster = task; one pair per task)."""
    vals = []
    for _ in range(NBOOT):
        v = fn(rng.integers(0, n, n))
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(P: list[dict[str, Any]], label: str) -> dict[str, Any]:
    n = len(P)
    out: dict[str, Any] = {"n_pairs": n}
    if n == 0:
        return out
    pb = np.array([r["bf16"]["p_iso"] for r in P])
    pq = np.array([r["q4"]["p_iso"] for r in P])
    rb = np.array([r["bf16"]["raw"] for r in P])
    rq = np.array([r["q4"]["raw"] for r in P])
    d = pq - pb
    rng = np.random.default_rng(SEED)
    out["p_iso_bf16_mean"], out["p_iso_q4_mean"] = float(pb.mean()), float(pq.mean())
    out["paired_diff_p_mean"] = float(d.mean())
    out["paired_diff_p_ci95"] = boot(lambda i: float(d[i].mean()), n, np.random.default_rng(SEED))
    out["paired_diff_p_median"] = float(np.median(d))
    out["mean_abs_diff_p"] = float(np.abs(d).mean())
    out["mean_abs_diff_p_ci95"] = boot(lambda i: float(np.abs(d[i]).mean()), n, np.random.default_rng(SEED))
    out["max_abs_diff_p"] = float(np.abs(d).max())
    out["mean_abs_diff_raw"] = float(np.abs(rq - rb).mean())
    out["spearman_p"] = spearman(pb, pq)
    out["spearman_p_ci95"] = boot(lambda i: spearman(pb[i], pq[i]), n, np.random.default_rng(SEED))
    out["spearman_raw"] = spearman(rb, rq)
    # gate verdicts (same gate code on both answers) and truncation
    gb = [r["bf16"].get("gate_regate", r["bf16"]["gate"]) for r in P]
    gq = [r["q4"]["gate"] for r in P]
    diff = np.array([a != b for a, b in zip(gb, gq)], float)
    out["gate_verdict_differs_frac"] = float(diff.mean())
    out["gate_verdict_differs_ci95"] = boot(lambda i: float(diff[i].mean()), n, np.random.default_rng(SEED))
    cross: dict[str, int] = {}
    for a, b in zip(gb, gq):
        cross[f"{a}->{b}"] = cross.get(f"{a}->{b}", 0) + 1
    out["gate_cross_bf16_to_q4"] = dict(sorted(cross.items()))
    out["gate_stored_vs_recomputed_bf16_differs"] = int(sum(r["bf16"]["gate"] != r["bf16"].get("gate_regate", r["bf16"]["gate"]) for r in P))
    tb = np.array([bool(r["bf16"]["truncated_candidate"]) for r in P], float)
    tq = np.array([bool(r["q4"]["truncated_candidate"]) for r in P], float)
    out["truncated_frac_bf16"], out["truncated_frac_q4"] = float(tb.mean()), float(tq.mean())
    out["truncated_differs_pairs"] = int((tb != tq).sum())
    out["truncated_frac_diff_ci95"] = boot(lambda i: float((tq[i] - tb[i]).mean()), n, np.random.default_rng(SEED))
    # supporting descriptives
    ident = np.array([bool(r["q4"].get("text_identical_to_bf16")) for r in P])
    out["answer_text_identical_frac"] = float(ident.mean())
    for k in ("mean_logprob", "min_logprob"):
        lb = np.array([r["bf16"][k] for r in P], float)
        lq = np.array([r["q4"][k] for r in P], float)
        out[f"{k}_shift_q4_minus_bf16_mean"] = float((lq - lb).mean())
        out[f"{k}_bf16_mean"], out[f"{k}_q4_mean"] = float(lb.mean()), float(lq.mean())
    out["p_diff_identical_text_n"] = int(ident.sum())
    if ident.sum():
        out["p_diff_identical_text_mean"] = float(d[ident].mean())
        out["p_diff_identical_text_mean_abs"] = float(np.abs(d[ident]).mean())
    if (~ident).sum():
        out["p_diff_different_text_mean_abs"] = float(np.abs(d[~ident]).mean())
    cb = [r["bf16"].get("correct") for r in P]
    cq = [r["q4"].get("correct") for r in P]
    both = [(a, b) for a, b in zip(cb, cq) if a is not None and b is not None]
    out["hidden_correct_bf16"] = int(sum(a for a, _ in both))
    out["hidden_correct_q4"] = int(sum(b for _, b in both))
    out["hidden_correct_n_scored"] = len(both)
    out["hidden_correct_disagree"] = int(sum(a != b for a, b in both))
    # router start tier at tau 0.5 (prompt only: identical for both precisions by construction)
    rp = [r["router_p2b"] for r in P if r.get("router_p2b") is not None]
    if rp:
        out["router_p_2b_mean"] = float(np.mean(rp))
        out["router_starts_at_2b_tau0.5"] = int(sum(p >= TAU for p in rp))
    out["label"] = label
    return out


def analyze(a: argparse.Namespace) -> int:
    rows = _jsonl(a.rows)
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["task_id"], {})[r["arm"]] = r
    P = []
    for tid, arms in sorted(by.items()):
        if "bf16" in arms and "q4" in arms:
            P.append({"task_id": tid, "domain": arms["bf16"]["domain"], "bf16": arms["bf16"], "q4": arms["q4"],
                      "bf16_regate": arms.get("bf16_regate"),
                      "router_p2b": (arms.get("router", {}).get("p_tier") or {}).get(TIER)})
    for r in P:
        r["q4"]["text_identical_to_bf16"] = bool(r["q4"].get("text_identical_to_bf16"))
    res: dict[str, Any] = {"plan": "docs/ANALYSIS_PLAN_E_TPU.md Addendum A19 (written and committed before this was run)",
                           "status": "descriptive and exploratory; in-sample (the 80 tasks are in E, which Laya trained on); no performance claim",
                           "bootstrap": {"resamples": NBOOT, "seed": SEED, "cluster": "task", "ci": "95% percentile"},
                           "tau_router": TAU, "tier": TIER, "q4_arm": "Qwen3.5-2B Q4_K_M, llama.cpp b11476, CPU, greedy",
                           "bf16_arm": "stored E training row, vLLM-TPU bf16 (signals as trained)"}
    res["n_rows"] = len(rows)
    res["python"] = summarize([r for r in P if r["domain"] == "python"], "python (primary)")
    res["rust"] = summarize([r for r in P if r["domain"] == "rust"], "rust (descriptive; Laya is gate-only for Rust by default)")
    res["math_protocol_confounded"] = summarize([r for r in P if r["domain"] == "math"],
                                                "math: the night Q4_K_M generations used the code-only prompt path (python programs, "
                                                "no PoT gate on the Q4 side); NOT a quantisation contrast")
    # sensitivity: bf16 gate recomputed under D25 (python and rust)
    sens = []
    for r in P:
        if r["domain"] == "math":
            continue
        use = r["bf16_regate"] or r["bf16"]
        sens.append({"task_id": r["task_id"], "domain": r["domain"], "bf16": use, "q4": r["q4"], "router_p2b": r["router_p2b"]})
    res["sensitivity_bf16_gate_recomputed"] = {d: summarize([s for s in sens if s["domain"] == d], d + ": bf16 gate recomputed under the D25 payload")
                                               for d in ("python", "rust")}
    res["sensitivity_bf16_gate_recomputed"]["note"] = "bf16 rows whose recomputed gate differs from the stored one were re-scored"
    # compact per-domain table
    keys = ["n_pairs", "paired_diff_p_mean", "paired_diff_p_ci95", "mean_abs_diff_p", "spearman_p", "gate_verdict_differs_frac",
            "truncated_frac_bf16", "truncated_frac_q4", "answer_text_identical_frac"]
    res["table"] = {d: {k: res[d].get(k) for k in keys} for d in ("python", "rust", "math_protocol_confounded")}
    res["per_task"] = [{"task_id": r["task_id"], "domain": r["domain"], "p_bf16": r["bf16"]["p_iso"], "p_q4": r["q4"]["p_iso"],
                       "gate_bf16": r["bf16"]["gate"], "gate_bf16_recomputed": r["bf16"].get("gate_regate"), "gate_q4": r["q4"]["gate"],
                       "trunc_bf16": r["bf16"]["truncated_candidate"], "trunc_q4": r["q4"]["truncated_candidate"],
                       "mean_lp_bf16": r["bf16"]["mean_logprob"], "mean_lp_q4": r["q4"]["mean_logprob"],
                       "text_identical": r["q4"]["text_identical_to_bf16"]} for r in P]
    # run3 (bf16 TPU, no log-probs): text identity and hidden verdict agreement only
    run3 = {}
    for name in ("res_qwen35_2b",):
        f = RES / "tpu_run3" / f"{name}.rescored.jsonl"
        full = RES / "tpu_run3" / f"{name}.jsonl"
        if f.exists() and full.exists():
            txt = {r["task_id"]: r.get("text") for r in _jsonl(full)}
            st = {r["task_id"]: r["status"] for r in _jsonl(f)}
            root = Path(os.environ.get("GWAYA_DATA_ROOT", "/mnt/data/home/xavkal/gwaya-data"))
            q4 = {r["task_id"]: r["text"] for r in _jsonl(root / "night" / "cache" / "candidates_2b.jsonl")}
            bfs = {}
            for r in _jsonl(root / "laya_data" / "laya_train" / "calibrator.jsonl"):
                if r["tier"] == TIER:
                    bfs[f'{r["source"]}/{r["id"]}'] = r["candidate"]
            run3[name] = {"n": len(txt), "text_identical_to_E_tpu_bf16": sum(txt[t] == bfs.get(t) for t in txt),
                          "text_identical_to_q4": sum(txt[t] == q4.get(t) for t in txt),
                          "E_tpu_bf16_identical_to_q4": sum(bfs[t] == q4.get(t) for t in bfs if t in txt),
                          "status_counts": {s: sum(v == s for v in st.values()) for s in sorted(set(st.values()))},
                          "note": "run3 carries no token log-probs; Laya was not run on it (A19)"}
    res["run3_text_identity"] = run3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res["table"], indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["score", "analyze"])
    ap.add_argument("--rows", type=Path, default=RES / "laya_runtime" / "quant_shift_probe_rows.jsonl")
    ap.add_argument("--out", type=Path, default=RES / "laya_runtime" / "quant_shift_probe.json")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    return score(a) if a.stage == "score" else analyze(a)


if __name__ == "__main__":
    sys.exit(main())
