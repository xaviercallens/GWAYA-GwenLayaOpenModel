#!/usr/bin/env python3
"""A18: score TPU-generated Lean proofs with the kernel gate (locally) and summarize per docs/ANALYSIS_PLAN_E_TPU.md A18.

Inputs: tasks_lean.jsonl (scripts/lean/build_minif2f_tasks.py) and a raw dir with raw_<served>.jsonl (greedy) and
raw_sample_<served>.jsonl (k draws) from scripts/tpu/gen_batch.py. Identical answer texts for one task are checked once.
Outputs in --out: scored_<file>.jsonl (one row per raw row + status/reason) and summary_a18.json.
  usage: score_lean.py --tasks T --raw RAW_DIR --out OUT [--tiers qwen3.5-2b-bf16,qwen3.5-4b-bf16,qwen3.5-9b-bf16]
         [--chips qwen3.5-9b-bf16=4] [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gwaya.domains.checkers import check  # noqa: E402
from gwaya.domains.task import Task  # noqa: E402

PRICE_CHIP_H = 1.20  # v5e on-demand list price, USD per chip-hour (the ledger's price source)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def answer_text(raw_text: str) -> str:
    # gen_batch prefilled "```lean\n"; the generator stops at the closing fence, so re-wrap (as OllamaGenerator.wrap_output)
    return f"```lean4\n{raw_text.rstrip()}\n```"


def dedupe(rows: list[dict]) -> list[dict]:
    """A resumed sample run can repeat a task's draws: keep the first row per (task, sample)."""
    seen, out = set(), []
    for r in rows:
        k = (r["task_id"], r.get("sample"))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def attribute_seconds(rows: list[dict]) -> None:
    """Accelerator-seconds per row = chunk wall time x the row's share of the chunk's generated tokens (A12 rule)."""
    tot: dict[tuple, int] = {}
    for r in rows:
        tot[(r.get("kind"), r.get("chunk"))] = tot.get((r.get("kind"), r.get("chunk")), 0) + r.get("completion_tokens", 0)
    for r in rows:
        t = tot[(r.get("kind"), r.get("chunk"))]
        r["accel_s"] = r.get("chunk_wall_s", 0.0) * (r.get("completion_tokens", 0) / t if t else 0.0)


def score_rows(rows: list[dict], tasks: dict[str, Task], workers: int, cache: dict) -> None:
    todo = {}
    for r in rows:
        key = (r["task_id"], r.get("text", ""))
        if key not in cache and r.get("finish_reason") != "prompt_too_long":
            todo[key] = r
    def run(key):
        tid, text = key
        res = check(tasks[tid], answer_text(text))
        return key, res.status, res.evidence.get("reason") or (str(res.evidence.get("error", ""))[:200] or None)
    with ThreadPoolExecutor(workers) as ex:
        for key, status, reason in ex.map(run, list(todo)):
            cache[key] = (status, reason)
    for r in rows:
        if r.get("finish_reason") == "prompt_too_long":
            r["status"], r["reason"] = "UNVERIFIED", "prompt_too_long"
        else:
            r["status"], r["reason"] = cache[(r["task_id"], r.get("text", ""))]


def boot_ci(values: list[float], reps: int = 10_000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n = len(values)
    if not n:
        return {"point": None, "lo": None, "hi": None}
    stats = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return {"point": sum(values) / n, "lo": stats[int(0.025 * reps)], "hi": stats[int(0.975 * reps) - 1]}


def summarize(tiers: list[str], greedy: dict, sampled: dict, ids: list[str], chips: dict) -> dict:
    out: dict = {"n_tasks": len(ids), "tiers": {}, "price_chip_h": PRICE_CHIP_H}
    proved_g, cost_g = {}, {}
    for t in tiers:
        g = {r["task_id"]: r for r in greedy.get(t, [])}
        s: dict[str, list] = {}
        for r in sampled.get(t, []):
            s.setdefault(r["task_id"], []).append(r)
        if not g and not s:
            continue
        pg = {i: g.get(i, {}).get("status") == "VERIFIED" for i in ids}
        ps = {i: any(r["status"] == "VERIFIED" for r in s.get(i, [])) for i in ids}
        proved_g[t] = pg
        cost_g[t] = {i: g.get(i, {}).get("accel_s", 0.0) * chips.get(t, 1) * PRICE_CHIP_H / 3600 for i in ids}
        reasons: dict[str, int] = {}
        for r in list(g.values()) + [x for v in s.values() for x in v]:
            if r["status"] != "VERIFIED":
                k = "truncated" if r.get("finish_reason") == "length" else (r.get("reason") or "lean_error")
                k = k if k in ("truncated", "statement_not_preserved", "empty_response", "prompt_too_long",
                               "lean_environment_missing_dependency", "forbidden_construct", "unknown_import",
                               "timeout", "target_theorem_missing", "target_not_theorem", "statement_mismatch",
                               "nonstandard_axioms", "kernel_replay_failed", "no_formal_statement",
                               "lean_project_missing", "reference_not_elaborated") \
                    else ("forbidden_construct" if "Unsound" in k else "lean_error")
                reasons[k] = reasons.get(k, 0) + 1
        all_rows = list(g.values()) + [x for v in s.values() for x in v]
        out["tiers"][t] = {
            "greedy_rows": len(g), "sample_rows": sum(len(v) for v in s.values()),
            "draws_per_task": sorted({len(v) for v in s.values()}),
            "greedy_proof_rate": boot_ci([float(pg[i]) for i in ids]) if g else None,
            "any_of_k_proof_rate": boot_ci([float(ps[i]) for i in ids]) if s else None,
            "greedy_proved": sorted(i for i in ids if pg[i]), "sampled_proved_n": sum(ps.values()),
            "union_proved_n": sum(pg[i] or ps[i] for i in ids),
            "not_verified_reasons": reasons,
            "greedy_tokens": sum(r.get("completion_tokens", 0) for r in g.values()),
            "sample_tokens": sum(r.get("completion_tokens", 0) for v in s.values() for r in v),
            "greedy_usd": sum(cost_g[t].values()),
            "sample_usd": sum(r.get("accel_s", 0.0) for v in s.values() for r in v) * chips.get(t, 1) * PRICE_CHIP_H / 3600,
            "chips": chips.get(t, 1),
            "rows_total": len(all_rows),
        }
    have = [t for t in tiers if t in proved_g]
    if len(have) >= 2:
        top = have[-1]
        casc_proved, casc_cost = 0, 0.0
        for i in ids:
            for t in have:
                casc_cost += cost_g[t][i]
                if proved_g[t][i]:
                    casc_proved += 1
                    break
        out["cascade_greedy"] = {"order": have, "proved": casc_proved, "usd": casc_cost,
                                 "always_top": {"tier": top, "proved": sum(proved_g[top].values()), "usd": sum(cost_g[top].values())},
                                 "cost_ratio_vs_top": casc_cost / sum(cost_g[top].values()) if sum(cost_g[top].values()) else None}
        out["greedy_overlap"] = {f"{a}&{b}": sum(proved_g[a][i] and proved_g[b][i] for i in ids)
                                 for k, a in enumerate(have) for b in have[k + 1:]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", default="qwen3.5-2b-bf16,qwen3.5-4b-bf16,qwen3.5-9b-bf16")
    ap.add_argument("--chips", default="qwen3.5-9b-bf16=4", help="chips per tier for cost (default 1)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    tasks = {r["task_id"]: Task(r["domain"], r["task_id"], r["prompt"], r["checker_payload"]) for r in read_jsonl(Path(args.tasks))}
    ids = sorted(tasks)
    tiers = args.tiers.split(",")
    chips = {k: int(v) for k, v in (x.split("=") for x in args.chips.split(",") if x)}
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    greedy, sampled, cache = {}, {}, {}
    for t in tiers:
        for kind, store, name in (("answer", greedy, f"raw_{t}.jsonl"), ("sample", sampled, f"raw_sample_{t}.jsonl")):
            rows = [r for r in dedupe(read_jsonl(raw / name)) if r["task_id"] in tasks]
            if not rows:
                continue
            attribute_seconds(rows)
            score_rows(rows, tasks, args.workers, cache)
            store[t] = rows
            (out / f"scored_{name}").write_text("".join(json.dumps({k: v for k, v in r.items() if k != "token_logprobs"}) + "\n"
                                                        for r in rows))
            print(f"{name}: {len(rows)} rows, {sum(r['status'] == 'VERIFIED' for r in rows)} VERIFIED", flush=True)
    summary = summarize(tiers, greedy, sampled, ids, chips)
    (out / "summary_a18.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({t: {k: v for k, v in d.items() if k in ("greedy_proof_rate", "any_of_k_proof_rate", "union_proved_n")}
                      for t, d in summary["tiers"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
