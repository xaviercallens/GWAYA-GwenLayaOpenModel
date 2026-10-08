#!/usr/bin/env python3
"""Import remotely generated candidates (e.g. scripts/tpu/gen_batch.py output) into a study cache.

Writes <out>/gens.jsonl in exactly the format scripts/run_study.py produces (same cache keys, same
seeds, same hash chain), so `run_study.py --mode score` finds every generation as a cache hit and
scoring, arm replay and the analysis run unchanged on a machine where the sandbox works.

Cache key: "<served>|<quant>|<domain>|<task_id>|<seed>|t<temperature>" with
seed = 1000 * (rank of (domain, task_id) in the sorted task list + 1) + call_index (call 0 here).

Accelerator-seconds (gpu_s / eval_s / wall_s) are the chunk wall time split by each call's share of
generated + prompt tokens in that chunk (the registered attribution rule). They are chip-seconds of
the remote accelerator (one chip per chunk), not GPU-seconds; the unit is recorded in meta.json.

  python scripts/import_remote_gens.py --tasks tasks.jsonl --raw raw.jsonl --served qwen3.5-4b \
      --quant bf16 --out runs/E_tpu --accelerator "TPU v5e x1"
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_study as rs  # noqa: E402
from gwaya.generators import OllamaGenerator  # noqa: E402


def cache_key(served: str, quant: str, domain: str, task_id: str, seed: int, temperature: float) -> str:
    return f"{served}|{quant}|{domain}|{task_id}|{seed}|t{temperature}"


def import_gens(tasks_path: Path, raw_path: Path, served: str, quant: str, out_dir: Path, *,
                temperature: float = 0.0, max_tokens: dict[str, int] | None = None,
                accelerator: str = "unspecified", seconds_from: Path | None = None) -> dict[str, Any]:
    max_tokens = max_tokens or {"python": 1024, "rust": 1024, "math": 1024, "lean4": 1024}
    tasks = rs.load_tasks_jsonl(tasks_path)
    ordered = sorted(tasks, key=lambda t: (t.domain, t.task_id))
    item_index = {(t.domain, t.task_id): i for i, t in enumerate(ordered)}

    raw_rows = [json.loads(line) for line in Path(raw_path).read_text().splitlines() if line.strip()]
    unknown = [r for r in raw_rows if (r["domain"], r["task_id"]) not in item_index]
    if unknown:
        raise SystemExit(f"{len(unknown)} raw rows are not in the task file, e.g. {unknown[0]['task_id']!r}")

    # accelerator-seconds: chunk wall time split by token share within the chunk
    share_total: dict[int, int] = defaultdict(int)
    chunk_wall: dict[int, float] = {}
    for r in raw_rows:
        share_total[r["chunk"]] += r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
        chunk_wall[r["chunk"]] = r["chunk_wall_s"]

    # Re-imports of a subset (serial re-scoring) must keep the ORIGINAL attributed seconds: splitting a whole
    # chunk's wall time over a handful of tasks would inflate their cost by orders of magnitude.
    orig_secs: dict[tuple[str, str, str], dict[str, float]] = {}
    if seconds_from:
        for rec in rs.ChainedLog(Path(seconds_from)).records:
            orig_secs[(rec["model"], rec["domain"], rec["task_id"])] = {k: rec.get(k) for k in ("wall_s", "eval_s", "gpu_s")}
    gen = OllamaGenerator(model=served)
    log = rs.ChainedLog(out_dir / "gens.jsonl")
    have = {rec["key"] for rec in log.records}
    added = skipped = 0
    for r in raw_rows:
        domain, task_id = r["domain"], r["task_id"]
        seed = 1000 * (item_index[(domain, task_id)] + 1)
        key = cache_key(served, quant, domain, task_id, seed, temperature)
        if key in have:
            skipped += 1
            continue
        toks = r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
        secs = chunk_wall[r["chunk"]] * toks / share_total[r["chunk"]] if share_total[r["chunk"]] else 0.0
        keep = orig_secs.get((served, domain, task_id))
        wall_s, eval_s, gpu_s = (keep["wall_s"], keep["eval_s"], keep["gpu_s"]) if keep else (round(secs, 6),) * 3
        log.append({
            "key": key, "model": served, "quant": quant, "domain": domain, "task_id": task_id, "seed": seed,
            "call_index": 0, "temperature": temperature, "max_tokens": int(max_tokens.get(domain, 1024)),
            "wall_s": wall_s, "text": gen.wrap_output(r.get("text", ""), domain),
            "prompt_tokens": int(r.get("prompt_tokens", 0)), "completion_tokens": int(r.get("completion_tokens", 0)),
            "eval_s": eval_s, "gpu_s": gpu_s, "mean_logprob": r.get("mean_logprob"),
            "done_reason": r.get("finish_reason", "stop"), "token_logprobs": r.get("token_logprobs"),
            "cpu_seconds": None,
        })
        have.add(key)
        added += 1
    missing = len(tasks) - len({(r["domain"], r["task_id"]) for r in raw_rows})
    meta = {"served": served, "quant": quant, "temperature": temperature, "accelerator": accelerator,
            "seconds_unit": "accelerator chip-seconds (chunk wall time split by token share), not GPU-seconds",
            "added": added, "skipped_existing": skipped, "tasks_without_generation": missing,
            "source_raw": str(raw_path)}
    (out_dir / "import_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--served", required=True, help="tier name used as the cache-key model, e.g. qwen3.5-4b-bf16")
    ap.add_argument("--quant", required=True, help="quant label, e.g. bf16")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--accelerator", default="unspecified")
    ap.add_argument("--seconds-from", type=Path, help="original gens.jsonl whose attributed seconds are kept (subset re-imports)")
    args = ap.parse_args(argv)
    meta = import_gens(args.tasks, args.raw, args.served, args.quant, args.out,
                       temperature=args.temperature, accelerator=args.accelerator, seconds_from=args.seconds_from)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
