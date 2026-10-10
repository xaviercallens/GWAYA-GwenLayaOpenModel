#!/usr/bin/env python3
"""Batch candidate generation on a TPU VM (vLLM-TPU), using the study's raw protocol prompts.

Why this exists: scripts/run_study.py sends one request at a time, which leaves a TPU idle.
This script generates the same prompts in large batches and writes one raw JSONL row per task;
scripts/import_remote_gens.py then converts the rows into run_study's hash-chained gens.jsonl so
scoring and arm replay use the unchanged study code, on a machine where the sandbox works.

Protocol (identical to the llama.cpp / Ollama path): OllamaGenerator.build_raw_prompt (raw Qwen
ChatML, code fence prefill for code domains, \\boxed prose prompt for math), stop sequences from
OllamaGenerator.stop_sequences, greedy decoding, max_tokens from plan generation_settings (1024),
thinking disabled. Nothing is scored here.

Usage (on the TPU VM, inside the vllm venv):
  SKIP_JAX_PRECOMPILE=1 python gen_batch.py --model Qwen/Qwen3.5-4B --served qwen3.5-4b \
      --tasks tasks.jsonl --out raw_qwen3.5-4b.jsonl [--chunk 192] [--max-seqs 128] [--limit N]

Resumable: rows already present in --out (by domain+task_id) are skipped, so a preempted or
time-boxed run continues where it stopped. Each row carries its chunk's wall time so the importer
can attribute accelerator-seconds by token share (the registered attribution rule).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
for cand in (HERE.parents[2], Path.home() / "smoke" / "repo"):
    if (cand / "gwaya").is_dir():
        sys.path.insert(0, str(cand))
        break

MAX_NEW_TOKENS = {"python": 1024, "rust": 1024, "math": 1024, "lean4": 1024}
MAX_PROMPT_TOKENS = 2000


def read_tasks(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def done_keys(path: Path, need: int = 1) -> set[tuple[str, str]]:
    """(domain, task_id) of tasks with at least `need` rows written (need = --samples for --kind sample)."""
    if not path.exists():
        return set()
    count: dict[tuple[str, str], int] = {}
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue  # torn last line from a kill mid-write
        key = (d["domain"], d["task_id"])
        count[key] = count.get(key, 0) + (need if d.get("finish_reason") == "prompt_too_long" else 1)
    return {k for k, n in count.items() if n >= need}


def sample_params(kind: str, samples: int, temperature: float, top_p: float, seed: int) -> dict:
    """Greedy for answer/pot; for sample, n independent draws (the registered Lean protocol: k=8, T=0.8, top_p 0.95).
    vLLM-TPU (JAX) rejects a per-request seed, so `seed` is applied to the engine instead (LLM(seed=...)), not here."""
    if kind != "sample":
        return {"temperature": 0.0}
    return {"n": samples, "temperature": temperature, "top_p": top_p}


class NoLogprobsError(RuntimeError):
    """The engine returned no per-token log-probabilities (the confidence baselines need them)."""


def require_logprobs(rows: list[dict]) -> None:
    """Fail fast after the first chunk: a run without log-probs cannot serve the B3/raw-confidence arms."""
    gen = [r for r in rows if r.get("completion_tokens", 0) > 0]
    if gen and not any(r.get("mean_logprob") is not None for r in gen):
        raise NoLogprobsError("engine returned no log-probabilities for any generated row; stopping before the "
                              "rest of the run is spent on data without confidence scores")


def mean_and_list(output) -> tuple[float | None, list[float]]:
    """Per-token logprob of the sampled tokens (logprobs=1 returns the sampled token plus the top-1)."""
    lps: list[float] = []
    for tid, step in zip(output.token_ids, output.logprobs or []):
        lp = step.get(tid) if isinstance(step, dict) else None
        if lp is not None:
            lps.append(float(getattr(lp, "logprob", lp)))
    return (sum(lps) / len(lps) if lps else None), [round(x, 4) for x in lps]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="HF repo id, e.g. Qwen/Qwen3.5-4B")
    ap.add_argument("--served", required=True, help="served/tier name used in cache keys, e.g. qwen3.5-4b")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=192)
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--max-batched-tokens", type=int, default=1024,
                    help="prefill buffer; 2048 starves the KV/state cache of hybrid Qwen3.5 models on a 16 GB v5e chip")
    ap.add_argument("--limit", type=int, help="max tasks per domain (smoke tests)")
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--kind", choices=["answer", "pot", "sample"], default="answer",
                    help="pot: program-of-thought programs for the math gate (math tasks only); "
                         "sample: --samples draws per task at --temperature/--top-p (one row per draw)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from gwaya.generators import OllamaGenerator

    gen = OllamaGenerator(model=args.served)
    tasks = read_tasks(args.tasks)
    if args.kind == "pot":
        tasks = [t for t in tasks if t["domain"] == "math"]
    if args.limit:
        seen: dict[str, int] = {}
        kept = []
        for t in tasks:
            seen[t["domain"]] = seen.get(t["domain"], 0) + 1
            if seen[t["domain"]] <= args.limit:
                kept.append(t)
        tasks = kept
    out = Path(args.out)
    done = done_keys(out, args.samples if args.kind == "sample" else 1)
    todo = [t for t in tasks if (t["domain"], t["task_id"]) not in done]
    print(f"[gen_batch] {args.served}: {len(tasks)} tasks, {len(done)} already done, {len(todo)} to do", flush=True)
    if not todo:
        return 0

    from vllm import LLM, SamplingParams

    t0 = time.time()
    kwargs = dict(model=args.model, max_model_len=args.max_model_len, max_num_seqs=args.max_seqs, seed=args.seed,
                  max_num_batched_tokens=args.max_batched_tokens, tensor_parallel_size=args.tensor_parallel)
    if "Qwen3.5" in args.model or "Qwen3.8" in args.model:  # multimodal repos: text only
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kwargs)
    tok = llm.get_tokenizer()
    print(f"[gen_batch] engine ready in {time.time() - t0:.0f}s", flush=True)

    with out.open("a") as fh:
        for ci in range(0, len(todo), args.chunk):
            chunk = todo[ci:ci + args.chunk]
            if args.kind == "pot":  # the problem text without the boxed-answer instruction
                prompts = [gen.build_raw_prompt(re.sub(r"\s*Put the final answer in \\boxed\{\}\.?\s*$", "", t["prompt"]),
                                                "math_pot") for t in chunk]
            else:
                prompts = [gen.build_raw_prompt(t["prompt"], t["domain"]) for t in chunk]
            lens = [len(tok.encode(p)) for p in prompts]
            ok = [i for i, n in enumerate(lens) if n <= MAX_PROMPT_TOKENS]
            sp = sample_params(args.kind, args.samples, args.temperature, args.top_p, args.seed)
            params = [SamplingParams(**sp, max_tokens=MAX_NEW_TOKENS[chunk[i]["domain"]],
                                     stop=gen.stop_sequences("math_pot" if args.kind == "pot" else chunk[i]["domain"]),
                                     logprobs=1) for i in ok]
            t1 = time.time()
            outs = llm.generate([prompts[i] for i in ok], params) if ok else []
            wall = time.time() - t1
            by_idx = dict(zip(ok, outs))
            for i, t in enumerate(chunk):
                base = {"domain": t["domain"], "task_id": t["task_id"], "prompt_tokens": lens[i], "kind": args.kind,
                        "chunk": ci // args.chunk, "chunk_wall_s": round(wall, 3), "chunk_size": len(chunk)}
                if i in by_idx:
                    rows = []
                    for j, o in enumerate(by_idx[i].outputs):
                        mean_lp, lp_list = mean_and_list(o)
                        rows.append({**base, **({"sample": j} if args.kind == "sample" else {}), "text": o.text,
                                     "completion_tokens": len(o.token_ids), "finish_reason": o.finish_reason or "stop",
                                     "mean_logprob": mean_lp, "token_logprobs": lp_list})
                else:  # prompt longer than we can serve: recorded, never silently dropped
                    rows = [{**base, "text": "", "completion_tokens": 0, "finish_reason": "prompt_too_long",
                             "mean_logprob": None, "token_logprobs": []}]
                # all draws of a task in one write, so a resumed run never sees half a task
                fh.write("".join(json.dumps(row) + "\n" for row in rows))
            fh.flush()
            os.fsync(fh.fileno())
            if ci == 0:
                try:
                    require_logprobs([json.loads(l) for l in out.read_text().splitlines() if l.strip()])
                except NoLogprobsError as exc:
                    print(f"[gen_batch] FATAL: {exc}", flush=True)
                    return 3
            toks = sum(len(o.token_ids) for i in ok for o in by_idx[i].outputs)
            print(f"[gen_batch] chunk {ci // args.chunk + 1}/{(len(todo) + args.chunk - 1) // args.chunk}: "
                  f"{len(chunk)} tasks, {toks} tokens, {wall:.0f}s, {toks / max(wall, 1e-6):.0f} tok/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
