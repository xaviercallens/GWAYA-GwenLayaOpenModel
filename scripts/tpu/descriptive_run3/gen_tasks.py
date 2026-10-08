"""TPU run #2: generate the night tasks (python + rust) with vLLM-TPU, score with the GWAYA gate.

Usage: python gen_tasks.py <hf_model_id> <tasks_jsonl> <out_jsonl> [--domains python,rust] [--limit N]
Writes one JSON row per task plus a summary line on stdout.
"""
import json
import os
import sys
import time
from collections import Counter

SYSTEM = {
    "python": "You are a careful Python programmer. Reply with one complete Python code block only. No explanations.",
    "rust": "You are a careful Rust programmer. Reply with one complete Rust code block containing the full function "
            "(with its signature) and no main function. No explanations.",
}


def main() -> None:
    model, tasks_path, out_path = sys.argv[1:4]
    domains = sys.argv[sys.argv.index("--domains") + 1].split(",") if "--domains" in sys.argv else ["python", "rust", "math"]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    sys.path.insert(0, os.path.expanduser("~/smoke/repo"))
    from gwaya.domains.checkers import check_math_task, check_python, check_rust
    from gwaya.domains.task import Task
    from gwaya.generators import _MATH_SYSTEM_PROMPT
    SYSTEM["math"] = _MATH_SYSTEM_PROMPT  # same prompt as the fixed run_study path

    tasks = [json.loads(l) for l in open(tasks_path)]
    tasks = [t for t in tasks if t["domain"] in domains][:limit]
    t0 = time.time()
    from vllm import LLM, SamplingParams

    kwargs = dict(model=model, max_model_len=4096, max_num_seqs=16, max_num_batched_tokens=1024)
    if "Qwen3.5" in model or "Qwen3.8" in model:
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kwargs)
    load_s = time.time() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=1024)
    msgs = [[{"role": "system", "content": SYSTEM[t["domain"]]}, {"role": "user", "content": t["prompt"]}] for t in tasks]
    t1 = time.time()
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    gen_s = time.time() - t1
    toks = sum(len(o.outputs[0].token_ids) for o in outs)

    check = {"python": check_python, "rust": check_rust, "math": check_math_task}
    rows = []
    with open(out_path, "w") as fh:
        for t, o in zip(tasks, outs):
            text = o.outputs[0].text
            r = check[t["domain"]](Task(t["domain"], t["task_id"], t["prompt"], t["checker_payload"]), text)
            reason = (r.evidence.get("details") or {}).get("reason", r.evidence.get("reason"))
            row = {"model": model, "domain": t["domain"], "task_id": t["task_id"], "status": r.status,
                   "reason": reason, "completion_tokens": len(o.outputs[0].token_ids),
                   "finish_reason": o.outputs[0].finish_reason, "text": text}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
    summary = {"model": model, "n": len(rows), "load_s": round(load_s, 1), "gen_s": round(gen_s, 1),
               "completion_tokens": toks, "tok_per_s": round(toks / max(gen_s, 1e-6), 1),
               "by_domain": {d: dict(Counter(r["status"] for r in rows if r["domain"] == d)) for d in domains}}
    print("SUMMARY " + json.dumps(summary))
    json.dump(summary, open(out_path + ".summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
