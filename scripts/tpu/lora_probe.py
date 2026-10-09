#!/usr/bin/env python3
"""Bounded probe: can a Qwen3.5 dense model take LoRA gradient steps on a TPU via torch_xla + peft?

Run on a TPU VM with torch_xla. Reports what happened (first step time, steps/s, or the exact failure) as JSON; it
claims nothing about adapter quality. Exit code 0 even on a measured failure so the driver keeps the evidence.
"""
import argparse
import json
import time
import traceback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="lora_probe.json")
    a = ap.parse_args()
    res = {"model": a.model, "steps_requested": a.steps, "seq": a.seq, "batch": a.batch, "ok": False}
    t0 = time.time()
    try:
        import torch
        import torch_xla.core.xla_model as xm
        import transformers
        import peft
        res["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
        dev = xm.xla_device()
        tok = transformers.AutoTokenizer.from_pretrained(a.model)
        model = transformers.AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
        res["load_s"] = round(time.time() - t0, 1)
        names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
        targets = [t for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if t in names]
        res["lora_targets"] = targets
        model = peft.get_peft_model(model, peft.LoraConfig(r=16, lora_alpha=32, target_modules=targets, task_type="CAUSAL_LM"))
        model.to(dev)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        ids = tok(["def add(a, b):\n    return a + b\n" * 200] * a.batch, return_tensors="pt", truncation=True,
                  max_length=a.seq, padding="max_length")["input_ids"].to(dev)
        times, losses = [], []
        for i in range(a.steps):
            t = time.time()
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            xm.mark_step()
            losses.append(float(out.loss.item()))
            times.append(time.time() - t)
        res.update(ok=True, first_step_s=round(times[0], 1), median_step_s=round(sorted(times)[len(times) // 2], 3),
                   loss_first=losses[0], loss_last=losses[-1], steps_done=len(times))
    except Exception as e:  # noqa: BLE001 - the failure IS the measurement
        res["error"] = f"{type(e).__name__}: {str(e)[:800]}"
        res["traceback_tail"] = traceback.format_exc()[-1500:]
    res["wall_s"] = round(time.time() - t0, 1)
    open(a.out, "w").write(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
