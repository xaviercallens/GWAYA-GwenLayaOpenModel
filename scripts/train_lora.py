#!/usr/bin/env python3
"""QLoRA SFT / DPO trainer for the GwenLaya v4 generator tiers (pre-registration section 5).

Everything that does not need torch (recipe resolution, architecture detection, LoRA target
discovery, VRAM/step estimates, checkpoint scheduling, GCS sync, resume) is plain Python, so
`--print-plan` and the unit tests run without torch, transformers, peft or trl. Heavy imports
happen lazily inside `run_training`.

  train_lora.py --recipe qwen3.5-4b-sft --gpu l4 --train-file sft.jsonl --out-dir RUN --print-plan
  train_lora.py --recipe qwen3.5-4b-sft --gpu l4 --train-file sft.jsonl --out-dir RUN \
      --gcs-sync gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/adapters/qwen35-4b-sft

Data: SFT rows `{"messages": [user, assistant]}`, DPO rows `{"prompt": [...], "chosen": [...],
"rejected": [...]}` as written by scripts/data/build_sft_dataset.py / build_dpo_pairs.py.

Nothing here is a measured result. VRAM numbers are a weights-only heuristic, not a measurement;
the S4c probe (`--max-steps 100`) records the real peak in run_manifest.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "experiments" / "plan.json"
SEED = 1234  # plan.json seeds.training
GIB = 1024 ** 3

# Hyper-parameters are the pre-registration section 5 table. Batch geometry is NOT specified there
# and is this script's default (effective batch 16). `anchor` is the v3 continuity model whose
# values come from lora_configs/lora_qwen2.5-coder_1.5b.json (3 epochs, lr 2e-4, r 16).
RECIPES: dict[str, dict[str, Any]] = {
    "qwen3.5-4b-sft": dict(stage="S4a", method="sft", model="Qwen/Qwen3.5-4B", r=16, alpha=32, lr=2e-4,
                           epochs=2, seq=2048, bs=2, ga=8),
    "qwen3.5-4b-dpo": dict(stage="S4a", method="dpo", model="Qwen/Qwen3.5-4B", r=16, alpha=32, lr=5e-6,
                           epochs=1, seq=2048, bs=1, ga=16, beta=0.1),
    "qwen3.5-9b-sft": dict(stage="S4b", method="sft", model="Qwen/Qwen3.5-9B", r=16, alpha=32, lr=1e-4,
                           epochs=2, seq=2048, bs=1, ga=16),
    "qwen3.8-27b-sft": dict(stage="S4d", method="sft", model="Qwen/Qwen3.8-27B", r=8, alpha=16, lr=5e-5,
                            epochs=1, seq=1024, bs=1, ga=16, conditional="S4c GO: peak VRAM <= 22 GiB in 100 steps"),
    "qwen2.5-coder-1.5b-sft": dict(stage=None, method="sft", model="Qwen/Qwen2.5-Coder-1.5B-Instruct", r=16,
                                   alpha=32, lr=2e-4, epochs=3, seq=2048, bs=2, ga=4, anchor=True),
}

# Parameter counts (billions) from the task context (HF model cards, 2026-10-07); the 1.5B anchor is nominal.
PARAMS_B = {"Qwen/Qwen3.5-4B": 4.7, "Qwen/Qwen3.5-9B": 9.7, "Qwen/Qwen3.8-27B": 27.8,
            "Qwen/Qwen2.5-Coder-1.5B-Instruct": 1.5}

GPUS = {  # name -> (vram GiB, compute dtype, max trainable params in B per the plan's fit notes)
    "t4": dict(vram_gib=16, dtype="float16", max_params_b=4.7),
    "l4": dict(vram_gib=24, dtype="bfloat16", max_params_b=27.8),
}

ALLOWED_GCS_PREFIX_DEFAULT = "gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/"
S4C_VRAM_GATE_GIB = 22

LORA_EXCLUDE = ("vision", "visual", "merger", "mm_projector", "lm_head", "embed", "router", "audio")
LORA_EXCLUDE_LEAF = {"gate", "router"}  # MoE routers, not the SwiGLU gate_proj


# ---- plan / recipe resolution ------------------------------------------------------------------

def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def stage_of(plan: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for s in plan["stages"]:
        if s["id"] == stage_id:
            return s
    raise KeyError(f"stage {stage_id} not in plan")


def resolve_recipe(name: str, gpu: str, plan: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    """Merge recipe + gpu profile + CLI overrides; cross-check against plan.json. Raises ValueError."""
    if name not in RECIPES:
        raise ValueError(f"unknown recipe {name!r}; known: {sorted(RECIPES)}")
    if gpu not in GPUS:
        raise ValueError(f"unknown --gpu {gpu!r}; known: {sorted(GPUS)}")
    cfg = dict(RECIPES[name])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg.update(recipe=name, gpu=gpu, compute_dtype=GPUS[gpu]["dtype"], seed=SEED, quant="nf4+double_quant")
    pb = PARAMS_B.get(cfg["model"])
    cfg["params_b"] = pb if pb is not None else "TBD"
    if pb is not None and pb > GPUS[gpu]["max_params_b"]:
        raise ValueError(f"{cfg['model']} ({pb}B) does not fit QLoRA on {gpu} per the plan's fit notes")
    if cfg.get("stage") and plan is not None:
        st = stage_of(plan, cfg["stage"])
        if cfg["model"] not in st["models"]:
            raise ValueError(f"{cfg['model']} not in plan stage {cfg['stage']} models {st['models']}")
        cfg["plan_stage"] = {k: st.get(k) for k in ("id", "where", "cap_usd", "max_hours", "est_hours")}
    return cfg


def estimate_vram(cfg: dict[str, Any]) -> dict[str, Any]:
    """Weights-only lower bound for the 4-bit base (0.5 byte/param). Activations/optimizer/KV are TBD:
    they depend on hidden size and layer count, which are read from config.json at run time."""
    pb = cfg["params_b"]
    gpu_gib = GPUS[cfg["gpu"]]["vram_gib"]
    if pb == "TBD":
        return {"weights_4bit_gib_lower_bound": "TBD", "gpu_vram_gib": gpu_gib, "activations_gib": "TBD",
                "note": "unknown parameter count"}
    w = round(pb * 1e9 * 0.5 / GIB, 2)
    return {"weights_4bit_gib_lower_bound": w, "gpu_vram_gib": gpu_gib, "activations_gib": "TBD",
            "weights_fraction_of_vram": round(w / gpu_gib, 3),
            "note": "heuristic lower bound, NOT measured; S4c / --max-steps probe records peak VRAM"}


def estimate_steps(n_examples: int | None, cfg: dict[str, Any]) -> dict[str, Any]:
    eff = cfg["bs"] * cfg["ga"]
    if not n_examples:
        return {"effective_batch": eff, "steps_per_epoch": "TBD", "total_steps": "TBD"}
    spe = math.ceil(n_examples / eff)
    return {"effective_batch": eff, "n_examples": n_examples, "steps_per_epoch": spe,
            "total_steps": spe * cfg["epochs"]}


# ---- architecture detection / LoRA targets -----------------------------------------------------

def detect_text_backbone(config: dict[str, Any]) -> dict[str, Any]:
    """Decide how to load a repo from its config.json. Image-text-to-text repos (Qwen3.5) nest the language
    model under `text_config` next to a `vision_config`; we train only that text backbone."""
    archs = config.get("architectures") or []
    text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else None
    multimodal = bool(
        config.get("vision_config") or text_cfg
        or any("ConditionalGeneration" in a or "ImageTextToText" in a or "VL" in a for a in archs))
    inner = text_cfg or config
    return {
        "architectures": archs,
        "model_type": config.get("model_type"),
        "multimodal": multimodal,
        "text_model_type": inner.get("model_type"),
        "hidden_size": inner.get("hidden_size", "TBD"),
        "num_hidden_layers": inner.get("num_hidden_layers", "TBD"),
        "is_moe": bool(inner.get("num_experts") or inner.get("num_local_experts")),
        "load_text_config_only": multimodal and text_cfg is not None,
        "freeze_prefixes": ["visual", "vision_tower", "multi_modal_projector"] if multimodal else [],
    }


def discover_lora_targets(named_linear: Iterable[str]) -> list[str]:
    """Pick LoRA targets from `model.named_modules()` names of nn.Linear (or bnb Linear4bit) modules:
    every linear projection of the text backbone, excluding vision/embedding/lm_head/MoE-router modules.
    Reading names (not hard-coding q_proj...) covers Qwen3.5's non-standard attention blocks."""
    out = []
    for n in named_linear:
        low = n.lower()
        if any(x in low for x in LORA_EXCLUDE) or n.rsplit(".", 1)[-1] in LORA_EXCLUDE_LEAF:
            continue
        out.append(n)
    if not out:
        raise ValueError("no LoRA target modules found; refusing to train zero parameters")
    return sorted(set(out))


# ---- data --------------------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{i}: bad JSON ({e})") from e
    return rows


def count_rows(path: str | Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def validate_rows(rows: Sequence[dict[str, Any]], method: str, allow_dry_run: bool = False) -> None:
    """Fail loudly on malformed rows and on dry-run (fake-generator) data."""
    if not rows:
        raise ValueError("training file has no rows")
    for i, r in enumerate(rows):
        if method == "sft":
            m = r.get("messages")
            ok = isinstance(m, list) and len(m) >= 2 and m[0].get("role") == "user" and m[-1].get("role") == "assistant"
        else:
            ok = all(isinstance(r.get(k), list) and r[k] for k in ("prompt", "chosen", "rejected"))
        if not ok:
            raise ValueError(f"row {i}: not a valid {method} row")
        prov = r.get("provenance") or {}
        flat = [prov] + [v for v in prov.values() if isinstance(v, dict)]
        if not allow_dry_run and any(p.get("dry_run") for p in flat):
            raise ValueError(f"row {i}: dry_run data (fake generator); pass --allow-dry-run only for smoke tests")


def sft_to_prompt_completion(row: dict[str, Any]) -> dict[str, Any]:
    """Conversational prompt/completion format -> TRL masks the prompt (loss on completion tokens only)."""
    m = row["messages"]
    return {"prompt": m[:-1], "completion": [m[-1]]}


# ---- checkpointing -----------------------------------------------------------------------------

COMMITTED = "COMMITTED"
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


class SaveSchedule:
    """Save every `every_steps` steps or `every_seconds` seconds, whichever comes first."""

    def __init__(self, every_steps: int = 200, every_seconds: float = 900.0,
                 clock: Callable[[], float] = time.monotonic):
        self.every_steps, self.every_seconds, self.clock = every_steps, every_seconds, clock
        self._last_t = clock()
        self._last_step = 0

    def should_save(self, step: int) -> bool:
        due = (self.every_steps and step - self._last_step >= self.every_steps) or \
              (self.every_seconds and self.clock() - self._last_t >= self.every_seconds)
        if due and step > self._last_step:
            self._last_step, self._last_t = step, self.clock()
            return True
        return False


def latest_committed(ckpt_dir: str | Path) -> Path | None:
    """Highest checkpoint-N directory that carries a COMMITTED marker (a half-written one is ignored)."""
    d = Path(ckpt_dir)
    if not d.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for p in d.iterdir():
        m = _CKPT_RE.match(p.name)
        if m and p.is_dir() and (p / COMMITTED).exists() and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    return best[1] if best else None


def mark_committed(ckpt: str | Path) -> None:
    (Path(ckpt) / COMMITTED).write_text("ok\n")


def check_gcs_uri(uri: str, allowed_prefix: str = ALLOWED_GCS_PREFIX_DEFAULT) -> str:
    """Writes are allowed only under the plan's gwenlaya_v4/ prefix."""
    if not uri.startswith(allowed_prefix) or ".." in uri:
        raise ValueError(f"refusing gcs path {uri!r}: must be under {allowed_prefix}")
    return uri.rstrip("/")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


class GcsSync:
    """Upload each local checkpoint, then its COMMITTED marker last, so a preempted upload is never resumed."""

    def __init__(self, remote: str, runner: Callable[[list[str]], Any] = _run,
                 allowed_prefix: str = ALLOWED_GCS_PREFIX_DEFAULT):
        self.remote = check_gcs_uri(remote, allowed_prefix)
        self.run = runner

    def push(self, ckpt: str | Path) -> None:
        ckpt = Path(ckpt)
        dest = f"{self.remote}/{ckpt.name}"
        self.run(["gcloud", "storage", "rsync", "--recursive", "--exclude", f"^{COMMITTED}$", str(ckpt), dest])
        self.run(["gcloud", "storage", "cp", str(ckpt / COMMITTED), f"{dest}/{COMMITTED}"])

    def latest_remote_committed(self) -> str | None:
        try:
            out = self.run(["gcloud", "storage", "ls", f"{self.remote}/checkpoint-*/{COMMITTED}"])
        except subprocess.CalledProcessError:
            return None  # nothing there yet
        steps = [int(m.group(1)) for line in (out.stdout or "").splitlines()
                 if (m := re.search(r"/checkpoint-(\d+)/" + COMMITTED + r"\s*$", line))]
        return f"checkpoint-{max(steps)}" if steps else None

    def pull(self, name: str, local_dir: str | Path) -> Path:
        dest = Path(local_dir) / name
        dest.mkdir(parents=True, exist_ok=True)
        self.run(["gcloud", "storage", "rsync", "--recursive", f"{self.remote}/{name}", str(dest)])
        return dest


def find_resume(ckpt_dir: str | Path, sync: GcsSync | None = None) -> Path | None:
    """Local committed checkpoint wins unless the remote one is newer; else pull the remote one."""
    local = latest_committed(ckpt_dir)
    if sync is not None:
        remote = sync.latest_remote_committed()
        if remote:
            rstep = int(remote.split("-")[1])
            if local is None or rstep > int(local.name.split("-")[1]):
                return sync.pull(remote, ckpt_dir)
    return local


# ---- training (lazy heavy imports) -------------------------------------------------------------

def _need(mod: str) -> Any:
    try:
        return importlib.import_module(mod)
    except ImportError as e:
        raise SystemExit(f"missing dependency {mod!r}: pip install -e '.[train]'  ({e})") from e


def _filter_kwargs(fn: Callable, kw: dict[str, Any]) -> dict[str, Any]:
    import inspect
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


def run_training(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    torch = _need("torch")
    transformers = _need("transformers")
    peft = _need("peft")
    trl = _need("trl")
    datasets = _need("datasets")
    _need("bitsandbytes")

    out = Path(args.out_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sync = GcsSync(args.gcs_sync) if args.gcs_sync else None

    rows = read_jsonl(args.train_file)
    validate_rows(rows, cfg["method"], args.allow_dry_run)
    dtype = getattr(torch, cfg["compute_dtype"])
    base = args.base_model_path or cfg["model"]

    acfg = transformers.AutoConfig.from_pretrained(base)
    arch = detect_text_backbone(acfg.to_dict())
    bnb = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    load_kw = dict(quantization_config=bnb, torch_dtype=dtype, device_map={"": 0})
    if arch["load_text_config_only"]:
        # image-text-to-text repo: try the full config through the causal-LM auto class first, then the
        # nested text config. Which one maps weights correctly was not verified for Qwen3.5 offline.
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(base, **load_kw)
        except (ValueError, KeyError):
            model = transformers.AutoModelForCausalLM.from_pretrained(base, config=acfg.get_text_config(), **load_kw)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(base, **load_kw)
    tok = transformers.AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_mod = importlib.import_module("bitsandbytes")
    linear_types = (torch.nn.Linear, bnb_mod.nn.Linear4bit)
    targets = discover_lora_targets(n for n, m in model.named_modules() if isinstance(m, linear_types))
    model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lcfg = peft.LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.05, target_modules=targets,
                           bias="none", task_type="CAUSAL_LM")

    resume = None if args.no_resume else find_resume(ckpt_dir, sync)
    schedule = SaveSchedule(args.save_steps, args.save_minutes * 60)

    class _Ckpt(transformers.TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if schedule.should_save(state.global_step):
                control.should_save = True
            return control

        def on_save(self, a, state, control, **kw):
            if not state.is_world_process_zero:
                return
            p = ckpt_dir / f"checkpoint-{state.global_step}"
            if p.is_dir():
                mark_committed(p)
                if sync:
                    sync.push(p)

    common = dict(
        output_dir=str(ckpt_dir), per_device_train_batch_size=cfg["bs"], gradient_accumulation_steps=cfg["ga"],
        learning_rate=cfg["lr"], num_train_epochs=cfg["epochs"], max_steps=args.max_steps or -1,
        lr_scheduler_type="cosine", warmup_ratio=0.03, optim="paged_adamw_8bit", gradient_checkpointing=True,
        bf16=cfg["compute_dtype"] == "bfloat16", fp16=cfg["compute_dtype"] == "float16",
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2, logging_steps=10,
        seed=SEED, report_to="none")
    if cfg["method"] == "sft":
        ds = datasets.Dataset.from_list([sft_to_prompt_completion(r) for r in rows])
        targs = trl.SFTConfig(**_filter_kwargs(trl.SFTConfig, dict(common, max_length=cfg["seq"], completion_only_loss=True)))
        trainer_cls = trl.SFTTrainer
    else:
        ds = datasets.Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows])
        targs = trl.DPOConfig(**_filter_kwargs(trl.DPOConfig, dict(common, beta=cfg["beta"], max_length=cfg["seq"])))
        trainer_cls = trl.DPOTrainer
    tkw = dict(model=model, args=targs, train_dataset=ds, peft_config=lcfg, callbacks=[_Ckpt()],
               processing_class=tok, tokenizer=tok)
    trainer = trainer_cls(**_filter_kwargs(trainer_cls.__init__, tkw))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / GIB
    final = out / "adapter"
    trainer.model.save_pretrained(str(final))
    tok.save_pretrained(str(final))
    if sync:
        sync.run(["gcloud", "storage", "rsync", "--recursive", str(final), f"{sync.remote}/adapter"])
    manifest = {
        "recipe": cfg, "arch": arch, "lora_target_count": len(targets), "resumed_from": str(resume) if resume else None,
        "n_train_rows": len(rows), "global_step": trainer.state.global_step, "train_metrics": dict(res.metrics),
        "wall_seconds": round(wall, 1), "peak_vram_gib_measured": round(peak, 2),
        "versions": {m: getattr(importlib.import_module(m), "__version__", "?")
                     for m in ("torch", "transformers", "peft", "trl", "bitsandbytes")},
    }
    if args.max_steps:
        manifest["probe_vram_gate_gib"] = S4C_VRAM_GATE_GIB
        manifest["probe_vram_ok"] = peak <= S4C_VRAM_GATE_GIB
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


# ---- CLI ---------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--gpu", required=True, choices=sorted(GPUS), help="t4 -> fp16 compute, l4 -> bf16 compute")
    ap.add_argument("--train-file", help="SFT or DPO JSONL from scripts/data/")
    ap.add_argument("--num-examples", type=int, help="for --print-plan when no file is present")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--base-model-path", help="override base (e.g. an SFT-merged model for the DPO stage)")
    ap.add_argument("--max-steps", type=int, help="probe run, e.g. 100 for the S4c gate")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seq", type=int)
    ap.add_argument("--bs", type=int)
    ap.add_argument("--ga", type=int)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--save-minutes", type=float, default=15.0)
    ap.add_argument("--gcs-sync", help="gs://.../gwenlaya_v4/... directory for checkpoints (spot preemption)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--allow-dry-run", action="store_true")
    ap.add_argument("--plan", default=str(PLAN_PATH))
    ap.add_argument("--print-plan", action="store_true", help="print the resolved plan; imports no heavy library")
    return ap


def make_plan(args: argparse.Namespace) -> dict[str, Any]:
    cfg = resolve_recipe(args.recipe, args.gpu, load_plan(Path(args.plan)), lr=args.lr, epochs=args.epochs,
                         seq=args.seq, bs=args.bs, ga=args.ga)
    n = args.num_examples
    if n is None and args.train_file and Path(args.train_file).exists():
        n = count_rows(args.train_file)
    return {"config": cfg, "vram_estimate": estimate_vram(cfg), "steps": estimate_steps(n, cfg),
            "checkpoint": {"every_steps": args.save_steps, "every_minutes": args.save_minutes,
                           "gcs_sync": args.gcs_sync, "resume": not args.no_resume}}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = make_plan(args)
        if args.gcs_sync:
            check_gcs_uri(args.gcs_sync)
    except (ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.print_plan:
        print(json.dumps(plan, indent=1))
        return 0
    if not args.train_file or not args.out_dir:
        print("error: --train-file and --out-dir are required to train", file=sys.stderr)
        return 2
    run_training(plan["config"], args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
