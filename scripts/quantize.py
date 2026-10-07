#!/usr/bin/env python3
"""Quantization pipeline for tuned GwenLaya tiers (pre-registration sections 5 and 6, plan stage C4).

  merge LoRA into the bf16 base -> HF dir
  -> llama.cpp convert_hf_to_gguf (f16) -> [llama-imatrix on train text] -> llama-quantize q8_0, q4_K_M
  -> optional AWQ (vLLM) -> Ollama Modelfile per GGUF -> manifest.json with sha256 + sizes

  quantize.py plan --base Qwen/Qwen3.5-4B --adapter RUN/adapter --out-dir DIR      # prints the plan, runs nothing
  quantize.py run  --base ... --adapter ... --out-dir DIR [--install-llama-cpp] [--awq]
  quantize.py transfer --source src_C.json --target tgt_C.json [--target-eval tgt_E.json]

Plan mode imports no heavy library and touches no disk. `run` executes each step and skips steps
whose output already exists. Heavy imports (torch, peft, awq) are lazy. Merge is refused above 9B
parameters (pre-registration: the 27B adapter stays unmerged and is served with llama.cpp --lora).

Quant evaluation: `transfer` implements the section 6 calibrator transfer on {scores, correct}
JSON files ({"scores": [...], "correct": [...]}), one per quant level, produced by the study runner:
(T) calibrator fit on the source quant (bf16) C and applied to the target quant;
(R) the same calibrator family refit on the target quant C; both are scored on the target eval
file and both are always reported. (F) full Laya retrain is not run here and is reported as such.
Nothing here is a measured result; metrics are computed only from files passed in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(os.environ.get("GWAYA_DATA_ROOT", os.path.expanduser("~/gwaya-data")))
LLAMA_CPP_DIR = DATA_ROOT / "llama.cpp"
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp"
QUANTS = ("q8_0", "q4_K_M")          # GGUF levels after the f16 intermediate; pre-registration section 5
EVAL_QUANTS = ("bf16",) + QUANTS      # levels the study runner evaluates (ablation factor, section 2)
MAX_MERGE_PARAMS_B = 9.7              # 31 GB local RAM; 27B is never merged
PARAMS_B = {"Qwen/Qwen3.5-2B": 2.0, "Qwen/Qwen3.5-4B": 4.7, "Qwen/Qwen3.5-9B": 9.7, "Qwen/Qwen3.8-27B": 27.8,
            "Qwen/Qwen2.5-Coder-1.5B-Instruct": 1.5}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def check_path(p: Path) -> None:
    """Large artifacts must not land on the nearly full root disk."""
    if not str(p.resolve()).startswith(("/mnt/data", "/tmp", "/var/tmp")) and "pytest" not in str(p):
        raise ValueError(f"{p}: write large artifacts under /mnt/data (root disk is nearly full)")


def modelfile_text(gguf_name: str, *, system: str | None = None, temperature: float = 0.0, num_ctx: int = 4096) -> str:
    lines = [f"FROM ./{gguf_name}", f"PARAMETER temperature {temperature:g}", f"PARAMETER num_ctx {num_ctx}"]
    if system:
        lines.append('SYSTEM """' + system.replace('"""', "'''") + '"""')
    return "\n".join(lines) + "\n"


def build_plan(base: str, adapter: str | None, out_dir: Path, *, name: str | None = None,
               quants: Sequence[str] = QUANTS, imatrix_text: str | None = None, awq: bool = False,
               llama_cpp: Path = LLAMA_CPP_DIR) -> dict[str, Any]:
    """Ordered steps with commands and output paths. Pure function; no I/O."""
    for q in quants:
        if q not in QUANTS:
            raise ValueError(f"unsupported quant {q!r}; choose from {QUANTS}")
    if adapter and PARAMS_B.get(base, 0.0) > MAX_MERGE_PARAMS_B:
        raise ValueError(f"{base}: merge refused above {MAX_MERGE_PARAMS_B}B params; "
                         "serve the unmerged adapter with llama.cpp --lora over the q4 base GGUF")
    name = name or base.split("/")[-1].lower()
    merged, f16 = out_dir / "merged_bf16", out_dir / f"{name}-f16.gguf"
    steps: list[dict[str, Any]] = []
    if adapter:
        steps.append({"id": "merge", "kind": "python", "output": str(merged),
                      "desc": f"merge_and_unload {adapter} into {base} in bf16 (CPU)"})
    src = str(merged) if adapter else base
    steps.append({"id": "convert_f16", "output": str(f16),
                  "cmd": [sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), src,
                          "--outfile", str(f16), "--outtype", "f16"]})
    imat = None
    if imatrix_text:
        imat = out_dir / f"{name}.imatrix"
        steps.append({"id": "imatrix", "output": str(imat),
                      "cmd": [str(llama_cpp / "build/bin/llama-imatrix"), "-m", str(f16), "-f", imatrix_text,
                              "-o", str(imat)]})
    for q in quants:
        out = out_dir / f"{name}-{q}.gguf"
        cmd = [str(llama_cpp / "build/bin/llama-quantize")]
        if imat:
            cmd += ["--imatrix", str(imat)]
        steps.append({"id": f"quantize_{q}", "output": str(out), "cmd": cmd + [str(f16), str(out), q]})
        steps.append({"id": f"modelfile_{q}", "output": str(out_dir / f"Modelfile.{q}"), "kind": "python",
                      "gguf": out.name})
    if awq:
        steps.append({"id": "awq", "kind": "python", "output": str(out_dir / f"{name}-awq"),
                      "desc": "AutoAWQ w4 g128 GEMM for vLLM (needs a GPU)"})
    return {"base": base, "adapter": adapter, "name": name, "out_dir": str(out_dir), "quants": list(quants),
            "imatrix_text": imatrix_text, "steps": steps, "eval_quants": list(EVAL_QUANTS),
            "ollama_tags": {q: f"{name}:{q}" for q in quants}}


def install_llama_cpp(dest: Path = LLAMA_CPP_DIR, runner=subprocess.run) -> None:
    check_path(dest)
    if not (dest / ".git").exists():
        runner(["git", "clone", "--depth", "1", LLAMA_CPP_URL, str(dest)], check=True)
    runner(["cmake", "-S", str(dest), "-B", str(dest / "build"), "-DGGML_CUDA=OFF"], check=True)
    runner(["cmake", "--build", str(dest / "build"), "--target", "llama-quantize", "llama-imatrix", "-j"], check=True)


def merge_adapter(base: str, adapter: str, out: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def awq_quantize(src: str, out: Path) -> None:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    m = AutoAWQForCausalLM.from_pretrained(src)
    tok = AutoTokenizer.from_pretrained(src)
    m.quantize(tok, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"})
    m.save_quantized(str(out))
    tok.save_pretrained(out)


def file_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def execute(plan: dict[str, Any], *, runner=subprocess.run, system: str | None = None) -> dict[str, Any]:
    out_dir = Path(plan["out_dir"])
    check_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done, skipped = [], []
    for st in plan["steps"]:
        out = Path(st["output"])
        if out.exists() and not st["id"].startswith("modelfile_"):
            skipped.append(st["id"])
            continue
        if st["id"] == "merge":
            merge_adapter(plan["base"], plan["adapter"], out)
        elif st["id"] == "awq":
            awq_quantize(str(out_dir / "merged_bf16") if plan["adapter"] else plan["base"], out)
        elif st["id"].startswith("modelfile_"):
            out.write_text(modelfile_text(st["gguf"], system=system))
        else:
            runner(st["cmd"], check=True)
        done.append(st["id"])
    files = {}
    for p in sorted(out_dir.glob("*.gguf")) + sorted(out_dir.glob("*.imatrix")) + sorted(out_dir.glob("Modelfile.*")):
        files[p.name] = file_record(p)
    manifest = {"base": plan["base"], "adapter": plan["adapter"], "ollama_tags": plan["ollama_tags"],
                "steps_run": done, "steps_skipped_existing": skipped, "files": files,
                "awq_dir": str(out_dir / f"{plan['name']}-awq") if any(s["id"] == "awq" for s in plan["steps"]) else None}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


# ---- calibrator transfer across quant levels (pre-registration section 6) ----------------------

def calibration_metrics(p: Sequence[float], y: Sequence[float], bins: int = 10) -> dict[str, Any]:
    n = len(p)
    if n == 0 or n != len(y):
        raise ValueError("p and y must be equal-length and non-empty")
    eps = 1e-12
    brier = sum((a - b) ** 2 for a, b in zip(p, y)) / n
    ll = -sum(b * math.log(max(a, eps)) + (1 - b) * math.log(max(1 - a, eps)) for a, b in zip(p, y)) / n
    ece = 0.0
    for k in range(bins):
        idx = [i for i, a in enumerate(p) if min(int(a * bins), bins - 1) == k]
        if idx:
            ece += len(idx) / n * abs(sum(p[i] for i in idx) / len(idx) - sum(y[i] for i in idx) / len(idx))
    return {"n": n, "brier": brier, "log_loss": ll, "ece": ece}


def transfer_report(source_c: dict, target_c: dict, target_eval: dict | None = None, *, method: str = "auto",
                    source_label: str = "bf16", target_label: str = "q4_K_M") -> dict[str, Any]:
    """(T) calibrator fit on source C applied to target; (R) refit on target C. Both scored on
    target_eval (default: target_c, which is then in-sample for R and labelled so)."""
    from gwaya.gwenlaya import fit_calibration
    ev = target_eval or target_c
    t_art = fit_calibration(source_c["scores"], source_c["correct"], method=method)
    r_art = fit_calibration(target_c["scores"], target_c["correct"], method=method)
    y = [float(v) for v in ev["correct"]]
    out: dict[str, Any] = {"source": source_label, "target": target_label,
                           "eval_is_target_c": target_eval is None, "F_full_retrain": "not_run"}
    for tag, art in (("T_transfer", t_art), ("R_refit", r_art)):
        out[tag] = {"calibrator": art.calibrator, "sha256": art.sha256,
                    **calibration_metrics(art.model().predict([float(s) for s in ev["scores"]]), y)}
    return out


# ---- CLI ----------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--base", required=True, help="HF id of the bf16 base, e.g. Qwen/Qwen3.5-4B")
        p.add_argument("--adapter", help="LoRA adapter dir; omit to quantize the base only")
        p.add_argument("--out-dir", required=True)
        p.add_argument("--name")
        p.add_argument("--quants", nargs="+", default=list(QUANTS))
        p.add_argument("--imatrix-text", help="TRAIN-split text file for the imatrix")
        p.add_argument("--awq", action="store_true")
        p.add_argument("--llama-cpp", default=str(LLAMA_CPP_DIR))
        p.add_argument("--system", help="SYSTEM prompt for the Modelfile")
        if name == "run":
            p.add_argument("--install-llama-cpp", action="store_true")
    t = sub.add_parser("transfer")
    t.add_argument("--source", required=True)
    t.add_argument("--target", required=True)
    t.add_argument("--target-eval")
    t.add_argument("--method", default="auto")
    t.add_argument("--source-label", default="bf16")
    t.add_argument("--target-label", default="q4_K_M")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "transfer":
            load = lambda f: json.loads(Path(f).read_text())  # noqa: E731
            rep = transfer_report(load(a.source), load(a.target), load(a.target_eval) if a.target_eval else None,
                                  method=a.method, source_label=a.source_label, target_label=a.target_label)
            print(json.dumps(rep, indent=1))
            return 0
        plan = build_plan(a.base, a.adapter, Path(a.out_dir), name=a.name, quants=a.quants,
                          imatrix_text=a.imatrix_text, awq=a.awq, llama_cpp=Path(a.llama_cpp))
        if a.cmd == "plan":
            print(json.dumps(plan, indent=1))
            return 0
        if a.install_llama_cpp:
            install_llama_cpp(Path(a.llama_cpp))
        if not (Path(a.llama_cpp) / "convert_hf_to_gguf.py").exists():
            print("error: llama.cpp not found; pass --install-llama-cpp", file=sys.stderr)
            return 2
        print(json.dumps(execute(plan, system=a.system), indent=1))
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
