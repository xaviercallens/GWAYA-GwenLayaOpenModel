#!/usr/bin/env bash
# Runs ON a TPU VM (A28): train Laya v2 (router + calibrator LoRA, ModernBERT) on feedback-retrofitted data, score the
# held-out F fold, and run the generator-LoRA probe (Qwen3.5-0.8B/2B, torch_xla + peft; a measured failure is a result).
# Inputs next to this script: repo.tgz, calibrator_v2_train.jsonl, router_v2_train.jsonl, calibrator_F_heldout.jsonl,
# router_F_heldout.jsonl. Output: out.tgz (out/calibrator, out/router, lora_probe_*.json, status.jsonl, *.log).
set -uo pipefail
R=~/smoke; cd "$R"
export HF_HOME=$R/hf PATH=$HOME/.local/bin:$PATH PJRT_DEVICE=TPU
st() { printf '{"step":"%s","rc":%s,"secs":%s}\n' "$1" "$2" "$3" >> $R/status.jsonl; }
step() { local name=$1 tmo=$2; shift 2; local t0=$SECONDS; timeout "$tmo" bash -c "$*" > "$R/$name.log" 2>&1; local rc=$?; st "$name" "$rc" $((SECONDS-t0)); return $rc; }
mkdir -p repo && tar xzf repo.tgz -C repo
step uv 300 "curl -LsSf https://astral.sh/uv/install.sh | sh"
step venv 1500 "uv venv -q -p 3.11 tx-venv && VIRTUAL_ENV=$R/tx-venv uv pip install -q 'torch==2.8.0' 'torch_xla[tpu]==2.8.0' -f https://storage.googleapis.com/libtpu-wheels/index.html && VIRTUAL_ENV=$R/tx-venv uv pip install -q transformers peft safetensors accelerate"
PY=$R/tx-venv/bin/python
TIERS="qwen3.5-2b-bf16,qwen3.5-4b-bf16,qwen3.5-9b-bf16"
COMMON="--tiers $TIERS --device xla --bf16 --batch-size 16 --max-len 512 --no-slice-filter"
# 0. timing probe first: if the projection is bad the real steps are skipped by the timeouts below
step time_cal 600 "$PY repo/scripts/train_laya.py --mode calibrator --data calibrator_v2_train.jsonl --out-dir out/time $COMMON --timing-steps 50"
step train_calibrator 3600 "$PY repo/scripts/train_laya.py --mode calibrator --data calibrator_v2_train.jsonl --out-dir out/calibrator $COMMON --predict F_heldout=calibrator_F_heldout.jsonl"
step train_router 2400 "$PY repo/scripts/train_laya.py --mode router --data router_v2_train.jsonl --out-dir out/router $COMMON --predict F_heldout=router_F_heldout.jsonl"
# generator LoRA feasibility on TPU (no quality claim)
for m in 0.8B 2B; do
  step lora_probe_$m 900 "$PY repo/scripts/tpu/lora_probe.py --model Qwen/Qwen3.5-$m --steps 20 --out lora_probe_$m.json"
done
tar czf out.tgz out status.jsonl *.log lora_probe_*.json 2>/dev/null
echo SMOKE_DONE
