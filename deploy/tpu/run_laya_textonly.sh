#!/usr/bin/env bash
# Runs ON a TPU VM: A16 text-only calibrator ablation (train once, then score C, E' and the fresh Rust set R').
# Inputs: repo.tgz, calibrator.jsonl (train), calibrator_calib.jsonl, calibrator_eprime.jsonl, calibrator_rfresh.jsonl.
set -uo pipefail
R=~/smoke; cd "$R"
export HF_HOME=$R/hf PATH=$HOME/.local/bin:$PATH PJRT_DEVICE=TPU
st() { printf '{"step":"%s","rc":%s,"secs":%s}\n' "$1" "$2" "$3" >> $R/status.jsonl; }
step() { local name=$1 tmo=$2; shift 2; local t0=$SECONDS; timeout "$tmo" bash -c "$*" > "$R/$name.log" 2>&1; local rc=$?; st "$name" "$rc" $((SECONDS-t0)); return $rc; }
mkdir -p repo && tar xzf repo.tgz -C repo
step uv 300 "curl -LsSf https://astral.sh/uv/install.sh | sh"
step venv 1500 "uv venv -q -p 3.11 tx-venv && VIRTUAL_ENV=$R/tx-venv uv pip install -q 'torch==2.8.0' 'torch_xla[tpu]==2.8.0' -f https://storage.googleapis.com/libtpu-wheels/index.html && VIRTUAL_ENV=$R/tx-venv uv pip install -q transformers peft safetensors accelerate"
PY=$R/tx-venv/bin/python
ARGS="--tiers qwen3.5-2b-bf16,qwen3.5-4b-bf16,qwen3.5-9b-bf16 --device xla --bf16 --epochs 3 --batch-size 16 --max-len 512 --no-slice-filter --text-only"
step train_textonly 5400 "$PY repo/scripts/train_laya.py --mode calibrator --data calibrator.jsonl --out-dir out/textonly $ARGS --predict calib=calibrator_calib.jsonl eprime=calibrator_eprime.jsonl rfresh=calibrator_rfresh.jsonl" || { echo '{"step":"ABORT_AFTER_train_textonly"}' >> $R/status.jsonl; tar czf out.tgz out status.jsonl; exit 1; }
tar czf out.tgz out status.jsonl 2>/dev/null
echo SMOKE_DONE
