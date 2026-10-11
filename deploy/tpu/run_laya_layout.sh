#!/usr/bin/env bash
# Runs ON a TPU VM: A22 layout ablation. Train the Laya CALIBRATOR ONLY (the router sees no candidate) with the candidate
# placed before the prompt (--layout candidate_first), same recipe as the frozen v1 and as the A16 text-only run
# (deploy/tpu/run_laya_textonly.sh): ModernBERT-large, LoRA r=16, 3 epochs, lr 2e-4 (script default), batch 16,
# max length 512, seed 1234 (script constant), bf16, same 9,243 training rows, --no-slice-filter, the numeric features kept.
# The ONLY difference to v1 is the text layout. Then score C, E' (python+math) and the fresh Rust set R' with it.
# Inputs next to this script: repo.tgz, calibrator.jsonl (train: laya_train/calibrator.jsonl), calibrator_calib.jsonl
# (laya_calib), calibrator_eprime.jsonl (laya_eprime), calibrator_rfresh.jsonl (laya_rfresh).
# Output: out.tgz (out/calibrator_v2/{adapter,heads.safetensors,head_config.json,run_manifest.json,preds_*.jsonl}, logs, status).
set -uo pipefail
R=~/smoke; cd "$R"
export HF_HOME=$R/hf PATH=$HOME/.local/bin:$PATH PJRT_DEVICE=TPU
st() { printf '{"step":"%s","rc":%s,"secs":%s}\n' "$1" "$2" "$3" >> $R/status.jsonl; }
step() { local name=$1 tmo=$2; shift 2; local t0=$SECONDS; timeout "$tmo" bash -c "$*" > "$R/$name.log" 2>&1; local rc=$?; st "$name" "$rc" $((SECONDS-t0)); return $rc; }
mkdir -p repo && tar xzf repo.tgz -C repo
step uv 300 "curl -LsSf https://astral.sh/uv/install.sh | sh"
step venv 1500 "uv venv -q -p 3.11 tx-venv && VIRTUAL_ENV=$R/tx-venv uv pip install -q 'torch==2.8.0' 'torch_xla[tpu]==2.8.0' -f https://storage.googleapis.com/libtpu-wheels/index.html && VIRTUAL_ENV=$R/tx-venv uv pip install -q transformers peft safetensors accelerate"
PY=$R/tx-venv/bin/python
# no --text-only (the [signals] line and the 16 features are kept, as v1); no --shuffle-labels
ARGS="--tiers qwen3.5-2b-bf16,qwen3.5-4b-bf16,qwen3.5-9b-bf16 --device xla --bf16 --epochs 3 --batch-size 16 --max-len 512 --no-slice-filter --layout candidate_first"
step plan 120 "$PY repo/scripts/train_laya.py --mode calibrator --data calibrator.jsonl --out-dir out/calibrator_v2 $ARGS --print-plan"
step train_layout 5400 "$PY repo/scripts/train_laya.py --mode calibrator --data calibrator.jsonl --out-dir out/calibrator_v2 $ARGS --predict calib=calibrator_calib.jsonl eprime=calibrator_eprime.jsonl rfresh=calibrator_rfresh.jsonl" || { echo '{"step":"ABORT_AFTER_train_layout"}' >> $R/status.jsonl; tar czf out.tgz out status.jsonl *.log; exit 1; }
tar czf out.tgz out status.jsonl *.log 2>/dev/null
echo SMOKE_DONE
