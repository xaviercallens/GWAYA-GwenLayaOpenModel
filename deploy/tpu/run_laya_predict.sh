#!/usr/bin/env bash
# Runs ON a TPU VM: score row files with the FROZEN Laya (adapter + heads from laya_v1.tgz), no training.
# Inputs: repo.tgz, laya_v1.tgz (router/ calibrator/), router_eprime.jsonl, calibrator_eprime.jsonl, router.jsonl, calibrator.jsonl (only for the train-data schema check).
set -uo pipefail
R=~/smoke; cd "$R"
export HF_HOME=$R/hf PATH=$HOME/.local/bin:$PATH PJRT_DEVICE=TPU
st() { printf '{"step":"%s","rc":%s,"secs":%s}\n' "$1" "$2" "$3" >> $R/status.jsonl; }
step() { local name=$1 tmo=$2; shift 2; local t0=$SECONDS; timeout "$tmo" bash -c "$*" > "$R/$name.log" 2>&1; local rc=$?; st "$name" "$rc" $((SECONDS-t0)); return $rc; }
mkdir -p repo laya && tar xzf repo.tgz -C repo && tar xzf laya_v1.tgz -C laya
step uv 300 "curl -LsSf https://astral.sh/uv/install.sh | sh"
step venv 1500 "uv venv -q -p 3.11 tx-venv && VIRTUAL_ENV=$R/tx-venv uv pip install -q 'torch==2.8.0' 'torch_xla[tpu]==2.8.0' -f https://storage.googleapis.com/libtpu-wheels/index.html && VIRTUAL_ENV=$R/tx-venv uv pip install -q transformers peft safetensors accelerate"
PY=$R/tx-venv/bin/python
ARGS="--tiers qwen3.5-2b-bf16,qwen3.5-4b-bf16,qwen3.5-9b-bf16 --device xla --bf16 --batch-size 16 --max-len 512 --no-slice-filter"
step pred_calibrator 2400 "$PY repo/scripts/train_laya.py --mode calibrator --data calibrator_eprime.jsonl --out-dir out/calibrator $ARGS --load-dir laya/calibrator --predict eprime=calibrator_eprime.jsonl"
step pred_router 2400 "$PY repo/scripts/train_laya.py --mode router --data router_eprime.jsonl --out-dir out/router $ARGS --load-dir laya/router --predict eprime=router_eprime.jsonl"
tar czf out.tgz out status.jsonl 2>/dev/null
echo SMOKE_DONE
