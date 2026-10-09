#!/usr/bin/env bash
# Runs ON a TPU VM: LoRA-train Laya (ModernBERT router + calibrator heads) with torch_xla, plus a bounded generator-LoRA probe.
# Inputs next to this script: repo.tgz, router.jsonl, calibrator.jsonl, router_calib.jsonl, calibrator_calib.jsonl.  Env: EPOCHS, BATCH, MAXLEN, TIMING_ONLY=1, PROBE=1.
set -uo pipefail
R=~/smoke; cd "$R"
export HF_HOME=$R/hf PATH=$HOME/.local/bin:$PATH PJRT_DEVICE=TPU
st() { printf '{"step":"%s","rc":%s,"secs":%s,"note":%s}\n' "$1" "$2" "$3" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1][-500:]))' "${4:-}")" >> $R/status.jsonl; }
step() { local name=$1 tmo=$2; shift 2; local t0=$SECONDS; timeout "$tmo" bash -c "$*" > "$R/$name.log" 2>&1; local rc=$?; st "$name" "$rc" $((SECONDS-t0)) "$(grep -E 'train_laya|Error|Traceback|torch_xla' $R/$name.log | tail -4)"; return $rc; }
mkdir -p repo && tar xzf repo.tgz -C repo
step uv 300 "curl -LsSf https://astral.sh/uv/install.sh | sh"
step venv 1500 "uv venv -q -p 3.11 tx-venv && VIRTUAL_ENV=$R/tx-venv uv pip install -q 'torch==2.8.0' 'torch_xla[tpu]==2.8.0' -f https://storage.googleapis.com/libtpu-wheels/index.html && VIRTUAL_ENV=$R/tx-venv uv pip install -q transformers peft safetensors accelerate && $R/tx-venv/bin/python -c 'import torch_xla.core.xla_model as xm, torch; print(torch.__version__, xm.xla_device())'"
PY=$R/tx-venv/bin/python
ARGS="--device xla --bf16 --epochs ${EPOCHS:-3} --batch-size ${BATCH:-16} --max-len ${MAXLEN:-512} --no-slice-filter"
step timing_cal 1200 "$PY repo/scripts/train_laya.py --mode calibrator --data calibrator.jsonl --out-dir out/timing $ARGS --timing-steps 30"
if [ "${TIMING_ONLY:-0}" != 1 ]; then
  step train_router 5400 "$PY repo/scripts/train_laya.py --mode router --data router.jsonl --out-dir out/router $ARGS --predict calib=router_calib.jsonl"
  step train_calibrator 7200 "$PY repo/scripts/train_laya.py --mode calibrator --data calibrator.jsonl --out-dir out/calibrator $ARGS --predict calib=calibrator_calib.jsonl"
  step train_router_sh 5400 "$PY repo/scripts/train_laya.py --mode router --data router.jsonl --out-dir out/router_sh $ARGS --shuffle-labels --predict calib=router_calib.jsonl"
  step train_calibrator_sh 7200 "$PY repo/scripts/train_laya.py --mode calibrator --data calibrator.jsonl --out-dir out/calibrator_sh $ARGS --shuffle-labels --predict calib=calibrator_calib.jsonl"
fi
[ "${PROBE:-1}" = 1 ] && step lora_probe 1500 "$PY repo/scripts/tpu/lora_probe.py --out out/lora_probe.json"
tar czf out.tgz out status.jsonl 2>/dev/null
echo SMOKE_DONE
