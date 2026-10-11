#!/usr/bin/env bash
# Runs ON a TPU VM: score the fresh-set-F row files with the FROZEN Laya v1 (adapter + heads from laya_v1.tgz), no training
# (A21; same recipe as deploy/tpu/run_laya_predict.sh, which scored E'). The adapter/head hashes are checked against
# docs/LAYA_FREEZE.txt (uploaded as LAYA_FREEZE.txt) before anything is scored; a mismatch aborts.
# Inputs next to this script: repo.tgz, laya_v1.tgz (router/ calibrator/), LAYA_FREEZE.txt, router_F.jsonl, calibrator_F.jsonl.
# Output: out.tgz (out/calibrator/preds_F.jsonl, out/router/preds_F.jsonl, predict_manifest.json, status.jsonl).
set -uo pipefail
R=~/smoke; cd "$R"
export HF_HOME=$R/hf PATH=$HOME/.local/bin:$PATH PJRT_DEVICE=TPU
st() { printf '{"step":"%s","rc":%s,"secs":%s}\n' "$1" "$2" "$3" >> $R/status.jsonl; }
step() { local name=$1 tmo=$2; shift 2; local t0=$SECONDS; timeout "$tmo" bash -c "$*" > "$R/$name.log" 2>&1; local rc=$?; st "$name" "$rc" $((SECONDS-t0)); return $rc; }
mkdir -p repo laya && tar xzf repo.tgz -C repo && tar xzf laya_v1.tgz -C laya
bad=0
for f in router/adapter/adapter_model.safetensors router/heads.safetensors router/head_config.json \
         calibrator/adapter/adapter_model.safetensors calibrator/heads.safetensors calibrator/head_config.json; do
  want=$(awk -v f="$f" '$2 == f {print $1}' LAYA_FREEZE.txt)
  got=$(sha256sum "laya/$f" | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "HASH MISMATCH $f want=$want got=$got" | tee -a hash_check.log; bad=1; }
done
if [ "$bad" != 0 ]; then echo '{"step":"ABORT_hash_mismatch"}' >> status.jsonl; tar czf out.tgz status.jsonl hash_check.log; echo SMOKE_DONE; exit 1; fi
echo '{"step":"hash_check","rc":0,"secs":0}' >> status.jsonl
step uv 300 "curl -LsSf https://astral.sh/uv/install.sh | sh"
step venv 1500 "uv venv -q -p 3.11 tx-venv && VIRTUAL_ENV=$R/tx-venv uv pip install -q 'torch==2.8.0' 'torch_xla[tpu]==2.8.0' -f https://storage.googleapis.com/libtpu-wheels/index.html && VIRTUAL_ENV=$R/tx-venv uv pip install -q transformers peft safetensors accelerate"
PY=$R/tx-venv/bin/python
ARGS="--tiers qwen3.5-2b-bf16,qwen3.5-4b-bf16,qwen3.5-9b-bf16 --device xla --bf16 --batch-size 16 --max-len 512 --no-slice-filter"
step pred_calibrator 2400 "$PY repo/scripts/train_laya.py --mode calibrator --data calibrator_F.jsonl --out-dir out/calibrator $ARGS --load-dir laya/calibrator --predict F=calibrator_F.jsonl"
step pred_router 2400 "$PY repo/scripts/train_laya.py --mode router --data router_F.jsonl --out-dir out/router $ARGS --load-dir laya/router --predict F=router_F.jsonl"
tar czf out.tgz out status.jsonl *.log 2>/dev/null
echo SMOKE_DONE
