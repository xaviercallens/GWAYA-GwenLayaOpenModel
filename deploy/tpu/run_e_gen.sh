#!/usr/bin/env bash
# Runs ON the TPU VM: batch-generate candidates for the E set (prompts only, no tests) per tier.
# Env: MODELS="hf_id|served_name ..." (default: coder anchor, Qwen3.5-2B, Qwen3.5-4B), MAX_SEQS, LIMIT.
set -uo pipefail
R=~/smoke; cd "$R"
export HF_HOME=$R/hf PATH=$HOME/.local/bin:$PATH SKIP_JAX_PRECOMPILE=1
MODELS="${MODELS:-Qwen/Qwen2.5-Coder-1.5B-Instruct|qwen2.5-coder-1.5b-bf16 Qwen/Qwen3.5-2B|qwen3.5-2b-bf16 Qwen/Qwen3.5-4B|qwen3.5-4b-bf16}"
st() { printf '{"step":"%s","rc":%s,"secs":%s,"note":%s}\n' "$1" "$2" "$3" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1][-500:]))' "${4:-}")" >> $R/status.jsonl; }
step() { local name=$1 tmo=$2; shift 2; local t0=$SECONDS; timeout "$tmo" bash -c "$*" > "$R/$name.log" 2>&1; local rc=$?; st "$name" "$rc" $((SECONDS-t0)) "$(grep -E 'gen_batch|Error|Traceback' $R/$name.log | tail -4)"; return $rc; }

# repo (for gwaya.generators prompt building)
mkdir -p repo && tar xzf repo.tgz -C repo
step uv 300 "curl -LsSf https://astral.sh/uv/install.sh | sh"
step venv 1500 "uv venv -q -p 3.12 vllm-venv && VIRTUAL_ENV=$R/vllm-venv uv pip install -q vllm-tpu && $R/vllm-venv/bin/python -c 'import vllm, jax; print(vllm.__version__, jax.devices())'"
PY=$R/vllm-venv/bin/python
# A33: spot runs push partial outputs to the bucket every 5 min so a preemption loses minutes, not the stage
# (best effort: a VM without storage scope just skips it). gen_batch resumes from any raw file uploaded back.
if [ -n "${SYNC_URI:-}" ]; then
  ( while sleep 300; do gcloud storage cp $R/raw_*.jsonl $R/status.jsonl "$SYNC_URI/" >/dev/null 2>&1 || gsutil -q cp $R/raw_*.jsonl $R/status.jsonl "$SYNC_URI/" >/dev/null 2>&1; done ) &
fi
# Hybrid (Qwen3.5) models can refuse to start when max-seqs is too high for the 16 GB chip
# ("Mamba and attention pools together exceed the HBM budget"). Try a ladder; output is resumable.
KINDS="${KINDS:-${KIND:-answer}}"            # answer | pot (program-of-thought programs for the math gate) | sample (SAMPLES draws, T 0.8); several allowed
for spec in $MODELS; do
  hf="${spec%%|*}"; served="${spec##*|}"
  for KIND in $KINDS; do
    OUTPFX=raw_; [ "$KIND" = pot ] && OUTPFX=raw_pot_; [ "$KIND" = sample ] && OUTPFX=raw_sample_; [ "$KIND" = tests ] && OUTPFX=raw_tests_
    for seqs in ${MAX_SEQS_LADDER:-96 32 16 8}; do
      step "gen_${KIND}_${served}_s$seqs" ${STEP_TIMEOUT:-3600} "$PY repo/scripts/tpu/gen_batch.py --model $hf ${GEN_EXTRA:-} --served $served --tasks tasks_prompts.jsonl --out ${OUTPFX}$served.jsonl --kind $KIND --samples $([ "$KIND" = tests ] && echo ${TEST_SAMPLES:-2} || echo ${SAMPLES:-8}) --max-seqs $seqs --tensor-parallel ${TP:-1} ${LIMIT:+--limit $LIMIT}"
      rc=$?
      [ $rc -eq 0 ] && break
      if [ $rc -eq 3 ]; then echo '{"step":"ABORT_NO_LOGPROBS","rc":3}' >> $R/status.jsonl; exit 3; fi
    done
  done
  # free the disk before the next (large) checkpoint; the 35B-A3B alone is ~70 GB in bf16
  [ -n "${CLEAN_HF:-}" ] && rm -rf "$HF_HOME/hub/models--${hf//\//--}"
done
echo SMOKE_DONE >> $R/status.jsonl
