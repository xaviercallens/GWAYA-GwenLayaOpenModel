#!/usr/bin/env bash
# Runs ON a v5litepod-4 TPU VM: time-boxed, INSTRUMENTED load probe of Qwen3.5-27B (bf16, TP=4).
# The first attempt (ledger tpu_b27-tp4-122313, 33 min, $3.28) printed nothing after "Using FLASH_ATTN backend" and
# never generated a token. This probe records WHY: memory, disk and HF-cache growth every 30 s, the engine process CPU,
# and a py-spy stack dump at fixed minutes, then stops at the time box. Env: PROBE_MIN (default 22), MODEL.
set -uo pipefail
R=~/smoke; cd "$R"
export HF_HOME=$R/hf PATH=$HOME/.local/bin:$PATH SKIP_JAX_PRECOMPILE=1
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"; PROBE_MIN="${PROBE_MIN:-22}"
st() { printf '{"step":"%s","rc":%s,"secs":%s,"note":%s}\n' "$1" "$2" "$3" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1][-500:]))' "${4:-}")" >> $R/status.jsonl; }
mkdir -p repo && tar xzf repo.tgz -C repo
t0=$SECONDS; curl -LsSf https://astral.sh/uv/install.sh | sh > uv.log 2>&1; st uv $? $((SECONDS-t0))
t0=$SECONDS; (uv venv -q -p 3.12 vllm-venv && VIRTUAL_ENV=$R/vllm-venv uv pip install -q vllm-tpu py-spy) > venv.log 2>&1; st venv $? $((SECONDS-t0))
PY=$R/vllm-venv/bin/python
( while true; do
    echo "== $(date -Is) up=$SECONDS"; free -g | sed -n 2p; df -h ~ | tail -1; du -sh "$R/hf" 2>/dev/null
    ps -eo pid,etime,pcpu,rss,cmd --sort=-rss | head -4 | cut -c1-160
    sleep 30
  done ) > monitor.log 2>&1 &
MON=$!
( prev=0; for m in 6 12 18; do
    sleep $(( (m - prev) * 60 )); prev=$m
    for pid in $(pgrep -f "EngineCore" | head -2); do
      sudo -n "$R/vllm-venv/bin/py-spy" dump --pid "$pid" > "pyspy_${m}min_$pid.txt" 2>&1 || "$R/vllm-venv/bin/py-spy" dump --pid "$pid" > "pyspy_${m}min_$pid.txt" 2>&1 || true
    done
  done ) &
SPY=$!
t0=$SECONDS
timeout $(( PROBE_MIN * 60 )) $PY repo/scripts/tpu/gen_batch.py --model "$MODEL" --served qwen3.5-27b-bf16 \
  --tasks tasks_prompts.jsonl --out raw_qwen3.5-27b-bf16.jsonl --max-seqs 8 --max-model-len 2048 \
  --tensor-parallel 4 --limit 2 > gen_answer_27b_probe.log 2>&1
rc=$?; st gen_27b_probe $rc $((SECONDS-t0)) "$(grep -E 'gen_batch|Error|Traceback|engine ready' gen_answer_27b_probe.log | tail -4)"
kill $MON $SPY 2>/dev/null
echo SMOKE_DONE >> $R/status.jsonl
