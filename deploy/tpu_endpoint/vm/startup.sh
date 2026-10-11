#!/usr/bin/env bash
# Runs ON the TPU VM at boot (TPU metadata startup-script-url, as root). Serves ONE model with `vllm serve`
# (OpenAI API) on 0.0.0.0:$PORT behind a bearer key, then runs the idle/TTL watchdog that deletes this VM.
# Configuration comes from TPU node metadata keys gwaya-* (set by the gateway when it creates the node).
set -uo pipefail
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
md() { curl -fsS -H 'Metadata-Flavor: Google' "$MD/$1" 2>/dev/null || true; }
MODEL_HF="$(md gwaya-model-hf)";   SERVED="$(md gwaya-model-served)"; KEY="$(md gwaya-vllm-key)"
PORT="$(md gwaya-vllm-port)";      PORT="${PORT:-8000}"
IDLE_MIN="$(md gwaya-idle-min)";   IDLE_MIN="${IDLE_MIN:-10}"
MAX_UP="$(md gwaya-max-up-min)";   MAX_UP="${MAX_UP:-120}"
GCS="$(md gwaya-weights-gcs)";     PROJECT="$(md gwaya-project)"; NODE="$(md gwaya-node)"
[ -n "$MODEL_HF" ] && [ -n "$SERVED" ] && [ -n "$KEY" ] || { echo "missing gwaya-* metadata" >&2; exit 2; }
R=/opt/gwaya; mkdir -p "$R"; cd "$R"
export HOME=/root PATH=/root/.local/bin:$PATH HF_HOME=$R/hf SKIP_JAX_PRECOMPILE=1
log() { echo "[$(date -Is)] $*" | tee -a $R/startup.log; }

# unattended-upgrades holds the apt lock on fresh TPU images and is not needed here
systemctl stop unattended-upgrades 2>/dev/null || true

# 1. weights: GCS copy of the HF snapshot (fast) or HF once, then cached to GCS for the next boot
LOCAL="$R/models/$SERVED"; mkdir -p "$LOCAL"
if [ -n "$GCS" ] && gcloud storage ls "$GCS/$SERVED/config.json" >/dev/null 2>&1; then
  log "weights from $GCS/$SERVED"; gcloud storage cp -r "$GCS/$SERVED/*" "$LOCAL/" >/dev/null 2>&1
  FROM_GCS=1
else
  FROM_GCS=0
fi
# 2. python env (vllm-tpu pinned; same recipe as deploy/tpu/run_e_gen.sh)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x $R/venv/bin/python ] || { uv venv -q -p 3.12 $R/venv && VIRTUAL_ENV=$R/venv uv pip install -q "vllm-tpu==${VLLM_TPU_VERSION:-0.31.0}" huggingface_hub; }
PY=$R/venv/bin/python
if [ "$FROM_GCS" = 0 ]; then
  log "weights from HF $MODEL_HF"
  $PY -c "from huggingface_hub import snapshot_download as s; s('$MODEL_HF', local_dir='$LOCAL')" >>$R/startup.log 2>&1
  [ -n "$GCS" ] && gcloud storage cp -r "$LOCAL/*" "$GCS/$SERVED/" >/dev/null 2>&1 && log "cached to $GCS/$SERVED"
fi

# 3. server
EXTRA=""
$R/venv/bin/vllm serve --help 2>/dev/null | grep -q default-chat-template-kwargs && \
  EXTRA="--default-chat-template-kwargs {\"enable_thinking\":false}"
log "starting vllm serve $SERVED"
nohup $R/venv/bin/vllm serve "$LOCAL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" --api-key "$KEY" \
  --max-model-len 8192 --max-num-seqs 16 --max-num-batched-tokens 1024 \
  --limit-mm-per-prompt '{"image":0,"video":0}' $EXTRA > $R/vllm.log 2>&1 &
echo $! > $R/vllm.pid

# 4. watchdog: delete this node after IDLE_MIN without requests or MAX_UP minutes of uptime (cost safety net)
md gwaya-watchdog-py > $R/watchdog.py
nohup $PY $R/watchdog.py "$PORT" "$KEY" "$IDLE_MIN" "$MAX_UP" "$PROJECT" "$NODE" > $R/watchdog.log 2>&1 &
