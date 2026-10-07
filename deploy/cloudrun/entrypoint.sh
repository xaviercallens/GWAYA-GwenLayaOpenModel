#!/usr/bin/env bash
# Start llama-server (loopback) on the GGUF found under $MODEL_DIR, then the FastAPI front on $PORT.
# The front answers /healthz 503 until llama-server is healthy, so Cloud Run's startup probe gates traffic.
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-/models}"
LLAMA_PORT="${LLAMA_PORT:-8081}"
PORT="${PORT:-8080}"

if [ -n "${MODEL_FILE:-}" ]; then
  MODEL="$MODEL_DIR/$MODEL_FILE"
else
  # first *.gguf by name; deploy.sh sets MODEL_FILE so this is only a convenience
  MODEL="$(find "$MODEL_DIR" -maxdepth 2 -name '*.gguf' 2>/dev/null | sort | head -n1 || true)"
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "FATAL: no GGUF model at ${MODEL:-$MODEL_DIR/*.gguf}. Mount the quantized/ prefix at $MODEL_DIR or set MODEL_FILE." >&2
  exit 2
fi
echo "model: $MODEL"

LLAMA_BIN="$(command -v llama-server || true)"
[ -z "$LLAMA_BIN" ] && LLAMA_BIN=/app/llama-server
# context is shared across slots by llama.cpp: per-slot ctx = LLAMA_CTX / LLAMA_PARALLEL
"$LLAMA_BIN" -m "$MODEL" --host 127.0.0.1 --port "$LLAMA_PORT" \
  --parallel "${LLAMA_PARALLEL:-8}" -c "${LLAMA_CTX:-8192}" -ngl "${LLAMA_NGL:-99}" \
  --alias "${MODEL_ALIAS:-gwenlaya-gen}" ${LLAMA_EXTRA_ARGS:-} &
LLAMA_PID=$!
trap 'kill "$LLAMA_PID" 2>/dev/null || true' TERM INT EXIT

exec python3 -m uvicorn front:app --app-dir /app --host 0.0.0.0 --port "$PORT" --no-access-log
