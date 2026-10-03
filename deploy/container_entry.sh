#!/usr/bin/env bash
# Runs inside the container. /data is a host mount that already holds:
#   /data/model/ollama_models.tar      (restored from the data lake)
#   /data/datasets/mbpp-sanitized/     (restored from the data lake)
#   /data/out/                         (resume state; synced back to the lake by the host)
# Env: GWAYA_N (problems), GWAYA_REQUIRE_GPU (1 = abort if Ollama is not on the GPU).
set -euo pipefail
N="${GWAYA_N:-60}"
REQUIRE_GPU="${GWAYA_REQUIRE_GPU:-1}"
mkdir -p /data/ollama /data/out

if [ ! -d /data/ollama/models/blobs ]; then
  echo "restoring Ollama models from /data/model/ollama_models.tar"
  tar xf /data/model/ollama_models.tar -C /data/ollama
fi

ollama serve > /data/out/ollama_serve.log 2>&1 &
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:11434/api/tags > /dev/null && break
  sleep 1
done
curl -sf http://127.0.0.1:11434/api/tags | python -c "import sys,json; print('models:', [m['name'] for m in json.load(sys.stdin)['models']])"

export GWAYA_OLLAMA_VERSION="$(ollama --version 2>&1 | tail -1)"

# Warm the model and verify where it actually runs (this is what validates any speed-up claim).
curl -s http://127.0.0.1:11434/api/generate -d '{"model":"qwen2.5-coder:1.5b","prompt":"hi","stream":false,"options":{"num_predict":4}}' > /dev/null
ollama ps | tee /data/out/ollama_ps.txt
if [ "$REQUIRE_GPU" = "1" ] && ! grep -q "GPU" /data/out/ollama_ps.txt; then
  echo "FATAL: GWAYA_REQUIRE_GPU=1 but Ollama is not running the model on the GPU" >&2
  exit 3
fi

# Hardware tag derived from what Ollama ACTUALLY used (nvidia-smi is absent in the raw-device fallback).
if grep -q "GPU" /data/out/ollama_ps.txt; then
  GPU_NAME="$(grep -o 'name="[^"]*"' /data/out/ollama_serve.log | grep -iv 'cpu' | tail -1 | sed 's/name=//; s/"//g')"
  [ -n "$GPU_NAME" ] || GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  export GWAYA_HARDWARE="${GPU_NAME:-GPU(unnamed)} (ollama ps: $(grep -o '[0-9]*% GPU' /data/out/ollama_ps.txt | head -1)) / $(nproc) vCPU"
else
  export GWAYA_HARDWARE="cpu-only / $(nproc) vCPU"
fi
echo "hardware: $GWAYA_HARDWARE | $GWAYA_OLLAMA_VERSION"

cd /app
exec python scripts/run_low_tier_benchmark.py --n "$N" --out-dir /data/out --mbpp-dir /data/datasets/mbpp-sanitized
