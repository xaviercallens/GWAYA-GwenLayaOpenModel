#!/usr/bin/env bash
# ==============================================================================
# GWAYA: Consumer RTX GPU & Ollama Deployment Helper
# ==============================================================================
# Automatically detects local NVIDIA GPU (e.g. RTX 3060/3080/4070/4080/4090/5090),
# checks VRAM, pulls the optimal Qwen2.5-Coder model, and verifies 100% GPU offload.
#
# Usage:
#   ./scripts/deploy_rtx_ollama.sh             # Auto-detects and suggests model
#   ./scripts/deploy_rtx_ollama.sh 1.5b        # Pulls qwen2.5-coder:1.5b (research baseline)
#   ./scripts/deploy_rtx_ollama.sh 7b          # Pulls qwen2.5-coder:7b
#   ./scripts/deploy_rtx_ollama.sh 32b         # Pulls qwen2.5-coder:32b (for 24GB+ GPUs)
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo "  GWAYA: Deploying Qwen on Local NVIDIA RTX GPU via Ollama"
echo "======================================================================"

# 1. Check NVIDIA Driver & GPU
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[-] nvidia-smi not found. Running in CPU mode."
  GPU_NAME="CPU"
  VRAM_MB=0
else
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  echo "[+] Detected GPU: $GPU_NAME (${VRAM_MB} MB VRAM, Driver $DRIVER)"
fi

# 2. Check Ollama installation
if ! command -v ollama >/dev/null 2>&1; then
  echo "[-] Ollama is not installed."
  echo "    Install with: curl -fsSL https://ollama.ai/install.sh | sh"
  exit 1
fi
echo "[+] Ollama binary found: $(ollama --version 2>&1 | tail -1)"

# 3. Ensure Ollama server is running
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "[*] Starting local Ollama server in background..."
  ollama serve > /tmp/ollama_rtx.log 2>&1 &
  sleep 3
fi

# 4. Model selection
REQUESTED="${1:-auto}"
if [ "$REQUESTED" = "auto" ]; then
  if [ "$VRAM_MB" -ge 22000 ]; then
    MODEL_TAG="32b"
  elif [ "$VRAM_MB" -ge 11000 ]; then
    MODEL_TAG="14b"
  elif [ "$VRAM_MB" -ge 6000 ]; then
    MODEL_TAG="7b"
  elif [ "$VRAM_MB" -ge 3000 ]; then
    MODEL_TAG="3b"
  else
    MODEL_TAG="1.5b"
  fi
  echo "[*] Auto-selected model tag based on VRAM: $MODEL_TAG"
else
  MODEL_TAG="$REQUESTED"
fi

MODEL="qwen2.5-coder:${MODEL_TAG}"
echo "[*] Pulling model: $MODEL (this may take a few minutes on first run)..."
ollama pull "$MODEL"

# 5. Warm model and verify GPU offload
echo "[*] Warming model and verifying GPU layer offloading..."
curl -s http://127.0.0.1:11434/api/generate -d "{\"model\":\"$MODEL\",\"prompt\":\"def fibonacci(n):\",\"stream\":false,\"options\":{\"num_predict\":8}}" > /dev/null

echo "----------------------------------------------------------------------"
echo "Ollama Active Process Table:"
ollama ps
echo "----------------------------------------------------------------------"

if ollama ps | grep -q "100% GPU"; then
  echo "[✓] SUCCESS: Model $MODEL is offloaded 100% to GPU ($GPU_NAME)!"
elif ollama ps | grep -q "GPU"; then
  echo "[!] PARTIAL GPU: Model is partially offloaded to GPU."
else
  echo "[!] NOTE: Model is executing on CPU."
fi

echo ""
echo "You can now run GWAYA with:"
echo "  1. Benchmark: python scripts/run_benchmark.py --model $MODEL"
echo "  2. MCP Server: fastmcp run mcp_server.py"
echo "======================================================================"
