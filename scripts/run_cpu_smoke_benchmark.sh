#!/usr/bin/env bash
# CPU-only smoke test benchmark: quick (n<=10) validation that scripts/run_benchmark.py
# works locally with Ollama running on the same box.
#
# Usage:
#   ./scripts/run_cpu_smoke_benchmark.sh [--model <model>] [--n <n>]
#
# Defaults:
#   --model qwen2.5-coder:0.5b
#   --n 10
#
# Exits with:
#   0 = success
#   1 = argument error, missing Ollama, or benchmark failure
#   2 = model not available in Ollama /api/tags
#   3 = Ollama service not running on localhost:11434

set -euo pipefail

# Defaults
MODEL="qwen2.5-coder:0.5b"
N=10

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --n)
      N="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--model <model>] [--n <n>]" >&2
      exit 1
      ;;
  esac
done

# Validate --n is <= 10
if [[ $N -gt 10 ]]; then
  echo "ERROR: --n must be <= 10 for smoke test (got $N)" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/results/local_cpu_smoke"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "GWAYA CPU Smoke Benchmark"
echo "=========================================="
echo "Model:  $MODEL"
echo "Tasks:  $N"
echo "Output: $OUT_DIR"
echo ""

# 1. Check that Ollama is running
echo "[1/4] Checking Ollama service at localhost:11434..."
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "ERROR: Ollama not running at http://127.0.0.1:11434" >&2
  exit 3
fi
echo "✓ Ollama is responding"
echo ""

# 2. Check that the model is available
echo "[2/4] Checking if model '$MODEL' is available..."
MODELS_JSON=$(curl -s http://127.0.0.1:11434/api/tags)
if ! echo "$MODELS_JSON" | grep -q "\"name\":\"$MODEL\""; then
  echo "ERROR: Model '$MODEL' not found in Ollama /api/tags" >&2
  echo "Available models:"
  echo "$MODELS_JSON" | jq -r '.models[]?.name // empty' | sed 's/^/  /' || true
  exit 2
fi
echo "✓ Model '$MODEL' is available"
echo ""

# 3. Run the benchmark
echo "[3/4] Running benchmark (n=$N)..."
cd "$REPO_ROOT"

# Use the venv from the scratchpad if available, else system python
PYTHON_BIN="${VENV_PYTHON:-python3}"
if [ -n "${VENV_PYTHON:-}" ] && [ -x "$VENV_PYTHON" ]; then
  PYTHON_BIN="$VENV_PYTHON"
fi

if ! $PYTHON_BIN scripts/run_benchmark.py \
  --model "$MODEL" \
  --n "$N" \
  --out-dir "$OUT_DIR" \
  2>&1; then
  echo "ERROR: Benchmark failed" >&2
  exit 1
fi
echo ""

# 4. Extract and display results
echo "[4/4] Extracting results..."
RESULTS_FILE="$OUT_DIR/results.json"
if [ ! -f "$RESULTS_FILE" ]; then
  echo "ERROR: Results file not found: $RESULTS_FILE" >&2
  exit 1
fi

# Extract and print the pass@1 table
echo ""
echo "========== PASS@1 TABLE =========="
python3 - <<'EOF' "$RESULTS_FILE"
import json
import sys

results = json.loads(open(sys.argv[1]).read())
arms = results.get("arms", {})
for arm in ["A0", "A2", "A3", "A4"]:
  if arm in arms:
    pass_at_1 = arms[arm].get("pass_at_1")
    coverage = arms[arm].get("coverage_pct")
    precision = arms[arm].get("precision_pct")
    print(f"{arm:3} | pass@1={pass_at_1:6.2f}% | coverage={coverage:6.2f}% | precision={precision!s:>7}")

# Gate
gates = results.get("gates", {})
if "G3" in gates:
  g3 = gates["G3"]
  print(f"\nGate G3 (A3): {g3.get('verdict', 'UNKNOWN')}")
  print(f"  Precision >= 92.0%: {g3.get('precision_pct')}")
  print(f"  Coverage >= 75.0%:  {g3.get('coverage_pct')}")
EOF

echo ""
echo "✓ Smoke test completed successfully"
echo "  Full results: $OUT_DIR/results.json"
echo "  Raw rows:    $OUT_DIR/rows.jsonl"
