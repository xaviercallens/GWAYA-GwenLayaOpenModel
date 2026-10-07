#!/usr/bin/env bash
# Runs inside the stage container (via container_entry.sh when GWAYA_STAGE is set).
# /data holds: datasets/ (from the lake), out/ (resume state = lake runs/<stage>), adapters/, quantized/.
# The commands come from experiments/plan.json through deploy/stage_launch.py, so `run_on_gcp.sh --stage X`
# in dry-run prints exactly what runs here. Each command is fail-fast; the exit code is the stage result.
set -uo pipefail
STAGE="${GWAYA_STAGE:?GWAYA_STAGE not set}"
export GWAYA_GPU="${GWAYA_GPU:-l4}" GWAYA_DATA_DIR="${GWAYA_DATA_DIR:-/data}" OLLAMA_MODELS=/data/ollama/models
cd /app
mkdir -p /data/out /data/adapters /data/quantized "/data/results/$STAGE" /data/ollama/models

if [ "${GWAYA_REQUIRE_GPU:-1}" = "1" ] && ! nvidia-smi -L > /dev/null 2>&1; then
  echo "FATAL: GWAYA_REQUIRE_GPU=1 but no GPU is visible" >&2; exit 3
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "/data/results/$STAGE/gpu.csv" 2>&1 || true

MODELS="$(python deploy/stage_launch.py ollama-models "$STAGE")" || exit 2
USES_STUDY=0; case "$STAGE" in S1|S2|S5|S6) USES_STUDY=1;; esac
if [ "$USES_STUDY" = 1 ]; then
  ollama serve > /data/out/ollama_serve.log 2>&1 &
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:11434/api/tags > /dev/null && break; sleep 1; done
  # tuned GGUFs produced by the quantize step are registered under their directory name
  for mf in /data/quantized/*/Modelfile; do
    [ -e "$mf" ] && ollama create "$(basename "$(dirname "$mf")")" -f "$mf"
  done
  for m in $MODELS; do ollama pull "$m" || { echo "FATAL: cannot pull $m" >&2; exit 4; }; done
  ollama list > "/data/results/$STAGE/ollama_list.txt"
fi

python deploy/stage_launch.py entry "$STAGE" > /data/out/.entry_commands || exit 2
rc=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  echo "+ $cmd"
  eval "$cmd" || { rc=$?; echo "FATAL: stage command failed rc=$rc: $cmd" >&2; break; }
done < /data/out/.entry_commands
exit "$rc"
