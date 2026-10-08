#!/usr/bin/env bash
# Import TPU-generated candidates into the study cache and score them locally (bwrap sandbox),
# one process per model so the 8 cores are used. Usage:
#   scripts/tpu/score_local.sh <raw_dir> <tasks_full.jsonl> <out_root> [suffix=bf16]
# Per model it writes <out_root>/<served>/{gens,rows}.jsonl. The larger tier also runs the gate-only
# arm (B5). Needs: python with the repo deps, bwrap. Never sets GWAYA_ALLOW_UNISOLATED.
set -uo pipefail
RAW="$1"; TASKS="$2"; OUT="$3"; Q="${4:-bf16}"
PY="${PYTHON:-python3}"
PLAN="${PLAN:-experiments/night/plan_night.json}"
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" GWAYA_DATA_ROOT="${GWAYA_DATA_ROOT:-$HOME/gwaya-data}"
unset GWAYA_ALLOW_UNISOLATED
mkdir -p "$OUT"
score_one() {  # served arms...
  local served="$1"; shift
  local d="$OUT/$served"
  mkdir -p "$d"
  $PY scripts/import_remote_gens.py --tasks "$TASKS" --raw "$RAW/raw_$served.jsonl" --served "$served" \
      --quant "$Q" --out "$d" --accelerator "TPU v5e x1" > "$d/import.log" 2>&1 || { echo "import failed: $served"; return 1; }
  $PY scripts/run_study.py --plan "$PLAN" --stage night_L2 --backend openai --backend-url http://127.0.0.1:1/v1 \
      --tasks "$TASKS" --models "$served" --quants "$Q" --arms "$@" --mode score --no-vram --check-workers "${CHECK_WORKERS:-1}" --out "$d" > "$d/score.log" 2>&1
  echo "$served: exit $? ($(date -Is))"
}
for spec in ${SCORE_SPECS:-"qwen2.5-coder-1.5b-bf16:base" "qwen3.5-2b-bf16:base" "qwen3.5-4b-bf16:base,gate_only"}; do
  served="${spec%%:*}"; arms="${spec##*:}"
  [ -f "$RAW/raw_$served.jsonl" ] || { echo "skip $served (no raw file)"; continue; }
  score_one "$served" ${arms//,/ } &
done
wait
