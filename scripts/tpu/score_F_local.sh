#!/usr/bin/env bash
# A21 step 1: import the F candidates (python + math answers, math PoT programs) and score them locally in the bwrap sandbox,
# exactly as E' (scripts/tpu/score_local.sh): per tier one import + one `run_study.py --mode score` with arms base,gate_only.
# Lean rows are NOT handled here (scripts/lean/score_lean.py, --timeout-s 300). Never sets GWAYA_ALLOW_UNISOLATED.
#   PYTHON=<python with sympy> scripts/tpu/score_F_local.sh [OUT_ROOT]
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" GWAYA_DATA_ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}"
unset GWAYA_ALLOW_UNISOLATED
PY="${PYTHON:-python3}"
SRC="${F_STAGE:-$GWAYA_DATA_ROOT/eval_sets_F_tpu}"
OUT="${1:-$GWAYA_DATA_ROOT/eval_sets_F_scored}"
S=results/gwenlaya_v4/eval_sets_F
mkdir -p "$OUT"
TASKS="$OUT/tasks_F_pm.jsonl"
cat "$S/tasks_F_python.jsonl" "$S/tasks_F_math.jsonl" > "$TASKS"
one() {  # served rawdir chips
  local served="$1" raw="$2" chips="$3" d="$OUT/$1"
  mkdir -p "$d"
  # python + math rows only (the Lean rows are scored by score_lean.py)
  $PY -c "
import json, sys
open(sys.argv[2], 'w').writelines(l for l in open(sys.argv[1]) if json.loads(l)['domain'] in ('python', 'math'))
" "$raw/raw_$served.jsonl" "$d/raw_answer_pm.jsonl"
  $PY scripts/import_remote_gens.py --tasks "$TASKS" --raw "$d/raw_answer_pm.jsonl" --served "$served" --quant bf16 \
      --out "$d" --accelerator "TPU v5e x$chips" --chips "$chips" > "$d/import.log" 2>&1 || { echo "import failed $served"; return 1; }
  $PY scripts/import_remote_gens.py --tasks "$TASKS" --raw "$raw/raw_pot_$served.jsonl" --served "$served" --quant bf16 \
      --out "$d" --accelerator "TPU v5e x$chips" --chips "$chips" --kind pot > "$d/import_pot.log" 2>&1 || { echo "import pot failed $served"; return 1; }
  $PY scripts/run_study.py --plan experiments/night/plan_night.json --stage night_L2 --backend openai --backend-url http://127.0.0.1:1/v1 \
      --tasks "$TASKS" --models "$served" --quants bf16 --arms base gate_only --mode score --no-vram \
      --check-workers "${CHECK_WORKERS:-1}" --out "$d" > "$d/score.log" 2>&1
  echo "$served: exit $? ($(date -Is))"
}
one qwen3.5-2b-bf16 "$SRC/out_small" 1 &
one qwen3.5-4b-bf16 "$SRC/out_small" 1 &
one qwen3.5-9b-bf16 "$SRC/out_9b" 4 &
wait
