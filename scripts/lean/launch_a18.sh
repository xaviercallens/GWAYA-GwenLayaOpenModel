#!/usr/bin/env bash
# A18 Lean miniF2F generation on TPU: small tiers (0.8B/2B/4B, one v5e chip) and 9B (v5litepod-4, TP=4) in parallel,
# each greedy + 8 draws (T 0.8, top_p 0.95, seed 0). Dry run by default (drive.sh prints plan, worst case, ledger).
#   real run:  GWAYA_CONFIRM_SPEND=1 scripts/lean/launch_a18.sh --yes-spend
# Worst case: small 120 min x 1 chip + 9B 100 min x 4 chips at $1.20/chip-h = ~$10.2. Cap: GWAYA_CAP_USD (default 47.76).
set -uo pipefail
cd "$(dirname "$0")/../.."
export GWAYA_DATA_ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}" GWAYA_CAP_USD="${GWAYA_CAP_USD:-47.76}"
B="${A18_STAGE:-$GWAYA_DATA_ROOT/lean_a18_tpu}"
mkdir -p "$B"
[ -f results/gwenlaya_v4/lean_a18/tasks_lean.jsonl ] || { echo "run scripts/lean/build_minif2f_tasks.py first" >&2; exit 2; }
cp results/gwenlaya_v4/lean_a18/tasks_lean.jsonl "$B/tasks_prompts.jsonl"
git archive --format=tar.gz -o "$B/repo.tgz" HEAD
COMMON=(--script deploy/tpu/run_e_gen.sh --upload "$B/repo.tgz $B/tasks_prompts.jsonl" --download "raw_*.jsonl status.jsonl *.log")
deploy/tpu/drive.sh --name lean-small "${COMMON[@]}" --max-minutes 120 --zones "us-west4-a us-central1-a us-east5-b" \
  --remote-env "MODELS='Qwen/Qwen3.5-0.8B|qwen3.5-0.8b-bf16 Qwen/Qwen3.5-2B|qwen3.5-2b-bf16 Qwen/Qwen3.5-4B|qwen3.5-4b-bf16' KINDS='answer sample' MAX_SEQS_LADDER='32 16 8'" \
  --out "$B/out_small" "$@" > "$B/drive_small.log" 2>&1 &
deploy/tpu/drive.sh --name lean-9b --accel v5litepod-4 "${COMMON[@]}" --max-minutes 100 --zones "us-south1-a us-central1-a us-east5-b" \
  --remote-env "MODELS='Qwen/Qwen3.5-9B|qwen3.5-9b-bf16' TP=4 KINDS='answer sample' MAX_SEQS_LADDER='32 16 8'" \
  --out "$B/out_9b" "$@" > "$B/drive_9b.log" 2>&1 &
wait
tail -n 15 "$B/drive_small.log" "$B/drive_9b.log"
