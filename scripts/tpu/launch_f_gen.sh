#!/usr/bin/env bash
# A21 stage S3: candidates (greedy answer + math PoT program) for the fresh sets F on 2B/4B (one v5e chip) and 9B (v5litepod-4, TP=4).
# Dry run by default (drive.sh prints plan, worst case, ledger).
#   real run:  GWAYA_CONFIRM_SPEND=1 scripts/tpu/launch_f_gen.sh --yes-spend
#   F_ONLY=small|9b runs one slice.  F_MODELS_SMALL overrides the small-tier model list.
# Worst case: small 90 min x 1 chip + 9B 70 min x 4 chips at $1.20/chip-h = ~$7.4. Cap: GWAYA_CAP_USD (default 47.76).
set -uo pipefail
cd "$(dirname "$0")/../.."
export GWAYA_DATA_ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}" GWAYA_CAP_USD="${GWAYA_CAP_USD:-47.76}"
B="${F_STAGE:-$GWAYA_DATA_ROOT/eval_sets_F_tpu}"
S=results/gwenlaya_v4/eval_sets_F
mkdir -p "$B"
for d in python math lean; do [ -f "$S/tasks_F_$d.jsonl" ] || { echo "missing $S/tasks_F_$d.jsonl" >&2; exit 2; }; done
cat "$S"/tasks_F_python.jsonl "$S"/tasks_F_math.jsonl "$S"/tasks_F_lean.jsonl > "$B/tasks_prompts.jsonl"
git archive --format=tar.gz -o "$B/repo.tgz" HEAD
COMMON=(--script deploy/tpu/run_e_gen.sh --upload "$B/repo.tgz $B/tasks_prompts.jsonl" --download "raw_*.jsonl status.jsonl *.log")
SMALL="${F_MODELS_SMALL:-Qwen/Qwen3.5-2B|qwen3.5-2b-bf16 Qwen/Qwen3.5-4B|qwen3.5-4b-bf16}"
[ "${F_ONLY:-small}" = 9b ] || deploy/tpu/drive.sh --name f-small "${COMMON[@]}" --max-minutes 90 --zones "us-west4-a us-central1-a us-east5-b" \
  --remote-env "MODELS='$SMALL' KINDS='answer pot' MAX_SEQS_LADDER='32 16 8'" \
  --out "$B/out_small" "$@" > "$B/drive_small.log" 2>&1 &
[ "${F_ONLY:-9b}" = small ] || deploy/tpu/drive.sh --name f-9b --accel v5litepod-4 "${COMMON[@]}" --max-minutes 70 --zones "us-south1-a us-central1-a us-east5-b" \
  --remote-env "MODELS='Qwen/Qwen3.5-9B|qwen3.5-9b-bf16' TP=4 KINDS='answer pot' MAX_SEQS_LADDER='32 16 8'" \
  --out "$B/out_9b" "$@" > "$B/drive_9b.log" 2>&1 &
wait
tail -n 15 "$B/drive_small.log" "$B/drive_9b.log"
