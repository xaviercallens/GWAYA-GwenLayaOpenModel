#!/usr/bin/env bash
# A30 stage T4: 8 sampled candidates per F python task on Qwen3.5-9B (v5litepod-4, TP=4). Dry run by default.
#   real run (the USER, own terminal):
#   GWAYA_CONFIRM_SPEND=1 setsid nohup scripts/tpu/launch_f_sample_9b.sh --yes-spend > /mnt/data/home/xavkal/gwaya-data/f_sample_9b_launch.log 2>&1 &
# Worst case 70 min x 4 chips x $1.20/h = $5.6 (+ drive.sh margin). Cap: GWAYA_CAP_USD. VM is deleted by drive.sh on exit.
set -uo pipefail
cd "$(dirname "$0")/../.."
export GWAYA_DATA_ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}" GWAYA_CAP_USD="${GWAYA_CAP_USD:-47.76}"
B="${F_SAMPLE9_STAGE:-$GWAYA_DATA_ROOT/eval_sets_F_sample_9b}"
S=results/gwenlaya_v4/eval_sets_F
mkdir -p "$B"
[ -f "$S/tasks_F_python.jsonl" ] || { echo "missing $S/tasks_F_python.jsonl" >&2; exit 2; }
cp "$S/tasks_F_python.jsonl" "$B/tasks_prompts.jsonl"
git archive --format=tar.gz -o "$B/repo.tgz" HEAD
deploy/tpu/drive.sh --name f-sample-9b --script deploy/tpu/run_e_gen.sh --accel v5litepod-4 \
  --upload "$B/repo.tgz $B/tasks_prompts.jsonl" --download "raw_*.jsonl status.jsonl *.log" \
  --max-minutes 70 --zones "us-south1-a us-central1-a us-east5-b" \
  --remote-env "MODELS='Qwen/Qwen3.5-9B|qwen3.5-9b-bf16' TP=4 KINDS='sample' SAMPLES=8 MAX_SEQS_LADDER='32 16 8'" \
  --out "$B/out" "$@"
