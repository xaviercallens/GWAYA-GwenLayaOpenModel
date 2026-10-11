#!/usr/bin/env bash
# S1 of the 2026-10-10 next run: time-boxed, instrumented load probe of Qwen3.5-27B on one v5litepod-4 (bf16, TP=4).
# Dry run by default (drive.sh prints plan, worst case, ledger). Real run, started by the USER:
#   GWAYA_CONFIRM_SPEND=1 setsid nohup scripts/tpu_probe_27b.sh --yes-spend > $GWAYA_DATA_ROOT/probe27b_launch.log 2>&1 &
# Worst case 40 min x 4 chips x $1.20/h = $3.2 (+0.1 margin). Abort rule written in advance: if the engine is not
# ready within PROBE_MIN (22) minutes the run is stopped; the monitor and py-spy files say where it stalled.
set -uo pipefail
cd "$(dirname "$0")/.."
export GWAYA_DATA_ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}" GWAYA_CAP_USD="${GWAYA_CAP_USD:-47.76}"
B="${PROBE_STAGE:-$GWAYA_DATA_ROOT/probe27b}"; mkdir -p "$B"
cp "$GWAYA_DATA_ROOT/night/results/tasks_E_primary.d25.jsonl" "$B/tasks_prompts.jsonl"
git archive --format=tar.gz -o "$B/repo.tgz" HEAD
deploy/tpu/drive.sh --name probe27b --accel v5litepod-4 --script deploy/tpu/run_27b_probe.sh \
  --upload "$B/repo.tgz $B/tasks_prompts.jsonl" --download "raw_*.jsonl status.jsonl *.log pyspy_*.txt" \
  --max-minutes 40 --zones "us-south1-a us-west4-a us-central1-a" \
  --remote-env "PROBE_MIN=22" --out "$B/out" "$@"
