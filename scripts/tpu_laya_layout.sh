#!/usr/bin/env bash
# S2 of the 2026-10-10 next run: A22 Laya candidate-first layout ablation (train the calibrator once, predict C, E', R').
# Dry run by default. Real run, started by the USER:
#   GWAYA_CONFIRM_SPEND=1 setsid nohup scripts/tpu_laya_layout.sh --yes-spend > $GWAYA_DATA_ROOT/laya_layout_launch.log 2>&1 &
# Expected ~25-30 min and ~$0.55 (the A16 text-only run took 1,548 s, $0.516); worst case 60 min = $1.30.
set -uo pipefail
cd "$(dirname "$0")/.."
export GWAYA_DATA_ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}" GWAYA_CAP_USD="${GWAYA_CAP_USD:-47.76}"
L="$GWAYA_DATA_ROOT/laya_data"; B="${LAYOUT_STAGE:-$GWAYA_DATA_ROOT/laya_layout}"; mkdir -p "$B"
# the four row files share the basename calibrator.jsonl: stage them under the names run_laya_layout.sh expects
cp "$L/laya_train/calibrator.jsonl"  "$B/calibrator.jsonl"
cp "$L/laya_calib/calibrator.jsonl"  "$B/calibrator_calib.jsonl"
cp "$L/laya_eprime/calibrator.jsonl" "$B/calibrator_eprime.jsonl"
cp "$L/laya_rfresh/calibrator.jsonl" "$B/calibrator_rfresh.jsonl"
git archive --format=tar.gz -o "$B/repo.tgz" HEAD   # the COMMITTED tree (a prior audit flagged shipping the working tree)
deploy/tpu/drive.sh --name laya-layout --script deploy/tpu/run_laya_layout.sh --accel v5litepod-1 \
  --upload "$B/repo.tgz $B/calibrator.jsonl $B/calibrator_calib.jsonl $B/calibrator_eprime.jsonl $B/calibrator_rfresh.jsonl" \
  --download "out.tgz status.jsonl" --max-minutes 60 --zones "us-west4-a us-central1-a us-east5-b" \
  --out "$B/out" "$@"
