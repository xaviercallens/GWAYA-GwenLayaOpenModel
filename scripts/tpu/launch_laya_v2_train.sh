#!/usr/bin/env bash
# A28 stage T1: Laya v2 training on one TPU v5e chip with feedback-retrofitted data + generator-LoRA probe.
# Dry run by default (drive.sh prints plan, worst case, ledger). Real run, started by the USER:
#   GWAYA_CONFIRM_SPEND=1 setsid nohup scripts/tpu/launch_laya_v2_train.sh --yes-spend > $GWAYA_DATA_ROOT/laya_v2_launch.log 2>&1 &
# Cost: max 90 min on one chip, about $1.80 at list price (drive.sh adds its own margin). Cap: GWAYA_CAP_USD.
set -uo pipefail
cd "$(dirname "$0")/../.."
export GWAYA_DATA_ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}" GWAYA_CAP_USD="${GWAYA_CAP_USD:-47.76}"
B="${LAYA_V2_STAGE:-$GWAYA_DATA_ROOT/laya_v2}"
S=results/gwenlaya_v4/eval_sets_F/laya_rows
T="$GWAYA_DATA_ROOT/laya_data/laya_train"
mkdir -p "$B"
for f in "$T/calibrator.jsonl" "$T/router.jsonl" "$S/calibrator_F.jsonl" "$S/router_F.jsonl"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }
done
( cd "$S" && sha256sum -c SHA256SUMS_F_rows.txt ) || { echo "ABORT: F rows differ from manifest" >&2; exit 3; }
python3 scripts/phase2/build_feedback_set.py --base-calibrator "$T/calibrator.jsonl" --base-router "$T/router.jsonl" \
  --f-calibrator "$S/calibrator_F.jsonl" --f-router "$S/router_F.jsonl" --out-dir "$B" \
  ${HUMAN_FEEDBACK:+--human-feedback "$HUMAN_FEEDBACK"} || exit 3
git archive --format=tar.gz -o "$B/repo.tgz" HEAD   # the COMMITTED tree
deploy/tpu/drive.sh --name laya-v2 --script deploy/tpu/run_laya_v2_train.sh --accel v5litepod-1 \
  --upload "$B/repo.tgz $B/calibrator_v2_train.jsonl $B/router_v2_train.jsonl $B/calibrator_F_heldout.jsonl $B/router_F_heldout.jsonl" \
  --download "out.tgz status.jsonl" --max-minutes 90 --zones "us-west4-a us-central1-a us-east5-b" \
  --out "$B/out" "$@"
