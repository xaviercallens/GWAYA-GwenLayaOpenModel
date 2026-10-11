#!/usr/bin/env bash
# A21 stage S4: score the fresh sets F (python + math, tiers 2B/4B/9B) with the FROZEN Laya v1 on one TPU v5e chip.
# Dry run by default (drive.sh prints plan, worst case, ledger). Real run, started by the USER:
#   GWAYA_CONFIRM_SPEND=1 setsid nohup scripts/tpu/launch_f_laya_predict.sh --yes-spend > $GWAYA_DATA_ROOT/f_laya_predict_launch.log 2>&1 &
# Cost: inference only on 2,724 calibrator rows + 908 router rows (ModernBERT-large, 512 tokens); E' (2,484 + 828 rows) took a
# few minutes after setup. Max 30 min on one chip = worst case $0.70 (drive.sh adds a $0.10 margin). Cap: GWAYA_CAP_USD.
# The frozen adapters/heads are hash-checked against docs/LAYA_FREEZE.txt here AND on the VM before scoring.
set -uo pipefail
cd "$(dirname "$0")/../.."
export GWAYA_DATA_ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}" GWAYA_CAP_USD="${GWAYA_CAP_USD:-47.76}"
S=results/gwenlaya_v4/eval_sets_F/laya_rows
L="${LAYA_V1_TGZ:-$GWAYA_DATA_ROOT/laya_data/laya_v1.tgz}"
B="${F_LAYA_STAGE:-$GWAYA_DATA_ROOT/eval_sets_F_laya}"
mkdir -p "$B"
for f in router_F.jsonl calibrator_F.jsonl; do [ -f "$S/$f" ] || { echo "missing $S/$f (scripts/data/build_F_laya_rows.py)" >&2; exit 2; }; done
[ -f "$L" ] || { echo "missing $L" >&2; exit 2; }
# 1. the row files are the ones in the committed manifest
( cd "$S" && sha256sum -c SHA256SUMS_F_rows.txt ) || { echo "ABORT: row files differ from $S/SHA256SUMS_F_rows.txt" >&2; exit 3; }
# 2. the frozen adapters and heads are the ones in docs/LAYA_FREEZE.txt
CHK="$(mktemp -d)"; trap 'rm -rf "$CHK"' EXIT
tar xzf "$L" -C "$CHK"
for f in router/adapter/adapter_model.safetensors router/heads.safetensors router/head_config.json \
         calibrator/adapter/adapter_model.safetensors calibrator/heads.safetensors calibrator/head_config.json; do
  want="$(awk -v f="$f" '$2 == f {print $1}' docs/LAYA_FREEZE.txt)"
  got="$(sha256sum "$CHK/$f" | cut -d' ' -f1)"
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "ABORT: $f sha256 $got != frozen $want" >&2; exit 3; }
done
echo "frozen Laya v1 verified against docs/LAYA_FREEZE.txt (6 files)"
cp "$S/router_F.jsonl" "$S/calibrator_F.jsonl" "$B/"
cp "$L" "$B/laya_v1.tgz"
cp docs/LAYA_FREEZE.txt "$B/LAYA_FREEZE.txt"
git archive --format=tar.gz -o "$B/repo.tgz" HEAD   # the COMMITTED tree
deploy/tpu/drive.sh --name f-laya --script deploy/tpu/run_laya_predict_F.sh --accel v5litepod-1 \
  --upload "$B/repo.tgz $B/laya_v1.tgz $B/LAYA_FREEZE.txt $B/router_F.jsonl $B/calibrator_F.jsonl" \
  --download "out.tgz status.jsonl" --max-minutes 30 --zones "us-west4-a us-central1-a us-east5-b" \
  --out "$B/out" "$@"
