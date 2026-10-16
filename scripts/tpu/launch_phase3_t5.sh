#!/usr/bin/env bash
# A33 stage T5: Qwen3.5-35B-A3B (answer + 8 samples + synthesized tests) then Qwen3.5-27B (answer + 8 samples) on
# F python + G, bf16, TP=8, v5litepod-8. Spot with a US zone ladder by default; PROVISIONING=on-demand to override.
# Dry run by default. Real run (the operator):
#   GWAYA_CONFIRM_SPEND=1 setsid nohup scripts/tpu/launch_phase3_t5.sh --yes-spend > $GWAYA_DATA_ROOT/phase3_t5.log 2>&1 &
# Worst case: 150 min x 8 chips x 0.4942 = 9.9 USD spot (24 USD on-demand: refused by the 47 USD cap unless the ledger allows).
set -uo pipefail
cd "$(dirname "$0")/../.."
export GWAYA_DATA_ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}" GWAYA_CAP_USD="${GWAYA_CAP_USD:-47}"
B="${T5_STAGE:-$GWAYA_DATA_ROOT/phase3_t5}"; mkdir -p "$B"
F=results/gwenlaya_v4/eval_sets_F/tasks_F_python.jsonl; G=results/gwenlaya_v4/eval_set_G/tasks_G_python.jsonl
for f in $F $G; do [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }; done
cat $F $G > "$B/tasks_prompts.jsonl"
git archive --format=tar.gz -o "$B/repo.tgz" HEAD
PROVFLAG="--spot"; [ "${PROVISIONING:-spot}" = on-demand ] && PROVFLAG=""
RUNNAME="p3-t5-$(date +%m%d%H%M)"
UPL="$B/repo.tgz $B/tasks_prompts.jsonl"; for r in "$B"/out/raw_*.jsonl; do [ -f "$r" ] && UPL="$UPL $r"; done   # resume after preemption
deploy/tpu/drive.sh --name p3-t5 --script deploy/tpu/run_e_gen.sh --accel ${T5_ACCEL:-v5litepod-4} $PROVFLAG \
  --upload "$UPL" --download "raw_*.jsonl status.jsonl *.log" --max-minutes "${T5_MAX_MIN:-150}" \
  --zones "${T5_ZONES:-us-south1-a us-south1-b us-east5-a us-east5-b us-east5-c us-east1-b us-east1-c us-west4-a us-west4-b us-west1-c us-east4-b us-east7-b}" \
  --remote-env "MODELS='${T5_MODELS:-Qwen/Qwen3.5-35B-A3B|qwen3.5-35b-a3b-bf16|answer,sample,tests Qwen/Qwen3.5-27B|qwen3.5-27b-bf16|answer,sample}' TP=${T5_TP:-4} SAMPLES=${T5_SAMPLES:-4} TEST_SAMPLES=2 MAX_SEQS_LADDER='${T5_LADDER:-16 8}' STEP_TIMEOUT=4200 CLEAN_HF=1 SYNC_URI=gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/tpu_runs/partial/$RUNNAME" \
  --out "$B/out" "$@"
