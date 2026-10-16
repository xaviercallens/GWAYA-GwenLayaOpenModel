#!/usr/bin/env bash
# A33 stage T6: 2B/4B (one v5e chip) and 9B (v5litepod-4, TP=4): greedy answers on G + the pool-O 6,000-task subset
# (Laya track), 8 samples on G only. Dry run by default. Real run (the operator):
#   GWAYA_CONFIRM_SPEND=1 setsid nohup scripts/tpu/launch_phase3_t6.sh --yes-spend > $GWAYA_DATA_ROOT/phase3_t6.log 2>&1 &
# T6_ONLY=small|9b runs one half. Worst case spot: small 150 min x 1 x 0.4942 = 1.3 USD; 9b 90 min x 4 x 0.4942 = 3.0 USD.
set -uo pipefail
cd "$(dirname "$0")/../.."
export GWAYA_DATA_ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}" GWAYA_CAP_USD="${GWAYA_CAP_USD:-47}"
B="${T6_STAGE:-$GWAYA_DATA_ROOT/phase3_t6${T6_O_ONLY:+_O}}"; mkdir -p "$B"
G=results/gwenlaya_v4/eval_set_G/tasks_G_python.jsonl; O="$GWAYA_DATA_ROOT/pool_O/tasks_O_subset6000.jsonl"
if [ -n "${T6_O_ONLY:-}" ]; then   # Laya track only: greedy on the pool-O subset (G runs in a later session)
  [ -f "$O" ] || { echo "missing $O" >&2; exit 2; }
  cp $O "$B/tasks_prompts.jsonl"; : > "$B/tasks_sample.jsonl"; KS=answer
else
  for f in $G $O; do [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }; done
  cat $G $O > "$B/tasks_prompts.jsonl"; cp $G "$B/tasks_sample.jsonl"; KS=answer,sample
fi
git archive --format=tar.gz -o "$B/repo.tgz" HEAD
PROVFLAG="--spot"; [ "${PROVISIONING:-spot}" = on-demand ] && PROVFLAG=""
ZL="${T6_ZONES:-us-south1-a us-south1-b us-east5-a us-east5-b us-east5-c us-east1-b us-east1-c us-west4-a us-west4-b us-west1-c us-east4-b us-east7-b}"
run() {  # $1 name $2 accel $3 minutes $4 remote env
  local o="$B/out_$1"; local upl="$B/repo.tgz $B/tasks_prompts.jsonl $B/tasks_sample.jsonl"
  for r in "$o"/raw_*.jsonl; do [ -f "$r" ] && upl="$upl $r"; done
  deploy/tpu/drive.sh --name "p3-t6-$1" --script deploy/tpu/run_e_gen.sh --accel "$2" $PROVFLAG --upload "$upl" \
    --download "raw_*.jsonl status.jsonl *.log" --max-minutes "$3" --zones "$ZL" --remote-env "$4" --out "$o" "${ARGS[@]}"
}
ARGS=("$@")
SY=gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/tpu_runs/partial/p3-t6
[ "${T6_ONLY:-small}" = 9b ] || run small v5litepod-1 150 "MODELS='Qwen/Qwen3.5-2B|qwen3.5-2b-bf16|$KS Qwen/Qwen3.5-4B|qwen3.5-4b-bf16|$KS' SAMPLES=8 SAMPLE_TASKS=tasks_sample.jsonl MAX_SEQS_LADDER='32 16 8' STEP_TIMEOUT=5400 SYNC_URI=$SY-small" &
[ "${T6_ONLY:-9b}" = small ] || run 9b v5litepod-4 90 "MODELS='Qwen/Qwen3.5-9B|qwen3.5-9b-bf16|$KS' TP=4 SAMPLES=8 SAMPLE_TASKS=tasks_sample.jsonl MAX_SEQS_LADDER='32 16 8' STEP_TIMEOUT=4800 SYNC_URI=$SY-9b" &
wait
