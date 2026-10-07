#!/usr/bin/env bash
# Launch the GWAYA benchmark on a spot GPU VM, wait, and PROVE the resources were released.
# Supports T4 (16GB) and L4 (24GB) GPUs with spot pricing discounts.
#
# EXAMPLES (dry-run is default):
#   deploy/run_on_gcp.sh --gpu l4 --n 257                    # print gcloud commands, cost estimate, zones to try
#   deploy/run_on_gcp.sh --gpu l4 --n 257 --dry-run           # explicit dry-run
#   GWAYA_CONFIRM_SPEND=1 deploy/run_on_gcp.sh --gpu l4 --yes-spend --n 257  # actually launch
#   deploy/run_on_gcp.sh --gpu t4 --model-ladder qwen2.5-coder:7b
#
# GwenLaya v4 staged mode (experiments/plan.json; one GPU at a time, spot T4/L4 only, never A100):
#   deploy/run_on_gcp.sh --stage S2                                   # dry-run: cap, max run, zones, entry commands
#   GWAYA_CONFIRM_SPEND=1 deploy/run_on_gcp.sh --stage S2 --yes-spend # launch (read-only preflight first)
#   Cap and max-run-duration come from plan.json (cap / planning rate); state resumes from
#   gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/runs/<stage> after a preemption.
set -uo pipefail
ORIG_ARGS="$*"

PROJECT="${GCP_PROJECT:-gen-lang-client-0625573011}"
LAKE="gs://socrateai-datalake-gen-lang-client-0625573011/gwaya_bench"
IMAGE="us-east4-docker.pkg.dev/$PROJECT/gwaya-bench/runner:v1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_LAKE="gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4"
STAGE_IMAGE="us-east4-docker.pkg.dev/$PROJECT/gwenlaya/runner:v4"
MAX_ATTEMPTS="${GWAYA_MAX_ATTEMPTS:-3}"

# Read GPU pricing and configuration from JSON
read_gpu_config() {
  local prices_file=$1
  local gpu=$2
  python3 << PYEOF
import json, sys
with open('$prices_file') as f:
  cfg = json.load(f)
  gpu_cfg = cfg.get('gpus', {}).get('$gpu')
  if not gpu_cfg:
    print(f'ERROR: GPU $gpu not found in gpu_prices.json', file=sys.stderr)
    sys.exit(1)
  print(gpu_cfg.get('machine_type', ''))
  print(gpu_cfg.get('price_per_hour_usd', 0))
  print(','.join(gpu_cfg.get('zones', [])))
  print(','.join(gpu_cfg.get('default_models', [])))
PYEOF
}

# Defaults
GPU="l4"
N=257
MODEL_LADDER=""
CAP_USD=5
MAX_HOURS=2
DRY_RUN=1
YES_SPEND=0
TAG=""
ZONE=""
STAGE=""
GPU_SET=0
PURPOSE="gwaya-bench"

# Parse arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)
      GPU="$2"
      GPU_SET=1
      shift 2
      ;;
    --n)
      N="$2"
      shift 2
      ;;
    --model-ladder)
      MODEL_LADDER="$2"
      shift 2
      ;;
    --cap-usd)
      CAP_USD="$2"
      shift 2
      ;;
    --max-hours)
      MAX_HOURS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --yes-spend)
      YES_SPEND=1
      DRY_RUN=0
      shift
      ;;
    --zone)
      ZONE="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --smoke)
      # Smoke test: 5 problems on L4, 1 hour
      GPU="l4"
      N=5
      MAX_HOURS=1
      TAG="smoke_l4_n5"
      shift
      ;;
    *)
      echo "unknown arg $1"
      exit 2
      ;;
  esac
done

# Staged mode: everything (GPU, cap, max run, zones) comes from experiments/plan.json
if [ -n "$STAGE" ]; then
  STAGE_ARGS=(info "$STAGE")
  [ "$GPU_SET" -eq 1 ] && STAGE_ARGS+=(--gpu "$GPU")
  STAGE_OUT=$(python3 "$SCRIPT_DIR/stage_launch.py" "${STAGE_ARGS[@]}") || { echo "ERROR: stage $STAGE cannot be launched on a spot VM"; exit 2; }
  eval "$STAGE_OUT"
  GPU="$S_GPU"; CAP_USD="$S_CAP"; PURPOSE="gwenlaya"; TAG="$STAGE"
  LAKE="$STAGE_LAKE"; IMAGE="${GWAYA_STAGE_IMAGE:-$STAGE_IMAGE}"
fi

# Generate tag if not provided
if [ -z "$TAG" ]; then
  TAG="gcp_${GPU}_n${N}_$(date +%s)"
fi

# Read GPU configuration using temporary file (process substitution doesn't work reliably with multi-line output)
TMPFILE=$(mktemp)
read_gpu_config "$SCRIPT_DIR/gpu_prices.json" "$GPU" > "$TMPFILE" 2>&1
MACHINE_TYPE=$(sed -n '1p' "$TMPFILE")
PRICE_PER_HOUR=$(sed -n '2p' "$TMPFILE")
ZONES_STR=$(sed -n '3p' "$TMPFILE")
DEFAULT_MODELS=$(sed -n '4p' "$TMPFILE")
rm -f "$TMPFILE"

# Check for errors
if [ -z "$MACHINE_TYPE" ] || [ -z "$PRICE_PER_HOUR" ]; then
  echo "ERROR: Failed to read GPU configuration for $GPU"
  exit 1
fi

# Handle model ladder selection
if [ -z "$MODEL_LADDER" ]; then
  # Use first default model for the GPU
  MODEL_LADDER=$(echo "$DEFAULT_MODELS" | cut -d',' -f1)
fi

# Convert comma-separated zones to array
[ -n "$STAGE" ] && ZONES_STR="${S_ZONES// /,}"
IFS=',' read -ra ZONES_ARRAY <<< "$ZONES_STR"

# If zone is not specified, use all available zones
if [ -z "$ZONE" ]; then
  ZONES="${ZONES_ARRAY[*]}"
else
  ZONES="$ZONE"
fi

# Validate inputs
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
  echo "ERROR: --n must be a positive integer"
  exit 1
fi

if ! [[ "$MAX_HOURS" =~ ^[0-9]+$ ]] || [ "$MAX_HOURS" -lt 1 ]; then
  echo "ERROR: --max-hours must be a positive integer"
  exit 1
fi

# Calculate cost estimate
MAX_SECONDS=$(( MAX_HOURS * 3600 ))
WORST=$(python3 << PYTHON_EOF
worst = round(float("$MAX_HOURS") * float("$PRICE_PER_HOUR") * 1.25 + 0.15, 2)
print(worst)
PYTHON_EOF
)
ACCEL_FLAG=""
[ "$GPU" = "t4" ] && ACCEL_FLAG="--accelerator=type=nvidia-tesla-t4,count=1"
if [ -n "$STAGE" ]; then
  # plan.json planning rate (all-in, with margin) replaces gpu_prices.json, which is self-labelled unverified
  MAX_SECONDS="$S_MAX_SECONDS"; WORST="$S_WORST"; PRICE_PER_HOUR="$S_RATE"
  MODEL_LADDER="(per stage entry, experiments/plan.json)"
fi

echo "=== GCP Spot GPU Benchmark Configuration ==="
echo "GPU:              $GPU ($MACHINE_TYPE, ${PRICE_PER_HOUR} USD/hour)"
if [ -n "$STAGE" ]; then
  echo "Stage:            $STAGE (experiments/plan.json; depends on: ${S_DEPENDS:-none})"
  echo "Lake:             $LAKE (state: runs/$STAGE, resumes after preemption, up to $MAX_ATTEMPTS attempts)"
  echo "Planning rate:    ${PRICE_PER_HOUR} USD/h (plan.json: $S_PRICE_SOURCE; not re-read this session)"
  echo "Expected outputs: $S_OUTPUTS"
else
  echo "Problems:         $N (MBPP)"
fi
echo "Model:            $MODEL_LADDER"
echo "Max run:          ${MAX_SECONDS}s (--max-run-duration)"
if [ -n "$STAGE" ]; then
  echo "Worst-case cost:  \$$WORST (max run x planning rate; disk/egress are on the misc caps)"
else
  echo "Worst-case cost:  \$$WORST (25% margin + disk/egress)"
fi
echo "Cost cap:         \$$CAP_USD"
echo "Zones to try:     $ZONES"
echo "Dry-run:          $DRY_RUN"
echo ""

# Validate cost cap
python3 << PYTHON_EOF || exit 1
import sys
if float("$WORST") > float("$CAP_USD"):
  print(f"ABORT: worst-case cost \${WORST} exceeds cap \${CAP_USD}", file=sys.stderr)
  sys.exit(1)
PYTHON_EOF

# Generate and display gcloud commands
LTAG=$(echo "$TAG" | tr 'A-Z_' 'a-z-')
NAME="${PURPOSE}-${LTAG}-$(date +%H%M%S)"
LABELS="purpose=$PURPOSE,run=$LTAG${STAGE:+,stage=$LTAG}"
if [ -n "$STAGE" ]; then
  METADATA="install-nvidia-driver=True,IMAGE=$IMAGE,LAKE=$LAKE,RUN_TAG=$TAG,STAGE=$STAGE,GPU_KIND=$GPU,REQUIRE_GPU=1"
else
  METADATA="install-nvidia-driver=True,IMAGE=$IMAGE,LAKE=$LAKE,RUN_TAG=$TAG,N=$N,MODEL=$MODEL_LADDER,REQUIRE_GPU=1"
fi
# $1 = zone, $2 = max-run-duration seconds
create_cmd() {
  echo "gcloud compute instances create \"$NAME\" --project=\"$PROJECT\" --zone=$1 --machine-type=\"$MACHINE_TYPE\" $ACCEL_FLAG --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration=\"${2}s\" --maintenance-policy=TERMINATE --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release --boot-disk-size=120GB --boot-disk-type=pd-balanced --boot-disk-auto-delete --scopes=cloud-platform --labels=$LABELS --metadata=$METADATA --metadata-from-file=startup-script=\"$SCRIPT_DIR/vm_startup.sh\""
}

echo "=== gcloud Commands (in order) ==="
echo ""

echo "1. Create spot VM in one of these zones (script tries in order):"
for Z in $ZONES; do
  echo "   $(create_cmd "$Z" "$MAX_SECONDS")"
  echo ""
done

echo "2. Wait for results (polls every 60s):"
echo "   gcloud compute instances describe \"$NAME\" --zone CREATED_ZONE --project=\"$PROJECT\""
if [ -n "$STAGE" ]; then
  echo "   gcloud storage cat \"$LAKE/runs/$STAGE/DONE\"   # COMPLETE/FAILED marker; absent after a preemption -> relaunch with the remaining time"
else
  echo "   gcloud storage cat \"$LAKE/runs/$TAG/out/rows.jsonl\""
fi
echo ""

echo "3. Cleanup:"
echo "   gcloud compute instances delete \"$NAME\" --zone CREATED_ZONE --project=\"$PROJECT\" --quiet"
echo "   gcloud compute disks list --project=\"$PROJECT\" --filter='labels.purpose=$PURPOSE' --format='value(name,zone.basename())' | while read n z; do gcloud compute disks delete \$n --zone \$z --project=\"$PROJECT\" --quiet; done"
echo ""

# Exit early if dry-run
if [ "$DRY_RUN" -eq 1 ]; then
  echo "=== DRY-RUN MODE (no resources created) ==="
  if [ -n "$STAGE" ]; then
    echo "=== Stage entry commands (deploy/stage_entry.sh, from plan.json) ==="
    python3 "$SCRIPT_DIR/stage_launch.py" entry "$STAGE" | sed 's/^/  /'
    echo "=== Spend-time preflight (read-only gcloud, skipped in dry-run) ==="
    echo "  gcloud compute project-info describe --project=$PROJECT --format=json   # GPUS_ALL_REGIONS usage must be 0"
    echo "  gcloud compute instances list --project=$PROJECT --format=json          # no running accelerator VM"
    echo "  gcloud storage cat $LAKE/runs/LEDGER.jsonl                              # ledger + cap must stay <= planned cap"
    echo "  gcloud artifacts docker images describe $IMAGE"
  fi
  echo "To actually launch, run:"
  echo "  GWAYA_CONFIRM_SPEND=1 $0 $ORIG_ARGS --yes-spend"
  exit 0
fi

# Validate spend confirmation
if [ "${GWAYA_CONFIRM_SPEND:-}" != "1" ]; then
  echo "ERROR: --yes-spend requires GWAYA_CONFIRM_SPEND=1 env var"
  exit 1
fi

if [ -z "$STAGE" ]; then
# Run preflight checks
echo "=== Preflight: checking lake and registry assets ==="
for o in model/ollama_models.tar datasets/mbpp-sanitized code/code_latest.tar.gz images/gwaya-bench.tar.gz; do
  if ! gcloud storage ls "$LAKE/$o" > /dev/null 2>&1; then
    echo "ERROR: missing in lake: $o"
    exit 1
  fi
done

if ! gcloud artifacts docker images describe "$IMAGE" --project="$PROJECT" > /dev/null 2>&1; then
  echo "ERROR: image not in registry"
  exit 1
fi
fi

# Release resources on exit (label-based: every VM and disk labelled purpose=$PURPOSE)
release() {
  echo "=== Release check ==="
  for i in $(seq 1 3); do
    left=$(gcloud compute instances list --project="$PROJECT" --filter="labels.purpose=$PURPOSE" --format="value(name,zone.basename())")
    if [ -z "$left" ]; then
      break
    fi
    echo "Still present, deleting: $left"
    echo "$left" | while read -r n z; do
      gcloud compute instances delete "$n" --zone "$z" --project="$PROJECT" --quiet
    done
  done
  echo "Instances labelled purpose=$PURPOSE:"
  gcloud compute instances list --project="$PROJECT" --filter="labels.purpose=$PURPOSE" --format="table(name,zone.basename(),status)"
  if [ -n "$STAGE" ]; then
    # a VM that is gone must not leave a billing disk behind
    gcloud compute disks list --project="$PROJECT" --filter="labels.purpose=$PURPOSE" --format="value(name,zone.basename())" \
      | while read -r n z; do [ -n "$n" ] && gcloud compute disks delete "$n" --zone "$z" --project="$PROJECT" --quiet; done
  fi
  echo "Disks labelled purpose=$PURPOSE:"
  gcloud compute disks list --project="$PROJECT" --filter="labels.purpose=$PURPOSE" --format="table(name,zone.basename(),status)"
}
trap 'echo "interrupted"; release; exit 130' INT TERM

# Staged mode: read-only preflight, then up to MAX_ATTEMPTS spot sessions that resume from the lake
run_stage() {
  echo "=== Preflight (read-only) ==="
  quota=$(gcloud compute project-info describe --project="$PROJECT" --format=json) || { echo "ABORT: cannot read quotas"; return 1; }
  echo "$quota" | python3 "$SCRIPT_DIR/stage_launch.py" check-quota || { echo "ABORT: GPU slot not free (single GPU quota); stop the other VM yourself or wait"; return 1; }
  inst=$(gcloud compute instances list --project="$PROJECT" --format=json) || { echo "ABORT: cannot list instances"; return 1; }
  echo "$inst" | python3 "$SCRIPT_DIR/stage_launch.py" check-accel || { echo "ABORT: an accelerator VM is running"; return 1; }
  if ! gcloud artifacts docker images describe "$IMAGE" --project="$PROJECT" > /dev/null 2>&1; then
    echo "ABORT: runner image $IMAGE not in registry (see experiments/launch_packet.md checklist)"; return 1
  fi
  if ! gcloud storage ls "$LAKE/datasets/" > /dev/null 2>&1; then
    echo "ABORT: no inputs under $LAKE/datasets/"; return 1
  fi
  spent=$(gcloud storage cat "$LAKE/runs/LEDGER.jsonl" 2>/dev/null | python3 "$SCRIPT_DIR/stage_launch.py" ledger-total) || { echo "ABORT: unreadable ledger"; return 1; }
  python3 -c "
import sys
plan, spent, cap = 40.0, float('${spent:-0}'), float('$CAP_USD')
print(f'ledger upper bound so far \${spent}, stage cap \${cap}, planned cap \${plan}')
sys.exit(1 if spent + cap > plan else 0)" || { echo "ABORT: ledger + stage cap exceeds the planned cap"; return 1; }
  if [ -n "$(gcloud storage ls "$LAKE/runs/$STAGE/DONE" 2>/dev/null)" ]; then
    echo "Stage $STAGE already has a DONE marker: $(gcloud storage cat "$LAKE/runs/$STAGE/DONE")"; return 0
  fi
  remaining="$MAX_SECONDS"
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    if [ "$remaining" -lt 600 ]; then echo "PARTIAL: stage cap exhausted; not extending"; return 3; fi
    CREATED=""
    for Z in $ZONES; do
      echo "=== Attempt $attempt: spot VM $NAME in $Z ($MACHINE_TYPE, max-run-duration ${remaining}s, DELETE on termination) ==="
      eval "$(create_cmd "$Z" "$remaining")" 2>&1 | tail -4
      if gcloud compute instances describe "$NAME" --zone "$Z" --project="$PROJECT" > /dev/null 2>&1; then
        CREATED="$Z"; break
      fi
      echo "No capacity or creation failed in $Z"
    done
    if [ -z "$CREATED" ]; then echo "ABORT: no zone had spot $GPU capacity"; return 1; fi
    START=$(date +%s)
    DEADLINE=$(( START + remaining + 600 ))
    while [ "$(date +%s)" -lt "$DEADLINE" ]; do
      sleep 60
      if ! gcloud compute instances describe "$NAME" --zone "$CREATED" --project="$PROJECT" > /dev/null 2>&1; then
        echo "VM is gone (self-deleted, preempted or expired)"; break
      fi
      echo "$(( ($(date +%s)-START)/60 )) min elapsed (attempt $attempt)"
    done
    used=$(( $(date +%s) - START ))
    release
    done_msg=$(gcloud storage cat "$LAKE/runs/$STAGE/DONE" 2>/dev/null || true)
    status="PREEMPTED_OR_EXPIRED"; [ -n "$done_msg" ] && status="$done_msg"
    { gcloud storage cat "$LAKE/runs/LEDGER.jsonl" 2>/dev/null; python3 "$SCRIPT_DIR/stage_launch.py" ledger-line --stage "$STAGE" --seconds "$used" --rate "$PRICE_PER_HOUR" --attempt "$attempt" --status "$status"; } \
      | gcloud storage cp - "$LAKE/runs/LEDGER.jsonl" > /dev/null 2>&1 || echo "WARNING: ledger write failed; record this session by hand: stage=$STAGE seconds=$used"
    remaining=$(( remaining - used ))
    if [ -n "$done_msg" ]; then echo "Stage finished: $done_msg"; return 0; fi
    echo "No DONE marker: preemption or max-run expiry; remaining ${remaining}s of the stage cap"
    NAME="${PURPOSE}-${LTAG}-$(date +%H%M%S)"
  done
  echo "PARTIAL: $MAX_ATTEMPTS attempts used without DONE"
  return 3
}

if [ -n "$STAGE" ]; then
  run_stage; rc=$?
  release
  echo "=== Result in lake ==="
  gcloud storage ls -l "$LAKE/runs/$STAGE/" 2>&1 | head -20
  exit "$rc"
fi

# Try to create VM in each zone
CREATED=""
for Z in $ZONES; do
  echo "=== Trying spot VM $NAME in $Z ($MACHINE_TYPE, max-run-duration ${MAX_SECONDS}s, DELETE on termination) ==="
  GCLOUD_CMD="$(create_cmd "$Z" "$MAX_SECONDS")"

  if eval "$GCLOUD_CMD" 2>&1 | tail -4; then
    if gcloud compute instances describe "$NAME" --zone "$Z" --project="$PROJECT" > /dev/null 2>&1; then
      CREATED="$Z"
      break
    fi
  fi
  echo "No capacity or creation failed in $Z"
done

if [ -z "$CREATED" ]; then
  echo "ABORT: no zone had spot $GPU capacity"
  release
  exit 1
fi

ZONE="$CREATED"
echo "VM created in $ZONE"

# Wait for completion
DEADLINE=$(( $(date +%s) + MAX_HOURS*3600 + 600 ))
START=$(date +%s)
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 60
  if ! gcloud compute instances describe "$NAME" --zone "$ZONE" --project="$PROJECT" > /dev/null 2>&1; then
    echo "VM is gone (self-deleted or expired)"
    break
  fi
  rows=$(gcloud storage cat "$LAKE/runs/$TAG/out/rows.jsonl" 2>/dev/null | wc -l)
  echo "$(( ($(date +%s)-START)/60 )) min elapsed, problems completed in lake: $rows/$N"
done

release
echo "=== Result in lake ==="
gcloud storage ls -l "$LAKE/runs/$TAG/" "$LAKE/runs/$TAG/out/" 2>&1 | head -20
