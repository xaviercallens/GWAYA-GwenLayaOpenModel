#!/usr/bin/env bash
# Build and deploy the GwenLaya generation service (plan.json stage S7) to Cloud Run GPU (L4).
# DRY-RUN IS THE DEFAULT: prints every command, runs no gcloud. A real run needs BOTH --yes-spend
# and GWAYA_CONFIRM_SPEND=1, and also exactly the commands the user approved (prereg section 11).
#
#   deploy/cloudrun/deploy.sh --model-file qwen3.5-9b-q4_K_M.gguf            # dry-run
#   GWAYA_CONFIRM_SPEND=1 deploy/cloudrun/deploy.sh --yes-spend --model-file X.gguf
set -uo pipefail

PROJECT="${GCP_PROJECT:-gen-lang-client-0625573011}"
REGION="us-central1"          # the only region whose L4 quota was read (prereg section 11); do not change blindly
SERVICE="gwenlaya-gen"
BUCKET="socrateai-datalake-gen-lang-client-0625573011"
QUANT_PREFIX="gwenlaya_v4/quantized"
REPO="gwenlaya"               # Artifact Registry repo in $REGION; it must already exist (never created here)
IMAGE=""
MODEL_FILE=""
LAYA_URL=""
SERVICE_ACCOUNT=""
SKIP_BUILD=0
DRY_RUN=1
YES_SPEND=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() { sed -n '2,8p' "$0"; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --model-file) MODEL_FILE="$2"; shift 2 ;;
    --image) IMAGE="$2"; SKIP_BUILD=1; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --laya-url) LAYA_URL="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --dry-run) DRY_RUN=1; YES_SPEND=0; shift ;;
    --yes-spend) YES_SPEND=1; DRY_RUN=0; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 2 ;;
  esac
done

case "$SERVICE" in
  gwenlaya-*) ;;
  *) echo "REFUSED: service name must start with gwenlaya- (never touch other services, e.g. anse-serverless-laya)" >&2; exit 2 ;;
esac
[ -z "$IMAGE" ] && IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/$SERVICE:v1"
if [ -z "$MODEL_FILE" ]; then
  if [ "$DRY_RUN" = 0 ]; then echo "REFUSED: --model-file (GGUF name under gs://$BUCKET/$QUANT_PREFIX/) is required for a real deploy" >&2; exit 2; fi
  MODEL_FILE="<MODEL_FILE.gguf>"
fi

ENV_VARS="MODEL_DIR=/models,MODEL_FILE=$MODEL_FILE,LLAMA_PARALLEL=8,MODEL_ALIAS=$SERVICE"
[ -n "$LAYA_URL" ] && ENV_VARS="$ENV_VARS,LAYA_SERVICE_URL=$LAYA_URL"

BUILD_CMD=(gcloud builds submit "$SCRIPT_DIR" --project "$PROJECT" --region "$REGION" --tag "$IMAGE")
DEPLOY_CMD=(gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" --image "$IMAGE"
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy
  --cpu 4 --memory 16Gi --no-cpu-throttling
  --min-instances 0 --max-instances 1 --concurrency 8 --timeout 600
  --no-allow-unauthenticated --labels purpose=gwenlaya
  --execution-environment gen2
  --add-volume "name=models,type=cloud-storage,bucket=$BUCKET,readonly=true,mount-options=only-dir=$QUANT_PREFIX"
  --add-volume-mount volume=models,mount-path=/models
  --set-env-vars "$ENV_VARS"
  --startup-probe "httpGet.path=/healthz,httpGet.port=8080,initialDelaySeconds=0,periodSeconds=5,timeoutSeconds=3,failureThreshold=120")
[ -n "$SERVICE_ACCOUNT" ] && DEPLOY_CMD+=(--service-account "$SERVICE_ACCOUNT")
QUOTA_CMD=(gcloud beta quotas info list --service=run.googleapis.com --project "$PROJECT" --format=json)

show() { printf '  '; printf '%q ' "$@"; printf '\n'; }

if [ "$DRY_RUN" = 1 ]; then
  echo "DRY-RUN: no gcloud command is executed."
  echo "Read-only quota preflight (runs first in a real deploy; needs NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion[$REGION] >= 1):"
  show "${QUOTA_CMD[@]}"
  if [ "$SKIP_BUILD" = 0 ]; then echo "Build (Cloud Build, billed to the misc: build line):"; show "${BUILD_CMD[@]}"; fi
  echo "Deploy (L4 Cloud Run, about \$1.0465/h only while an instance is up; planning rate in experiments/plan.json):"
  show "${DEPLOY_CMD[@]}"
  echo "Notes: bucket is US-EAST4 and the service is in $REGION, so the model read is cross-region; see docs/CLOUDRUN_ENDPOINT.md."
  echo "To run for real: GWAYA_CONFIRM_SPEND=1 $0 --yes-spend --model-file ..."
  exit 0
fi

if [ "${GWAYA_CONFIRM_SPEND:-}" != "1" ]; then
  echo "REFUSED: --yes-spend also requires GWAYA_CONFIRM_SPEND=1" >&2
  exit 3
fi

echo "[1/3] read-only quota preflight"
QUOTA_JSON="$("${QUOTA_CMD[@]}")" || { echo "ABORT: quota check failed; S7 is dropped, not retried blindly" >&2; exit 4; }
QUOTA_VAL="$(printf '%s' "$QUOTA_JSON" | python3 -c '
import json, sys
region = sys.argv[1]
for q in json.load(sys.stdin):
    if q.get("quotaId") == "NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion":
        for d in q.get("dimensionsInfos", []):
            if d.get("dimensions", {}).get("region") == region:
                print(d.get("details", {}).get("value", "")); sys.exit(0)
print("")' "$REGION")"
if ! [[ "$QUOTA_VAL" =~ ^[0-9]+$ ]] || [ "$QUOTA_VAL" -lt 1 ]; then
  echo "ABORT: no Cloud Run L4 quota read for $REGION (got '${QUOTA_VAL}'). S7 is dropped. Quota is never requested." >&2
  exit 4
fi
echo "quota NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion[$REGION] = $QUOTA_VAL (other services may already use some)"

if [ "$SKIP_BUILD" = 0 ]; then
  echo "[2/3] build"; show "${BUILD_CMD[@]}"; "${BUILD_CMD[@]}" || { echo "build failed" >&2; exit 5; }
fi
echo "[3/3] deploy"; show "${DEPLOY_CMD[@]}"; "${DEPLOY_CMD[@]}" || { echo "deploy failed" >&2; exit 6; }
echo "Deployed $SERVICE. Run smoke_test.py next, then teardown.sh."
