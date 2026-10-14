#!/usr/bin/env bash
# Deploy the GWAYA/GwenLaya serverless endpoint: Cloud Run CPU gateway (min-instances 0, OpenAI format, API key)
# that creates a Cloud TPU VM running `vllm serve` on demand and lets it delete itself when idle.
#
#   deploy/tpu_endpoint/deploy.sh                       # dry run: prints every command, runs nothing
#   GWAYA_CONFIRM_SPEND=1 deploy/tpu_endpoint/deploy.sh --yes-spend
#
# Resources created (all named gwenlaya-*): service account, custom IAM role, bucket (weights cache), firewall rule,
# Artifact Registry repo, Cloud Run service. Nothing outside those names is modified (no data-lake IAM, no other
# service). The API key is generated once and written to $GWAYA_DATA_ROOT/endpoint/keys.json (mode 600); it is never
# committed or printed unless --show-key.
set -uo pipefail
cd "$(dirname "$0")"
PROJECT="${GCP_PROJECT:-gen-lang-client-0625573011}"
REGION="${CLOUD_RUN_REGION:-us-central1}"
SERVICE="gwenlaya-endpoint"
SA_NAME="gwenlaya-endpoint"; SA="$SA_NAME@$PROJECT.iam.gserviceaccount.com"
ROLE="gwenlayaEndpointTpu"
BUCKET="gwenlaya-endpoint-$PROJECT"
REPO="gwenlaya"; IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/$SERVICE:v1"
ZONES="${TPU_ZONES:-us-west4-a us-central1-a us-east5-b}"
ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}"; KEYS="$ROOT/endpoint/keys.json"
YES=0; SHOW=0
for a in "$@"; do case "$a" in --yes-spend) YES=1 ;; --show-key) SHOW=1 ;; -h|--help) sed -n 2,12p "$0"; exit 0 ;; *) echo "unknown arg $a" >&2; exit 2 ;; esac; done
RUN() { if [ "$YES" = 1 ]; then printf '+ '; printf '%q ' "$@" | sed -e "s#$API_KEY#<api-key>#g" -e "s#$VLLM_KEY#<vllm-key>#g"; echo; "$@"; else printf '  '; printf '%q ' "$@"; echo; fi; }
if [ "$YES" = 1 ] && [ "${GWAYA_CONFIRM_SPEND:-}" != 1 ]; then echo "REFUSED: --yes-spend also needs GWAYA_CONFIRM_SPEND=1" >&2; exit 3; fi
[ "$YES" = 1 ] || echo "DRY RUN: nothing is executed. Real run: GWAYA_CONFIRM_SPEND=1 $0 --yes-spend"

echo "== keys"
if [ "$YES" = 1 ]; then
  mkdir -p "$(dirname "$KEYS")"; umask 077
  [ -f "$KEYS" ] || python3 - "$KEYS" <<'PY'
import json, secrets, sys
json.dump({"api_key": "gwk_" + secrets.token_urlsafe(32), "vllm_key": secrets.token_urlsafe(32)}, open(sys.argv[1], "w"))
PY
  API_KEY="$(python3 -c "import json;print(json.load(open('$KEYS'))['api_key'])")"
  VLLM_KEY="$(python3 -c "import json;print(json.load(open('$KEYS'))['vllm_key'])")"
else
  API_KEY="<generated at real run>"; VLLM_KEY="<generated at real run>"
fi
echo "  keys file: $KEYS"

echo "== 1. service account and least-privilege role (create/get/list/delete TPU nodes only)"
RUN gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" --display-name "GwenLaya endpoint (gateway + TPU VM)"
RUN gcloud iam roles create "$ROLE" --project "$PROJECT" --title "GwenLaya endpoint TPU lifecycle" \
  --permissions tpu.nodes.create,tpu.nodes.get,tpu.nodes.list,tpu.nodes.delete,tpu.operations.get,tpu.operations.list,compute.networks.get,compute.subnetworks.use,compute.subnetworks.get
RUN gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA" --role "projects/$PROJECT/roles/$ROLE" --condition=None
RUN gcloud iam service-accounts add-iam-policy-binding "$SA" --project "$PROJECT" --member "serviceAccount:$SA" --role roles/iam.serviceAccountUser

echo "== 2. weights cache bucket (own bucket; the data lake is not touched)"
RUN gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location us --uniform-bucket-level-access
RUN gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member "serviceAccount:$SA" --role roles/storage.objectAdmin

echo "== 3. firewall: gateway (Direct VPC egress, 10.128.0.0/9 on the default network) -> TPU VM tag, tcp:8000 only"
RUN gcloud compute firewall-rules create gwenlaya-endpoint-8000 --project "$PROJECT" --network default --direction INGRESS \
  --allow tcp:8000 --source-ranges 10.128.0.0/9 --target-tags gwenlaya-endpoint

echo "== 4. gateway image"
RUN gcloud artifacts repositories create "$REPO" --project "$PROJECT" --location "$REGION" --repository-format docker
RUN gcloud builds submit . --project "$PROJECT" --region "$REGION" --tag "$IMAGE"

echo "== 5. Cloud Run service: min 0, HTTP autoscale, public URL, API key enforced by the app"
RUN gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --service-account "$SA" --allow-unauthenticated \
  --min-instances 0 --max-instances 5 --concurrency 80 --timeout 3600 --cpu 1 --memory 512Mi --cpu-boost \
  --network default --subnet default --vpc-egress private-ranges-only \
  --set-env-vars "GCP_PROJECT=$PROJECT,TPU_ZONES=${ZONES// /+},TPU_SERVICE_ACCOUNT=$SA,TPU_ACCEL=v5litepod-1,MODEL_HF=Qwen/Qwen3.5-4B,MODEL_SERVED=gwenlaya-4b,IDLE_MIN=10,MAX_UP_MIN=120,HOLD_S=540,GWAYA_API_KEYS=$API_KEY,VLLM_KEY=$VLLM_KEY" \
  --labels purpose=gwenlaya

if [ "$YES" = 1 ]; then
  URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)')"
  echo "endpoint: $URL"; echo "key file: $KEYS"
  [ "$SHOW" = 1 ] && echo "api key:  $API_KEY"
fi
