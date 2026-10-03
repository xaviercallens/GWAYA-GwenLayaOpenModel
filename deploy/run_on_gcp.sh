#!/usr/bin/env bash
# Launch the GWAYA benchmark on a spot GPU VM, wait, and PROVE the resources were released.
#   deploy/gwaya_bench/run_on_gcp.sh --smoke                 # 5 problems, checks GPU use + speed-up
#   deploy/gwaya_bench/run_on_gcp.sh --tag gcp_l4_n257 --n 257
#   deploy/gwaya_bench/run_on_gcp.sh --tag gcp_l4_n257 --n 257   # same tag again = resume from the lake
set -uo pipefail
PROJECT="${GCP_PROJECT:-gen-lang-client-0625573011}"
LAKE="gs://socrateai-datalake-gen-lang-client-0625573011/gwaya_bench"
IMAGE="us-east4-docker.pkg.dev/$PROJECT/gwaya-bench/runner:v1"
ZONE="us-east4-a"; MAX_HOURS=2; PRICE=0.45; CAP=5; TAG="gcp_l4_n257"; N=257; REQUIRE_GPU=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

while [ $# -gt 0 ]; do case "$1" in
  --smoke) TAG="smoke_l4_n5"; N=5; MAX_HOURS=1 ;;
  --tag) TAG="$2"; shift ;; --n) N="$2"; shift ;; --zone) ZONE="$2"; shift ;;
  --max-hours) MAX_HOURS="$2"; shift ;; --price-per-hour) PRICE="$2"; shift ;; --cap-usd) CAP="$2"; shift ;;
  *) echo "unknown arg $1"; exit 2 ;; esac; shift; done

NAME="gwaya-bench-$(echo "$TAG" | tr '_' '-')-$(date +%H%M%S)"
WORST=$(python3 -c "print(round($MAX_HOURS*$PRICE*1.25+0.15,2))")   # 25% price margin + disk/egress allowance
echo "worst-case cost estimate: \$$WORST (cap \$$CAP; price \$$PRICE/h is an unverified third-party figure)"
python3 -c "import sys; sys.exit(0 if $WORST <= $CAP else 1)" || { echo "ABORT: worst case exceeds cap"; exit 1; }

echo "== preflight: lake and registry assets"
for o in model/ollama_models.tar datasets/mbpp-sanitized code/code_latest.tar.gz images/gwaya-bench.tar.gz; do
  gcloud storage ls "$LAKE/$o" > /dev/null 2>&1 || { echo "missing in lake: $o"; exit 1; }
done
gcloud artifacts docker images describe "$IMAGE" --project="$PROJECT" > /dev/null 2>&1 || { echo "image not in registry"; exit 1; }

release() {
  echo "== release check"
  for i in $(seq 1 3); do
    left=$(gcloud compute instances list --project="$PROJECT" --filter="labels.purpose=gwaya-bench" --format="value(name,zone.basename())")
    [ -z "$left" ] && break
    echo "still present, deleting: $left"
    echo "$left" | while read -r n z; do gcloud compute instances delete "$n" --zone "$z" --project="$PROJECT" --quiet; done
  done
  echo "-- instances labelled purpose=gwaya-bench:"; gcloud compute instances list --project="$PROJECT" --filter="labels.purpose=gwaya-bench" --format="table(name,zone.basename(),status)"
  echo "-- disks labelled purpose=gwaya-bench:";     gcloud compute disks list     --project="$PROJECT" --filter="labels.purpose=gwaya-bench" --format="table(name,zone.basename(),status)"
}
trap 'echo "interrupted"; release; exit 130' INT TERM

ZONES="${ZONE_LIST:-$ZONE us-east4-c us-central1-a us-central1-b us-central1-c}"
CREATED=""
for Z in $ZONES; do
  echo "== trying spot VM $NAME in $Z (g2-standard-8 + 1xL4, max-run-duration ${MAX_HOURS}h, DELETE on termination)"
  if gcloud compute instances create "$NAME" --project="$PROJECT" --zone="$Z" \
    --machine-type=g2-standard-8 --provisioning-model=SPOT --instance-termination-action=DELETE \
    --max-run-duration="${MAX_HOURS}h" --maintenance-policy=TERMINATE \
    --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
    --boot-disk-size=120GB --boot-disk-type=pd-balanced --boot-disk-auto-delete \
    --scopes=cloud-platform --labels=purpose=gwaya-bench,run="$TAG" \
    --metadata=install-nvidia-driver=True,IMAGE="$IMAGE",LAKE="$LAKE",RUN_TAG="$TAG",N="$N",REQUIRE_GPU="$REQUIRE_GPU" \
    --metadata-from-file=startup-script="$SCRIPT_DIR/vm_startup.sh" 2>&1 | tail -4; then
    if gcloud compute instances describe "$NAME" --zone "$Z" --project="$PROJECT" > /dev/null 2>&1; then CREATED="$Z"; break; fi
  fi
  echo "no capacity/creation failed in $Z"
done
[ -n "$CREATED" ] || { echo "ABORT: no zone had spot L4 capacity"; release; exit 1; }
ZONE="$CREATED"; echo "VM created in $ZONE"

DEADLINE=$(( $(date +%s) + MAX_HOURS*3600 + 600 ))
START=$(date +%s)
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 60
  if ! gcloud compute instances describe "$NAME" --zone "$ZONE" --project="$PROJECT" > /dev/null 2>&1; then echo "VM is gone (self-deleted or expired)"; break; fi
  rows=$(gcloud storage cat "$LAKE/runs/$TAG/out/rows.jsonl" 2>/dev/null | wc -l)
  echo "$(( ($(date +%s)-START)/60 )) min elapsed, problems completed in lake: $rows/$N"
done
release
echo "== result in lake:"; gcloud storage ls -l "$LAKE/runs/$TAG/" "$LAKE/runs/$TAG/out/" 2>&1 | head -20
