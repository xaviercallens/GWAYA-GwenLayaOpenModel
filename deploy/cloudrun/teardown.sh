#!/usr/bin/env bash
# Delete the GwenLaya Cloud Run service(s). DRY-RUN BY DEFAULT. A real run needs --yes and GWAYA_CONFIRM_SPEND=1.
# Only services named gwenlaya-* are ever touched (never anse-serverless-laya / deepseek-prover-v2).
# The container image in Artifact Registry is kept unless --delete-image is passed.
set -uo pipefail
PROJECT="${GCP_PROJECT:-gen-lang-client-0625573011}"
REGION="us-central1"
SERVICES=()
IMAGE=""
DELETE_IMAGE=0
DRY_RUN=1

while [ $# -gt 0 ]; do
  case "$1" in
    --service) SERVICES+=("$2"); shift 2 ;;
    --delete-image) IMAGE="$2"; DELETE_IMAGE=1; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes|--yes-spend) DRY_RUN=0; shift ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ ${#SERVICES[@]} -eq 0 ] && SERVICES=(gwenlaya-gen)
for s in "${SERVICES[@]}"; do
  case "$s" in gwenlaya-*) ;; *) echo "REFUSED: $s does not start with gwenlaya-" >&2; exit 2 ;; esac
done

CMDS=()
for s in "${SERVICES[@]}"; do
  CMDS+=("gcloud run services delete $s --project $PROJECT --region $REGION --quiet")
done
[ "$DELETE_IMAGE" = 1 ] && CMDS+=("gcloud artifacts docker images delete $IMAGE --project $PROJECT --quiet")

if [ "$DRY_RUN" = 1 ]; then
  echo "DRY-RUN: no gcloud command is executed."; printf '  %s\n' "${CMDS[@]}"
  echo "To run for real: GWAYA_CONFIRM_SPEND=1 $0 --yes"; exit 0
fi
[ "${GWAYA_CONFIRM_SPEND:-}" = "1" ] || { echo "REFUSED: --yes also requires GWAYA_CONFIRM_SPEND=1" >&2; exit 3; }
rc=0
for c in "${CMDS[@]}"; do echo "+ $c"; $c || rc=1; done
exit $rc
