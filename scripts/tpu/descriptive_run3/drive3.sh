#!/usr/bin/env bash
# Local driver: spot v5litepod-1 smoke test. Deletes the TPU VM on every exit path.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/home/xavkal/xdev/GWAYA-GwenLayaOpenModel
LAKE=gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/tpu_smoke
LEDGER="${GWAYA_LEDGER:-${GWAYA_DATA_ROOT:-$HOME/gwaya-data}/spend_ledger.jsonl}"
NAME="gwenlaya-tpu-run3-$(date +%H%M%S)"
ZONES="${ZONES:-us-central1-a us-west4-a us-east5-b us-south1-a europe-west4-b}"
PROV_FLAG="${PROV_FLAG---spot}"
ZONE=""; T_START=0
log() { echo "[$(date -Is)] $*"; }

cleanup() {
  [ -n "$ZONE" ] || return 0
  log "deleting $NAME in $ZONE"
  scp_back
  gcloud compute tpus tpu-vm delete "$NAME" --zone "$ZONE" --quiet 2>&1 | tail -2
  local secs=$(( $(date +%s) - T_START ))
  log "remaining TPU VMs in $ZONE:"; gcloud compute tpus tpu-vm list --zone "$ZONE" --format="table(name,state)" 2>&1
  local usd; usd=$(python3 -c "print(round($secs*1.20/3600,4))")
  echo "{\"ts\":\"$(date -Is)\",\"item\":\"tpu_run3\",\"zone\":\"$ZONE\",\"accel\":\"v5litepod-1 ondemand\",\"seconds\":$secs,\"usd_estimate\":$usd,\"resource\":\"TPU VM $NAME (v5litepod-1 on-demand, $ZONE), deleted\",\"price_source\":\"Cloud Billing Catalog TpuV5e on-demand 1.20 USD/chip-h\"}" >> "$LEDGER"
  ZONE=""
}
scp_back() {
  mkdir -p "$HERE/out3"
  gcloud compute tpus tpu-vm scp "$NAME:~/smoke/status.jsonl" "$NAME:~/smoke/res_*" "$NAME:~/smoke/*.log" "$HERE/out3/" --zone "$ZONE" --quiet 2>/dev/null
  gcloud storage cp "$HERE"/out3/* "$LAKE/$NAME/" >/dev/null 2>&1 && log "uploaded results to $LAKE/$NAME/"
}
trap cleanup EXIT
trap 'log "interrupted"; exit 130' INT TERM

mkdir -p "$(dirname "$LEDGER")"
tar czf "$HERE/repo.tgz" --exclude=.git --exclude='__pycache__' --exclude='*.egg-info' -C "$REPO" .

for Z in $ZONES; do
  log "creating $NAME (v5litepod-1, ${PROV_FLAG:-on-demand}) in $Z"
  if gcloud compute tpus tpu-vm create "$NAME" --zone "$Z" --accelerator-type v5litepod-1 \
       --version v2-alpha-tpuv5-lite $PROV_FLAG --labels purpose=gwenlaya-tpu-smoke --quiet 2>&1 | tail -3; then
    if gcloud compute tpus tpu-vm describe "$NAME" --zone "$Z" >/dev/null 2>&1; then ZONE="$Z"; break; fi
  fi
done
[ -n "$ZONE" ] || { log "ABORT: no spot v5e capacity in $ZONES"; exit 1; }
T_START=$(date +%s)
log "created in $ZONE"

ssh_run() { gcloud compute tpus tpu-vm ssh "$NAME" --zone "$ZONE" --quiet --command "$1"; }
for i in 1 2 3 4 5 6; do ssh_run "mkdir -p ~/smoke/repo" && break; sleep 20; done
gcloud compute tpus tpu-vm scp "$HERE/repo.tgz" "$HERE/run3.sh" "$HERE/gen_tasks.py" "$HERE/tasks.jsonl" "$NAME:~/smoke/" --zone "$ZONE" --quiet
ssh_run "cd ~/smoke && tar xzf repo.tgz -C repo && chmod +x run3.sh"
log "running smoke.sh (hard cap 100 min)"
timeout 6000 gcloud compute tpus tpu-vm ssh "$NAME" --zone "$ZONE" --quiet --command "cd ~/smoke && ./run3.sh" 2>&1 | tail -20
log "smoke finished (rc=$?)"
