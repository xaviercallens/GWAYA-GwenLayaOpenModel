#!/usr/bin/env bash
# Run one script on a Cloud TPU VM, copy results back, delete the VM on every exit path.
#
#   deploy/tpu/drive.sh --name e-gen --script deploy/tpu/run_e_gen.sh \
#       --upload "repo.tgz tasks.jsonl" --download "raw_*.jsonl status.jsonl" --max-minutes 120
#
# Dry run by default: prints the plan, worst-case cost and ledger total, creates nothing.
# Real run needs --yes-spend AND GWAYA_CONFIRM_SPEND=1, and ledger total + worst case must stay
# under GWAYA_CAP_USD (default 45). Spend is appended to the ledger WITH usd_estimate, resource and
# price_source (list-price estimate, not the bill).
set -uo pipefail

NAME="run"; SCRIPT=""; UPLOAD=""; DOWNLOAD=""; MAX_MIN=90; ACCEL="v5litepod-1"; VERSION="v2-alpha-tpuv5-lite"
PROV="on-demand"; ZONES="us-central1-a us-west4-a us-east5-b us-south1-a"; YES=0
LAKE="${GWAYA_LAKE:-gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/tpu_runs}"
LEDGER="${GWAYA_LEDGER:-${GWAYA_DATA_ROOT:-$HOME/gwaya-data}/spend_ledger.jsonl}"
CAP="${GWAYA_CAP_USD:-45}"
PRICE_ONDEMAND="${GWAYA_TPU_PRICE_ONDEMAND:-1.20}"   # USD per chip-hour, v5e (Cloud Billing Catalog, read 2026-10-08)
PRICE_SPOT="${GWAYA_TPU_PRICE_SPOT:-0.4942}"
OUT_DIR="${GWAYA_TPU_OUT:-./tpu_out}"

while [ $# -gt 0 ]; do case "$1" in
  --name) NAME="$2"; shift ;; --script) SCRIPT="$2"; shift ;; --upload) UPLOAD="$2"; shift ;;
  --download) DOWNLOAD="$2"; shift ;; --max-minutes) MAX_MIN="$2"; shift ;; --accel) ACCEL="$2"; shift ;;
  --spot) PROV="spot" ;; --zones) ZONES="$2"; shift ;; --out) OUT_DIR="$2"; shift ;; --yes-spend) YES=1 ;;
  *) echo "unknown arg $1" >&2; exit 2 ;; esac; shift; done
[ -n "$SCRIPT" ] || { echo "--script is required" >&2; exit 2; }

CHIPS="$(echo "$ACCEL" | sed -E 's/.*-([0-9]+)$/\1/')"
if [ "$PROV" = spot ]; then PRICE="$PRICE_SPOT"; PROV_FLAG="--spot"; else PRICE="$PRICE_ONDEMAND"; PROV_FLAG=""; fi
WORST="$(python3 -c "print(round($MAX_MIN/60*$CHIPS*$PRICE + 0.1,2))")"   # +0.1: setup/teardown margin
LEDGER_TOTAL="$(python3 - "$LEDGER" <<'PY'
import json, sys
t = 0.0
try:
    for line in open(sys.argv[1]):
        if line.strip():
            t += float(json.loads(line).get("usd_estimate", 0) or 0)
except FileNotFoundError:
    pass
print(round(t, 4))
PY
)"
echo "plan: $NAME on $ACCEL ($PROV, $CHIPS chip x \$$PRICE/h) max ${MAX_MIN} min; worst case \$$WORST; ledger total \$$LEDGER_TOTAL; cap \$$CAP"
python3 -c "import sys; sys.exit(0 if $LEDGER_TOTAL + $WORST <= $CAP else 1)" || { echo "ABORT: ledger + worst case exceeds cap" >&2; exit 1; }
if [ "$YES" != 1 ] || [ "${GWAYA_CONFIRM_SPEND:-}" != 1 ]; then
  echo "DRY RUN (no resources created). Real run: --yes-spend with GWAYA_CONFIRM_SPEND=1"; exit 0
fi

TPU="gwenlaya-tpu-$NAME-$(date +%H%M%S)"; ZONE=""; T_START=0; WATCHDOG_PID=""
log() { echo "[$(date -Is)] $*"; }
cleanup() {
  [ -n "$ZONE" ] || return 0
  log "copying results and deleting $TPU ($ZONE)"
  mkdir -p "$OUT_DIR"
  for pat in $DOWNLOAD; do gcloud compute tpus tpu-vm scp "$TPU:~/smoke/$pat" "$OUT_DIR/" --zone "$ZONE" --quiet 2>/dev/null; done
  gcloud storage cp "$OUT_DIR"/* "$LAKE/$TPU/" >/dev/null 2>&1 && log "uploaded to $LAKE/$TPU/"
  gcloud compute tpus tpu-vm delete "$TPU" --zone "$ZONE" --quiet 2>&1 | tail -1
  [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null
  local secs=$(( $(date +%s) - T_START ))
  local usd; usd="$(python3 -c "print(round($secs/3600*$CHIPS*$PRICE, 4))")"
  mkdir -p "$(dirname "$LEDGER")"
  python3 - "$LEDGER" "$TPU" "$ZONE" "$ACCEL" "$PROV" "$secs" "$usd" "$PRICE" <<'PY'
import json, sys, datetime
led, tpu, zone, accel, prov, secs, usd, price = sys.argv[1:9]
rec = {"ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "item": "tpu_" + tpu.split("gwenlaya-tpu-")[1],
       "zone": zone, "accel": f"{accel} {prov}", "seconds": int(secs), "usd_estimate": float(usd),
       "resource": f"TPU VM {tpu} ({accel} {prov}, {zone}), deleted",
       "price_source": f"Cloud Billing Catalog TpuV5e {prov} {price} USD/chip-h (list-price estimate, not the bill)"}
open(led, "a").write(json.dumps(rec) + "\n")
print("ledger:", rec["usd_estimate"], "USD")
PY
  log "remaining TPU VMs in $ZONE:"; gcloud compute tpus tpu-vm list --zone "$ZONE" --format="table(name,state)" 2>&1
  ZONE=""
}
trap cleanup EXIT
trap 'log interrupted; exit 130' INT TERM

for Z in $ZONES; do
  log "creating $TPU ($ACCEL, $PROV) in $Z"
  if gcloud compute tpus tpu-vm create "$TPU" --zone "$Z" --accelerator-type "$ACCEL" --version "$VERSION" $PROV_FLAG \
       --labels purpose=gwenlaya-tpu --quiet 2>&1 | tail -2 && gcloud compute tpus tpu-vm describe "$TPU" --zone "$Z" >/dev/null 2>&1; then
    ZONE="$Z"; break
  fi
  log "no capacity in $Z"
done
[ -n "$ZONE" ] || { log "ABORT: no capacity in: $ZONES"; exit 1; }
T_START="$(date +%s)"
# independent watchdog: deletes the VM even if this driver is killed
nohup bash -c "sleep $(( (MAX_MIN + 15) * 60 )); gcloud compute tpus tpu-vm delete $TPU --zone $ZONE --quiet" >/dev/null 2>&1 &
WATCHDOG_PID=$!
log "created in $ZONE; watchdog pid $WATCHDOG_PID deletes it after $((MAX_MIN + 15)) min"

ssh_run() { gcloud compute tpus tpu-vm ssh "$TPU" --zone "$ZONE" --quiet --command "$1"; }
for i in 1 2 3 4 5 6; do ssh_run "mkdir -p ~/smoke" && break; sleep 20; done
FILES="$SCRIPT"; for f in $UPLOAD; do FILES="$FILES $f"; done
gcloud compute tpus tpu-vm scp $FILES "$TPU:~/smoke/" --zone "$ZONE" --quiet
RS="$(basename "$SCRIPT")"
log "running $RS (hard cap ${MAX_MIN} min)"
timeout "$(( MAX_MIN * 60 ))" gcloud compute tpus tpu-vm ssh "$TPU" --zone "$ZONE" --quiet --command "cd ~/smoke && chmod +x $RS && ./$RS" 2>&1 | tail -30
log "remote script finished (rc=${PIPESTATUS[0]})"
