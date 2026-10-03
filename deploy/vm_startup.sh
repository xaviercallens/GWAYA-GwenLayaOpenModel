#!/usr/bin/env bash
# VM startup script (metadata `startup-script`). Runs as root on the spot GPU VM.
# Guarantees: results are synced to the data lake every 90 s and at the end, and the instance deletes
# itself on ANY exit path (trap). Instance also carries --max-run-duration + termination-action DELETE.
set -uo pipefail
md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
IMAGE="$(md attributes/IMAGE)"; LAKE="$(md attributes/LAKE)"; TAG="$(md attributes/RUN_TAG)"
N="$(md attributes/N)"; REQUIRE_GPU="$(md attributes/REQUIRE_GPU)"
NAME="$(md name)"; ZONE="$(md zone | awk -F/ '{print $NF}')"
LOG=/var/log/gwaya_startup.log
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

finish() {
  rc=$?
  log "finishing (rc=$rc); final sync then self-delete"
  [ -d /data/out ] && gcloud storage rsync -r /data/out "$LAKE/runs/$TAG/out" >> "$LOG" 2>&1
  gcloud storage cp "$LOG" "$LAKE/runs/$TAG/startup.log" > /dev/null 2>&1
  echo "rc=$rc finished=$(date -Is)" | gcloud storage cp - "$LAKE/runs/$TAG/DONE" > /dev/null 2>&1
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet >> "$LOG" 2>&1
}
trap finish EXIT

log "waiting for NVIDIA driver (DL image installs it on first boot)"
for i in $(seq 1 90); do nvidia-smi -L > /dev/null 2>&1 && break; sleep 10; done
nvidia-smi -L 2>&1 | tee -a "$LOG"
if [ "$REQUIRE_GPU" = "1" ] && ! nvidia-smi -L > /dev/null 2>&1; then log "no GPU after 15 min; aborting"; exit 4; fi

if ! command -v docker > /dev/null; then
  log "installing docker + nvidia-container-toolkit (DL image ships neither)"
  export DEBIAN_FRONTEND=noninteractive
  { echo "--- held packages:"; apt-mark showhold; echo "--- relevant installed:"; dpkg -l | grep -i -E 'docker|containerd|nvidia-container|libnvidia-container' | cut -c1-120; } >> "$LOG" 2>&1
  apt-get update -qq > /dev/null 2>&1
  APT="apt-get install -y -qq --allow-change-held-packages -o Dpkg::Options::=--force-confold"
  if ! $APT docker.io >> "$LOG" 2>&1; then
    log "docker.io failed (see apt output above); trying the official convenience script"
    curl -fsSL https://get.docker.com | sh >> "$LOG" 2>&1 || { log "docker install failed"; exit 5; }
  fi
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg >> "$LOG" 2>&1
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq > /dev/null 2>&1
  if $APT nvidia-container-toolkit >> "$LOG" 2>&1; then
    nvidia-ctk runtime configure --runtime=docker >> "$LOG" 2>&1
    systemctl restart docker >> "$LOG" 2>&1
    GPU_FLAGS="--gpus all"
  else
    log "nvidia-container-toolkit failed; falling back to raw device + host driver library mounts"
    systemctl restart docker >> "$LOG" 2>&1
    GPU_FLAGS="--device /dev/nvidiactl --device /dev/nvidia0 --device /dev/nvidia-uvm"
    for lib in /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1; do
      [ -e "$lib" ] && GPU_FLAGS="$GPU_FLAGS -v $(readlink -f $lib):$lib:ro"
    done
  fi
fi
GPU_FLAGS="${GPU_FLAGS:---gpus all}"
command -v docker > /dev/null || { log "docker missing"; exit 5; }
mkdir -p /data/model /data/datasets /data/out
log "restoring model, dataset and resume state from $LAKE"
gcloud storage cp "$LAKE/model/ollama_models.tar" /data/model/ >> "$LOG" 2>&1 || exit 6
gcloud storage cp -r "$LAKE/datasets/mbpp-sanitized" /data/datasets/ >> "$LOG" 2>&1 || exit 6
gcloud storage rsync -r "$LAKE/runs/$TAG/out" /data/out >> "$LOG" 2>&1 || log "no prior state for $TAG (fresh run)"

gcloud auth configure-docker us-east4-docker.pkg.dev -q >> "$LOG" 2>&1
log "pulling $IMAGE"
docker pull "$IMAGE" >> "$LOG" 2>&1 || {
  log "registry pull failed; loading image tar from the lake"
  gcloud storage cp "$LAKE/images/gwaya-bench.tar.gz" /data/ >> "$LOG" 2>&1 && docker load -i /data/gwaya-bench.tar.gz >> "$LOG" 2>&1 || exit 7
}

( while true; do sleep 90; gcloud storage rsync -r /data/out "$LAKE/runs/$TAG/out" > /dev/null 2>&1; done ) &
SYNC_PID=$!

log "running benchmark (N=$N)"
# bwrap needs user namespaces; this VM is disposable so relax the container's seccomp/apparmor profile.
docker run --rm $GPU_FLAGS --security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt systempaths=unconfined \
  --cap-add SYS_ADMIN -v /data:/data -e GWAYA_N="$N" -e GWAYA_REQUIRE_GPU="$REQUIRE_GPU" "$IMAGE" >> "$LOG" 2>&1
RC=$?
kill "$SYNC_PID" 2>/dev/null
log "benchmark container exited rc=$RC"
exit "$RC"
