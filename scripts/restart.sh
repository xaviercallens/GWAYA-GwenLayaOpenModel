#!/usr/bin/env bash
# One-command restart for GWAYA / GwenLaya work: prints where things stand and the exact next commands.
# READ-ONLY: no spend, no deletes, no pushes. Safe to run at the start of any session.
#
#   scripts/restart.sh            # status + next steps
#   scripts/restart.sh --env      # also print the exports a shell needs (eval "$(scripts/restart.sh --env-only)")
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}"
CAP="${GWAYA_CAP_USD:-50}"
PROJECT="${GCP_PROJECT:-gen-lang-client-0625573011}"
if [ "${1:-}" = "--env-only" ]; then
  echo "export GWAYA_DATA_ROOT=$ROOT GWAYA_CAP_USD=47.76 GCP_PROJECT=$PROJECT"
  echo "export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  exit 0
fi
hr() { printf '\n== %s\n' "$1"; }

hr "git"
echo "branch: $(git rev-parse --abbrev-ref HEAD)  head: $(git rev-parse --short HEAD)"
echo "latest tag: $(git describe --tags --abbrev=0 2>/dev/null || echo none)"
git status --short | grep -v '^??' | head -5
git worktree list | sed 's#^#  wt: #' | head -20
command -v gh >/dev/null && { echo "open PRs:"; gh pr list --state open --json number,title,headRefName --jq '.[] | "  #\(.number) \(.headRefName): \(.title)"' 2>/dev/null; }

hr "spend ledger (list-price estimates, not the bill)"
python3 - "$ROOT/spend_ledger.jsonl" "$CAP" <<'PY'
import json, sys
t = 0.0; n = 0; last = None
try:
    for ln in open(sys.argv[1]):
        if ln.strip():
            d = json.loads(ln); t += float(d.get("usd_estimate", 0) or 0); n += 1; last = d
except FileNotFoundError:
    print("no ledger at", sys.argv[1]); sys.exit(0)
cap = float(sys.argv[2])
print(f"{n} lines, total ${t:.2f} of ${cap:.0f} cap, remaining ${cap - t:.2f}")
if last:
    print("last:", last.get("ts"), last.get("item"), "$", last.get("usd_estimate"))
PY

hr "cloud resources that bill while they exist"
for z in us-west4-a us-central1-a us-east5-b us-south1-a; do
  gcloud compute tpus tpu-vm list --zone "$z" --project "$PROJECT" --format="value(name,acceleratorType,state)" 2>/dev/null | sed "s#^#  TPU $z: #"
done
echo "(gwenlaya-tpu-* and gwenlaya-endpoint are ours; gwenlaya-tpu-1 and other VMs are not - never delete those)"
gcloud run services describe gwenlaya-endpoint --region us-central1 --project "$PROJECT" --format="value(status.url)" 2>/dev/null \
  | sed 's#^#  endpoint: #' | grep . || echo "  endpoint: not deployed (deploy/tpu_endpoint/deploy.sh, run by the user)"

hr "local prerequisites"
for f in "$ROOT/laya_data/laya_v1.tgz" "$ROOT/night/models/Qwen3.5-2B-Q4_K_M.gguf" "$HOME/.config/zenodo/token" "$ROOT/endpoint/keys.json"; do
  [ -f "$f" ] && echo "  ok      $f" || echo "  missing $f"
done
python3 -c "import sklearn, numpy" 2>/dev/null && echo "  python3 has sklearn/numpy" || echo "  python3 lacks sklearn/numpy"
df -h "$ROOT" | tail -1 | awk '{print "  data disk free: "$4" ("$5" used)"}'

hr "next commands (the user runs anything that spends money or changes IAM, in their own terminal)"
cat <<'EOF'
  tests            PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider -q   (22 env-only failures are known)
  endpoint deploy  GWAYA_CONFIRM_SPEND=1 deploy/tpu_endpoint/deploy.sh --yes-spend
  endpoint test    GWAYA_ENDPOINT=<url> GWAYA_API_KEY=<key> python3 deploy/tpu_endpoint/smoke.py --stop
  TPU batch stage  GWAYA_CONFIRM_SPEND=1 setsid nohup scripts/tpu/launch_*.sh --yes-spend > $GWAYA_DATA_ROOT/<stage>.log 2>&1 &
  zenodo           python3 scripts/release/zenodo_new_version.py            # draft; add --publish <id> only after review
  open work        see docs/TPU_ENDPOINT.md (measure cold start) and the README "Open Call" tracks; 27B tier never loaded
EOF
