#!/usr/bin/env bash
# Build the pinned Lean 4 + Mathlib lake project used by Lean4CompilerOracle(project_dir=...).
# Everything goes under ROOT on the second disk (the root disk is nearly full).
# Needs network (GitHub, Mathlib cache); fetches prebuilt oleans, no source build.
#   usage: scripts/setup_lean_mathlib.sh [ROOT]
# The pin below is the last Mathlib tag whose miniF2F headers (e.g. Mathlib.Algebra.BigOperators.Basic,
# removed from later tags) still resolve; checked by HTTP 200/404 on tag file listings on 2026-10-07.
set -euo pipefail
ROOT="${1:-${GWAYA_DATA_ROOT:-$HOME/gwaya-data}/lean-mathlib}"
MATHLIB_REV="${MATHLIB_REV:-v4.7.0}"
TOOLCHAIN="leanprover/lean4:${MATHLIB_REV}"
mkdir -p "$ROOT/elan/bin" "$ROOT/project"
export ELAN_HOME="$ROOT/elan"
cp -n "$HOME"/.elan/bin/elan "$ELAN_HOME/bin/elan"
for t in lake lean; do ln -sf elan "$ELAN_HOME/bin/$t"; done
export PATH="$ELAN_HOME/bin:$PATH"
cd "$ROOT/project"
echo "$TOOLCHAIN" > lean-toolchain
cat > lakefile.lean <<EOF
import Lake
open Lake DSL

package «gwenlaya_lean» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "$MATHLIB_REV"
EOF
elan toolchain install "$TOOLCHAIN"
lake update
lake exe cache get
MATHLIB_REV="$MATHLIB_REV" python3 - <<'PY'
import json, datetime, os
m = json.load(open("lake-manifest.json"))
rev = next(p["rev"] for p in m["packages"] if p["name"] == "mathlib")
json.dump({"lean_toolchain": open("lean-toolchain").read().strip(), "mathlib_tag": os.environ["MATHLIB_REV"],
           "mathlib_commit": rev, "cache": "lake exe cache get",
           "created": datetime.date.today().isoformat()},
          open("gwenlaya_lean_project.json", "w"), indent=1)
PY
echo "done: export GWAYA_LEAN_MATHLIB_DIR=$ROOT/project"
