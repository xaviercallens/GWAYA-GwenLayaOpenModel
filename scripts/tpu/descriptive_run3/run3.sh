#!/usr/bin/env bash
# Runs ON the TPU VM (run #2). Each step time-boxed; status in ~/smoke/status.jsonl.
set -uo pipefail
R=~/smoke; cd "$R"
export HF_HOME=$R/hf PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
st() { printf '{"step":"%s","rc":%s,"secs":%s,"note":%s}\n' "$1" "$2" "$3" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1][-800:]))' "${4:-}")" >> $R/status.jsonl; }
step() { local name=$1 tmo=$2; shift 2; local t0=$SECONDS; timeout "$tmo" bash -c "$*" > "$R/$name.log" 2>&1; local rc=$?; st "$name" "$rc" $((SECONDS-t0)) "$(grep -E 'SUMMARY|Error|error|Traceback' $R/$name.log | tail -5; tail -c 400 $R/$name.log)"; return $rc; }

# 1. apt: the image runs unattended-upgrades at boot and holds the dpkg lock (run #1 failure)
step apt 900 "sudo systemctl stop unattended-upgrades apt-daily.service apt-daily-upgrade.service 2>/dev/null; \
  for i in \$(seq 1 60); do sudo fuser /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock >/dev/null 2>&1 || break; sleep 5; done; \
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq bubblewrap build-essential && \
  bwrap --ro-bind / / --dev /dev true && echo bwrap-ok"
step rust 600 "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && rustc --version"
step uv 300 "curl -LsSf https://astral.sh/uv/install.sh | sh"
step venv 1500 "uv venv -q -p 3.12 vllm-venv && VIRTUAL_ENV=$R/vllm-venv uv pip install -q vllm-tpu pyflakes requests sympy && $R/vllm-venv/bin/python -c 'import vllm, jax; print(vllm.__version__, jax.devices())'"
PY=$R/vllm-venv/bin/python
step gate_selftest 300 "cd $R/repo && GWAYA_DATA_ROOT=$R/data $PY -c \"
from gwaya.oracles import RustCompilerOracle as R
from gwaya.test_harness import run_candidate_tests as T
print('rust_ok', R().verify_with_test('pub fn add(a:i32,b:i32)->i32{a+b}','assert_eq!(add(1,2),3);',timeout_s=60).success)
print('rust_wrong_rejected', not R().verify_with_test('pub fn add(a:i32,b:i32)->i32{a-b}','assert_eq!(add(1,2),3);',timeout_s=60).success)
h=T('def f(x):\n    return x+1\n','assert f(1)==2\n'); print('python_ok', h.success, h.isolation)\""
# 2. all 80 night tasks (python, rust, math with the fixed prompt) per model; lazy compile for Qwen3.5
step qwen35_2b 1500 "SKIP_JAX_PRECOMPILE=1 $PY gen_tasks.py Qwen/Qwen3.5-2B tasks.jsonl $R/res_qwen35_2b.jsonl"
step qwen35_4b 1800 "SKIP_JAX_PRECOMPILE=1 $PY gen_tasks.py Qwen/Qwen3.5-4B tasks.jsonl $R/res_qwen35_4b.jsonl"
step qwen25_coder_1p5b 1200 "$PY gen_tasks.py Qwen/Qwen2.5-Coder-1.5B-Instruct tasks.jsonl $R/res_qwen25_1p5b.jsonl"
echo SMOKE_DONE >> $R/status.jsonl
