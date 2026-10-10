#!/usr/bin/env bash
# Local llama.cpp serving for GwenLaya (log-probs on): scripts/gwenlaya_serve.sh start|stop|status [2b|4b ...]
#
# Flags are the registered study serving protocol (docs/STAGE_LOG.md L0/L2, docs/DEVIATIONS.md D14): loopback only,
# --parallel 2, -c 8192 (4096 per slot), --jinja, --reasoning-budget 0 (thinking off); the client sends greedy
# temperature 0 and the raw prompt (gwaya/llama_tier.py). Threads default to 3 (the study used -t 4).
# No sudo, no network (the binary and the GGUFs must already be on disk). Servers are tracked by pidfile and only
# those pids are ever signalled: another job's llama-server is never touched.
#
# Env (all optional):
#   GWAYA_DATA_ROOT      default /mnt/data/home/xavkal/gwaya-data (logs and pidfiles go to $GWAYA_DATA_ROOT/serve/)
#   LLAMA_SERVER_BIN     default $GWAYA_DATA_ROOT/night/bin/llama-b11476/llama-server
#   GWENLAYA_GGUF_DIR    default $GWAYA_DATA_ROOT/night/models
#   GWENLAYA_THREADS     default 3
#   GWENLAYA_PARALLEL    default 2
#   GWENLAYA_CTX         default 8192
#   GWENLAYA_PORT_2B     default 8091      GWENLAYA_PORT_4B  default 8092
#   GWENLAYA_START_WAIT  seconds to wait for /health, default 300
set -u

ROOT="${GWAYA_DATA_ROOT:-/mnt/data/home/xavkal/gwaya-data}"
BIN="${LLAMA_SERVER_BIN:-$ROOT/night/bin/llama-b11476/llama-server}"
MODELS="${GWENLAYA_GGUF_DIR:-$ROOT/night/models}"
THREADS="${GWENLAYA_THREADS:-3}"
PARALLEL="${GWENLAYA_PARALLEL:-2}"
CTX="${GWENLAYA_CTX:-8192}"
WAIT="${GWENLAYA_START_WAIT:-300}"
RUN="$ROOT/serve"

cmd="${1:-}"; shift || true
tiers=("$@"); [ ${#tiers[@]} -eq 0 ] && tiers=(2b 4b)

port_of()  { case "$1" in 2b) echo "${GWENLAYA_PORT_2B:-8091}";; 4b) echo "${GWENLAYA_PORT_4B:-8092}";; *) return 1;; esac; }
gguf_of()  { case "$1" in 2b) echo "$MODELS/Qwen3.5-2B-Q4_K_M.gguf";; 4b) echo "$MODELS/Qwen3.5-4B-Q4_K_M.gguf";; *) return 1;; esac; }
pidfile()  { echo "$RUN/llama_$1.pid"; }
alive()    { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null && grep -q llama-server "/proc/$1/cmdline" 2>/dev/null; }
mypid()    { local f; f="$(pidfile "$1")"; [ -f "$f" ] && cat "$f" || true; }
healthy()  { curl -fsS --max-time 3 "http://127.0.0.1:$(port_of "$1")/health" >/dev/null 2>&1; }
port_busy(){ (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

start_one() {
  local t="$1" port gguf pid log
  port="$(port_of "$t")" || { echo "unknown tier '$t' (use 2b or 4b)" >&2; return 2; }
  gguf="$(gguf_of "$t")"
  [ -x "$BIN" ] || { echo "llama-server not found or not executable: $BIN" >&2; return 1; }
  [ -f "$gguf" ] || { echo "model not found: $gguf" >&2; return 1; }
  pid="$(mypid "$t")"
  if alive "$pid"; then echo "$t: already running (pid $pid, port $port)"; return 0; fi
  if port_busy "$port"; then
    echo "$t: port $port is already in use by another process (not started; set GWENLAYA_PORT_${t^^} to use another)" >&2
    return 1
  fi
  mkdir -p "$RUN"; log="$RUN/llama_$t.log"
  # the binary dir holds its shared libraries
  (cd "$(dirname "$BIN")" || exit 1
   nohup "$BIN" -m "$gguf" --host 127.0.0.1 --port "$port" -t "$THREADS" \
      --parallel "$PARALLEL" -c "$CTX" --jinja --reasoning-budget 0 >"$log" 2>&1 </dev/null &
   echo $! >"$(pidfile "$t")")
  pid="$(mypid "$t")"
  local i=0
  while [ "$i" -lt "$WAIT" ]; do
    alive "$pid" || { echo "$t: llama-server exited during start, see $log" >&2; rm -f "$(pidfile "$t")"; return 1; }
    if healthy "$t"; then echo "$t: healthy (pid $pid, port $port, -t $THREADS, log $log)"; return 0; fi
    sleep 2; i=$((i + 2))
  done
  echo "$t: not healthy after ${WAIT}s (pid $pid left running; run '$0 stop'), see $log" >&2
  return 1
}

stop_one() {
  local t="$1" pid; pid="$(mypid "$t")"
  if [ -z "$pid" ]; then echo "$t: no pidfile (nothing of ours to stop)"; return 0; fi
  if alive "$pid"; then
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 20); do alive "$pid" || break; sleep 0.5; done
    alive "$pid" && kill -9 "$pid" 2>/dev/null
    echo "$t: stopped pid $pid"
  else
    echo "$t: pid $pid not running"
  fi
  rm -f "$(pidfile "$t")"
}

status_one() {
  local t="$1" pid port; pid="$(mypid "$t")"; port="$(port_of "$t")"
  if alive "$pid"; then
    if healthy "$t"; then echo "$t: up (pid $pid, port $port, health ok)"; else echo "$t: process $pid up, health not ok (port $port)"; fi
  elif healthy "$t"; then
    echo "$t: port $port answers but it is not ours (no live pidfile)"
  else
    echo "$t: down (port $port)"
  fi
}

rc=0
case "$cmd" in
  start)  for t in "${tiers[@]}"; do start_one "$t" || rc=1; done
          if [ $rc -eq 0 ]; then
            echo "export GWENLAYA_BACKEND=llamacpp GWENLAYA_SERVED_QUANT=q4_k_m"
            u=""; for t in "${tiers[@]}"; do u+="${u:+,}qwen3.5-${t}-bf16=http://127.0.0.1:$(port_of "$t")"; done
            echo "export GWENLAYA_LLAMA_URLS=\"$u\""
          fi ;;
  stop)   for t in "${tiers[@]}"; do stop_one "$t"; done ;;
  status) for t in "${tiers[@]}"; do status_one "$t"; done ;;
  *)      echo "usage: $0 start|stop|status [2b|4b ...]" >&2; exit 2 ;;
esac
exit $rc
