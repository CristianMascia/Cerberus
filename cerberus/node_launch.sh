#!/usr/bin/env bash
#
# Cerberus — per-node launcher (run via srun from deploy.py on each node).
#
# Reads the plan, starts this node's llama.cpp servers inside the CUDA sandbox
# (each on a free port, bound to 0.0.0.0), reports "label<TAB>host<TAB>port" back
# to the shared endpoints file, then waits (keeping the srun step, and thus the
# servers, alive) until the step is killed.
#
# Usage:  node_launch.sh <node_idx> <plan.tsv> <endpoints.parts>
#
# Config comes from the environment (exported by deploy.py, propagated by srun).
#
set -uo pipefail

NODE_IDX="${1:?usage: node_launch.sh <node_idx> <plan.tsv> <endpoints.parts>}"
PLAN="${2:?missing plan.tsv}"
PARTS="${3:?missing endpoints.parts}"

SANDBOX="${CERBERUS_SANDBOX:-$HOME/tools/cuda-build}"
BIN="${CERBERUS_BIN:-$HOME/tools/llama.cpp-master/build/bin}"
BIND_DIR="${CERBERUS_BIND:-$HOME/tools}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
NGL="${CERBERUS_NGL:-999}"
THREADS="${CERBERUS_THREADS:-8}"
THREADS_HTTP="${CERBERUS_THREADS_HTTP:-4}"
PORT_MIN="${CERBERUS_PORT_MIN:-8081}"
PORT_MAX="${CERBERUS_PORT_MAX:-8999}"
LOGDIR="${CERBERUS_LOGDIR:-$(pwd)/logs}"

mkdir -p "$LOGDIR"
HOST="$(hostname)"
log() { echo "[node $NODE_IDX @ $HOST] $*"; }

SINGULARITY="$(command -v singularity 2>/dev/null || command -v apptainer 2>/dev/null || true)"
[ -n "$SINGULARITY" ] || { log "ERROR: singularity/apptainer not found on this node"; exit 1; }

PIDS=()
cleanup() {
    log "stopping servers..."
    for pid in "${PIDS[@]:-}"; do [ -n "$pid" ] && kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Find a free TCP port in [PORT_MIN, PORT_MAX] on this node.
free_port() {
    python3 - "$PORT_MIN" "$PORT_MAX" <<'PY'
import socket, sys, random
lo, hi = int(sys.argv[1]), int(sys.argv[2])
for _ in range(200):
    p = random.randint(lo, hi)
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", p)); s.close(); print(p); break
    except OSError:
        s.close()
else:
    sys.exit("no free port")
PY
}

started=0
while IFS=$'\t' read -r node label container device ctx parallel kv reasoning reasoning_format rbudget; do
    [ "$node" = "$NODE_IDX" ] || continue

    port="$(free_port)" || { log "no free port for $label"; continue; }

    srv=( -m "$container" -dev "$device"
          -ngl "$NGL" --fit off
          -c "$ctx" -np "$parallel"
          -ctk "$kv" -ctv "$kv" --jinja
          --reasoning-format "${reasoning_format:-deepseek}" --reasoning "$reasoning"
          --host 0.0.0.0 --port "$port"
          -t "$THREADS" --threads-http "$THREADS_HTTP" )
    [ -n "$rbudget" ] && srv+=( --reasoning-budget "$rbudget" )

    log "start '$label' port $port device $device ctx $ctx"
    "$SINGULARITY" exec --nv \
        --bind "$BIND_DIR":"$BIND_DIR" \
        --bind "$HF_CACHE":/hf \
        --env LD_LIBRARY_PATH="$BIN" \
        "$SANDBOX" "$BIN/llama-server" "${srv[@]}" \
        > "$LOGDIR/$label.log" 2>&1 &
    PIDS+=("$!")

    # report the endpoint back to the master (atomic append)
    printf '%s\t%s\t%s\t%s\n' "$label" "$HOST" "$port" "$reasoning" >> "$PARTS"
    started=$((started + 1))
done < "$PLAN"

[ "$started" -gt 0 ] || { log "no models for this node"; exit 0; }
log "$started server(s) started; waiting."
wait
