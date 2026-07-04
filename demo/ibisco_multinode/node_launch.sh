#!/usr/bin/env bash
#
# Cerberus multinode — launcher eseguito su UN nodo (via srun da run_demo_multinode.sh).
#
# Avvia i server dei modelli assegnati a questo nodo, leggendo il manifest
# generato dal master, poi resta in attesa (wait) così lo step srun vive finché i
# server sono attivi. Alla terminazione (fine job o kill dello step) i server
# vengono chiusi.
#
# Uso:  node_launch.sh <node_idx> <runtime.tsv>
#
# Il manifest .runtime.tsv ha colonne (TAB):
#   node_idx  nome  gguf_container  porta  ctx  device_arg
#
# La configurazione (sandbox, binari, cache, thread) arriva dall'ambiente,
# propagato da srun; i default coincidono con quelli del master.
#
set -euo pipefail

NODE_IDX="${1:?uso: node_launch.sh <node_idx> <runtime.tsv>}"
RUNTIME="${2:?manifest runtime.tsv mancante}"

SANDBOX="${CERBERUS_SANDBOX:-$HOME/tools/cuda-build}"
BIN="${CERBERUS_BIN:-$HOME/tools/llama.cpp-master/build/bin}"
BIND_DIR="${CERBERUS_BIND:-$HOME/tools}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
NGL="${NGL:-99}"
THREADS="${THREADS:-8}"
THREADS_HTTP="${THREADS_HTTP:-4}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
LOG_DIR="${LOG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs}"

mkdir -p "$LOG_DIR"
log() { echo "[node $NODE_IDX @ $(hostname)] $*"; }

SINGULARITY="$(command -v singularity || command -v apptainer)"

# ---- Arresto pulito dei server di questo nodo --------------------------------
PIDS=()
cleanup() {
    log "arresto dei server..."
    for pid in "${PIDS[@]:-}"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- Avvio dei server assegnati a questo nodo --------------------------------
log "avvio dei server assegnati (da $RUNTIME)..."
started=0
while IFS=$'\t' read -r node name container port ctx dev_arg; do
    [ "$node" = "$NODE_IDX" ] || continue

    srv_args=(
        -m "$container"
        --device "$dev_arg"
        --host "$SERVER_HOST" --port "$port"
        -ngl "$NGL" -c "$ctx" --jinja
        -t "$THREADS" --threads-http "$THREADS_HTTP"
    )
    # Split multi-GPU (device con virgola): ripartizione uniforme di default.
    log "-> '$name' porta $port device $dev_arg"
    "$SINGULARITY" exec --nv \
        --bind "$BIND_DIR":"$BIND_DIR" \
        --bind "$HF_CACHE":/hf \
        --env LD_LIBRARY_PATH="$BIN" \
        "$SANDBOX" \
        "$BIN/llama-server" \
        "${srv_args[@]}" \
        > "$LOG_DIR/$name.log" 2>&1 &
    PIDS+=("$!")
    started=$((started + 1))
done < "$RUNTIME"

[ "$started" -gt 0 ] || { log "nessun modello per questo nodo."; exit 0; }

log "$started server avviati; in attesa (lo step srun resta vivo)."
# Mantiene vivo lo step finché i server sono attivi (o finché lo step è ucciso).
wait
