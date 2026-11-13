#!/usr/bin/env bash
#
# Cerberus — per-node launcher (run via srun from deploy.py on each node).
#
# Reads the plan, starts this node's servers (llama-server, or `vllm serve` for
# backend=vllm, from the vLLM .sif or from a Python env on the host; each on a
# free port, bound to 0.0.0.0), reports "instance<TAB>host<TAB>port" back
# to the shared endpoints file, then waits (keeping the srun step, and thus the
# servers, alive) until the step is killed. A server that exits on its own (e.g.
# vLLM "Engine core initialization failed") is reported to <endpoints>.dead as
# "instance<TAB>host<TAB>gpus<TAB>exit_code<TAB>backend", so deploy.py stops
# waiting for it at once instead of at the health timeout.
#
# Usage:  node_launch.sh <node_idx> <plan.tsv> <endpoints.parts>
#
# Config comes from the environment (exported by deploy.py, propagated by srun).
#
set -uo pipefail

NODE_IDX="${1:?usage: node_launch.sh <node_idx> <plan.tsv> <endpoints.parts>}"
PLAN="${2:?missing plan.tsv}"
PARTS="${3:?missing endpoints.parts}"
DEAD="${PARTS%.parts}.dead"

# Two ways to run llama-server:
#   CERBERUS_IMAGE set  -> a self-contained .sif with llama-server inside
#                          (e.g. ~/containers/llamacpp_h100.sif on the H100s);
#   otherwise           -> CUDA sandbox + llama.cpp binaries on the host (IBiSCo).
IMAGE="${CERBERUS_IMAGE:-}"
VLLM_IMAGE="${CERBERUS_VLLM_IMAGE:-}"
# backend=vllm without a container: the prefix of a conda env / venv with vllm
# installed (<env>/bin/vllm). When set it wins over CERBERUS_VLLM_IMAGE; the
# model's own `vllm_env` (last plan column) wins over both.
VLLM_ENV_DEFAULT="${CERBERUS_VLLM_ENV:-}"
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

# Singularity/Apptainer is required only by the servers that run in a container
# (all of llama.cpp, vLLM without vllm_env / CERBERUS_VLLM_ENV): checked when one starts.
SINGULARITY="$(command -v singularity 2>/dev/null || command -v apptainer 2>/dev/null || true)"
need_singularity() {
    [ -n "$SINGULARITY" ] || { log "ERROR: singularity/apptainer not found on this node"; exit 1; }
}
[ -z "$IMAGE" ] || [ -f "$IMAGE" ] || { log "ERROR: CERBERUS_IMAGE '$IMAGE' not found"; exit 1; }
# Env vars that reach the container through --cleanenv: APPTAINERENV_X / SINGULARITYENV_X.
case "$(basename "${SINGULARITY:-apptainer}")" in apptainer) ENVP=APPTAINERENV_ ;; *) ENVP=SINGULARITYENV_ ;; esac

PIDS=()
INSTS=() GPUS=() BACKENDS=()   # parallel to PIDS: what each server is, for .dead
# record the server just started in the background ($!)
track() { PIDS+=("$!"); INSTS+=("$inst"); GPUS+=("$1"); BACKENDS+=("$2"); }
# Each server runs in its own process group (setsid): stopping it signals the whole
# group, so vLLM's worker processes do not outlive the server.
CLEANED=0
cleanup() {
    [ "$CLEANED" = 1 ] && return
    CLEANED=1
    log "stopping servers..."
    local pid
    for pid in "${PIDS[@]:-}"; do [ -n "$pid" ] && kill -TERM -- "-$pid" 2>/dev/null; done
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        alive=0
        for pid in "${PIDS[@]:-}"; do [ -n "$pid" ] && kill -0 -- "-$pid" 2>/dev/null && alive=1; done
        [ "$alive" = 0 ] && break
        sleep 1
    done
    for pid in "${PIDS[@]:-}"; do [ -n "$pid" ] && kill -KILL -- "-$pid" 2>/dev/null; done
    wait 2>/dev/null || true
}
trap cleanup EXIT
# leave right after: the servers stopped here are not crashes for the watch loop
trap 'cleanup; exit 143' INT TERM

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

# "CUDA0,CUDA1" (logical GPUs of this step) -> physical ids for CUDA_VISIBLE_DEVICES,
# through the step's own CUDA_VISIBLE_DEVICES when SLURM set one.
cuda_devices() {
    local vis=() out=() d i
    IFS=, read -ra vis <<< "${CUDA_VISIBLE_DEVICES:-}"
    for d in ${1//,/ }; do
        i="${d#CUDA}"
        out+=("${vis[$i]:-$i}")
    done
    (IFS=,; echo "${out[*]}")
}

# backend=vllm: `vllm serve` on the cached snapshot, one process per model,
# tensor-parallel over its GPUs. In the container, --cleanenv keeps the host's
# LD_LIBRARY_PATH, PYTHONPATH and conda variables out (CUDA conflicts); on the
# host (vllm_env) the same variables are dropped with `env -u`.
# Without container the FlashInfer sampler is off by default: it JIT-compiles its
# kernels with nvcc, which a plain vllm env does not have (export
# VLLM_USE_FLASHINFER_SAMPLER=1 to turn it back on).
start_vllm() {
    local gpus; gpus="$(cuda_devices "$device")"
    local model="$container"
    # the plan has container paths (/hf/...): on the host they live under HF_CACHE
    [ -n "$VLLM_ENV" ] && model="$HF_CACHE/${container#/hf/}"
    local args=( serve "$model" --served-model-name "$label"
                 --host 0.0.0.0 --port "$port"
                 --tensor-parallel-size "$tp"
                 --gpu-memory-utilization "$gmu"
                 --max-model-len "$ctx" --max-num-seqs "$parallel" )
    [ "$kv" = fp8 ] && args+=( --kv-cache-dtype fp8 )
    [ -n "$rparser" ] && args+=( --reasoning-parser "$rparser" )
    args+=( "${vextra[@]}" )   # vllm_args from models.conf, verbatim
    if [ -n "$VLLM_ENV" ]; then
        log "start '$inst' (vllm, $VLLM_ENV) port $port gpus $gpus tp $tp max-model-len $ctx"
        setsid env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP \
            -u VIRTUAL_ENV -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
            PATH="$VLLM_ENV/bin:$PATH" HF_HOME="$HF_CACHE" HF_HUB_OFFLINE=1 \
            CUDA_VISIBLE_DEVICES="$gpus" OMP_NUM_THREADS="$THREADS" \
            PYTHONNOUSERSITE=1 VLLM_NO_USAGE_STATS=1 \
            VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
            "$VLLM_ENV/bin/vllm" "${args[@]}" \
            > "$LOGDIR/$inst.log" 2>&1 &
        return
    fi
    need_singularity
    log "start '$inst' (vllm) port $port gpus $gpus tp $tp max-model-len $ctx"
    setsid env "${ENVP}HF_HOME=/hf" "${ENVP}HF_HUB_OFFLINE=1" \
        "${ENVP}CUDA_VISIBLE_DEVICES=$gpus" "${ENVP}OMP_NUM_THREADS=$THREADS" \
        "${ENVP}PYTHONNOUSERSITE=1" "${ENVP}VLLM_NO_USAGE_STATS=1" \
        "$SINGULARITY" exec --nv --cleanenv \
        --bind "$HF_CACHE":/hf \
        "$VLLM_IMAGE" vllm "${args[@]}" \
        > "$LOGDIR/$inst.log" 2>&1 &
}

started=0
while IFS=$'\t' read -r node label container device ctx parallel kv reasoning reasoning_format rbudget backend tp gmu rparser inst venv vargs; do
    [ "$node" = "$NODE_IDX" ] || continue
    # deploy.py writes "-" for empty columns (a tab IFS would merge empty fields)
    [ "$rbudget" = - ] && rbudget=""
    [ "${rparser:--}" = - ] && rparser=""
    inst="${inst:-$label}"   # server name: the label, or label#i for a replica
    [ "${venv:--}" = - ] && venv=""
    VLLM_ENV="${venv:-$VLLM_ENV_DEFAULT}"
    # vllm_args: one column, each argument followed by \x1f (see deploy.plan_row)
    vextra=()
    [ "${vargs:--}" = - ] || IFS=$'\x1f' read -ra vextra <<< "$vargs"

    port="$(free_port)" || { log "no free port for $label"; continue; }

    if [ "${backend:-llamacpp}" = vllm ]; then
        if [ -n "$VLLM_ENV" ]; then
            [ -x "$VLLM_ENV/bin/vllm" ] \
                || { log "ERROR: vllm env '$VLLM_ENV' has no bin/vllm"; exit 1; }
        else
            [ -n "$VLLM_IMAGE" ] && [ -f "$VLLM_IMAGE" ] \
                || { log "ERROR: CERBERUS_VLLM_IMAGE '$VLLM_IMAGE' not found"; exit 1; }
        fi
        start_vllm
        track "$(cuda_devices "$device")" vllm
        printf '%s\t%s\t%s\t%s\n' "$inst" "$HOST" "$port" "$reasoning" >> "$PARTS"
        started=$((started + 1))
        continue
    fi

    srv=( -m "$container" -dev "$device"
          -ngl "$NGL" --fit off
          -c "$ctx" -np "$parallel"
          -ctk "$kv" -ctv "$kv" --jinja
          --reasoning-format "${reasoning_format:-deepseek}" --reasoning "$reasoning"
          --host 0.0.0.0 --port "$port"
          -t "$THREADS" --threads-http "$THREADS_HTTP" )
    [ -n "$rbudget" ] && srv+=( --reasoning-budget "$rbudget" )

    need_singularity
    log "start '$inst' port $port device $device ctx $ctx"
    if [ -n "$IMAGE" ]; then
        setsid "$SINGULARITY" exec --nv \
            --bind "$HF_CACHE":/hf \
            "$IMAGE" llama-server "${srv[@]}" \
            > "$LOGDIR/$inst.log" 2>&1 &
    else
        setsid "$SINGULARITY" exec --nv \
            --bind "$BIND_DIR":"$BIND_DIR" \
            --bind "$HF_CACHE":/hf \
            --env LD_LIBRARY_PATH="$BIN" \
            "$SANDBOX" "$BIN/llama-server" "${srv[@]}" \
            > "$LOGDIR/$inst.log" 2>&1 &
    fi
    track "$(cuda_devices "$device")" llamacpp

    # report the endpoint back to the master (atomic append)
    printf '%s\t%s\t%s\t%s\n' "$inst" "$HOST" "$port" "$reasoning" >> "$PARTS"
    started=$((started + 1))
done < "$PLAN"

[ "$started" -gt 0 ] || { log "no models for this node"; exit 0; }
log "$started server(s) started; waiting."

# Watch the servers: report each one that exits, keep serving with the others,
# leave when none is left. (sleep in the background + wait: a TERM from srun
# runs the trap at once.)
done_=() failed=0
while :; do
    left=0
    for i in "${!PIDS[@]}"; do
        [ -n "${done_[$i]:-}" ] && continue
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then left=$((left + 1)); continue; fi
        wait "${PIDS[$i]}"; rc=$?
        done_[$i]=1
        [ "$rc" = 0 ] || failed=1
        log "ERROR: '${INSTS[$i]}' (${BACKENDS[$i]}, gpu ${GPUS[$i]}) exited with code $rc; log: $LOGDIR/${INSTS[$i]}.log"
        printf '%s\t%s\t%s\t%s\t%s\n' "${INSTS[$i]}" "$HOST" "${GPUS[$i]}" "$rc" \
            "${BACKENDS[$i]}" >> "$DEAD"
    done
    [ "$left" -gt 0 ] || break
    sleep 2 & wait $!
done
log "no server left."
exit "$failed"
