#!/usr/bin/env bash
#
# Cerberus multinode — demo su IBiSCo con 2 nodi di calcolo (3 GPU per nodo).
#
# Va eseguito sul NODO MASTER di un job SLURM che ha allocato >= 2 nodi con GPU
# (tipicamente da sbatch/salloc, vedi submit.sbatch). Avvia 4 modelli:
#   - i modelli con node=0 sul nodo master (localmente);
#   - i modelli con node=1 sul nodo worker (da remoto, via srun);
# ciascun server ascolta su --host 0.0.0.0 e viene raggiunto per <hostname>:<porta>.
# Poi interroga i 4 modelli in sequenza dal master e salva le risposte.
#
# Layout GPU per nodo (da models.conf): un modello su 1 GPU + uno splittato su 2.
#
# Variabili d'ambiente sovrascrivibili:
#   CERBERUS_SANDBOX  sandbox CUDA   (default: ~/tools/cuda-build)
#   CERBERUS_BIN      dir dei binari (default: ~/tools/llama.cpp-master/build/bin)
#   CERBERUS_BIND     dir montata in bind (default: ~/tools)
#   HF_HOME           root della cache HF (default: ~/.cache/huggingface)
#   GPUS_PER_NODE     GPU per nodo richieste allo step srun (default: 3)
#   NGL / THREADS / THREADS_HTTP / HEALTH_TIMEOUT   come nella demo single-node
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurazione (esportata: srun la propaga a node_launch.sh) ------------
export CERBERUS_SANDBOX="${CERBERUS_SANDBOX:-$HOME/tools/cuda-build}"
export CERBERUS_BIN="${CERBERUS_BIN:-$HOME/tools/llama.cpp-master/build/bin}"
export CERBERUS_BIND="${CERBERUS_BIND:-$HOME/tools}"
export HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
export NGL="${NGL:-99}"
export THREADS="${THREADS:-8}"
export THREADS_HTTP="${THREADS_HTTP:-4}"
export SERVER_HOST="0.0.0.0"          # raggiungibile dagli altri nodi
GPUS_PER_NODE="${GPUS_PER_NODE:-3}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"

HF_HUB="$HF_CACHE/hub"
MODELS_CONF="$SCRIPT_DIR/models.conf"
PROMPTS="$SCRIPT_DIR/prompts.txt"
NODE_LAUNCH="$SCRIPT_DIR/node_launch.sh"
OUT_DIR="$SCRIPT_DIR/outputs"
export LOG_DIR="$SCRIPT_DIR/logs"     # condivisa (Lustre): tutti i nodi ci scrivono
RUNTIME="$SCRIPT_DIR/.runtime.tsv"    # manifest generato per node_launch e client
ENDPOINTS="$SCRIPT_DIR/.endpoints.tsv"

mkdir -p "$OUT_DIR" "$LOG_DIR"

log() { echo "[cerberus] $*"; }
die() { echo "[cerberus] ERRORE: $*" >&2; exit 1; }

# ---- Arresto pulito: chiude gli step srun (che chiudono i server remoti) ------
SRUN_PIDS=()
cleanup() {
    echo ""
    log "arresto dei launcher srun e dei server..."
    for pid in "${SRUN_PIDS[@]:-}"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    log "fatto."
}
trap cleanup EXIT INT TERM

# ---- Preflight ---------------------------------------------------------------
command -v singularity >/dev/null 2>&1 || command -v apptainer >/dev/null 2>&1 \
    || die "singularity/apptainer non trovato."
[ -n "${SLURM_JOB_NODELIST:-}" ] || die "fuori da un job SLURM (SLURM_JOB_NODELIST assente).
       Lanciare da sbatch/salloc con >= 2 nodi (vedi submit.sbatch)."
command -v scontrol >/dev/null 2>&1 || die "scontrol non disponibile."
[ -e "$CERBERUS_SANDBOX" ] || die "sandbox non trovata: $CERBERUS_SANDBOX"
[ -x "$CERBERUS_BIN/llama-server" ] || die "binario non trovato: $CERBERUS_BIN/llama-server"
[ -d "$HF_HUB" ] || die "cache HF non trovata: $HF_HUB"
[ -f "$MODELS_CONF" ] || die "manca $MODELS_CONF"
[ -x "$NODE_LAUNCH" ] || die "manca o non eseguibile: $NODE_LAUNCH"

# ---- Nodi dell'allocazione ---------------------------------------------------
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NNODES="${#NODES[@]}"
[ "$NNODES" -ge 2 ] || die "servono almeno 2 nodi, allocati: $NNODES ($SLURM_JOB_NODELIST)."
log "nodi allocati ($NNODES): ${NODES[*]}"
log "nodo master: ${NODES[0]}"

# ---- Risoluzione dei modelli e generazione del manifest ----------------------
# -L segue i symlink: negli snapshot della cache HF i file sono link ai blob.
resolve_gguf() {
    find -L "$HF_HUB/$1/snapshots" -maxdepth 2 -type f -name "$2" 2>/dev/null | head -n1
}

: > "$RUNTIME"; : > "$ENDPOINTS"
declare -A NODE_USED
log "risoluzione dei modelli nella cache HF..."
while IFS= read -r raw; do
    line="${raw%%#*}"
    [ -z "${line//[[:space:]]/}" ] && continue
    IFS=';' read -r name repo glob port ctx maxtok node cudadev <<< "$line"
    name="$(echo "$name" | xargs)"; repo="$(echo "$repo" | xargs)"
    glob="$(echo "$glob" | xargs)"; port="$(echo "$port" | xargs)"
    ctx="$(echo "$ctx" | xargs)"; maxtok="$(echo "$maxtok" | xargs)"
    node="$(echo "$node" | xargs)"; cudadev="$(echo "$cudadev" | xargs)"
    [ -z "$name" ] && continue

    [ "$node" -lt "$NNODES" ] 2>/dev/null \
        || die "modello '$name': node=$node ma sono allocati $NNODES nodi."
    host="${NODES[$node]}"

    host_path="$(resolve_gguf "$repo" "$glob")"
    [ -n "$host_path" ] || die "GGUF non trovato per '$name' (repo=$repo glob=$glob).
       Scaricarlo con demo/ibisco/download_models.sh o vedi guides/llm_download.md."
    rel="${host_path#"$HF_CACHE"/}"
    container="/hf/$rel"

    # cuda_device "1,2" -> "CUDA1,CUDA2"
    dev_arg=""; IFS=',' read -ra parts <<< "$cudadev"
    for p in "${parts[@]}"; do dev_arg+="CUDA${p},"; done
    dev_arg="${dev_arg%,}"

    # manifest per node_launch: node, nome, container, porta, ctx, device
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$node" "$name" "$container" "$port" "$ctx" "$dev_arg" >> "$RUNTIME"
    # endpoints per il client: nome, host, porta, max_tokens
    printf '%s\t%s\t%s\t%s\n' "$name" "$host" "$port" "$maxtok" >> "$ENDPOINTS"

    NODE_USED["$node"]=1
    log "  $name -> nodo $node ($host), device $dev_arg, porta $port"
done < <(grep -v '^[[:space:]]*#' "$MODELS_CONF" | grep -v '^[[:space:]]*$')

# ---- Avvio dei launcher per nodo (via srun) ----------------------------------
# Un solo step srun per nodo: node_launch.sh avvia i server di quel nodo e resta
# in ascolto (wait), così lo step vive finché i server sono attivi.
log "avvio dei launcher (srun, ${GPUS_PER_NODE} GPU per nodo)..."
for idx in $(printf '%s\n' "${!NODE_USED[@]}" | sort -n); do
    node="${NODES[$idx]}"
    log "-> nodo $idx ($node)"
    srun --nodes=1 --ntasks=1 --nodelist="$node" \
         --gres=gpu:"$GPUS_PER_NODE" --overlap \
         bash "$NODE_LAUNCH" "$idx" "$RUNTIME" \
         > "$LOG_DIR/node_${idx}_srun.log" 2>&1 &
    SRUN_PIDS+=("$!")
done

# ---- Attesa che tutti i server siano pronti ----------------------------------
log "attesa health dei server..."
while IFS=$'\t' read -r name host port maxtok; do
    [ -z "$name" ] && continue
    waited=0
    while :; do
        if curl -s "http://$host:$port/health" 2>/dev/null | grep -q '"ok"'; then
            log "  $name pronto su http://$host:$port"; break
        fi
        sleep 2; waited=$((waited + 2))
        if [ "$waited" -ge "$HEALTH_TIMEOUT" ]; then
            log "  $name: timeout ($HEALTH_TIMEOUT s). Log:"
            tail -n 20 "$LOG_DIR/$name.log" 2>/dev/null | sed 's/^/      /' >&2
            die "avvio di '$name' fallito."
        fi
    done
done < "$ENDPOINTS"

log "tutti e ${#SRUN_PIDS[@]} i nodi attivi, 4 modelli su."
echo ""

# ---- Interrogazione dal master -----------------------------------------------
log "interrogazione dei 4 modelli in sequenza..."
python3 "$SCRIPT_DIR/query_models.py" \
    --endpoints "$ENDPOINTS" \
    --prompts "$PROMPTS" \
    --out "$OUT_DIR"

echo ""
log "demo completata. Risposte in: $OUT_DIR/"
# I server remoti/locali vengono chiusi da cleanup() (kill degli step srun).
