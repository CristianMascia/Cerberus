#!/usr/bin/env bash
#
# Cerberus — demo su IBiSCo.
#
# Avvia in sequenza tre modelli LLM (le tre "teste") con llama.cpp su IBiSCo,
# usando la sandbox CUDA e i binari compilati su Lustre (vedi
# guides/llamacpp_install.md, Parte 2), ciascuno da un modello GGUF nella cache
# di Hugging Face montata in bind, poi lancia lo script Python che interroga i
# tre modelli in sequenza e ne salva le risposte. Al termine (anche in caso di
# errore o Ctrl-C) arresta tutti i server.
#
# Va eseguito su un NODO DI CALCOLO con GPU, tipicamente all'interno di un job
# SLURM (salloc/sbatch) che ha già allocato le GPU. Esempio interattivo:
#
#   salloc --partition=<gpu_part> --gres=gpu:3 --time=02:00:00
#   ./run_demo_ibisco.sh
#
# Due modalità di assegnazione GPU (variabile CERBERUS_MODE):
#   manual (default) : ogni modello usa la GPU indicata da 'cuda_device' in
#                      models.conf (indice logico nell'allocazione).
#   auto             : i modelli sono distribuiti sulle GPU disponibili in base
#                      a 'est_mem_mb' (first-fit sulla memoria libera).
#
# Variabili d'ambiente sovrascrivibili:
#   CERBERUS_MODE     manual | auto                 (default: manual)
#   CERBERUS_SANDBOX  sandbox CUDA   (default: /lustre/home/$USER/cu122-sm70)
#   CERBERUS_BIN      dir dei binari (default: /lustre/home/$USER/llama.cpp-master/build/bin)
#   HF_HOME           root della cache HF           (default: ~/.cache/huggingface)
#   LUSTRE            radice Lustre da montare in bind (default: /lustre)
#   NGL               layer in offload su GPU        (default: 99)
#   HEALTH_TIMEOUT    attesa max avvio server (s)    (default: 300)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurazione ----------------------------------------------------------
MODE="${CERBERUS_MODE:-manual}"
SANDBOX="${CERBERUS_SANDBOX:-/lustre/home/$USER/cu122-sm70}"
BIN="${CERBERUS_BIN:-/lustre/home/$USER/llama.cpp-master/build/bin}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB="$HF_CACHE/hub"
LUSTRE="${LUSTRE:-/lustre}"
MODELS_CONF="$SCRIPT_DIR/models.conf"
PROMPTS="$SCRIPT_DIR/prompts.txt"
OUT_DIR="$SCRIPT_DIR/outputs"
LOG_DIR="$SCRIPT_DIR/logs"
HOST="0.0.0.0"          # raggiungibile da altri nodi (rete InfiniBand)
CLIENT_HOST="127.0.0.1" # il client gira sullo stesso nodo dei server
NGL="${NGL:-99}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# ---- Arresto pulito dei server al termine ------------------------------------
PIDS=()
cleanup() {
    echo ""
    echo "[cerberus] arresto dei server..."
    for pid in "${PIDS[@]:-}"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "[cerberus] fatto."
}
trap cleanup EXIT INT TERM

log() { echo "[cerberus] $*"; }
die() { echo "[cerberus] ERRORE: $*" >&2; exit 1; }

# ---- Preflight ---------------------------------------------------------------
command -v singularity >/dev/null 2>&1 || command -v apptainer >/dev/null 2>&1 \
    || die "singularity/apptainer non trovato nel PATH."
SINGULARITY="$(command -v singularity || command -v apptainer)"

case "$MODE" in manual|auto) ;; *) die "CERBERUS_MODE non valido: '$MODE' (manual|auto)";; esac
[ -e "$SANDBOX" ] || die "sandbox non trovata: $SANDBOX (vedi guides/llamacpp_install.md, Parte 2)."
[ -x "$BIN/llama-server" ] || die "binario non trovato: $BIN/llama-server"
[ -d "$HF_HUB" ] || die "cache HF non trovata: $HF_HUB"
[ -f "$MODELS_CONF" ] || die "manca $MODELS_CONF"
[ -f "$PROMPTS" ] || die "manca $PROMPTS"

log "modalità di allocazione GPU: $MODE"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] \
    && log "GPU allocate dal job (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES" \
    || log "CUDA_VISIBLE_DEVICES non impostata (fuori da un job SLURM?)."

# ---- Assegnazione delle GPU (manual | auto) ----------------------------------
# allocate_gpus.py emette righe "nome<TAB>device<TAB>tensor_split":
#   - device       : indice/i logici della/e GPU (es. "2" o "1,3");
#   - tensor_split : "-" per GPU singola, altrimenti le frazioni per lo split
#                    multi-GPU (es. "0.55,0.45"), nello stesso ordine dei device.
declare -A DEVICE_OF TSPLIT_OF
while IFS=$'\t' read -r a_name a_dev a_ts; do
    [ -z "$a_name" ] && continue
    DEVICE_OF["$a_name"]="$a_dev"
    TSPLIT_OF["$a_name"]="$a_ts"
done < <(python3 "$SCRIPT_DIR/allocate_gpus.py" --config "$MODELS_CONF" --mode "$MODE") \
    || die "assegnazione GPU fallita (vedi messaggi sopra)."

# ---- Risoluzione dei modelli nella cache HF ----------------------------------
# -L segue i symlink: negli snapshot della cache HF i file sono link ai blob.
resolve_gguf() {
    find -L "$HF_HUB/$1/snapshots" -maxdepth 2 -type f -name "$2" 2>/dev/null | head -n1
}

NAMES=(); PORTS=(); CTXS=(); GGUF_CONTAINER=(); DEVICES=(); TSPLITS=()

log "risoluzione dei modelli nella cache HF..."
while IFS= read -r raw; do
    line="${raw%%#*}"
    [ -z "${line//[[:space:]]/}" ] && continue
    IFS=';' read -r name repo glob port ctx maxtok cudadev estmem <<< "$line"
    name="$(echo "$name" | xargs)"; repo="$(echo "$repo" | xargs)"
    glob="$(echo "$glob" | xargs)"; port="$(echo "$port" | xargs)"
    ctx="$(echo "$ctx" | xargs)"
    [ -z "$name" ] && continue

    host_path="$(resolve_gguf "$repo" "$glob")"
    [ -n "$host_path" ] || die "GGUF non trovato per '$name' (repo=$repo glob=$glob).
       Verificare la cache; per scaricarlo vedi guides/llm_download.md."
    rel="${host_path#"$HF_CACHE"/}"

    device="${DEVICE_OF[$name]:-}"
    [ -n "$device" ] || die "nessuna GPU assegnata a '$name' (allocate_gpus.py)."
    # "1,3" -> "CUDA1,CUDA3"
    dev_arg=""
    IFS=',' read -ra parts <<< "$device"
    for p in "${parts[@]}"; do dev_arg+="CUDA${p},"; done
    dev_arg="${dev_arg%,}"

    tsplit="${TSPLIT_OF[$name]:--}"   # "-" se GPU singola

    NAMES+=("$name"); PORTS+=("$port"); CTXS+=("$ctx")
    GGUF_CONTAINER+=("/hf/$rel"); DEVICES+=("$dev_arg"); TSPLITS+=("$tsplit")
    if [ "$tsplit" = "-" ]; then
        log "  $name -> $(basename "$host_path")  (porta $port, device $dev_arg)"
    else
        log "  $name -> $(basename "$host_path")  (porta $port, device $dev_arg, split $tsplit)"
    fi
done < <(grep -v '^[[:space:]]*#' "$MODELS_CONF" | grep -v '^[[:space:]]*$')

[ "${#NAMES[@]}" -gt 0 ] || die "nessun modello valido in $MODELS_CONF"

# ---- Attesa che un server sia pronto -----------------------------------------
wait_health() {
    local port="$1" name="$2" waited=0
    while : ; do
        if curl -s "http://$CLIENT_HOST:$port/health" 2>/dev/null | grep -q '"ok"'; then
            log "  $name pronto su http://$CLIENT_HOST:$port"
            return 0
        fi
        sleep 2; waited=$((waited + 2))
        if [ "$waited" -ge "$HEALTH_TIMEOUT" ]; then
            log "  $name: timeout ($HEALTH_TIMEOUT s). Ultime righe di log:"
            tail -n 20 "$LOG_DIR/$name.log" 2>/dev/null | sed 's/^/      /' >&2
            return 1
        fi
    done
}

# ---- Avvio dei tre server in sequenza ----------------------------------------
log "avvio dei ${#NAMES[@]} server (sandbox: $(basename "$SANDBOX"))..."
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; port="${PORTS[$i]}"; ctx="${CTXS[$i]}"
    model="${GGUF_CONTAINER[$i]}"; device="${DEVICES[$i]}"; tsplit="${TSPLITS[$i]}"
    log "-> avvio '$name' (porta $port, ctx $ctx, device $device)"

    # Argomenti di llama-server; --tensor-split solo per lo split multi-GPU.
    srv_args=(
        -m "$model"
        --device "$device"
        --host "$HOST" --port "$port"
        -ngl "$NGL" -c "$ctx" --jinja
    )
    [ "$tsplit" != "-" ] && srv_args+=(--tensor-split "$tsplit")

    "$SINGULARITY" exec --nv \
        --bind "$LUSTRE":"$LUSTRE" \
        --bind "$HF_CACHE":/hf \
        --env LD_LIBRARY_PATH="$BIN" \
        "$SANDBOX" \
        "$BIN/llama-server" \
        "${srv_args[@]}" \
        > "$LOG_DIR/$name.log" 2>&1 &
    PIDS+=("$!")

    wait_health "$port" "$name" || die "avvio di '$name' fallito (vedi $LOG_DIR/$name.log)."
done

log "tutte e tre le teste di Cerberus sono attive."
echo ""

# ---- Interrogazione ----------------------------------------------------------
log "interrogazione dei modelli con query_models.py..."
python3 "$SCRIPT_DIR/query_models.py" \
    --config "$MODELS_CONF" \
    --prompts "$PROMPTS" \
    --host "$CLIENT_HOST" \
    --out "$OUT_DIR"

echo ""
log "demo completata. Risposte in: $OUT_DIR/"
# I server vengono arrestati automaticamente da cleanup() all'uscita.
