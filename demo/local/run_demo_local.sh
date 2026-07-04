#!/usr/bin/env bash
#
# Cerberus — demo locale.
#
# Avvia in sequenza tre modelli LLM (le tre "teste") con llama.cpp su
# Singularity, ciascuno da un modello GGUF nella cache di Hugging Face montata in
# bind, poi lancia lo script Python che interroga i tre modelli in sequenza e ne
# salva le risposte. Al termine (anche in caso di errore o Ctrl-C) arresta tutti
# i server.
#
# Prerequisiti:
#   - immagine Singularity locale di llama.cpp (vedi guides/llamacpp_install.md);
#   - modelli GGUF già presenti nella cache HF (vedi guides/llm_download.md);
#   - i modelli, le porte e i prompt sono configurati in models.conf e prompts.txt.
#
# Variabili d'ambiente sovrascrivibili:
#   CERBERUS_IMAGE   percorso dell'immagine .sif (default: ../../containers/llamacpp_local.sif)
#   HF_HOME          root della cache HF        (default: ~/.cache/huggingface)
#   NGL              layer in offload su GPU     (default: 99)
#   HEALTH_TIMEOUT   attesa max avvio server (s) (default: 180)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurazione ----------------------------------------------------------
IMAGE="${CERBERUS_IMAGE:-$SCRIPT_DIR/../../containers/llamacpp_local.sif}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB="$HF_CACHE/hub"
MODELS_CONF="$SCRIPT_DIR/models.conf"
PROMPTS="$SCRIPT_DIR/prompts.txt"
OUT_DIR="$SCRIPT_DIR/outputs"
LOG_DIR="$SCRIPT_DIR/logs"
HOST="127.0.0.1"
NGL="${NGL:-99}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"

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

[ -f "$IMAGE" ] || die "immagine non trovata: $IMAGE
       Costruirla secondo guides/llamacpp_install.md, oppure impostare CERBERUS_IMAGE."
[ -d "$HF_HUB" ] || die "cache HF non trovata: $HF_HUB"
[ -f "$MODELS_CONF" ] || die "manca $MODELS_CONF"
[ -f "$PROMPTS" ] || die "manca $PROMPTS"

# Risolve il percorso host del .gguf dato repo e glob (un solo match atteso).
# -L segue i symlink: negli snapshot della cache HF i file sono link ai blob.
resolve_gguf() {
    local repo="$1" glob="$2"
    find -L "$HF_HUB/$repo/snapshots" -maxdepth 2 -type f -name "$glob" 2>/dev/null \
        | head -n1
}

# ---- Lettura configurazione e risoluzione dei modelli ------------------------
NAMES=(); PORTS=(); CTXS=(); GGUF_CONTAINER=()

log "risoluzione dei modelli nella cache HF..."
while IFS= read -r raw; do
    line="${raw%%#*}"                       # rimuove eventuali commenti in coda
    [ -z "${line//[[:space:]]/}" ] && continue

    IFS=';' read -r name repo glob port ctx maxtok <<< "$line"
    name="$(echo "$name" | xargs)"; repo="$(echo "$repo" | xargs)"
    glob="$(echo "$glob" | xargs)"; port="$(echo "$port" | xargs)"
    ctx="$(echo "$ctx" | xargs)"
    [ -z "$name" ] && continue

    host_path="$(resolve_gguf "$repo" "$glob")"
    if [ -z "$host_path" ]; then
        die "GGUF non trovato per '$name' (repo=$repo glob=$glob).
       Verificare che il file sia in cache; se serve scaricarlo vedi guides/llm_download.md."
    fi
    # Percorso all'interno del container: la cache è montata su /hf.
    rel="${host_path#"$HF_CACHE"/}"
    container_path="/hf/$rel"

    NAMES+=("$name"); PORTS+=("$port"); CTXS+=("$ctx")
    GGUF_CONTAINER+=("$container_path")
    log "  $name -> $(basename "$host_path")  (porta $port)"
done < <(grep -v '^[[:space:]]*#' "$MODELS_CONF" | grep -v '^[[:space:]]*$')

[ "${#NAMES[@]}" -gt 0 ] || die "nessun modello valido in $MODELS_CONF"

# ---- Attesa che un server sia pronto -----------------------------------------
wait_health() {
    local port="$1" name="$2" waited=0
    while : ; do
        if curl -s "http://$HOST:$port/health" 2>/dev/null | grep -q '"ok"'; then
            log "  $name pronto su http://$HOST:$port"
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
log "avvio dei ${#NAMES[@]} server (immagine: $(basename "$IMAGE"))..."
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; port="${PORTS[$i]}"; ctx="${CTXS[$i]}"
    model="${GGUF_CONTAINER[$i]}"
    log "-> avvio '$name' (porta $port, ctx $ctx)"

    "$SINGULARITY" run --nv \
        -B "$HF_CACHE":/hf \
        "$IMAGE" \
        -m "$model" \
        --host "$HOST" --port "$port" \
        -ngl "$NGL" -c "$ctx" --jinja \
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
    --host "$HOST" \
    --out "$OUT_DIR"

echo ""
log "demo completata. Risposte in: $OUT_DIR/"
# I server vengono arrestati automaticamente da cleanup() all'uscita.
