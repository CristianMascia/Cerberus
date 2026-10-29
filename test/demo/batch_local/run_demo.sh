#!/usr/bin/env bash
#
# Cerberus — demo locale della batch API (senza SLURM).
#
# Avvia un llama-server con Qwen2.5-Coder-0.5B via Singularity, scrive a mano un
# endpoints.json che lo descrive, poi esercita l'intero flusso di `cerberus batch`:
#
#   1. validate             controllo di requests.jsonl contro models.conf
#   2. submit --dry-run     split in 2 shard x 2 ripetizioni, manifest e job.sbatch
#                           (calcola il placement: serve rete per leggere l'header GGUF)
#   3. run --shard 0 / 1    i due shard, sul server già attivo (--endpoints)
#   4. merge                results.jsonl nell'ordine dell'input
#   5. batch_check.py       verifica dei risultati
#
# Al termine (anche in caso di errore o Ctrl-C) il server viene arrestato.
#
# Variabili d'ambiente sovrascrivibili:
#   CERBERUS_IMAGE   immagine .sif     (default: <repo>/cerberus/containers/llamacpp_local.sif)
#   HF_HOME          root cache HF     (default: ~/.cache/huggingface)
#   PYTHON           interprete        (default: <repo>/.venv/bin/python se esiste, poi python3)
#   PORT             porta del server  (default: 8090)
#   NGL              layer su GPU      (default: 99)
#   HEALTH_TIMEOUT   attesa avvio (s)  (default: 180)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---- Configurazione ----------------------------------------------------------
IMAGE="${CERBERUS_IMAGE:-$REPO/cerberus/containers/llamacpp_local.sif}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
MODELS_CONF="$SCRIPT_DIR/models.conf"
REQUESTS="$SCRIPT_DIR/requests.jsonl"
OUT_DIR="$SCRIPT_DIR/outputs"
LOG_DIR="$SCRIPT_DIR/logs"
HOST="127.0.0.1"
PORT="${PORT:-8090}"
NGL="${NGL:-99}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
LABEL="qwen-coder"
HF_REPO="ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF"
GGUF="qwen2.5-coder-0.5b-q8_0.gguf"
PARALLEL=2
CTX=$(( (1024 + 256) * PARALLEL ))   # (max_input + max_output) * parallel, come Cerberus

if [ -z "${PYTHON:-}" ]; then
    if [ -x "$REPO/.venv/bin/python" ]; then PYTHON="$REPO/.venv/bin/python"; else PYTHON=python3; fi
fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="$OUT_DIR/batch_$STAMP"
ENDPOINTS="$OUT_DIR/endpoints_$STAMP.json"
mkdir -p "$OUT_DIR" "$LOG_DIR"

log() { echo "[cerberus] $*"; }
die() { echo "[cerberus] ERRORE: $*" >&2; exit 1; }
cerberus() { "$PYTHON" -m cerberus "$@"; }

# ---- Arresto pulito del server al termine ------------------------------------
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        log "arresto del server..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ---- Preflight ---------------------------------------------------------------
SINGULARITY="$(command -v singularity || command -v apptainer || true)"
[ -n "$SINGULARITY" ] || die "singularity/apptainer non trovato nel PATH."
[ -f "$IMAGE" ] || die "immagine non trovata: $IMAGE
       Impostare CERBERUS_IMAGE."
"$PYTHON" -c "import openai" 2>/dev/null \
    || die "il pacchetto 'openai' manca per $PYTHON (pip install -e $REPO)"
command -v curl >/dev/null 2>&1 || die "curl non trovato."
if curl -s "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    die "la porta $PORT è già in uso; impostare PORT=<altra porta>."
fi

# Il GGUF nella cache HF (snapshots/*/<file>); se manca lo scarica.
find_gguf() {
    find -L "$HF_CACHE/hub/models--${HF_REPO//\//--}/snapshots" -maxdepth 2 \
        -name "$GGUF" 2>/dev/null | head -n1
}
host_path="$(find_gguf)"
if [ -z "$host_path" ]; then
    log "modello non in cache: download con 'cerberus download'..."
    cerberus download -c "$MODELS_CONF"
    host_path="$(find_gguf)"
    [ -n "$host_path" ] || die "GGUF $GGUF non trovato in $HF_CACHE dopo il download."
fi
container_path="/hf/${host_path#"$HF_CACHE"/}"

# ---- Avvio del server --------------------------------------------------------
log "avvio di '$LABEL' su http://$HOST:$PORT (ctx $CTX, parallel $PARALLEL)..."
"$SINGULARITY" run --nv \
    -B "$HF_CACHE":/hf \
    "$IMAGE" \
    -m "$container_path" \
    --host "$HOST" --port "$PORT" \
    -ngl "$NGL" -c "$CTX" -np "$PARALLEL" --jinja \
    > "$LOG_DIR/$LABEL.log" 2>&1 &
SERVER_PID=$!

waited=0
until curl -s "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"ok"'; do
    kill -0 "$SERVER_PID" 2>/dev/null || {
        tail -n 20 "$LOG_DIR/$LABEL.log" >&2; die "il server è terminato (vedi $LOG_DIR/$LABEL.log)."; }
    sleep 2; waited=$((waited + 2))
    if [ "$waited" -ge "$HEALTH_TIMEOUT" ]; then
        tail -n 20 "$LOG_DIR/$LABEL.log" >&2
        die "timeout ($HEALTH_TIMEOUT s) in attesa del server."
    fi
done
log "server pronto."

# endpoints.json nel formato scritto da 'cerberus up'
cat > "$ENDPOINTS" <<EOF
{
  "job_id": "local-demo",
  "generated_at": "$(date -Iseconds)",
  "models": {
    "$LABEL": {"base_url": "http://$HOST:$PORT/v1", "host": "$HOST", "port": $PORT,
               "reasoning": false}
  }
}
EOF

# ---- Flusso batch ------------------------------------------------------------
echo ""
log "1/5 validate"
cerberus batch validate -c "$MODELS_CONF" -i "$REQUESTS" --model "$LABEL" --schema-only

echo ""
log "2/5 submit --dry-run (2 shard x 2 ripetizioni)"
cerberus batch submit -c "$MODELS_CONF" -i "$REQUESTS" --model "$LABEL" \
    --shards 2 --repeats 2 --batch-dir "$BATCH_DIR" --dry-run

echo ""
log "3/5 run degli shard sul server locale"
for shard in 0 1; do
    cerberus batch run --batch-dir "$BATCH_DIR" --shard "$shard" --endpoints "$ENDPOINTS"
done
cerberus batch status --batch-dir "$BATCH_DIR"

echo ""
log "4/5 merge"
cerberus batch merge --batch-dir "$BATCH_DIR"

echo ""
log "5/5 controllo dei risultati"
"$PYTHON" "$SCRIPT_DIR/../batch_check.py" --batch-dir "$BATCH_DIR"

echo ""
log "demo completata. Risultati: $BATCH_DIR/results.jsonl"
