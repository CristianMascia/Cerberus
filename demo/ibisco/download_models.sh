#!/usr/bin/env bash
#
# Cerberus — download dei modelli della demo IBiSCo.
#
# Scarica dalla cache di Hugging Face i tre modelli usati dalla demo, nella sola
# quantizzazione lanciata (i pattern sono letti da models.conf, così restano
# allineati alla demo). Su IBiSCo il download richiede l'ambiente conda dedicato
# 'hf_download' (già autenticato con la API key); vedi guides/llm_download.md.
#
# IMPORTANTE: eseguire sul LOGIN NODE, l'unico con accesso alla rete esterna.
#
# Variabili d'ambiente:
#   HF_HOME       root della cache HF. Su IBiSCo la home ha quota limitata:
#                 si consiglia /ibiscostorage/$USER/hf_cache (vedi la guida).
#   CONDA_SETUP   script di setup conda
#                 (default: /nfsexports/SOFTWARE/anaconda3.OK/setupconda.sh)
#   HF_ENV        nome dell'ambiente conda (default: hf_download)
#
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_CONF="$SCRIPT_DIR/models.conf"
CONDA_SETUP="${CONDA_SETUP:-/nfsexports/SOFTWARE/anaconda3.OK/setupconda.sh}"
HF_ENV="${HF_ENV:-hf_download}"

log() { echo "[cerberus] $*"; }
die() { echo "[cerberus] ERRORE: $*" >&2; exit 1; }

[ -f "$MODELS_CONF" ] || die "manca $MODELS_CONF"

# ---- Attivazione dell'ambiente conda hf_download -----------------------------
# Gli script conda usano variabili non inizializzate: si disattiva 'set -u'
# durante il source e l'activate.
[ -f "$CONDA_SETUP" ] || die "script di setup conda non trovato: $CONDA_SETUP
       Impostare CONDA_SETUP col percorso corretto (vedi guides/llm_download.md)."

log "attivazione ambiente conda '$HF_ENV'..."
set +u
# shellcheck disable=SC1090
source "$CONDA_SETUP"
conda activate "$HF_ENV" || die "impossibile attivare l'ambiente conda '$HF_ENV'."
set -u

command -v hf >/dev/null 2>&1 || die "comando 'hf' non disponibile nell'ambiente '$HF_ENV'."

# Verifica dell'autenticazione (l'ambiente è preconfigurato con la API key).
if ! hf auth whoami >/dev/null 2>&1; then
    log "ATTENZIONE: 'hf auth whoami' non conferma l'autenticazione."
    log "Per i repository gated contattare il titolare della API key (vedi la guida)."
fi

log "cache HF: ${HF_HOME:-$HOME/.cache/huggingface}"
[ -z "${HF_HOME:-}" ] && log "Suggerimento: impostare HF_HOME=/ibiscostorage/\$USER/hf_cache (quota home limitata)."

# ---- Download di ciascun modello dalla configurazione ------------------------
n=0
while IFS= read -r raw; do
    line="${raw%%#*}"
    [ -z "${line//[[:space:]]/}" ] && continue
    IFS=';' read -r name repo glob rest <<< "$line"
    name="$(echo "$name" | xargs)"; repo="$(echo "$repo" | xargs)"
    glob="$(echo "$glob" | xargs)"
    [ -z "$name" ] && continue

    # models--<org>--<repo>  ->  <org>/<repo>  (il primo '--' separa org e repo)
    repo_id="${repo#models--}"
    repo_id="${repo_id/--//}"

    log "-> $name : $repo_id  (include: $glob)"
    hf download "$repo_id" --include "$glob"
    n=$((n + 1))
done < <(grep -v '^[[:space:]]*#' "$MODELS_CONF" | grep -v '^[[:space:]]*$')

log "completato: $n modelli scaricati nella cache HF."
log "ora, da un nodo di calcolo con GPU, si può lanciare ./run_demo_ibisco.sh"
