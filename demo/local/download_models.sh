#!/usr/bin/env bash
#
# Cerberus — download dei modelli della demo locale.
#
# Scarica dalla cache di Hugging Face i tre modelli usati dalla demo, nella sola
# quantizzazione lanciata (i pattern sono letti direttamente da models.conf, così
# restano allineati alla demo). I file finiscono nella cache HF standard, pronti
# per run_demo_local.sh; nessun --local-dir (vedi guides/llm_download.md).
#
# Variabili d'ambiente:
#   HF_HOME   root della cache HF (default: ~/.cache/huggingface)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_CONF="$SCRIPT_DIR/models.conf"

log() { echo "[cerberus] $*"; }
die() { echo "[cerberus] ERRORE: $*" >&2; exit 1; }

# ---- Individuazione della CLI di Hugging Face --------------------------------
if command -v hf >/dev/null 2>&1; then
    HF=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF=(huggingface-cli)
else
    die "CLI di Hugging Face non trovata. Installare con: pip install -U huggingface_hub"
fi

[ -f "$MODELS_CONF" ] || die "manca $MODELS_CONF"
log "cache HF: ${HF_HOME:-$HOME/.cache/huggingface}"

# ---- Download di ciascun modello dalla configurazione ------------------------
n=0
while IFS= read -r raw; do
    line="${raw%%#*}"
    [ -z "${line//[[:space:]]/}" ] && continue
    IFS=';' read -r name repo glob port ctx maxtok <<< "$line"
    name="$(echo "$name" | xargs)"; repo="$(echo "$repo" | xargs)"
    glob="$(echo "$glob" | xargs)"
    [ -z "$name" ] && continue

    # models--<org>--<repo>  ->  <org>/<repo>  (il primo '--' separa org e repo)
    repo_id="${repo#models--}"
    repo_id="${repo_id/--//}"

    log "-> $name : $repo_id  (include: $glob)"
    "${HF[@]}" download "$repo_id" --include "$glob"
    n=$((n + 1))
done < <(grep -v '^[[:space:]]*#' "$MODELS_CONF" | grep -v '^[[:space:]]*$')

log "completato: $n modelli scaricati nella cache HF."
log "ora si può lanciare ./run_demo_local.sh"
