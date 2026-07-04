#!/usr/bin/env bash
#
# build_llamacpp_local.sh
# Costruisce l'immagine Singularity/Apptainer di llama.cpp (CUDA) a partire
# dal file .def presente NELLA STESSA directory di questo script.
#
# Uso:
#   ./build_llamacpp_local.sh [file.def] [output.sif]
# Default:
#   file.def   -> llamacpp_local.def
#   output.sif -> llamacpp_local.sif
#
set -euo pipefail

# --- Lavora sempre nella directory dello script (path relativi) -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Parametri --------------------------------------------------------------
DEF_FILE="${1:-llamacpp_local.def}"
OUTPUT="${2:-llamacpp_local.sif}"

if [[ ! -f "$DEF_FILE" ]]; then
    echo "ERRORE: file di definizione '$DEF_FILE' non trovato in $SCRIPT_DIR" >&2
    echo "Passa il nome corretto: ./build_llamacpp_local.sh nome.def" >&2
    exit 1
fi

# --- Seleziona il binario disponibile (apptainer o singularity) -------------
if command -v apptainer >/dev/null 2>&1; then
    SINGBIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
    SINGBIN=singularity
else
    echo "ERRORE: né 'apptainer' né 'singularity' trovati nel PATH." >&2
    exit 1
fi
echo ">> Uso: $SINGBIN"

# --- TMPDIR locale, per NON saturare /tmp (spesso tmpfs in RAM) --------------
# Vive nella directory dello script (stesso filesystem della home, con spazio).
BUILD_TMP="$SCRIPT_DIR/.build-tmp"
mkdir -p "$BUILD_TMP"
export APPTAINER_TMPDIR="$BUILD_TMP"
export SINGULARITY_TMPDIR="$BUILD_TMP"
echo ">> TMPDIR build: $BUILD_TMP"

# --- Fakeroot se non root (il %post fa apt-get install -> serve privilegio) --
BUILD_OPTS=()
if [[ "$(id -u)" -ne 0 ]]; then
    BUILD_OPTS+=(--fakeroot)
    echo ">> Non-root: aggiungo --fakeroot"
fi

# --- Build -------------------------------------------------------------------
echo ">> Costruzione: $OUTPUT  <-  $DEF_FILE"
if ! "$SINGBIN" build "${BUILD_OPTS[@]}" "$OUTPUT" "$DEF_FILE"; then
    echo "" >&2
    echo "ERRORE: build fallita." >&2
    echo "Se il problema è di privilegi, riprova con sudo:" >&2
    echo "    sudo $SINGBIN build $OUTPUT $DEF_FILE" >&2
    echo "(in tal caso NON serve --fakeroot)." >&2
    exit 1
fi

# --- Pulizia temp e riepilogo -----------------------------------------------
rm -rf "$BUILD_TMP"
echo ""
echo ">> Fatto. Immagine creata: $SCRIPT_DIR/$OUTPUT"
echo ""
echo "   Esecuzione (ricordati --nv per esporre il driver):"
echo "     $SINGBIN run --nv ./$OUTPUT -m modello.gguf -ngl 99 --host 0.0.0.0 --port 8080"