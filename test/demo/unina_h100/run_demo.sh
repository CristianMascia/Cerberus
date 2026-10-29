#!/usr/bin/env bash
#
# Cerberus demo (HPC UniNA, 1 nodo 2x H100) — end to end with the cerberus tool.
#
# Run this from INSIDE a SLURM allocation (the tool orchestrates via srun), e.g.:
#   conda activate cerberus
#   salloc --partition=ingnormal --nodes=1 --ntasks-per-node=1 --gpus-per-node=2 \
#          --cpus-per-task=10 --mem=200G --time=01:00:00
#   ./run_demo.sh
#
# --gpus-per-node is required: without it the job gets no GPU and the srun steps
# of `cerberus up` stay pending.
#
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# shellcheck source=../../../profiles/unina-h100.env
source "$DIR/../../../profiles/unina-h100.env"

command -v cerberus >/dev/null 2>&1 || { echo "activate the 'cerberus' env (pip install -e .)"; exit 1; }
[ -f "$CERBERUS_IMAGE" ] || { echo "image $CERBERUS_IMAGE missing: see docs/hpc_unina_h100.md"; exit 1; }

rm -f endpoints.json
cerberus download -c models.conf

cerberus up -c models.conf &
UP=$!
cleanup() { kill -INT "$UP" 2>/dev/null || true; wait "$UP" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[demo] waiting for endpoints.json ..."
for _ in $(seq 1 200); do [ -f endpoints.json ] && break; sleep 3; done
[ -f endpoints.json ] || { echo "[demo] servers did not come up (see .cerberus/*/logs)"; exit 1; }

python3 ../demo_client.py
echo "[demo] done — see outputs/. Tearing down."
