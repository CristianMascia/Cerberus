#!/usr/bin/env bash
#
# Cerberus demo (multi node) — end to end with the cerberus tool.
#
# Same as the single-node demo, but the allocation spans 2 nodes and Cerberus
# auto-assigns the 4 models across them. Run inside a SLURM allocation, e.g.:
#   conda activate cerberus
#   salloc --partition=gpus --nodes=2 --ntasks-per-node=1 --gpus-per-node=3 \
#          --cpus-per-task=8 --time=01:00:00
#   ./run_demo.sh
# or non-interactively:  sbatch submit.sbatch
#
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

command -v cerberus >/dev/null 2>&1 || { echo "activate the 'cerberus' env (pip install -e .)"; exit 1; }

rm -f endpoints.json
cerberus download -c models.conf

cerberus up -c models.conf &
UP=$!
cleanup() { kill -INT "$UP" 2>/dev/null || true; wait "$UP" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[demo] waiting for endpoints.json ..."
for _ in $(seq 1 300); do [ -f endpoints.json ] && break; sleep 3; done
[ -f endpoints.json ] || { echo "[demo] servers did not come up (see .cerberus/*/logs)"; exit 1; }

python3 ../demo_client.py
echo "[demo] done — see outputs/. Tearing down."
