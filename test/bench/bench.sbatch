#!/bin/bash
#
# Benchmark come job batch: avvia i server con cerberus up, misura ogni modello
# con scripts/bench.py, poi spegne tutto. Da lanciare in test/bench con l'env
# cerberus attivo (sbatch esporta l'ambiente):
#
#   UniNA:  BENCH_GPU=H100 sbatch -p ingnormal --gpus-per-node=2 --cpus-per-task=10 \
#               --mem=200G --time=03:00:00 bench.sbatch unina.conf \
#               qwen3-8b-llamacpp=unina-llamacpp qwen3-8b-vllm=unina-vllm
#   IBiSCo: BENCH_GPU=V100 sbatch -p gpus --gpus-per-node=1 --cpus-per-task=8 \
#               --time=02:00:00 bench.sbatch ibisco.conf qwen3-8b-llamacpp=ibisco-llamacpp
#
# Argomenti: <conf> <label>=<tag> ...   Variabili: CONC (default 1,4,16,32,64,128).
# Output: bench-<job>.out, up-<job>.log, risultati in bench_results.jsonl.
#
#SBATCH -J cerberus-bench
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o bench-%j.out

set -uo pipefail
CONF="${1:?uso: bench.sbatch <conf> <label>=<tag> ...}"
shift
[ "$#" -gt 0 ] || { echo "manca almeno un <label>=<tag>"; exit 1; }
CONC="${CONC:-1,4,16,32,64,128}"
cd "${SLURM_SUBMIT_DIR:-.}"
BENCH="$(cd ../.. && pwd)/scripts/bench.py"
command -v cerberus >/dev/null || { echo "attiva l'env cerberus prima di sbatch"; exit 1; }

rm -f endpoints.json
cerberus up -c "$CONF" > "up-${SLURM_JOB_ID:-local}.log" 2>&1 &
UP=$!
trap 'kill -INT $UP 2>/dev/null; wait $UP 2>/dev/null' EXIT

echo "[bench] attendo i server (vLLM: qualche minuto al primo avvio)..."
for _ in $(seq 1 180); do
    [ -f endpoints.json ] && break
    kill -0 "$UP" 2>/dev/null || { echo "[bench] cerberus up è uscito:"; tail -30 "up-${SLURM_JOB_ID:-local}.log"; exit 1; }
    sleep 10
done
[ -f endpoints.json ] || { echo "[bench] server non pronti dopo 30 minuti"; exit 1; }

for pair in "$@"; do
    label="${pair%%=*}"; tag="${pair#*=}"
    python "$BENCH" run --model "$label" --tag "$tag" --concurrency "$CONC"
done
echo "[bench] fatto: risultati in $(pwd)/bench_results.jsonl"
