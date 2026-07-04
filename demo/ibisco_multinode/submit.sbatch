#!/usr/bin/env bash
#
# Cerberus multinode — esempio di job SLURM: 2 nodi, 3 GPU per nodo.
#
# Adattare partizione e tempi alla configurazione di IBiSCo, quindi:
#   sbatch submit.sbatch
# oppure, in interattivo (lo script va poi eseguito su un nodo di calcolo):
#   salloc --partition=gpus --nodes=2 --ntasks-per-node=1 --gpus-per-node=3 \
#          --cpus-per-task=8 --time=02:00:00
#
#SBATCH --job-name=cerberus-mn
#SBATCH --partition=gpus
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
# In un job batch SLURM copia questo script in una spool dir ed esegue la copia:
# BASH_SOURCE punterebbe lì, non alla cartella di sottomissione. Si usa quindi
# SLURM_SUBMIT_DIR (impostata da SLURM alla dir da cui si è lanciato sbatch) per
# ritrovare gli script compagni; fallback a BASH_SOURCE per l'esecuzione diretta.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Se la cache HF è su storage dedicato, impostarla qui (vedi guides/llm_download.md):
# export HF_HOME=/ibiscostorage/$USER/hf_cache

# Il resto (avvio dei 4 server sui 2 nodi e interrogazione) è nello script master.
bash "$SCRIPT_DIR/run_demo_multinode.sh"
