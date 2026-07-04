#!/usr/bin/env bash
#
# Cerberus multinode — esempio di job SLURM: 2 nodi, 3 GPU per nodo.
#
# Adattare partizione e tempi alla configurazione di IBiSCo, quindi:
#   sbatch submit.sbatch
# oppure, in interattivo:
#   salloc --nodes=2 --ntasks-per-node=1 --gres=gpu:3 --time=01:00:00
#   ./run_demo_multinode.sh
#
#SBATCH --job-name=cerberus-mn
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Se la cache HF è su storage dedicato, impostarla qui (vedi guides/llm_download.md):
# export HF_HOME=/ibiscostorage/$USER/hf_cache

# Il resto (avvio dei 4 server sui 2 nodi e interrogazione) è nello script master.
bash "$SCRIPT_DIR/run_demo_multinode.sh"
