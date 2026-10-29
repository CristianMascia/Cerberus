#!/bin/bash
# Crea l'env vllm-base e prova la catena degli esperimenti offline, come job batch.
#   cd ~/Projects/Cerberus && sbatch envs/setup_vllm_base.sbatch
# Log in vllm-base-setup-<job>.out nella cartella da cui lanci sbatch. Se l'env esiste già, salta la creazione.
#SBATCH -J vllm-base-setup
#SBATCH -p ingnormal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o %x-%j.out

set -euo pipefail
REPO="${REPO:-$HOME/Projects/Cerberus}"
CONDA="${CONDA:-$HOME/miniforge3/bin/conda}"
BASE="$HOME/miniforge3/envs/vllm-base"
EXP="${EXP:-exp-prova}"

echo "[setup] $(date -Iseconds) su $(hostname)"
if [ -x "$BASE/bin/python" ]; then
    echo "[setup] $BASE esiste già: salto la creazione"
else
    "$CONDA" env create -f "$REPO/envs/vllm-base.yml"
fi
"$BASE/bin/python" -c "import vllm, torch; print('[setup]', vllm.__version__, torch.__version__, 'cuda', torch.cuda.is_available())"

# primo esperimento + smoke test (vLLM offline, Qwen2.5-0.5B)
if [ ! -d "$HOME/experiments/$EXP" ]; then
    bash "$REPO/scripts/new-exp.sh" "$EXP"
fi
source "$HOME/experiments/$EXP/env.sh"
cd "$HOME/experiments/$EXP"
python scripts/smoke.py
echo "[setup] fatto $(date -Iseconds)"
