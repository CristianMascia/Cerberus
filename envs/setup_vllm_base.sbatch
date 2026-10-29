#!/bin/bash
# Verifica l'env vllm-base (installato dal login node) e prova la catena degli
# esperimenti offline sulla GPU, come job batch.
#   ~/miniforge3/envs/vllm-base/bin/hf download Qwen/Qwen2.5-0.5B-Instruct   # login: pesi dello smoke test
#   cd ~/Projects/Cerberus && sbatch envs/setup_vllm_base.sbatch
# Log in vllm-base-setup-<job>.out nella cartella da cui lanci sbatch.
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
BASE="$HOME/miniforge3/envs/vllm-base"
EXP="${EXP:-exp-prova}"

echo "[setup] $(date -Iseconds) su $(hostname)"
if ! "$BASE/bin/python" -c "import vllm" 2>/dev/null; then
    # sui nodi di calcolo pip fallisce scrivendo su Lustre (Errno 14 Bad address):
    # l'env si installa dal login node, vedi docs/esperimenti_offline.md
    echo "[setup] in $BASE manca vllm: installalo dal login node (docs/esperimenti_offline.md, punto 1)" >&2
    exit 1
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
