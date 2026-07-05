#!/usr/bin/env bash
#
# Cerberus multinode demo — SLURM batch job: 2 nodes x 3 GPUs.
#   sbatch submit.sbatch
#
#SBATCH --job-name=cerberus-demo
#SBATCH --partition=gpus
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
# SLURM runs a copy of this script from a spool dir; use SLURM_SUBMIT_DIR to find
# the demo files.
DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Activate the cerberus env (adjust to your setup) and set HF_TOKEN before submit:
#   source /nfsexports/SOFTWARE/anaconda3.OK/setupconda.sh && conda activate cerberus
# export HF_TOKEN=hf_...

mkdir -p "$DIR/logs"
bash "$DIR/run_demo.sh"
