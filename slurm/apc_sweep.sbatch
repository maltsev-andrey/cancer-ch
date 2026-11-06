#!/bin/bash
#SBATCH --job-name=cancer-apc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%2
#SBATCH --output=logs/%x_%A_%a.out
#
# One APC fit per array task. Build the list first and size the array from it:
#   N=$(python pipeline/s06_apc.py --list | tail -1 | cut -d' ' -f1)
#   sbatch --array=0-$((N-1))%2 slurm/apc_sweep.sbatch                          # full fits + projection
#   EXTRA_ARGS="--holdout" sbatch --array=0-$((N-1))%2 slurm/apc_sweep.sbatch   # out-of-sample check
# %2 = at most two tasks at once on the single GPU; 4 vectorised chains each.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO/logs"
source "${CANCER_CH_VENV:-$HOME/cancer-ch-gpu}/bin/activate"   # venv with jax[cuda12] numpyro arviz

cd "$REPO"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # P100 only, never the V100
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # share the card between two tasks
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

python -u pipeline/s06_apc.py --index "$SLURM_ARRAY_TASK_ID" --future 2 --chains 4 \
       --warmup 1500 --samples 1500 ${EXTRA_ARGS:-}
