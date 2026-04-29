#!/usr/bin/env bash
#SBATCH --job-name=lewm-rflow-pusht
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/flash/Research/WM}"
cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/flash/Research/stable-wm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/flash/Research/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"

mkdir -p "$STABLEWM_HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR" logs

source .venv/bin/activate

DATA_CONFIG="${DATA_CONFIG:-pusht}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-6}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

OVERRIDES=(
  "data=$DATA_CONFIG"
  "trainer.max_epochs=$MAX_EPOCHS"
  "loader.batch_size=$BATCH_SIZE"
  "loader.num_workers=$NUM_WORKERS"
  "loss.residual_flow.enabled=true"
  "wandb.enabled=$WANDB_ENABLED"
)

if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "$EXTRA_OVERRIDES"
  OVERRIDES+=("${EXTRA_ARGS[@]}")
fi

python -c 'import torch; print("cuda_available=", torch.cuda.is_available()); print("device_count=", torch.cuda.device_count())'
python train.py "${OVERRIDES[@]}"
