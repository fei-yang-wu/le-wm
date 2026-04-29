#!/usr/bin/env bash
#SBATCH --job-name=lewm-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/flash/Research/WM}"
cd "$REPO_DIR"

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/flash/Research/stable-wm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/flash/Research/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"

mkdir -p "$STABLEWM_HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR" logs

source .venv/bin/activate

hostname
nvidia-smi -L || true
python -c 'import torch; print("torch=", torch.__version__); print("cuda_available=", torch.cuda.is_available()); print("device_count=", torch.cuda.device_count())'
python -c 'import train; import residual_flow; print("train import ok")'
