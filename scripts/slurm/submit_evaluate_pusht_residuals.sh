#!/usr/bin/env bash
set -euo pipefail

SLURM_BIN="${SLURM_BIN:-/opt/slurm/Ubuntu-20.04/current/bin}"
REPO_DIR="${REPO_DIR:-$HOME/flash/Research/WM}"
JOB_SCRIPT="$REPO_DIR/scripts/slurm/evaluate_pusht_residuals.sbatch"

cd "$REPO_DIR"
mkdir -p logs data/eval

"$SLURM_BIN/sbatch" "$@" "$JOB_SCRIPT"
