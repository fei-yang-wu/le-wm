# sky1 Setup

The project is cloned on the skynet login node at:

```text
fwu91@sky1:~/flash/Research/WM
```

This is the repository root, not a parent folder containing another `le-wm`
directory.

## Current Remote Setup

- Branch: `latent-residual-flow`
- Python environment: `.venv`
- Python version: uv-managed CPython 3.10.20
- Slurm binaries are available under:

```text
/opt/slurm/Ubuntu-20.04/current/bin
```

Important commands:

```bash
/opt/slurm/Ubuntu-20.04/current/bin/srun
/opt/slurm/Ubuntu-20.04/current/bin/squeue
/opt/slurm/Ubuntu-20.04/current/bin/sbatch
```

## Environment Setup

From `~/flash/Research/WM`:

```bash
uv venv --python=3.10
uv pip install 'stable-worldmodel[train]'
uv pip install 'datasets>=2.20,<3'
```

The explicit `datasets` upgrade fixes `stable-pretraining` imports. Without it,
uv may resolve `datasets==1.1.1`, which fails with:

```text
ImportError: cannot import name 'config' from 'datasets'
```

Use flash storage for datasets, checkpoints, and caches:

```bash
export STABLEWM_HOME=$HOME/flash/Research/stable-wm
export XDG_CACHE_HOME=$HOME/flash/Research/.cache
export HF_HOME=$XDG_CACHE_HOME/huggingface
export MPLCONFIGDIR=$XDG_CACHE_HOME/matplotlib
```

## Submitting a First PushT Job

Submit a small import/GPU smoke job first:

```bash
scripts/slurm/submit_smoke_import.sh
```

Submit the default residual-flow run:

```bash
scripts/slurm/submit_pusht_residual_flow.sh
```

Use a smaller batch size for lower-memory GPUs:

```bash
BATCH_SIZE=32 scripts/slurm/submit_pusht_residual_flow.sh
```

Pass Slurm options before the script path via the submit helper:

```bash
scripts/slurm/submit_pusht_residual_flow.sh --partition=gpu --time=06:00:00
```

Pass extra Hydra overrides through `EXTRA_OVERRIDES`:

```bash
EXTRA_OVERRIDES='loss.residual_flow.weight=0.3 output_model_name=lewm_rflow_w03' \
  scripts/slurm/submit_pusht_residual_flow.sh
```

Check jobs:

```bash
/opt/slurm/Ubuntu-20.04/current/bin/squeue -u "$USER"
```

Logs are written under:

```text
logs/
```

## Development Workflow

Use the login node for:

- cloning and pulling code,
- installing dependencies,
- preparing datasets/checkpoints,
- submitting and monitoring jobs,
- light smoke tests.

Use compute jobs for:

- training,
- evaluation rollouts,
- any GPU-heavy run.

Avoid long training commands directly on the login node.
