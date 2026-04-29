# Flow-Matched Residual Kernels Plan

Last updated: 2026-04-29

## Goal

Convert LeWorldModel from a deterministic latent transition model into a
stochastic, control-compatible transition kernel by jointly training:

```text
deterministic latent predictor + conditional flow-matched residual model
```

In latent form:

```math
z_{t+1} = \Phi_\psi(z_{\le t}, u_{\le t}) + S T_\theta^{z_{\le t},u_{\le t}}(\epsilon)
```

where LeWM provides the nominal predictor `Phi_psi`, and the new residual flow
learns a distribution over normalized latent residuals.

## Current Status

- Fork remote is set to `git@github.com:fei-yang-wu/le-wm.git`.
- Working branch is `latent-residual-flow`.
- Branch was pushed to the fork.
- Commit `ca701a0` added the first latent residual-flow training scaffold.
- Commit `7b23d64` added project instructions and this planning tracker.
- `sky1` clone path: `fwu91@sky1:~/flash/Research/WM`.
- `sky1` setup notes: `docs/sky1-setup.md`.
- Project data root: `~/flash/Research/WM/data`.
- Default Slurm target: `partition=wu-lab`, `qos=short`, `gpus-per-node=a40:1`,
  `cpus-per-task=6`.
- Overcap fallback: submit with `--partition=overcap --account=overcap` when
  `wu-lab` is busy.
- Vanilla LeWM behavior remains the default because residual flow is disabled in
  config unless `loss.residual_flow.enabled=true`.

## Implemented

- Added `residual_flow.py` with:
  - sinusoidal flow-time embedding,
  - conditional MLP vector field,
  - zero-initialized output head for stable startup.
- Extended `JEPA` in `jepa.py` with:
  - optional `residual_flow`,
  - EMA diagonal `residual_scale`,
  - residual normalization,
  - Euler residual sampling,
  - optional stochastic rollout path.
- Extended `train.py` with:
  - flow-matching residual loss,
  - residual scale updates,
  - residual flow construction from Hydra config,
  - joint objective `LeWM loss + lambda_fm * residual_fm_loss`.
- Added `config/train/lewm.yaml` options under `loss.residual_flow`.
- Added Slurm helpers under `scripts/slurm/` for PushT residual-flow training.
- Added `scripts/data/download_pusht.sh` for project-local PushT data setup.

## Verification So Far

- `python -m compileall jepa.py train.py residual_flow.py` passes.
- A minimal tensor smoke test for `ResidualFlow` passed after installing
  train-only dependencies.
- Full `train.py` import was blocked by dependency/version friction in
  `stable-pretraining` and `datasets` on the local Mac environment.
- Full `train.py` import passes on `sky1` after upgrading to
  `datasets==2.21.0`.
- PushT dataset is downloaded under `data/pusht_expert_train.h5`.
- Slurm smoke import job passed on overcap A40.
- Tiny residual-flow training smoke passed on overcap A40.

## Recommended Compute

LeWM reports training on a single NVIDIA L40S GPU. For this project:

- Preferred: 1x L40S, A100, A6000, or similar 40-48GB GPU.
- Minimum first attempt: 1x 24GB GPU with smaller batch size.
- Suggested first dataset: PushT.
- Suggested first run: 10 epochs, residual flow enabled.

Example:

```bash
python train.py data=pusht \
  trainer.max_epochs=10 \
  loss.residual_flow.enabled=true \
  wandb.enabled=false
```

For 24GB GPUs:

```bash
python train.py data=pusht \
  trainer.max_epochs=10 \
  loader.batch_size=32 \
  loss.residual_flow.enabled=true \
  wandb.enabled=false
```

## Milestones

### M0: Reproducible Setup

- Clone fork on a GPU/Linux machine.
- Install train dependencies.
- Download or mount LeWM datasets under `$STABLEWM_HOME`.
- Confirm vanilla LeWM training starts.
- Confirm residual-flow training starts.

### M1: First Valid Training Run

- Train vanilla LeWM on PushT for the same budget.
- Train LeWM + latent residual flow on PushT.
- Log training losses:
  - `pred_loss`
  - `sigreg_loss`
  - `residual_fm_loss`
  - total loss
- Save checkpoint paths and commit hash.

### M2: Prediction Distribution Evaluation

- Build an evaluation script for held-out latent residuals.
- Compare:
  - deterministic residual magnitude,
  - Gaussian residual baseline,
  - latent residual flow samples.
- Metrics:
  - latent MSE,
  - residual covariance match,
  - sample coverage,
  - weak metrics such as expected goal distance or contact indicators if
    available.

### M3: Stochastic Planning

- Add a config switch for stochastic rollout during planning.
- Compare deterministic CEM against stochastic expected-cost CEM.
- Try risk-sensitive variants:
  - mean cost,
  - mean plus variance,
  - CVaR-like elite cost.

### M4: Joint-Training Ablations

- `detach_residual_target=true` versus `false`.
- `detach_condition=true` versus `false`.
- residual-flow loss weight sweep.
- EMA residual scale versus fixed precomputed scale.
- one-step residual kernel versus multi-horizon residual kernels.

### M5: Research-Grade Results

- Run across PushT, Cube, Reacher, and TwoRoom if compute allows.
- Compare against:
  - deterministic LeWM,
  - Gaussian residual model,
  - mixture residual model if implemented,
  - diffusion residual model if implemented.
- Report:
  - prediction quality,
  - stochastic calibration,
  - MPC return/success,
  - planning latency,
  - number of function evaluations.

## Near-Term TODO

- Resolve GPU-machine setup and dependency versions.
- Confirmed `stable-worldmodel[train]` dependency install on `sky1` completes.
- Confirmed `datasets>=2.20,<3` is needed on top of the base train install.
- Download PushT into `data/pusht_expert_train.h5`.
- Submit a tiny Slurm smoke job on `sky1`.
- Add a dedicated residual evaluation script.
- Add a deterministic Gaussian residual baseline.
- Expose stochastic planning in `get_cost` or evaluation config.
- Decide whether to keep flow targets detached for the first real run.

## Experiment Log Template

```text
Date:
Commit:
Machine/GPU:
Dataset:
Command:
Config overrides:
Checkpoint:
Training result:
Evaluation result:
Notes:
Next action:
```

## Experiment Log

```text
Date: 2026-04-29
Commit: fe29db85
Machine/GPU: sky1 Slurm overcap, node shakey, 1x NVIDIA A40
Dataset: PushT, data/pusht_expert_train.h5
Command:
  MAX_EPOCHS=1 BATCH_SIZE=16 NUM_WORKERS=2 \
  EXTRA_OVERRIDES="+trainer.limit_train_batches=2 +trainer.limit_val_batches=1 output_model_name=lewm_rflow_smoke subdir=smoke_pusht_rflow" \
  scripts/slurm/submit_pusht_residual_flow.sh --partition=overcap --account=overcap --time=00:30:00
Checkpoint:
  data/smoke_pusht_rflow/lewm_rflow_smoke_epoch_1_object.ckpt
  data/smoke_pusht_rflow/lewm_rflow_smoke_weights.ckpt
Training result:
  Completed 2 train batches and 1 validation batch.
  fit/loss: 35.2742
  fit/pred_loss: 0.2305
  fit/residual_fm_loss: 34.5769
  validate/loss: 9.3510
  validate/pred_loss: 0.0654
  validate/residual_fm_loss: 8.6646
Notes:
  First job failed because new Hydra trainer keys require +trainer.* syntax.
  Resubmitted with +trainer.limit_train_batches and +trainer.limit_val_batches.
Next action:
  Submit a short but real PushT residual-flow run, then add evaluation tooling.
```

## Key Design Decisions

- Start in latent space, not pixels.
- Keep residual flow optional and off by default.
- Use diagonal EMA residual scale first.
- Train the stochastic model with flow matching, not likelihood.
- Treat stochastic planning as a later milestone after training and residual
  sampling are verified.
