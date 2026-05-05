# Project Status

Last updated: 2026-05-05

## Repository

- Local repo: `/Users/fwu/Documents/New project 2/le-wm`
- Branch: `latent-residual-flow`
- Fork remote: `git@github.com:fei-yang-wu/le-wm.git`
- Sky1 clone: `fwu91@sky1:~/flash/Research/WM`

## Implemented

- Optional latent residual flow in `residual_flow.py`.
- JEPA residual helpers in `jepa.py`:
  - residual condition construction,
  - EMA residual scale,
  - normalized residual sampling,
  - optional stochastic rollout.
- Joint residual flow-matching loss in `train.py`.
- Hydra config block under `loss.residual_flow` in `config/train/lewm.yaml`.
- Slurm helpers for import smoke tests, PushT residual-flow training, vanilla
  PushT training, and residual distribution evaluation.

## Cluster and Data

- Slurm defaults: `partition=wu-lab`, `qos=short`, `gpus-per-node=a40:1`,
  `cpus-per-task=6`.
- Working fallback when `wu-lab` is busy:
  `--partition=overcap --account=overcap`.
- Project data root on Sky1: `~/flash/Research/WM/data`.
- PushT data: `data/pusht_expert_train.h5`.
- Python environment on Sky1: `.venv` with `stable-worldmodel[train]` and
  `datasets==2.21.0`.

## Completed Runs

Tiny residual-flow smoke:

- Dataset: PushT.
- Scope: 1 epoch, 2 train batches, 1 validation batch.
- Result: completed on overcap A40 and saved object/weights checkpoints.

One-epoch residual-flow run:

- Slurm job: `3030237`.
- Result: completed in `01:42:04` with exit code `0:0`.
- Checkpoints:
  - `data/pusht_rflow_1epoch/lewm_rflow_pusht_1epoch_epoch_1_object.ckpt`
  - `data/pusht_rflow_1epoch/lewm_rflow_pusht_1epoch_weights.ckpt`
- Final validation metrics from the training log:
  - `validate/loss: 1.5134869813919067`
  - `validate/pred_loss: 0.06748738884925842`
  - `validate/residual_fm_loss: 1.2505505084991455`
  - `validate/sigreg_loss: 2.1714375019073486`

## Current Focus

M2 evaluation is active. We need to measure whether the learned flow samples
held-out latent residuals better than the diagonal Gaussian baseline induced by
the residual scale.

## Open Risks

- The 1-epoch checkpoint was trained before the new `time_scale=1000` time
  embedding default. The code keeps old object checkpoints loadable by falling
  back to `time_scale=1.0` when the attribute is missing.
- A 1-epoch run is only a smoke-quality model. It can validate the plumbing and
  basic metrics, but not a research claim.
- Vanilla PushT baseline comparison is still needed before interpreting control
  or prediction improvements.
