# Flow-Matched Residual Kernels Plan

Last updated: 2026-05-05

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
- Working branch is `latent-residual-flow`, pushed to the fork.
- Commit `ca701a0` added the first latent residual-flow training scaffold.
- Commit `7b23d64` added project instructions and this planning tracker.
- `sky1` clone path: `fwu91@sky1:~/flash/Research/WM`. Setup notes:
  `docs/sky1-setup.md`. Project data root: `~/flash/Research/WM/data`.
- Default Slurm target: `partition=wu-lab`, `qos=short`, `gpus-per-node=a40:1`,
  `cpus-per-task=6`. Overcap fallback: `--partition=overcap --account=overcap`.
- Vanilla LeWM behavior remains the default because residual flow is disabled in
  config unless `loss.residual_flow.enabled=true`.
- **M0 complete**: sky1 install verified, PushT downloaded, smoke import +
  tiny residual-flow training run passed on overcap A40
  (see Experiment Log entry `2026-04-29 / fe29db85`).
- **M1 smoke complete**: the first full-dataset, 1-epoch residual-flow PushT
  run completed on overcap A40 and saved checkpoints. The full fair comparison
  still needs a same-budget vanilla run and a post-`time_scale` residual-flow
  rerun.
- **M2 smoke complete**: residual distribution evaluation ran on the 1-epoch
  checkpoint. The flow improved covariance matching and 90% interval coverage
  versus the diagonal Gaussian baseline, but quantile ECE was slightly worse.

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
- Added Slurm helpers under `scripts/slurm/` for vanilla PushT training.
- Added `scripts/eval/evaluate_latent_residuals.py` for held-out latent
  residual distribution metrics.
- Added `scripts/data/download_pusht.sh` for project-local PushT data setup.
- Added `wiki/` as the compact project context front door.

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
- One-epoch residual-flow PushT job `3030237` completed on overcap A40 in
  `01:42:04` with finite validation metrics and saved checkpoints.
- Residual evaluation script passes syntax checks locally.
- Residual evaluation job `3080285` completed on `wu-lab` in `00:02:05` and
  wrote `data/eval/pusht_rflow_1epoch_residual_eval.json`.

## Recommended Compute

Active development runs on sky1 (`partition=wu-lab`, A40) with overcap
fallback. See `docs/sky1-setup.md` and `scripts/slurm/`.

First real run target: PushT, 10 epochs, residual flow enabled.

```bash
python train.py data=pusht \
  trainer.max_epochs=10 \
  loss.residual_flow.enabled=true \
  wandb.enabled=false
```

Future portability (other 24GB+ GPUs, e.g., L40S/A100/A6000): cap
`loader.batch_size=32` and rely on the same Hydra overrides.

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

**Success criteria:**

- `residual_fm_loss` decreases monotonically over the run and ends below the
  initial `‖x − ε‖²` plateau (≈ 2 in normalized space) by a clear margin.
- Vanilla `pred_loss` of the joint run stays within ±10% of the LeWM-only
  baseline (no regression of the deterministic objective).
- Both runs reach the same epoch count without crashes; checkpoints loadable.

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
  - per-dim quantile calibration (ECE),
  - NFE used to draw each sample,
  - weak metrics such as expected goal distance or contact indicators if
    available.

**Success criteria:**

- Flow residual covariance Frobenius error to held-out empirical covariance
  beats the per-dim Gaussian baseline by ≥ 20%.
- Per-dim quantile ECE for the flow ≤ Gaussian baseline ECE.
- Flow samples cover the empirical residual support (no obvious mode collapse
  on a 2D PCA projection).

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

Open items only — completed setup work has moved to **Current Status / M0**.

1. Re-run the residual-flow smoke on sky1 after the
   `SinusoidalTimeEmbedding` `time_scale` fix (commit lands in
   `residual_flow.py`); confirm `residual_fm_loss` dynamics change.
2. Add a shape-sanity unit test for `JEPA.residual_condition` covering
   `num_preds ∈ {1, 2}` and `history_size ∈ {3, 4}` so the
   `[ctx_emb, ctx_act, pred_emb]` concat is verified beyond the current
   `num_preds=1, history_size=3` happy path.
3. Decide and document the default for `loss.residual_flow.detach_condition`
   for the first real M1 run. Recommended starting point:
   `detach_condition=true` for an apples-to-apples comparison vs. vanilla
   LeWM, then ablate in M4.
4. Let the same-budget vanilla PushT job finish and record its metrics.
5. Add an optional full-covariance Gaussian oracle baseline for analysis only;
   keep the diagonal Gaussian baseline as the fair deployable baseline.
6. Expose a stochastic-rollout switch in `get_cost` / evaluation config so
   M3 planning experiments can be triggered without code changes.

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
NFE (sampling):
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

```text
Date: 2026-04-29
Commit: df52ab4
Machine/GPU: sky1 Slurm overcap, node deebot, 1x NVIDIA A40
Dataset: PushT, data/pusht_expert_train.h5
Command:
  MAX_EPOCHS=1 BATCH_SIZE=128 NUM_WORKERS=6 \
  scripts/slurm/submit_pusht_residual_flow.sh --partition=overcap --account=overcap --time=02:00:00
Checkpoint:
  data/pusht_rflow_1epoch/lewm_rflow_pusht_1epoch_epoch_1_object.ckpt
  data/pusht_rflow_1epoch/lewm_rflow_pusht_1epoch_weights.ckpt
Training result:
  Completed job 3030237 in 01:42:04 with exit code 0:0.
  fit/loss: 1.474353551864624
  fit/pred_loss: 0.07256205379962921
  fit/residual_fm_loss: 1.2396820783615112
  fit/sigreg_loss: 1.796875
  validate/loss: 1.5134869813919067
  validate/pred_loss: 0.06748738884925842
  validate/residual_fm_loss: 1.2505505084991455
  validate/sigreg_loss: 2.1714375019073486
Evaluation result:
  Slurm job 3080285 completed in 00:02:05 with exit code 0:0.
  JSON: data/eval/pusht_rflow_1epoch_residual_eval.json
  num_targets: 6144
  latent_dim: 192
  deterministic.latent_mse: 0.06709294766187668
  deterministic.normalized_residual_mse: 1.0047131776809692
  flow.cov_relative_frobenius: 0.4756399989128113
  gaussian.cov_relative_frobenius: 0.8401789665222168
  flow.interval_90_coverage: 0.8628132939338684
  gaussian.interval_90_coverage: 0.8166148066520691
  flow.quantile_ece: 0.02631089650094509
  gaussian.quantile_ece: 0.024060126394033432
  flow.eval_fm_loss: 1.2551902532577515
NFE (sampling):
  8
Notes:
  This checkpoint predates the time_scale=1000 time-embedding default. Object
  checkpoint loading preserves old behavior with a compatibility fallback.
  The flow has a real smoke-level distributional signal: covariance matching
  and interval coverage beat the diagonal Gaussian baseline. Quantile ECE does
  not beat Gaussian yet, so this is encouraging but not conclusive.
Next action:
  Let the vanilla PushT baseline finish, then re-run residual-flow training with
  the time_scale=1000 embedding fix and detach_condition=true default.
```

## Key Design Decisions

- Start in latent space, not pixels.
- Keep residual flow optional and off by default.
- Use diagonal EMA residual scale first.
- Train the stochastic model with flow matching, not likelihood.
- Treat stochastic planning as a later milestone after training and residual
  sampling are verified.
- Residual kernel is **one-step** (predicts `z_{t+1} − Φ(z_{≤t}, u_{≤t})`).
  Multi-horizon consistency is an explicit open problem — see M4
  ("one-step residual kernel versus multi-horizon residual kernels").
- For the first M1 run, default to **detached residual targets** and detached
  condition so the flow objective does not perturb the deterministic
  predictor; the joint-gradient variant is an M4 ablation.
- Time-conditioning uses a sinusoidal embedding scaled for τ ∈ [0, 1]
  (`time_scale=1000` so the standard `max_period=10000` regime applies);
  raw τ without scaling collapses the high-frequency channels.
