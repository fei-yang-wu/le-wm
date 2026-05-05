# Next Steps

Last updated: 2026-05-05

## Now

1. Let the same-budget vanilla PushT job finish and record its metrics.
2. Re-run residual-flow training after the `time_scale=1000` embedding fix and
   `detach_condition=true` default.
3. Evaluate the new checkpoint with the same residual-distribution script.

## Next Experiment Batch

1. Compare the completed vanilla run against the old 1-epoch residual-flow run.
2. Evaluate both the old 1-epoch checkpoint and the new checkpoint with the same
   evaluation script.
3. If the flow is competitive on covariance/calibration, start a 10-epoch
   residual-flow run.

## Implementation Backlog

- Add a shape-sanity test for `JEPA.residual_condition` with several
  `history_size` and `num_preds` settings.
- Add an optional full-covariance Gaussian oracle baseline for analysis only.
- Add PCA or UMAP plots of held-out residuals versus samples.
- Expose stochastic rollout in evaluation/planning configs.
- Add weak metrics tied to task cost, such as goal distance and constraint
  violation proxies.

## Decision Points

- Keep `detach_condition=true` for the first fair comparison, then ablate joint
  gradients later.
- Treat `time_scale=1000` as the default for new runs; old object checkpoints
  should retain their serialized behavior through the compatibility fallback.
- Do not claim control improvement until prediction-distribution metrics and
  vanilla baseline comparisons are in place.
