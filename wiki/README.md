# LeWM Residual-Flow Wiki

This wiki is the compact context front door for the Flow-Matched Residual
Kernels work in this fork. It should stay short enough that a future agent or
human can read it before making changes.

## Read Order

1. `wiki/status.md` - current repo, cluster, data, and checkpoint state.
2. `wiki/evaluation.md` - how we judge whether the residual kernel is useful.
3. `wiki/next-steps.md` - the next concrete implementation and experiment tasks.
4. `docs/project-plan.md` - fuller project plan and experiment log.
5. `docs/sky1-setup.md` - cluster-specific setup and Slurm details.

## Source of Truth

- Code behavior comes from the source files, not the wiki.
- Experiment paths and metrics should be copied from logs or JSON outputs.
- If a claim is uncertain, write it under open questions instead of presenting it
  as fact.
- Keep `AGENTS.md` for operating instructions and this wiki for project memory.

## Project in One Sentence

We are extending LeWorldModel with an optional conditional flow-matching model
over latent prediction residuals, so a deterministic latent transition can be
sampled as a stochastic transition kernel.
