#!/usr/bin/env python
"""Evaluate latent residual samples against held-out LeWM residuals."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from utils import get_column_normalizer, get_img_preprocessor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Object checkpoint path. Relative paths are resolved from STABLEWM_HOME.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Training config path. Defaults to config.yaml beside the checkpoint.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--device", default="cuda", help="Evaluation device.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--quantile-bins", type=int, default=10)
    return parser.parse_args()


def cache_dir():
    return Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))


def resolve_path(path_like, *, base=None):
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    candidates = []
    if base is not None:
        candidates.append(Path(base) / path)
    candidates.append(Path.cwd() / path)
    candidates.append(cache_dir() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_config(args, checkpoint):
    if args.config is not None:
        config = resolve_path(args.config)
        return OmegaConf.load(config), config

    config = checkpoint.parent / "config.yaml"
    if not config.exists():
        raise FileNotFoundError(
            f"No config supplied and no config.yaml found beside {checkpoint}"
        )
    return OmegaConf.load(config), config


def make_val_loader(cfg, args):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(
            source="pixels", target="pixels", img_size=cfg.get("img_size", 224)
        )
    ]

    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    generator = torch.Generator().manual_seed(args.seed)
    _, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.get("train_split", 0.9), 1 - cfg.get("train_split", 0.9)],
        generator=generator,
    )

    return torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def to_device(batch, device):
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def covariance(x):
    if x.size(0) < 2:
        return torch.zeros(x.size(1), x.size(1), device=x.device, dtype=x.dtype)
    centered = x - x.mean(dim=0, keepdim=True)
    return centered.T @ centered / (x.size(0) - 1)


def relative_frobenius(a, b, eps=1e-8):
    return (a - b).norm(p="fro") / b.norm(p="fro").clamp_min(eps)


def quantile_ece(samples, target, bins):
    ranks = (samples <= target.unsqueeze(0)).float().mean(dim=0).flatten()
    hist = torch.histc(ranks, bins=bins, min=0.0, max=1.0)
    freq = hist / hist.sum().clamp_min(1.0)
    expected = torch.full_like(freq, 1.0 / bins)
    return (freq - expected).abs().mean()


def interval_coverage(samples, target, lo=0.05, hi=0.95):
    q_lo = torch.quantile(samples, lo, dim=0)
    q_hi = torch.quantile(samples, hi, dim=0)
    return ((target >= q_lo) & (target <= q_hi)).float().mean()


def sample_normalized_flow_residuals(model, pred_emb, condition, args):
    sample_shape = (args.num_samples, *pred_emb.shape)
    noise = torch.randn(sample_shape, device=pred_emb.device, dtype=pred_emb.dtype)

    flat_noise = noise.flatten(0, 1)
    flat_pred = pred_emb.unsqueeze(0).expand(sample_shape).flatten(0, 1)
    flat_condition = (
        condition.unsqueeze(0)
        .expand(args.num_samples, *condition.shape)
        .flatten(0, 1)
    )

    residual = model.sample_residual(
        flat_pred,
        flat_condition,
        steps=args.flow_steps,
        noise=flat_noise,
    )
    scale = model.residual_scale.to(device=residual.device, dtype=residual.dtype)
    normalized = residual / scale.clamp_min(1e-8)
    return normalized.reshape(args.num_samples, -1, normalized.size(-1))


def evaluate(args):
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    checkpoint = resolve_path(args.checkpoint, base=cache_dir())
    cfg, config_path = load_config(args, checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = torch.load(checkpoint, map_location=device, weights_only=False)
    model = model.to(device).eval()

    loader = make_val_loader(cfg, args)
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds

    target_chunks = []
    flow_chunks = []
    gaussian_chunks = []
    pred_mse = []
    norm_residual_mse = []
    fm_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break

            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            batch = to_device(batch, device)
            output = model.encode(batch)

            emb = output["emb"]
            act_emb = output["act_emb"]
            ctx_emb = emb[:, :ctx_len]
            ctx_act = act_emb[:, :ctx_len]
            tgt_emb = emb[:, n_preds:]
            pred_emb = model.predict(ctx_emb, ctx_act)

            residual = tgt_emb - pred_emb
            target = model.normalize_residual(residual).flatten(0, 1)
            target_chunks.append(target.detach().cpu())
            pred_mse.append(residual.pow(2).mean().detach().cpu())
            norm_residual_mse.append(target.pow(2).mean().detach().cpu())

            gaussian = torch.randn(
                args.num_samples,
                target.size(0),
                target.size(1),
                device=device,
                dtype=target.dtype,
            )
            gaussian_chunks.append(gaussian.detach().cpu())

            if model.residual_flow is not None:
                condition = model.residual_condition(ctx_emb, ctx_act, pred_emb)
                flow = sample_normalized_flow_residuals(model, pred_emb, condition, args)
                flow_chunks.append(flow.detach().cpu())

                eps = torch.randn_like(target)
                tau = torch.rand(*target.shape[:-1], 1, device=device, dtype=target.dtype)
                z_tau = (1.0 - tau) * eps + tau * target
                target_velocity = target - eps
                pred_velocity = model.residual_flow(
                    tau,
                    z_tau,
                    condition.flatten(0, 1),
                )
                fm_losses.append((pred_velocity - target_velocity).pow(2).mean().cpu())

    target = torch.cat(target_chunks, dim=0)
    gaussian = torch.cat(gaussian_chunks, dim=1)

    target_cov = covariance(target)
    gaussian_cov = covariance(gaussian.flatten(0, 1))

    result = {
        "checkpoint": str(checkpoint),
        "config": str(config_path.resolve()),
        "num_targets": int(target.size(0)),
        "latent_dim": int(target.size(1)),
        "num_samples": args.num_samples,
        "flow_steps": args.flow_steps,
        "nfe": args.flow_steps,
        "deterministic": {
            "latent_mse": float(torch.stack(pred_mse).mean()),
            "normalized_residual_mse": float(torch.stack(norm_residual_mse).mean()),
        },
        "target_residual": {
            "mean_norm": float(target.mean(dim=0).norm()),
            "cov_trace": float(torch.trace(target_cov)),
        },
        "gaussian": {
            "mean_l2": float((gaussian.flatten(0, 1).mean(dim=0) - target.mean(dim=0)).norm()),
            "cov_relative_frobenius": float(relative_frobenius(gaussian_cov, target_cov)),
            "quantile_ece": float(quantile_ece(gaussian, target, args.quantile_bins)),
            "interval_90_coverage": float(interval_coverage(gaussian, target)),
            "nfe": 0,
        },
    }

    if flow_chunks:
        flow = torch.cat(flow_chunks, dim=1)
        flow_cov = covariance(flow.flatten(0, 1))
        result["flow"] = {
            "mean_l2": float((flow.flatten(0, 1).mean(dim=0) - target.mean(dim=0)).norm()),
            "cov_relative_frobenius": float(relative_frobenius(flow_cov, target_cov)),
            "quantile_ece": float(quantile_ece(flow, target, args.quantile_bins)),
            "interval_90_coverage": float(interval_coverage(flow, target)),
            "eval_fm_loss": float(torch.stack(fm_losses).mean()) if fm_losses else None,
            "nfe": args.flow_steps,
        }

    return result


def main():
    args = parse_args()
    result = evaluate(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)

    if args.output:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


if __name__ == "__main__":
    main()
