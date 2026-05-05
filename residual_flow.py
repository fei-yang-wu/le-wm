"""Conditional flow-matching residual model for LeWM latents."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Embed scalar flow time values in [0, 1].

    `time_scale` rescales tau before the standard transformer/diffusion
    sinusoidal embedding so that tau in [0, 1] spans the same effective
    frequency range as integer timesteps in [0, time_scale]. Without it,
    `max_period=10000` collapses the high-frequency channels because
    `tau * freq` never exceeds 1.
    """

    def __init__(
        self,
        dim: int,
        max_period: float = 10000.0,
        time_scale: float = 1000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.time_scale = time_scale

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        if tau.ndim == 0:
            tau = tau[None]
        if tau.size(-1) != 1:
            tau = tau.unsqueeze(-1)

        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=tau.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        # Older object checkpoints were saved before `time_scale` existed.
        # Preserve their original behavior when loading them for evaluation.
        time_scale = getattr(self, "time_scale", 1.0)
        angles = tau.float() * time_scale * freqs
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)

        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return emb.to(dtype=tau.dtype)


class ResidualFlow(nn.Module):
    """MLP vector field v_theta(tau, z_tau, condition)."""

    def __init__(
        self,
        *,
        residual_dim: int,
        condition_dim: int,
        hidden_dim: int = 512,
        depth: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.residual_dim = residual_dim
        self.condition_dim = condition_dim
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        input_dim = residual_dim + condition_dim + time_dim
        layers = []
        for i in range(depth):
            layers.extend(
                [
                    nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, residual_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        tau: torch.Tensor,
        z_tau: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        tau_emb = self.time_embed(tau).expand(*z_tau.shape[:-1], -1)
        x = torch.cat([z_tau, condition, tau_emb], dim=-1)
        return self.out(self.net(x))
