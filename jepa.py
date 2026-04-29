"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        residual_flow=None,
        residual_scale=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.residual_flow = residual_flow

        if residual_scale is None:
            residual_scale = torch.ones(1)
        self.register_buffer("residual_scale", residual_scale.float())
        self.register_buffer(
            "residual_scale_initialized", torch.tensor(False, dtype=torch.bool)
        )

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def residual_condition(self, emb, act_emb, pred_emb):
        """Build the conditioning vector for the latent residual flow."""
        return torch.cat([emb, act_emb, pred_emb], dim=-1)

    @torch.no_grad()
    def update_residual_scale(self, residual, decay: float = 0.99, eps: float = 1e-3):
        """Track a diagonal residual whitening scale with an EMA."""
        batch_scale = residual.detach().flatten(0, -2).std(dim=0, unbiased=False)
        batch_scale = batch_scale.clamp_min(eps).to(self.residual_scale)

        if not bool(self.residual_scale_initialized.item()):
            self.residual_scale.copy_(batch_scale)
            self.residual_scale_initialized.fill_(True)
        else:
            self.residual_scale.mul_(decay).add_(batch_scale, alpha=1.0 - decay)
            self.residual_scale.clamp_(min=eps)

    def normalize_residual(self, residual, eps: float = 1e-3):
        scale = self.residual_scale.to(device=residual.device, dtype=residual.dtype)
        return residual / scale.clamp_min(eps)

    def sample_residual(self, pred_emb, condition, steps: int = 8, noise=None):
        """Euler-sample a normalized residual and unwhiten it."""
        if self.residual_flow is None:
            raise RuntimeError("Residual flow is not attached to this JEPA model.")

        z = torch.randn_like(pred_emb) if noise is None else noise
        dt = 1.0 / steps
        for i in range(steps):
            tau = torch.full(
                (*z.shape[:-1], 1),
                (i + 0.5) * dt,
                device=z.device,
                dtype=z.dtype,
            )
            z = z + dt * self.residual_flow(tau, z, condition)

        scale = self.residual_scale.to(device=z.device, dtype=z.dtype)
        return z * scale.clamp_min(1e-3)

    def predict_stochastic(self, emb, act_emb, steps: int = 8, noise=None):
        pred_emb = self.predict(emb, act_emb)
        condition = self.residual_condition(emb, act_emb, pred_emb)
        return pred_emb + self.sample_residual(pred_emb, condition, steps, noise)

    ####################
    ## Inference only ##
    ####################

    def rollout(
        self,
        info,
        action_sequence,
        history_size: int = 3,
        stochastic: bool = False,
        flow_steps: int = 8,
    ):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            if stochastic and self.residual_flow is not None:
                pred_emb = self.predict_stochastic(
                    emb_trunc, act_trunc, steps=flow_steps
                )[:, -1:]
            else:
                pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        if stochastic and self.residual_flow is not None:
            pred_emb = self.predict_stochastic(
                emb_trunc, act_trunc, steps=flow_steps
            )[:, -1:]
        else:
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
