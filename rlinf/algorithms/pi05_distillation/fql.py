# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Fixed-teacher FQL, in the accepted pi05 normalized action coordinates.

Algorithm: seohongpark/fql e8cd16e, agents/fql.py. Pi05 adaptation: frozen
multitask prior; H50 endpoint anchor, C10 macro-action Q, finite-budget state.
"""

import torch
import torch.distributed as dist
from torch import nn


class TwinQ(nn.Module):
    """Independent four-layer critics; task instruction is in frozen VLM features."""

    def __init__(self, feature_dim, chunk=10, action_dim=14, width=512, depth=4):
        super().__init__()
        self.chunk, self.action_dim = chunk, action_dim
        inputs = feature_dim + chunk * action_dim + chunk + 1
        heads = []
        for _ in range(2):
            layers = []
            for i in range(depth):
                layers += [
                    nn.Linear(inputs if i == 0 else width, width),
                    nn.GELU(approximate="tanh"),
                    nn.LayerNorm(width, eps=1e-6),
                ]
            layers.append(nn.Linear(width, 1))
            heads.append(nn.Sequential(*layers))
        self.heads = nn.ModuleList(heads)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features, actions, mask, remaining):
        action = actions[:, : self.chunk, : self.action_dim].float() * mask[..., None]
        x = torch.cat(
            (
                features.float(),
                action.flatten(1),
                mask.float(),
                remaining[:, None].float(),
            ),
            -1,
        )
        return torch.cat([head(x) for head in self.heads], -1)


def prefix_mask(remaining, chunk=10):
    return (
        torch.arange(chunk, device=remaining.device)[None] < remaining[:, None]
    ).float()


def bellman_target(reward, bootstrap, next_q, discount=0.99, q_agg="mean"):
    if q_agg not in ("mean", "min"):
        raise ValueError(f"Unknown target Q aggregation: {q_agg}")
    future = next_q.min(-1).values if q_agg == "min" else next_q.mean(-1)
    return reward + discount * bootstrap * future


@torch.no_grad()
def q_normalization(q_values, eps=1e-6):
    """Official mean(abs(mean_heads(Q))), pooled over this global microbatch.

    Detaching the scale preserves the action gradient. The numerical floor only
    guards an exactly zero/near-zero critic; it is not a reward or Q clipping.
    """
    if eps <= 0:
        raise ValueError("Q normalization epsilon must be positive")
    per_sample = q_values.detach().float().mean(-1)
    stats = torch.stack(
        (per_sample.abs().sum(), per_sample.new_tensor(per_sample.numel()))
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats)
    mean_abs = stats[0] / stats[1]
    return mean_abs.clamp_min(eps).reciprocal(), mean_abs


def actor_objective(
    prediction, target, q_values, alpha=10.0, *, normalize_q_loss=False, q_norm_eps=1e-6
):
    mse = (prediction[..., :14].float() - target[..., :14].float()).square().mean()
    raw_q_loss = -q_values.float().mean()
    parts = {"endpoint_mse": mse}
    if normalize_q_loss:
        scale, mean_abs = q_normalization(q_values, q_norm_eps)
        q_loss = scale * raw_q_loss
        parts.update(
            q_actor_loss_raw=raw_q_loss,
            q_normalization_scale=scale,
            q_global_mean_abs=mean_abs,
            q_normalization_floor_hit=(mean_abs < q_norm_eps).float(),
        )
    else:
        q_loss = raw_q_loss
    parts["q_actor_loss"] = q_loss
    return alpha * mse + q_loss, parts


@torch.no_grad()
def update_target(target, source, tau=0.005):
    for tp, p in zip(target.parameters(), source.parameters(), strict=True):
        tp.lerp_(p, tau)
