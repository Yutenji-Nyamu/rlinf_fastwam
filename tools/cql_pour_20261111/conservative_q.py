# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Two-candidate conservative Q regularization, following verl-vla.

Reference: verl-project/verl-vla c1de826c, training_worker._calculate_critic_loss.
This is its data/policy-candidate penalty, not the full importance-sampled
CQL algorithm. Both critics receive gradients; candidate actions are detached
by the caller so critic updates cannot update the noise actor.
"""

import math

import torch
import torch.nn.functional as F


def conservative_q_loss(
    q_data: torch.Tensor, q_policy: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Return mean T*log(exp(Q_data/T)+exp(Q_policy/T))-Q_data."""
    if q_data.shape != q_policy.shape or q_data.ndim != 2:
        raise ValueError("Conservative Q expects matching [batch, heads] scores")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Conservative Q temperature must be finite and positive")
    gap = q_policy.float() - q_data.float()
    return (temperature * F.softplus(gap / temperature)).mean()
