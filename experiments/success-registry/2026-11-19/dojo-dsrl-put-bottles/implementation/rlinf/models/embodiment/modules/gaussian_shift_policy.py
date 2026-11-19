"""Unit-variance Gaussian steering with a bounded state-dependent mean.

Inspired by LP-DS zero-initialized bounded perturbations. This is a restricted
Gaussian-shift experiment, not the complete LP-DS algorithm (no action critic).
"""
import math
import torch
from torch import nn


class GaussianShiftPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128, 128),
                 shift_bound=0.5, **unused):
        super().__init__()
        if not math.isfinite(shift_bound) or shift_bound <= 0:
            raise ValueError('shift_bound must be positive and finite')
        self.output_dim = output_dim
        self.shift_bound = float(shift_bound)
        layers = []
        for width in hidden_dims:
            layers += [nn.Linear(input_dim, width), nn.LayerNorm(width), nn.ReLU()]
            input_dim = width
        self.shared_net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.mean_layer.weight)
        nn.init.zeros_(self.mean_layer.bias)

    def forward(self, features, sample=False, deterministic=False, noise_scale=None):
        mean = self.shift_bound * torch.tanh(self.mean_layer(self.shared_net(features)).float())
        variance = 1.0 + (0.0 if noise_scale is None else float(noise_scale)**2)
        dist = torch.distributions.Independent(torch.distributions.Normal(mean, torch.full_like(mean, math.sqrt(variance))), 1)
        if not sample:
            return dist
        if deterministic:
            raise ValueError('Gaussian-shift evaluation must retain independent Gaussian sampling')
        # Explicit standard-normal sampling supports exact paired-input checks.
        noise = mean + dist.stddev * torch.randn_like(mean)
        return noise.unsqueeze(1), dist.log_prob(noise)

    def sample(self, features, deterministic=False, noise_scale=None):
        return self.forward(features, sample=True, deterministic=deterministic, noise_scale=noise_scale)

    def evaluate_actions(self, features, actions):
        dist = self.forward(features)
        if actions.ndim == 3:
            actions = actions[:, 0]
        return dist.log_prob(actions), dist.entropy()
