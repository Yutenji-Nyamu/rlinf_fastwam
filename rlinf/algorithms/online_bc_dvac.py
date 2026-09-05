# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Action-level endpoint DVAC for success FM-BC; no RL objective or new forward.

Sources: GRPO action-DVAC endpoint variance / past-round log moments;
RLT per-horizon centered mean-one supervision; AttenA+ unreduced FM entry.
"""

import math
from collections import deque

import torch


def endpoint_variance(endpoints: torch.Tensor) -> torch.Tensor:
    """Population variance over denoising previews [B,L,H,D] -> [B,H]."""
    if endpoints.ndim != 4 or endpoints.shape[1] < 2:
        raise ValueError("DVAC requires at least two [B,L,H,D] endpoint previews.")
    endpoints = endpoints.detach().float()
    if not torch.isfinite(endpoints).all():
        raise ValueError("Non-finite DVAC endpoint preview.")
    return endpoints.var(dim=1, unbiased=False).sum(dim=-1)


def log_moments(variance: torch.Tensor, eps: float) -> torch.Tensor:
    """Small CPU float64 count/sum/sumsq packet, including failed episodes."""
    variance = variance.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("DVAC variance must be finite and non-negative.")
    values = (variance + eps).log()
    return torch.stack(
        (values.new_tensor(values.numel()), values.sum(), values.square().sum())
    )


class OnlineBCDvac:
    """Annotate new success records once, before replay archiving.

    Map using preceding completed rounds, then add this round's moments. Old
    replay weights stay fixed; the first uncalibrated round gets unit weights.
    """

    def __init__(
        self, *, window=5, alpha=0.25, z_clip=2.0, log_eps=1e-12, std_floor=1e-6
    ):
        if not all(math.isfinite(x) for x in (alpha, z_clip, log_eps, std_floor)):
            raise ValueError("DVAC settings must be finite.")
        if window < 1 or z_clip <= 0 or not 0 <= alpha <= 1 / (2 * z_clip):
            raise ValueError(
                "Require window>=1 and 0<=alpha<=1/(2*z_clip) for [0,2] weights."
            )
        if log_eps <= 0 or std_floor <= 0:
            raise ValueError("DVAC log epsilon and std floor must be positive.")
        self.settings = dict(
            window=int(window),
            alpha=float(alpha),
            z_clip=float(z_clip),
            log_eps=float(log_eps),
            std_floor=float(std_floor),
        )
        self.history = deque(maxlen=int(window))
        self.round_id = 0

    def annotate(self, episodes: list, moments: torch.Tensor) -> dict:
        moments = moments.detach().to(device="cpu", dtype=torch.float64)
        if moments.shape != (3,) or not torch.isfinite(moments).all() or moments[0] < 0:
            raise ValueError("Invalid DVAC round moments.")
        past = sum(self.history, torch.zeros(3, dtype=torch.float64))
        calibrated = past[0].item() > 0
        mean = past[1] / past[0] if calibrated else torch.tensor(0.0)
        std = (
            (
                (past[2] / past[0] - mean.square())
                .clamp_min(0)
                .sqrt()
                .clamp_min(self.settings["std_floor"])
            )
            if calibrated
            else torch.tensor(1.0)
        )
        weights = []
        for episode in episodes:
            for row in episode:
                v = row["dvac_v"].detach().to(dtype=torch.float64, device="cpu")
                q = (
                    row["action_valid_mask"]
                    .to(dtype=torch.float64, device="cpu")
                    .sum(-1)
                )
                if v.shape != q.shape or q.sum() <= 0:
                    raise ValueError("DVAC H axis must match valid supervised actions.")
                log_moments(v, self.settings["log_eps"])
                w = torch.ones_like(v)
                if calibrated:
                    z = ((v + self.settings["log_eps"]).log() - mean) / std
                    z = z.clamp(-self.settings["z_clip"], self.settings["z_clip"])
                    w = 1 + self.settings["alpha"] * (z - (z * q).sum() / q.sum())
                row["action_weights"] = w.float()
                row["dvac_calibration_round"] = torch.tensor(self.round_id)
                weights.append(w[q > 0])
        metrics = {
            "dvac/round": float(self.round_id + 1),
            "dvac/new_action_positions": moments[0].item(),
            "dvac/reference_positions": past[0].item(),
            "dvac/calibrated": float(calibrated),
            "dvac/log_mean": mean.item(),
            "dvac/log_std": std.item(),
        }
        if weights:
            w = torch.cat(weights)
            metrics.update(
                {
                    "dvac/weight_mean": w.mean().item(),
                    "dvac/weight_std": w.std(unbiased=False).item(),
                    "dvac/weight_min": w.min().item(),
                    "dvac/weight_max": w.max().item(),
                }
            )
        self.history.append(moments.clone())
        self.round_id += 1
        return metrics

    def state_dict(self):
        return {
            "settings": self.settings.copy(),
            "round_id": self.round_id,
            "history": list(self.history),
        }

    def load_state_dict(self, state):
        if state["settings"] != self.settings:
            raise ValueError(
                "DVAC checkpoint calibration settings differ from this run."
            )
        self.round_id = int(state["round_id"])
        self.history = deque(state["history"], maxlen=self.settings["window"])
