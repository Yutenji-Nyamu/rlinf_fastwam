# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Apply established DVCA batch mapping to frozen U or Norm measurements."""

import torch

from rlinf.algorithms.online_bc_dvac_controls import (
    apply_chunk_dropout,
    bc_controls_contract,
    effective_alphas,
)
from rlinf.algorithms.online_bc_dvac_two_level import compute_two_level_bc_weights


def signal_tau_contract(cfg: dict, signal_spec: dict) -> dict:
    """Freeze signal identity, temperatures and controls for safe resume."""
    kind = cfg.get("signal_kind")
    if kind not in ("ugrow_10_5", "norm_residual_t5_l3"):
        raise ValueError("Signal tau supports the existing U and Norm producers.")
    if not cfg.get("enabled") or cfg.get("normalization") != "two_level_batch":
        raise ValueError("Signal tau requires active two_level_batch mapping.")
    if any(k in cfg for k in ("window", "mapping", "alpha", "z_clip", "std_floor", "weight_min", "weight_max")):
        raise ValueError("Do not mix historical calibration with batch mapping.")
    if signal_spec.get("kind") != kind:
        raise ValueError("Signal producer and weighting contract differ.")
    if cfg.get("factor_mapping") != "exp_mean":
        raise ValueError("The temperature sweep uses existing exp_mean mapping.")
    settings = {
        "alpha_local": float(cfg.get("alpha_local", 1.0)),
        "alpha_chunk": float(cfg.get("alpha_chunk", 1.0)),
        "variance_eps": float(cfg.get("log_eps", 1e-12)),
        "range_eps": float(cfg.get("range_eps", 1e-6)),
        "factor_mapping": cfg["factor_mapping"],
        "temperature_local": float(cfg["temperature_local"]),
        "temperature_chunk": float(cfg["temperature_chunk"]),
    }
    compute_two_level_bc_weights(torch.ones(1, 1), torch.ones(1, 1, 1), **settings)
    return {
        "version": 1,
        "signal_spec": dict(signal_spec),
        "raw_key": "ugrow_u" if kind == "ugrow_10_5" else "norm_raw",
        "settings": settings,
        "controls": bc_controls_contract(cfg),
    }


@torch.no_grad()
def apply_signal_tau(inputs: dict, contract: dict, *, runner_step: int, update_step: int):
    """Return detached FM weights; never alter raw replay data or RNG state."""
    raw = inputs[contract["raw_key"]]
    mask = inputs["action_valid_mask"]
    if mask.ndim != 3 or (mask.sum(dim=(1, 2)) == 0).any():
        raise ValueError("Every sampled query needs valid action targets.")
    settings = dict(contract["settings"])
    controls = contract["controls"]
    settings["alpha_local"], settings["alpha_chunk"] = effective_alphas(
        settings["alpha_local"], settings["alpha_chunk"], controls,
        runner_step=runner_step,
    )
    weights, metrics = compute_two_level_bc_weights(raw, mask, **settings)
    weights, dropped = apply_chunk_dropout(
        weights, mask.bool().any(dim=(1, 2)), controls, update_step=update_step,
    )
    q = mask.sum(-1).to(weights.dtype)
    n = q.sum(-1)
    mean = ((weights * q).sum(-1) / n).mean()
    std = (((weights - mean).square() * q).sum(-1) / n).mean().sqrt()
    metrics.update(
        alpha_local=settings["alpha_local"], alpha_chunk=settings["alpha_chunk"],
        temperature_local=settings["temperature_local"],
        temperature_chunk=settings["temperature_chunk"],
        dropout_fraction=float(dropped.float().mean()),
        final_weight_mean=float(mean), final_weight_std=float(std),
        final_weight_min=float(weights.min()), final_weight_max=float(weights.max()),
        runner_round=float(runner_step + 1),
    )
    return weights, metrics
