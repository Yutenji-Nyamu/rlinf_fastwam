# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Conditional, balanced K4 OT for the separate Direct + GAD-OT variant.

OptiFlow motivates endpoint coupling. Here uniform equal-size marginals admit
an exact optimal permutation; this is not its Sinkhorn/row-argmax procedure.
Match complete H50 physical-action blocks inside each observation. Keep the
existing detached GAD weights, replay sampling and gradient scale.
"""

import itertools

import torch

from rlinf.algorithms.pi05_distillation.gad import weighted_mean

CONTRACT = {"version": 1, "samples": 4, "pairing": "balanced", "gradient_rows": 1}


def validate_config(config):
    """Never silently apply endpoint OT to CP or score-based objectives."""
    option = config.get("gad_ot")
    if option is None:
        return False
    if option != CONTRACT or config["method"] != "endpoint" or not config.get("gad"):
        raise ValueError("GAD-OT requires the explicit K4 endpoint contract")
    if config.get("loss_action_dim") != 14 or config.get("teacher_steps") not in (
        5,
        10,
    ):
        raise ValueError("GAD-OT preserves native teacher steps and physical 14D loss")
    return True


@torch.no_grad()
def match(student, teacher, weights):
    """Return one exact uniform-marginal OT permutation per observation.

    K<=4 is intentionally bounded: the production contract is K4, and small K
    supports checks of the assignment mathematics. No candidate is discarded.
    Ties use the first lexicographic optimum, including identity for constants.
    A per-observation importance multiplier leaves the minimizer unchanged.
    """
    if (
        student.ndim != 4
        or student.shape != teacher.shape
        or not 1 <= student.shape[1] <= 4
        or student.shape[-1] != 14
        or weights.shape != (student.shape[0], student.shape[2])
    ):
        raise ValueError("Expected matched [B,K,H,14] sets and [B,H] weights")
    s, t, w = student.float(), teacher.float(), weights.detach().float()
    cost = ((s[:, :, None] - t[:, None]).square().mean(-1) * w[:, None, None]).mean(-1)
    if not torch.isfinite(cost).all() or not torch.isfinite(w).all() or (w <= 0).any():
        raise FloatingPointError("Nonfinite transport cost or invalid action weight")
    k = s.shape[1]
    permutations = torch.tensor(
        list(itertools.permutations(range(k))), device=s.device, dtype=torch.long
    )
    rows = torch.arange(k, device=s.device)
    totals = cost[:, rows, permutations].mean(-1)
    chosen = permutations[totals.argmin(-1)]
    matched = cost.gather(2, chosen[..., None]).squeeze(-1).mean(-1)
    original = cost.diagonal(dim1=1, dim2=2).mean(-1)
    return chosen, {
        "ot_same_noise_weighted_mse": original.mean(),
        "ot_matched_weighted_mse": matched.mean(),
        "ot_cost_reduction": (original - matched).mean(),
        "ot_reassigned_fraction": (chosen != rows).float().mean(),
    }


@torch.no_grad()
def select(candidates, weights, *, seed):
    """Select a uniform student row and its assigned teacher; RNG is isolated.

    The seed is derived from the persisted run seed, update, rank and microbatch.
    Resume needs no unsaved mutable generator. Selection is uniform, not a
    second priority distribution; the K-row mean needs no extra factor of K.
    """
    assignment, metrics = match(candidates["student"], candidates["teacher"], weights)
    noise = candidates["noise"]
    if noise.shape[:3] != candidates["student"].shape[:3] or noise.shape[-1] != 32:
        raise ValueError("Retain the full 32D generation noise for recomputation")
    rng = torch.Generator(device=noise.device).manual_seed(int(seed))
    rows = torch.randint(
        noise.shape[1], (len(noise),), device=noise.device, generator=rng
    )
    batch = torch.arange(len(noise), device=noise.device)
    targets = assignment[batch, rows]
    return noise[batch, rows], candidates["teacher"][batch, targets], metrics


def endpoint_loss(student, candidates, context, weights, *, seed):
    """Recompute one selected student with gradients; keep teachers/plan detached.

    This estimates the frozen finite-batch assignment objective, not the exact
    population Wasserstein gradient. No teacher average or additional Q is used.
    """
    noise, target, metrics = select(candidates, weights, seed=seed)
    prediction = noise - student(
        noise, torch.ones(len(noise), device=noise.device), context
    )
    error = (prediction[..., :14].float() - target.detach().float()).square()
    loss = weighted_mean(error, weights)
    return loss, {
        **metrics,
        "endpoint_mse": loss,
        "joint_mse": error[..., [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].mean(),
        "gripper_mse": error[..., [6, 13]].mean(),
    }
