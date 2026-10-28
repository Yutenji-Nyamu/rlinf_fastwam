# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Mean Flow Distillation in native pi05 coordinates (noise=1, action=0).

Reference: happyw1nd/MFD@bd882924, train_sana_mfd.py. Preserve the detached
mean-velocity difference and auxiliary velocity regression. Continuous times,
physical action masking and conditional batched integration are pi05 adaptations.
"""

import math

import torch
import torch.distributed as dist

from rlinf.algorithms.pi05_distillation.losses import action_coordinates, expand


def interval_probabilities(step_size: float, device=None) -> torch.Tensor:
    """Probabilities of ceil(|U1-U2| / h) for two independent uniform times."""
    if not math.isfinite(step_size) or not 0 < step_size <= 1:
        raise ValueError("MFD integration step_size must lie in (0, 1]")
    edges = (
        torch.arange(math.ceil(1 / step_size) + 1, device=device, dtype=torch.float64)
        .mul_(step_size)
        .clamp_(max=1)
    )
    survival = (1 - edges).square()
    return survival[:-1] - survival[1:]


def sample_intervals(batch: int, device, *, step_size: float = 0.02):
    """Share K across ranks; independently sample each interval conditional on K.

    Every sample has the sorted-uniform marginal, although samples share K.
    This avoids variable-length padding to 50 forwards or serial sample loops.
    The scalar draw uses checkpointed Torch RNG; no extra RNG state is introduced.
    """
    probabilities = interval_probabilities(step_size, device)
    distributed = dist.is_available() and dist.is_initialized()
    if not distributed or dist.get_rank() == 0:
        count = torch.multinomial(probabilities, 1) + 1
    else:
        count = torch.zeros(1, device=device, dtype=torch.long)
    if distributed:
        dist.broadcast(count, src=0)
    steps = int(count.item())
    lo, hi = (steps - 1) * step_size, min(steps * step_size, 1.0)
    cdf_lo, cdf_hi = 1 - (1 - lo) ** 2, 1 - (1 - hi) ** 2
    u = torch.rand(batch, device=device, dtype=torch.float32)
    u = cdf_lo + (cdf_hi - cdf_lo) * u
    # Stable inverse CDF: 1-sqrt(1-u), including very short intervals.
    length = u / (1 + (1 - u).clamp_min(0).sqrt())
    end = torch.rand(batch, device=device, dtype=torch.float32) * (1 - length)
    return end + length, end, steps


@torch.no_grad()
def average_velocity(
    velocity, x, start, end, context, *, steps: int, diagnostics=None, action_dim=14
):
    """Euler mean along this model's OWN trajectory, with no backward graph.

    Each sample uses equal substeps (start-end)/K, as in the reference linspace.
    Averaging velocities equals the endpoint displacement divided by duration,
    but avoids endpoint cancellation and division by zero for tiny/zero duration.
    FP32 state/accumulation also prevents BF16 short steps being rounded away.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("MFD Euler steps must be a positive integer")
    if diagnostics is not None and not 0 < action_dim < x.shape[-1]:
        raise ValueError(
            "Padding diagnostics require both physical and padding coordinates"
        )
    position = x.detach().float()
    start, end = start.float(), end.float()
    delta = (start - end) / steps
    mean = torch.zeros_like(position)
    if diagnostics is not None:
        diagnostics["padding_start_rms"] = (
            position[..., action_dim:].square().mean().sqrt()
        )
        diagnostics["padding_position_abs_max"] = (
            position[..., action_dim:].abs().amax()
        )
        diagnostics["physical_position_abs_max"] = (
            position[..., :action_dim].abs().amax()
        )
        diagnostics["padding_velocity_abs_max"] = position.new_zeros(())
        diagnostics["physical_velocity_abs_max"] = position.new_zeros(())
    for index in range(steps):
        current_time = start - index * delta
        value = velocity(position, current_time, context).float()
        mean.add_(value / steps)
        position = position - expand(delta) * value
        if diagnostics is not None:
            for label, width in (
                ("physical", slice(None, action_dim)),
                ("padding", slice(action_dim, None)),
            ):
                for kind, data in (("position", position), ("velocity", value)):
                    key = f"{label}_{kind}_abs_max"
                    diagnostics[key] = torch.maximum(
                        diagnostics[key], data[..., width].abs().amax()
                    )
    if diagnostics is not None:
        diagnostics["padding_end_rms"] = (
            position[..., action_dim:].square().mean().sqrt()
        )
    return mean


def auxiliary_loss(
    fake, generated, context, *, action_dim: int = 14, loss_action_dim=None
):
    """Fit student velocity; v2 optionally scores only physical coordinates.

    Unlike OneDP epsilon MSE, there is no additional (1-t)^2 weighting. The
    generated action is detached: this update trains only the auxiliary model.
    """
    clean = generated.detach().float()
    times = torch.rand(len(clean), device=clean.device, dtype=torch.float32)
    noise = torch.randn_like(clean)
    noised = (1 - expand(times)) * clean + expand(times) * noise
    error = (fake(noised, times, context).float() - (noise - clean)).square()
    parts = {"fake_physical_mse": error[..., :action_dim].detach().mean()}
    if error.shape[-1] > action_dim:
        parts["fake_padding_mse"] = error[..., action_dim:].detach().mean()
    return action_coordinates(error, loss_action_dim).mean(), parts


def student_loss(
    generated,
    teacher,
    fake,
    context,
    *,
    step_size=0.02,
    action_dim=14,
    intervals=None,
    diagnostics=False,
):
    """Reference <stopgrad(u_teacher-u_fake), x_t>, over physical coordinates.

    No Jacobian passes through either integration; x_t retains its (1-t)
    derivative to the generated action. Optional intervals support analytic
    checks and identical-input comparisons, not a separate training mode.
    """
    if not 0 < action_dim <= generated.shape[-1]:
        raise ValueError("Invalid physical action dimension")
    start, end, steps = (
        sample_intervals(len(generated), generated.device, step_size=step_size)
        if intervals is None
        else intervals
    )
    clean = generated.float()
    noised = (1 - expand(start)) * clean + expand(start) * torch.randn_like(clean)
    traces = (
        {"teacher": {}, "fake": {}} if diagnostics else {"teacher": None, "fake": None}
    )
    teacher_mean = average_velocity(
        teacher,
        noised,
        start,
        end,
        context,
        steps=steps,
        diagnostics=traces["teacher"],
        action_dim=action_dim,
    )
    fake_mean = average_velocity(
        fake,
        noised,
        start,
        end,
        context,
        steps=steps,
        diagnostics=traces["fake"],
        action_dim=action_dim,
    )
    direction = (teacher_mean - fake_mean)[..., :action_dim]
    loss = (direction * noised[..., :action_dim]).mean()
    parts = {
        "mfd_direction_rms": direction.square().mean().sqrt(),
        "mfd_interval_mean": (start - end).mean(),
        "mfd_start_mean": start.mean(),
        "mfd_steps_per_path": loss.new_tensor(steps),
        "mfd_guidance_nfe": loss.new_tensor(2 * steps),
    }
    if diagnostics:
        for label, trace in traces.items():
            parts.update({f"mfd_{label}_{key}": value for key, value in trace.items()})
    return loss, parts
