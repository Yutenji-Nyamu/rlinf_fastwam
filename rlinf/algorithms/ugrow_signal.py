"""Same-noise, complete-solver U-GROW-style action disagreement."""

import math

import torch

UGROW_SIGNAL_SPEC = {
    "version": 1,
    "kind": "ugrow_10_5",
    "steps": [10, 5],
    "solver": "flow_ode",
    "coordinate_space": "normalized",
    "action_dim": 14,
    "epsilon": 1e-8,
}


def compute_ugrow_signal(
    main_actions: torch.Tensor,
    comparison_actions: torch.Tensor,
    action_dim: int = 14,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return detached float32 disagreement [B,H] over valid action coordinates.

    Both arguments are complete normalized action chunks from the same input
    and initial noise. The numerator is the population standard deviation of
    two predictions, abs(a / 2 - b / 2); the denominator is their RMS plus eps.
    Float64 hypot avoids intermediate overflow without clipping the signal.
    """
    if (
        not isinstance(action_dim, int)
        or isinstance(action_dim, bool)
        or action_dim <= 0
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("U-GROW requires a positive action_dim and finite eps > 0.")
    if (
        not isinstance(main_actions, torch.Tensor)
        or not isinstance(comparison_actions, torch.Tensor)
        or main_actions.ndim != 3
        or main_actions.shape != comparison_actions.shape
        or main_actions.shape[-1] < action_dim
        or main_actions.shape[1] == 0
        or main_actions.device != comparison_actions.device
        or not torch.is_floating_point(main_actions)
        or not torch.is_floating_point(comparison_actions)
    ):
        raise ValueError("U-GROW requires matching floating [B,H,D] action tensors.")
    a = main_actions.detach()[..., :action_dim].double()
    b = comparison_actions.detach()[..., :action_dim].double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("U-GROW valid action coordinates must be finite.")
    numerator = (a / 2.0 - b / 2.0).abs()
    denominator = torch.hypot(a / math.sqrt(2.0), b / math.sqrt(2.0)) + eps
    result = (numerator / denominator).mean(dim=-1).float()
    if not torch.isfinite(result).all():
        raise ValueError("U-GROW produced a non-finite signal.")
    return result.detach()
