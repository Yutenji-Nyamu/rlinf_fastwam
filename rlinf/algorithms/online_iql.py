"""Pure numerical contracts for online, chunk-level IQL.

One submitted action proposal is one MDP step.  RynnValue is a frozen
potential provider; it is not the learned IQL value function.
"""
from __future__ import annotations

import math

import torch


def iql_phase(round_number: int, warmup_rounds: int = 10) -> str:
    """Use one-based collection rounds: R1..R10 BC, R11 onward IQL."""
    if isinstance(round_number, bool) or int(round_number) != round_number or round_number < 1:
        raise ValueError("round_number must be a positive one-based integer")
    if isinstance(warmup_rounds, bool) or int(warmup_rounds) != warmup_rounds or warmup_rounds < 0:
        raise ValueError("warmup_rounds must be a nonnegative integer")
    return "bc" if round_number <= warmup_rounds else "iql"


def shape_reward(potential, next_potential, sparse_reward, bootstrap_mask,
                 gamma: float = 0.99, potential_scale: float = 0.1) -> torch.Tensor:
    """Return task reward plus PBRS; mask=0 uses absorbing potential zero.

    Raw terminal predictions remain in replay; only this computation uses zero.
    Inputs may be scalars or broadcastable tensors.  They are detached FP32.
    """
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite in [0,1]")
    if not math.isfinite(potential_scale) or potential_scale < 0:
        raise ValueError("potential_scale must be finite and nonnegative")
    inputs = (potential, next_potential, sparse_reward, bootstrap_mask)
    device = next((x.device for x in inputs if isinstance(x, torch.Tensor)), torch.device("cpu"))
    p, pn, r, mask = torch.broadcast_tensors(*[
        torch.as_tensor(x, dtype=torch.float32, device=device).detach() for x in inputs
    ])
    if not all(torch.isfinite(x).all() for x in (p, pn, r, mask)):
        raise ValueError("IQL reward inputs must be finite")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("bootstrap_mask must be binary")
    pn_used = torch.where(mask.bool(), pn, torch.zeros_like(pn))
    result = r + float(potential_scale) * (float(gamma) * pn_used - p)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite shaped reward")
    return result


def expectile_loss(diff: torch.Tensor, expectile: float = 0.8) -> torch.Tensor:
    """Unreduced expectile squared error; caller reduces over the batch."""
    if not math.isfinite(expectile) or not 0 < expectile < 1:
        raise ValueError("expectile must be in (0,1)")
    return torch.where(diff > 0, float(expectile), 1.0 - float(expectile)) * diff.square()


def advantage_weights(advantage: torch.Tensor, beta: float = 10.0,
                      max_weight: float = 100.0) -> torch.Tensor:
    """Detached exp(beta*A), capped in log space; no mean normalization."""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if not math.isfinite(max_weight) or max_weight <= 0:
        raise ValueError("max_weight must be finite and positive")
    a = advantage.detach().to(dtype=torch.float32)
    if not torch.isfinite(a).all():
        raise ValueError("IQL advantage must be finite before clipping")
    # Dividing the ceiling first also avoids overflow in beta*A for huge finite A.
    capped = a.clamp_max(math.log(max_weight) / beta)
    return torch.exp(capped * beta).clamp_max(max_weight)


def weight_diagnostics(weights: torch.Tensor) -> dict[str, float]:
    w = weights.detach().float().flatten()
    if not w.numel() or not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("Expected nonempty finite nonnegative IQL weights")
    total, squares = w.sum(), w.square().sum()
    return {
        "weight_mean": float(w.mean()), "weight_min": float(w.min()),
        "weight_max": float(w.max()), "weight_std": float(w.std(unbiased=False)),
        "weight_zero_fraction": float((w == 0).float().mean()),
        "weight_ess": float(total.square() / squares) if squares > 0 else 0.0,
        "weight_max_share": float(w.max() / total) if total > 0 else 0.0,
    }
