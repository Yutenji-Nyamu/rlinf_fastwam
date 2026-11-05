# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Small acceleration contracts; validation is never silently removed."""

from typing import Any

import torch


def validate_cpu_weights(weights: Any, batch_size: int) -> None:
    """Validate generated U weights before their unchanged device transfer."""
    if (
        not isinstance(weights, torch.Tensor)
        or weights.device.type != "cpu"
        or weights.shape != (batch_size, 1)
        or weights.requires_grad
        or not torch.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("DSRL U actor requires detached positive CPU weights [B,1]")
