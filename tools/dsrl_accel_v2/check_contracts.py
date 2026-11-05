# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Validate CPU U weights, including inputs that must still be rejected."""

import json

import torch

from tools.dsrl_accel_v2.contracts import validate_cpu_weights


def main() -> None:
    """Check positive CPU weights and reject invalid inputs."""
    validate_cpu_weights(torch.ones(8, 1), 8)
    invalid = [
        torch.ones(8),
        torch.zeros(8, 1),
        -torch.ones(8, 1),
        torch.full((8, 1), float("nan")),
        torch.full((8, 1), float("inf")),
        torch.ones(8, 1, requires_grad=True),
    ]
    for weights in invalid:
        try:
            validate_cpu_weights(weights, 8)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid U weights accepted")
    print(
        json.dumps(
            {"u_invalid_cases_rejected": len(invalid), "valid_cpu_weights": "pass"}
        )
    )


if __name__ == "__main__":
    main()
