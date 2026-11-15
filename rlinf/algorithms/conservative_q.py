"""CQL candidate penalty with optional coverage of the actual policy.

The legacy two-candidate loss follows verl-vla. The optional current-policy
candidate follows the candidate coverage used in the original CQL implementation:
https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py
"""

from typing import Optional

import torch


def conservative_q_loss(
    q_data: torch.Tensor,
    q_candidate: torch.Tensor,
    temperature: float = 1.0,
    *,
    q_policy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize candidate Q values relative to replay Q values.

    All inputs have the same batch/critic shape. With ``q_policy=None``, this
    preserves the legacy loss exactly. Adding a candidate changes the constant
    at equal Q values from ``temperature * log(2)`` to ``temperature * log(3)``;
    no candidate-count rescaling is applied to the gradients or configured weight.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    candidates = (q_data, q_candidate)
    if q_policy is not None:
        candidates += (q_policy,)
    return (
        torch.logsumexp(torch.stack(candidates) / temperature, dim=0) * temperature
        - q_data
    ).mean()
