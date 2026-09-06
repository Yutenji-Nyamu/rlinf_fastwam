# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Worker-owned C1 critic for the fixed-N RoboTwin QAM adaptation.

The policy graph deliberately does not import this module.  Every QAM actor
rank owns the complete logical ensemble, while corresponding heads are kept
identical across ranks by gradient averaging in the QAM worker.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class _IndependentQMLP(nn.Module):
    """One action-conditioned Q function with official-style MLP blocks."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")

        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_dims:
            width = int(width)
            layers.extend(
                (
                    nn.Linear(previous, width),
                    nn.GELU(approximate="tanh"),
                    nn.LayerNorm(width, eps=1e-6),
                )
            )
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._initialize_linear)

    @staticmethod
    def _initialize_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs).squeeze(-1)


class QAMCriticEnsemble(nn.Module):
    """Ten independent FP32 Q MLPs over fixed observation and action views.

    Args:
        feature_dim: Runtime width ``D`` of each of the four frozen prefix
            blocks.  This is discovered from the first valid online replay
            row instead of being hard-coded to a particular OpenPI checkpoint.
        proprio_dim: Normalized proprioception width.
        action_horizon: Planned macro-action horizon.
        action_dim: Active action width per planned slot.
        num_q_heads: Number of fully independent Q networks.
        hidden_dims: Hidden widths used independently by every Q network.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        proprio_dim: int = 14,
        action_horizon: int = 20,
        action_dim: int = 14,
        num_q_heads: int = 10,
        hidden_dims: Sequence[int] = (512, 512, 512, 512),
    ) -> None:
        super().__init__()
        for name, value in {
            "feature_dim": feature_dim,
            "proprio_dim": proprio_dim,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "num_q_heads": num_q_heads,
        }.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")

        self.feature_dim = int(feature_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.num_q_heads = int(num_q_heads)
        input_dim = (
            4 * self.feature_dim
            + self.proprio_dim
            + self.action_horizon * self.action_dim
        )
        self.q_functions = nn.ModuleList(
            _IndependentQMLP(input_dim, hidden_dims) for _ in range(self.num_q_heads)
        )
        # Critic arithmetic stays FP32 even when the frozen prefix is stored
        # as BF16 and the F1 action expert is FSDP mixed precision.
        self.float()

    def _flatten_inputs(
        self,
        feature: Tensor,
        proprio: Tensor,
        planned_action: Tensor,
    ) -> Tensor:
        if feature.ndim != 3 or feature.shape[1:] != (4, self.feature_dim):
            raise ValueError(
                "feature must have shape "
                f"[B, 4, {self.feature_dim}], got {tuple(feature.shape)}"
            )
        batch_size = feature.shape[0]
        if proprio.shape != (batch_size, self.proprio_dim):
            raise ValueError(
                "proprio must have shape "
                f"[B, {self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        if planned_action.shape != (
            batch_size,
            self.action_horizon,
            self.action_dim,
        ):
            raise ValueError(
                "planned_action must have shape "
                f"[B, {self.action_horizon}, {self.action_dim}], "
                f"got {tuple(planned_action.shape)}"
            )
        tensors = (
            feature.reshape(batch_size, -1),
            proprio,
            planned_action.reshape(batch_size, -1),
        )
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("QAM critic inputs must be finite")
        return torch.cat(
            [tensor.to(dtype=torch.float32) for tensor in tensors],
            dim=-1,
        )

    def forward(
        self,
        feature: Tensor,
        proprio: Tensor,
        planned_action: Tensor,
    ) -> Tensor:
        """Return all independent Q values with shape ``[num_q_heads, B]``."""
        inputs = self._flatten_inputs(feature, proprio, planned_action)
        return torch.stack(
            [q_function(inputs) for q_function in self.q_functions],
            dim=0,
        )
