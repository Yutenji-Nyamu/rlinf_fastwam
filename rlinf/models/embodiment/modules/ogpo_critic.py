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

"""FP32 action-conditioned Q ensemble for OGPO with frozen pi0 features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(device=tokens.device, dtype=torch.bool)
    weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    pooled = (tokens * weights).sum(dim=1) / denominator
    valid = mask.any(dim=1, keepdim=True)
    return torch.where(valid, pooled, torch.zeros_like(pooled))


def pool_ogpo_prefix_blocks(
    prefix_output: Tensor,
    prefix_pad_masks: Tensor,
    *,
    language_token_count: int,
    num_image_blocks: int = 3,
    image_token_counts: Sequence[int] | None = None,
) -> tuple[Tensor, tuple[int, ...]]:
    """Masked-mean pool image blocks and the trailing language block.

    OpenPI concatenates image-token blocks before the padded language tokens.
    The default derives equal image block lengths from the runtime shape; an
    exact dependency probe may instead pass image_token_counts explicitly.

    Returns a tensor shaped [B, num_image_blocks + 1, D] and the block lengths.
    """
    if prefix_output.ndim != 3:
        raise ValueError(
            "prefix_output must have shape [B, tokens, D], got "
            f"{tuple(prefix_output.shape)}"
        )
    if prefix_pad_masks.shape != prefix_output.shape[:2]:
        raise ValueError(
            "prefix_pad_masks must match the first two prefix dimensions"
        )
    if int(num_image_blocks) <= 0:
        raise ValueError("num_image_blocks must be positive")
    if int(language_token_count) <= 0:
        raise ValueError("language_token_count must be positive")

    image_tokens = prefix_output.shape[1] - int(language_token_count)
    if image_tokens <= 0:
        raise ValueError("language_token_count leaves no image tokens")
    if image_token_counts is None:
        if image_tokens % int(num_image_blocks) != 0:
            raise ValueError(
                "image tokens cannot be split into equal blocks; pass "
                "image_token_counts from the runtime prefix probe"
            )
        per_image = image_tokens // int(num_image_blocks)
        image_token_counts = (per_image,) * int(num_image_blocks)
    else:
        image_token_counts = tuple(int(count) for count in image_token_counts)
        if len(image_token_counts) != int(num_image_blocks):
            raise ValueError(
                "image_token_counts length must equal num_image_blocks"
            )
        if any(count <= 0 for count in image_token_counts):
            raise ValueError("every image token count must be positive")
        if sum(image_token_counts) != image_tokens:
            raise ValueError("image_token_counts do not cover the image prefix")

    block_lengths = (*image_token_counts, int(language_token_count))
    pooled_blocks = []
    start = 0
    for block_length in block_lengths:
        stop = start + block_length
        pooled_blocks.append(
            _masked_mean(
                prefix_output[:, start:stop],
                prefix_pad_masks[:, start:stop],
            )
        )
        start = stop
    return torch.stack(pooled_blocks, dim=1), tuple(block_lengths)


def pad_ogpo_actions(
    action: Tensor,
    *,
    horizon: int,
    action_dim: int,
    lengths: Tensor | None = None,
) -> Tensor:
    """Return [B, horizon, action_dim] with all invalid trailing slots zero."""
    if action.ndim != 3 or action.shape[2] != int(action_dim):
        raise ValueError(
            f"action must have shape [B, h, {action_dim}], got "
            f"{tuple(action.shape)}"
        )
    if not 1 <= action.shape[1] <= int(horizon):
        raise ValueError(f"action h must be in [1, {horizon}]")

    padded = F.pad(action, (0, 0, 0, int(horizon) - action.shape[1]))
    if lengths is None:
        lengths = torch.full(
            (action.shape[0],),
            action.shape[1],
            dtype=torch.long,
            device=action.device,
        )
    if lengths.shape != (action.shape[0],):
        raise ValueError(
            f"lengths must have shape [{action.shape[0]}], got "
            f"{tuple(lengths.shape)}"
        )
    lengths = lengths.to(device=action.device, dtype=torch.long)
    if torch.any(lengths < 1) or torch.any(lengths > action.shape[1]):
        raise ValueError("lengths must be in [1, provided action h]")
    valid = torch.arange(int(horizon), device=action.device).unsqueeze(0)
    valid = valid < lengths.unsqueeze(1)
    return padded * valid.unsqueeze(-1).to(dtype=padded.dtype)


class _IndependentQMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        hidden_dims = tuple(int(width) for width in hidden_dims)
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")

        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_dims:
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


class OGPOCriticEnsemble(nn.Module):
    """Independent FP32 Q functions over four prefix blocks and a C-step action.

    The default is the planned OGPO image critic: ten independent heads, each
    with five 512-wide hidden layers. Forward returns [B, M], where M is the
    number of Q heads.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        proprio_dim: int = 14,
        action_horizon: int = 10,
        action_dim: int = 14,
        num_q_heads: int = 10,
        hidden_dims: Sequence[int] = (512, 512, 512, 512, 512),
        num_feature_blocks: int = 4,
    ) -> None:
        super().__init__()
        for name, value in {
            "feature_dim": feature_dim,
            "proprio_dim": proprio_dim,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "num_q_heads": num_q_heads,
            "num_feature_blocks": num_feature_blocks,
        }.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")

        self.feature_dim = int(feature_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.num_q_heads = int(num_q_heads)
        self.num_feature_blocks = int(num_feature_blocks)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        input_dim = (
            self.num_feature_blocks * self.feature_dim
            + self.proprio_dim
            + self.action_horizon * self.action_dim
        )
        self.q_functions = nn.ModuleList(
            _IndependentQMLP(input_dim, self.hidden_dims)
            for _ in range(self.num_q_heads)
        )
        self.float()

    def _flatten_inputs(
        self,
        feature: Tensor,
        proprio: Tensor,
        action: Tensor,
        action_lengths: Tensor | None,
    ) -> Tensor:
        if feature.ndim != 3:
            raise ValueError(
                "feature must have shape "
                f"[B, {self.num_feature_blocks}, {self.feature_dim}], got "
                f"{tuple(feature.shape)}"
            )
        expected_feature = (
            feature.shape[0],
            self.num_feature_blocks,
            self.feature_dim,
        )
        if feature.shape != expected_feature:
            raise ValueError(
                "feature must have shape "
                f"[B, {self.num_feature_blocks}, {self.feature_dim}], got "
                f"{tuple(feature.shape)}"
            )
        batch_size = feature.shape[0]
        if proprio.shape != (batch_size, self.proprio_dim):
            raise ValueError(
                f"proprio must have shape [B, {self.proprio_dim}], got "
                f"{tuple(proprio.shape)}"
            )
        if action.shape[0] != batch_size:
            raise ValueError("feature and action batch sizes must match")

        padded_action = pad_ogpo_actions(
            action,
            horizon=self.action_horizon,
            action_dim=self.action_dim,
            lengths=action_lengths,
        )
        tensors = (
            feature.reshape(batch_size, -1),
            proprio,
            padded_action.reshape(batch_size, -1),
        )
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("OGPO critic inputs must be finite")
        return torch.cat(
            [tensor.to(dtype=torch.float32) for tensor in tensors],
            dim=-1,
        )

    def forward(
        self,
        feature: Tensor,
        proprio: Tensor,
        action: Tensor,
        action_lengths: Tensor | None = None,
    ) -> Tensor:
        """Return one scalar from every Q head with shape [B, M]."""
        inputs = self._flatten_inputs(
            feature,
            proprio,
            action,
            action_lengths,
        )
        return torch.stack(
            [q_function(inputs) for q_function in self.q_functions],
            dim=-1,
        )
