# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from rlinf.algorithms.advantages import compute_prism_rloo_advantages
from rlinf.algorithms.dvac_rank_reward import trajectory_dvac_quality


def test_executed_action_mean_and_tie_aware_reverse_rank() -> None:
    log_values = torch.tensor(
        [
            [
                [1.0, 1.0],
                [1.0, 20.0],
                [2.0, 20.0],
                [3.0, 20.0],
                [4.0, 20.0],
                [5.0, 20.0],
                [6.0, 20.0],
                [7.0, 20.0],
            ]
        ]
    )
    action_mask = torch.zeros_like(log_values, dtype=torch.bool)
    action_mask[..., 0] = True
    action_mask[:, 0, 1] = True

    cost, quality = trajectory_dvac_quality(
        torch.exp(log_values),
        action_mask,
        group_size=8,
        log_eps=1e-12,
    )

    assert torch.allclose(cost, torch.tensor([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    assert torch.allclose(
        quality,
        torch.tensor([13 / 14, 13 / 14, 5 / 7, 4 / 7, 3 / 7, 2 / 7, 1 / 7, 0.0]),
    )
    _, all_tied = trajectory_dvac_quality(
        torch.ones_like(log_values),
        torch.ones_like(action_mask),
        group_size=8,
        log_eps=1e-12,
    )
    assert torch.allclose(all_tied, torch.full((8,), 0.5))


def test_prism_rloo_has_zero_group_sum_without_std_normalization() -> None:
    quality = torch.tensor([0.0, 1.0, 6 / 7, 5 / 7, 4 / 7, 3 / 7, 2 / 7, 1 / 7])
    loss_mask = torch.ones(3, 8)

    advantages, returns = compute_prism_rloo_advantages(
        torch.zeros(1, 8),
        loss_mask,
        group_size=8,
        trajectory_quality=quality,
        quality_lambda=0.2,
    )

    assert returns is None
    assert torch.allclose(advantages.sum(dim=-1), torch.zeros(3), atol=1e-6)
    assert torch.count_nonzero(advantages) > 0
    all_success, _ = compute_prism_rloo_advantages(
        torch.ones(1, 8),
        loss_mask,
        group_size=8,
        trajectory_quality=quality,
        quality_lambda=0.2,
    )
    assert torch.allclose(all_success, advantages)
    constant_quality, _ = compute_prism_rloo_advantages(
        torch.zeros(1, 8),
        loss_mask,
        group_size=8,
        trajectory_quality=torch.full((8,), 0.5),
        quality_lambda=0.2,
    )
    assert torch.allclose(
        constant_quality, torch.zeros_like(constant_quality), atol=1e-7
    )

    mixed, _ = compute_prism_rloo_advantages(
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        loss_mask,
        group_size=8,
        trajectory_quality=quality,
        quality_lambda=0.2,
    )
    assert mixed[0, 0] > mixed[0, 1:].max()

    with pytest.raises(ValueError, match="binary episode scores"):
        compute_prism_rloo_advantages(
            torch.full((1, 8), 0.5),
            loss_mask,
            group_size=8,
            trajectory_quality=quality,
            quality_lambda=0.2,
        )
