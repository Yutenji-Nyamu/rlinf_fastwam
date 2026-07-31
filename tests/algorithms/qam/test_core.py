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

import torch
from torch import nn

from rlinf.algorithms.qam.core import (
    AMPath,
    adjoint_matching_loss,
    adjoint_matching_step_loss,
    clone_parameter_snapshot,
    ema_from_preupdate_,
    ensemble_critic_mse,
    flow_matching_loss,
    flow_ode_sample,
    pessimistic_ensemble_value,
    q_chunk_td_target,
    reverse_behavior_adjoint,
    sample_memoryless_am_path,
    terminal_mean_q_adjoint,
)


def test_population_std_and_fixed_horizon_target() -> None:
    q_values = torch.tensor(
        [
            [1.0, 3.0],
            [3.0, 7.0],
        ]
    )
    pessimistic = pessimistic_ensemble_value(q_values, rho=0.5)
    torch.testing.assert_close(pessimistic, torch.tensor([1.5, 4.0]))

    target = q_chunk_td_target(
        return_h=torch.tensor([0.25, 1.0]),
        bootstrap_mask=torch.tensor([1.0, 0.0]),
        next_target_q_values=q_values,
        discount_h=0.81,
        rho=0.5,
    )
    torch.testing.assert_close(target, torch.tensor([1.465, 1.0]))


def test_terminal_adjoint_uses_head_mean_batch_sum_and_clamp() -> None:
    action = torch.tensor(
        [
            [0.25, 1.5],
            [-0.5, 0.75],
        ]
    )

    def target_critic(candidate: torch.Tensor) -> torch.Tensor:
        per_sample = candidate.sum(dim=-1)
        return torch.stack([per_sample, 3.0 * per_sample], dim=0)

    adjoint = terminal_mean_q_adjoint(
        target_critic,
        action,
        inv_temp=0.5,
        clip_action=True,
    )
    expected = torch.tensor(
        [
            [-1.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    torch.testing.assert_close(adjoint, expected)


def test_am_path_uses_behavior_last_step_and_ignores_last_noise() -> None:
    initial_noise = torch.zeros(1, 1)
    step_noises = torch.tensor([[[0.0]], [[1000.0]]])

    def fine_velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.ones_like(state)

    def behavior_velocity(
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        del state
        return 10.0 + time

    path = sample_memoryless_am_path(
        fine_velocity,
        behavior_velocity,
        initial_noise,
        step_noises,
        flow_steps=2,
    )
    torch.testing.assert_close(
        path.states,
        torch.tensor([[[0.0]], [[1.0]]]),
    )
    torch.testing.assert_close(path.times, torch.tensor([[[0.0]], [[0.5]]]))
    torch.testing.assert_close(path.endpoint, torch.tensor([[6.25]]))

    ode_action = flow_ode_sample(
        fine_velocity,
        initial_noise,
        flow_steps=2,
    )
    torch.testing.assert_close(ode_action, torch.tensor([[1.0]]))


def test_reverse_uses_t_plus_h_and_am_base_uses_t() -> None:
    path = AMPath(
        states=torch.tensor([[[2.0]], [[3.0]]]),
        times=torch.tensor([[[0.0]], [[0.5]]]),
        sigmas=torch.ones(2, 1, 1),
        endpoint=torch.tensor([[0.0]]),
    )

    def behavior_velocity(
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return time * state

    reverse = reverse_behavior_adjoint(
        behavior_velocity,
        path,
        terminal_adjoint=torch.ones(1, 1),
    )
    torch.testing.assert_close(
        reverse,
        torch.tensor([[[0.75]], [[1.5]]]),
    )

    scale = nn.Parameter(torch.tensor(0.0))

    def fine_velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return scale * state

    loss = adjoint_matching_loss(
        fine_velocity,
        behavior_velocity,
        path,
        adjoints=torch.zeros_like(path.states),
    )
    torch.testing.assert_close(loss, torch.tensor(9.0))


def test_reverse_backward_vjp_matches_default_for_frozen_behavior() -> None:
    behavior = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        behavior.weight.copy_(
            torch.tensor(
                [
                    [0.5, -0.25],
                    [0.75, 0.125],
                ]
            )
        )
    behavior.requires_grad_(False)
    path = AMPath(
        states=torch.tensor(
            [
                [[0.5, -1.0], [1.5, 0.25]],
                [[-0.5, 0.75], [2.0, -1.5]],
            ]
        ),
        times=torch.tensor([[[0.0]], [[0.5]]]),
        sigmas=torch.ones(2, 2, 1),
        endpoint=torch.zeros(2, 2),
    )
    terminal = torch.tensor([[1.25, -0.5], [0.75, 2.0]])

    def behavior_velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return behavior(state)

    expected = reverse_behavior_adjoint(
        behavior_velocity,
        path,
        terminal,
    )
    actual = reverse_behavior_adjoint(
        behavior_velocity,
        path,
        terminal,
        use_backward_vjp=True,
    )

    torch.testing.assert_close(actual, expected)
    assert all(parameter.grad is None for parameter in behavior.parameters())


def test_sum_of_step_am_losses_matches_full_reduction_and_gradient() -> None:
    path = AMPath(
        states=torch.tensor(
            [
                [[0.5, -1.0], [1.5, 0.25]],
                [[-0.5, 0.75], [2.0, -1.5]],
            ]
        ),
        times=torch.tensor([[[0.0]], [[0.5]]]),
        sigmas=torch.tensor([[[1.5]], [[0.75]]]),
        endpoint=torch.zeros(2, 2),
    )
    adjoints = torch.tensor(
        [
            [[0.25, -0.5], [1.0, 0.75]],
            [[-0.25, 1.5], [0.5, -1.0]],
        ]
    )
    behavior = nn.Linear(2, 2, bias=False)
    fine = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        behavior.weight.copy_(torch.tensor([[0.5, 0.25], [-0.25, 0.75]]))
        fine.weight.copy_(torch.tensor([[0.125, -0.5], [0.25, 0.375]]))
    behavior.requires_grad_(False)

    def behavior_velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return behavior(state)

    def fine_velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return fine(state)

    full_loss = adjoint_matching_loss(
        fine_velocity,
        behavior_velocity,
        path,
        adjoints,
    )
    full_loss.backward()
    expected_gradient = fine.weight.grad.detach().clone()

    fine.weight.grad = None
    step_loss = torch.zeros(())
    for index in range(path.states.shape[0]):
        contribution = adjoint_matching_step_loss(
            fine_velocity,
            behavior_velocity,
            path.states[index],
            path.times[index],
            path.sigmas[index],
            adjoints[index],
        )
        contribution.backward()
        step_loss = step_loss + contribution.detach()

    torch.testing.assert_close(step_loss, full_loss.detach())
    torch.testing.assert_close(fine.weight.grad, expected_gradient)
    assert all(parameter.grad is None for parameter in behavior.parameters())


def test_official_loss_reductions_and_valid_gate() -> None:
    data_actions = torch.tensor(
        [
            [1.0, 3.0],
            [2.0, 4.0],
        ]
    )

    def zero_behavior(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.zeros_like(state)

    flow_loss = flow_matching_loss(
        zero_behavior,
        data_actions,
        initial_noise=torch.zeros_like(data_actions),
        times=torch.full((2, 1), 0.25),
        final_valid=torch.tensor([1.0, 0.0]),
    )
    torch.testing.assert_close(flow_loss, torch.tensor(2.5))

    critic_loss = ensemble_critic_mse(
        q_values=torch.tensor(
            [
                [1.0, 100.0],
                [3.0, 100.0],
            ]
        ),
        target=torch.tensor([2.0, 0.0]),
        final_valid=torch.tensor([1.0, 0.0]),
    )
    torch.testing.assert_close(critic_loss, torch.tensor(0.5))


def test_am_gradient_ownership_and_preupdate_ema() -> None:
    fine = nn.Linear(1, 1, bias=False)
    behavior = nn.Linear(1, 1, bias=False)
    critic = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        fine.weight.fill_(0.25)
        behavior.weight.fill_(0.5)
        critic.weight.fill_(2.0)

    path = AMPath(
        states=torch.tensor([[[1.0]], [[2.0]]]),
        times=torch.tensor([[[0.0]], [[0.5]]]),
        sigmas=torch.ones(2, 1, 1),
        endpoint=torch.tensor([[0.25]]),
    )

    def fine_velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return fine(state)

    def behavior_velocity(
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        del time
        return behavior(state)

    def target_critic(action: torch.Tensor) -> torch.Tensor:
        return critic(action).squeeze(-1).unsqueeze(0)

    terminal = terminal_mean_q_adjoint(
        target_critic,
        path.endpoint,
        inv_temp=0.5,
    )
    reverse = reverse_behavior_adjoint(
        behavior_velocity,
        path,
        terminal,
    )
    loss = adjoint_matching_loss(
        fine_velocity,
        behavior_velocity,
        path,
        reverse,
    )
    loss.backward()

    assert fine.weight.grad is not None
    assert torch.count_nonzero(fine.weight.grad).item() > 0
    assert behavior.weight.grad is None
    assert critic.weight.grad is None

    target = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        target.weight.zero_()
        fine.weight.fill_(1.0)
    preupdate = clone_parameter_snapshot(fine)
    with torch.no_grad():
        fine.weight.fill_(3.0)

    ema_from_preupdate_(target, preupdate, tau=0.25)
    torch.testing.assert_close(target.weight, torch.tensor([[0.25]]))
