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

"""Pure PyTorch numerical core for Q-learning with Adjoint Matching.

The executable semantics are a clean PyTorch translation of the MIT-licensed
implementation in ``ColinQiyangLi/qam`` at commit
``2726d767c9a0a7a46d49693f0391f73dc2cf58ac``. In particular, this module
matches ``agents/qam.py`` rather than introducing rollout, replay, or π0
integration policy.

Velocity and critic callables capture their observation conditioning. Actions
are flattened as ``[batch, action_dim]`` at this boundary. This keeps the
numerical core independent of the later π0 prefix-cache and action-projection
adapters.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

VelocityFn = Callable[[Tensor, Tensor], Tensor]
CriticFn = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class AMPath:
    """Detached auxiliary trajectory used by adjoint matching.

    Attributes:
        states: Pre-step states ``x_i`` with shape ``[K, B, D]``.
        times: QAM times ``t_i`` with shape ``[K, B, 1]``.
        sigmas: Memoryless-SDE scales with shape ``[K, B, 1]``.
        endpoint: Final state ``x_K`` with shape ``[B, D]``.
    """

    states: Tensor
    times: Tensor
    sigmas: Tensor
    endpoint: Tensor


def _time_tensor(x: Tensor, value: float) -> Tensor:
    """Create the official scalar-time batch shape for a flattened action."""
    return torch.full(
        (x.shape[0], 1),
        value,
        dtype=x.dtype,
        device=x.device,
    )


def flow_matching_loss(
    behavior_velocity: VelocityFn,
    data_actions: Tensor,
    initial_noise: Tensor,
    times: Tensor,
    final_valid: Tensor,
) -> Tensor:
    """Compute the official behavior flow-matching loss.

    The action-dimension MSE is reduced before applying the final-slot validity
    gate. The final mean still divides by the full batch size, including
    invalid samples whose contribution is zero.
    """
    interpolated = (1.0 - times) * initial_noise + times * data_actions
    target_velocity = data_actions - initial_noise
    predicted_velocity = behavior_velocity(interpolated, times)
    per_sample = (predicted_velocity - target_velocity).square().mean(dim=-1)
    valid = final_valid.reshape(data_actions.shape[0]).to(per_sample.dtype)
    return (per_sample * valid).mean()


@torch.no_grad()
def flow_ode_sample(
    velocity: VelocityFn,
    initial_noise: Tensor,
    *,
    flow_steps: int,
    clamp: tuple[float, float] | None = (-1.0, 1.0),
) -> Tensor:
    """Sample actions with the official forward Euler flow ODE."""
    if flow_steps <= 0:
        raise ValueError(f"flow_steps must be positive, got {flow_steps}")

    actions = initial_noise.detach()
    step_size = 1.0 / flow_steps
    for index in range(flow_steps):
        times = _time_tensor(actions, index * step_size)
        actions = actions + step_size * velocity(actions, times)

    if clamp is not None:
        actions = actions.clamp(*clamp)
    return actions.detach()


@torch.no_grad()
def sample_memoryless_am_path(
    fine_velocity: VelocityFn,
    behavior_velocity: VelocityFn,
    initial_noise: Tensor,
    step_noises: Tensor,
    *,
    flow_steps: int,
) -> AMPath:
    """Generate the detached auxiliary path from locked official semantics.

    The first ``K-1`` transitions use the fine-field memoryless SDE. The final
    transition uses a behavior-field Euler ODE at ``t_{K-1}``. Upstream still
    generates ``K`` noise tensors, so the final noise is accepted but
    intentionally unused.
    """
    if flow_steps <= 0:
        raise ValueError(f"flow_steps must be positive, got {flow_steps}")
    if step_noises.shape[0] != flow_steps:
        raise ValueError(
            "step_noises must contain one upstream noise tensor per flow step; "
            f"expected {flow_steps}, got {step_noises.shape[0]}"
        )

    state = initial_noise.detach()
    step_size = 1.0 / flow_steps
    sqrt_step = step_size**0.5
    states = []
    times = []
    sigmas = []

    for index in range(flow_steps):
        time = _time_tensor(state, index * step_size)
        sigma = torch.sqrt(2.0 * (1.0 - time + step_size) / (time + step_size))
        states.append(state)
        times.append(time)
        sigmas.append(sigma)

        if index != flow_steps - 1:
            fine = fine_velocity(state, time)
            drift = 2.0 * fine - state / (time + step_size)
            state = state + step_size * drift + sqrt_step * sigma * step_noises[index]
        else:
            state = state + step_size * behavior_velocity(state, time)

    return AMPath(
        states=torch.stack(states, dim=0).detach(),
        times=torch.stack(times, dim=0).detach(),
        sigmas=torch.stack(sigmas, dim=0).detach(),
        endpoint=state.detach(),
    )


def terminal_mean_q_adjoint(
    target_critic: CriticFn,
    action: Tensor,
    *,
    inv_temp: float,
    clip_action: bool = True,
) -> Tensor:
    """Initialize the terminal adjoint from target-ensemble mean Q.

    The official implementation averages only over the Q-head axis and then
    sums over the batch before differentiating. A batch mean would incorrectly
    scale every per-sample action gradient by ``1 / B``. The pessimistic
    ``rho * std`` term is deliberately absent here.
    """
    with torch.enable_grad():
        differentiable_action = action.detach().requires_grad_(True)
        critic_action = (
            differentiable_action.clamp(-1.0, 1.0)
            if clip_action
            else differentiable_action
        )
        q_values = target_critic(critic_action)
        mean_q_sum = q_values.mean(dim=0).sum()
        (action_gradient,) = torch.autograd.grad(
            mean_q_sum,
            differentiable_action,
            create_graph=False,
            retain_graph=False,
        )
    return (-float(inv_temp) * action_gradient).detach()


def reverse_behavior_adjoint(
    behavior_velocity: VelocityFn,
    path: AMPath,
    terminal_adjoint: Tensor,
    *,
    use_backward_vjp: bool = False,
) -> Tensor:
    """Propagate a terminal adjoint backward with input-only VJPs.

    For reverse index ``i``, the behavior field is evaluated at ``t_i + h``:

    ``F_i(x) = 2 f_beta(x, t_i + h) - x / (t_i + h)``.

    Returned adjoints and path states are detached. No graph through the full
    auxiliary trajectory is retained. ``use_backward_vjp`` uses ordinary
    ``Tensor.backward`` instead of ``autograd.grad`` because PyTorch FSDP does
    not support ``autograd.grad``. That route requires a frozen behavior model,
    as in the B1 production worker, so backward cannot accumulate parameter
    gradients.
    """
    flow_steps = path.states.shape[0]
    if flow_steps <= 0:
        raise ValueError("path must contain at least one flow step")

    step_size = 1.0 / flow_steps
    adjoint = terminal_adjoint.detach()
    reverse_adjoints = []

    for index in reversed(range(flow_steps)):
        with torch.enable_grad():
            state = path.states[index].detach().requires_grad_(True)
            reverse_time = path.times[index] + step_size
            reverse_drift = (
                2.0 * behavior_velocity(state, reverse_time) - state / reverse_time
            )
            if use_backward_vjp:
                reverse_drift.backward(gradient=adjoint)
                if state.grad is None:
                    raise RuntimeError(
                        "behavior input VJP backward did not populate state.grad"
                    )
                vector_jacobian_product = state.grad.detach()
            else:
                (vector_jacobian_product,) = torch.autograd.grad(
                    reverse_drift,
                    state,
                    grad_outputs=adjoint,
                    create_graph=False,
                    retain_graph=False,
                )
        adjoint = (adjoint + step_size * vector_jacobian_product).detach()
        reverse_adjoints.append(adjoint)

    return torch.stack(list(reversed(reverse_adjoints)), dim=0)


def adjoint_matching_loss(
    fine_velocity: VelocityFn,
    behavior_velocity: VelocityFn,
    path: AMPath,
    adjoints: Tensor,
) -> Tensor:
    """Compute the official non-residual fine-flow AM loss.

    The behavior output, path, and adjoints are treated as fixed targets. The
    reduction sums action dimensions and flow steps, then averages the batch.
    It does not multiply by the step size or by a replay validity gate.
    """
    states = path.states.detach()
    times = path.times.detach()
    sigmas = path.sigmas.detach()
    fixed_adjoints = adjoints.detach()

    fine = fine_velocity(states, times)
    with torch.no_grad():
        behavior = behavior_velocity(states, times)
    residual = 2.0 * (fine - behavior) / sigmas + sigmas * fixed_adjoints
    return residual.square().sum(dim=-1).sum(dim=0).mean()


def adjoint_matching_step_loss(
    fine_velocity: VelocityFn,
    behavior_velocity: VelocityFn,
    state: Tensor,
    time: Tensor,
    sigma: Tensor,
    adjoint: Tensor,
) -> Tensor:
    """Compute one flow-time contribution to the official AM reduction.

    The frozen behavior target is evaluated before the differentiable fine
    forward. This lets an FSDP caller backward each time step immediately,
    avoiding both unsupported ``autograd.grad`` use and retention of all flow
    step graphs at once. Summing these per-step batch means is exactly the
    reduction used by :func:`adjoint_matching_loss`.
    """
    fixed_state = state.detach()
    fixed_time = time.detach()
    fixed_sigma = sigma.detach()
    fixed_adjoint = adjoint.detach()
    with torch.no_grad():
        behavior = behavior_velocity(fixed_state, fixed_time)
    fine = fine_velocity(fixed_state, fixed_time)
    residual = 2.0 * (fine - behavior) / fixed_sigma + fixed_sigma * fixed_adjoint
    return residual.square().sum(dim=-1).mean()


def pessimistic_ensemble_value(q_values: Tensor, *, rho: float) -> Tensor:
    """Reduce ``[num_qs, batch]`` Q values with population standard deviation."""
    return q_values.mean(dim=0) - float(rho) * q_values.std(
        dim=0,
        correction=0,
    )


@torch.no_grad()
def q_chunk_td_target(
    return_h: Tensor,
    bootstrap_mask: Tensor,
    next_target_q_values: Tensor,
    *,
    discount_h: float,
    rho: float,
) -> Tensor:
    """Build the detached fixed-horizon pessimistic TD target."""
    next_q = pessimistic_ensemble_value(next_target_q_values, rho=rho)
    return (return_h + float(discount_h) * bootstrap_mask * next_q).detach()


def ensemble_critic_mse(
    q_values: Tensor,
    target: Tensor,
    final_valid: Tensor,
) -> Tensor:
    """Compute official ensemble critic MSE with the final-valid gate."""
    valid = final_valid.reshape(target.shape).to(q_values.dtype)
    return (
        (q_values - target.detach().unsqueeze(0)).square() * valid.unsqueeze(0)
    ).mean()


@torch.no_grad()
def clone_parameter_snapshot(module: nn.Module) -> dict[str, Tensor]:
    """Clone module parameters for an exact pre-update EMA source."""
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


@torch.no_grad()
def ema_from_preupdate_(
    target_module: nn.Module,
    online_preupdate_state: Mapping[str, Tensor],
    *,
    tau: float,
) -> None:
    """Update target parameters from a cloned pre-optimizer online snapshot."""
    target_parameters = dict(target_module.named_parameters())
    if target_parameters.keys() != online_preupdate_state.keys():
        raise ValueError("target and pre-update online parameter names differ")

    for name, target_parameter in target_parameters.items():
        source = online_preupdate_state[name].to(
            device=target_parameter.device,
            dtype=target_parameter.dtype,
        )
        target_parameter.mul_(1.0 - float(tau))
        target_parameter.add_(source, alpha=float(tau))
