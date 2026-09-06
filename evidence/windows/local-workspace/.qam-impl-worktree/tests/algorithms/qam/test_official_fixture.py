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

"""Pure-PyTorch parity checks against the locked official QAM fixture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rlinf.algorithms.qam.core import (
    adjoint_matching_loss,
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

EXPECTED_COMMIT = "2726d767c9a0a7a46d49693f0391f73dc2cf58ac"
FIXTURE_PATH = Path(__file__).with_name("oracle") / "qam_official_2726d767_v1.npz"
FORWARD_ATOL = 2e-5
FORWARD_RTOL = 3e-5
GRAD_ATOL = 4e-5
GRAD_RTOL = 5e-4


@dataclass(frozen=True)
class OfficialFixture:
    """In-memory numeric fixture and its decoded path manifest."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def tensor(self, name: str) -> Tensor:
        """Return an owned CPU tensor without sharing read-only NPZ storage."""
        return torch.from_numpy(np.array(self.arrays[name], copy=True))

    def tree(self, prefix: str) -> dict[str, Tensor]:
        """Decode one flattened JAX pytree by its recorded Flax paths."""
        return {
            entry["path"]: self.tensor(entry["key"])
            for entry in self.metadata["tree_manifest"][prefix]
        }


@pytest.fixture(scope="module")
def official_fixture() -> OfficialFixture:
    """Load the server-generated fixture without enabling pickle."""
    with np.load(FIXTURE_PATH, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    assert arrays
    assert all(not array.dtype.hasobject for array in arrays.values())
    metadata = json.loads(arrays["metadata_json_utf8"].tobytes().decode("utf-8"))
    assert metadata["source"]["source_commit"] == EXPECTED_COMMIT
    assert metadata["dimensions"] == {
        "action": 4,
        "batch": 2,
        "flow_steps": 3,
        "hidden_dims": [8, 8],
        "num_qs": 10,
        "observation": 3,
    }
    assert metadata["jax_platform"] == "cpu"
    assert metadata["jax_enable_x64"] is False
    return OfficialFixture(arrays=arrays, metadata=metadata)


class FlaxMLP(nn.Module):
    """Small PyTorch MLP with the locked Flax operation ordering."""

    def __init__(
        self,
        dimensions: tuple[int, ...],
        *,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(input_dim, output_dim)
            for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:])
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(output_dim, eps=1e-6)
            for output_dim in (dimensions[1:-1] if layer_norm else ())
        )
        self.layer_norm = layer_norm

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply Dense, tanh-approximate GELU, then optional LayerNorm."""
        value = inputs
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index != len(self.layers) - 1:
                value = F.gelu(value, approximate="tanh")
                if self.layer_norm:
                    value = self.norms[index](value)
        return value

    def load_flax(
        self,
        tree: dict[str, Tensor],
        root: str,
    ) -> None:
        """Load Flax Dense kernels with the required output/input transpose."""
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                layer.bias.copy_(tree[f"{root}/Dense_{index}/bias"])
                layer.weight.copy_(
                    tree[f"{root}/Dense_{index}/kernel"].transpose(-1, -2)
                )
                if index != len(self.layers) - 1 and self.layer_norm:
                    self.norms[index].bias.copy_(tree[f"{root}/LayerNorm_{index}/bias"])
                    self.norms[index].weight.copy_(
                        tree[f"{root}/LayerNorm_{index}/scale"]
                    )

    def flax_tensors(self, root: str, *, gradients: bool) -> dict[str, Tensor]:
        """Expose parameters or gradients in their original Flax orientation."""
        output: dict[str, Tensor] = {}
        for index, layer in enumerate(self.layers):
            bias = layer.bias.grad if gradients else layer.bias
            kernel = layer.weight.grad if gradients else layer.weight
            if bias is None or kernel is None:
                raise AssertionError(f"Missing MLP gradient at Dense_{index}")
            output[f"{root}/Dense_{index}/bias"] = bias
            output[f"{root}/Dense_{index}/kernel"] = kernel.transpose(-1, -2)
            if index != len(self.layers) - 1 and self.layer_norm:
                norm_bias = (
                    self.norms[index].bias.grad if gradients else self.norms[index].bias
                )
                norm_scale = (
                    self.norms[index].weight.grad
                    if gradients
                    else self.norms[index].weight
                )
                if norm_bias is None or norm_scale is None:
                    raise AssertionError(f"Missing MLP gradient at LayerNorm_{index}")
                output[f"{root}/LayerNorm_{index}/bias"] = norm_bias
                output[f"{root}/LayerNorm_{index}/scale"] = norm_scale
        return output


class FlaxActor(nn.Module):
    """Official action vector field with explicit observation conditioning."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = FlaxMLP((8, 8, 8, 4), layer_norm=False)

    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
        times: Tensor,
    ) -> Tensor:
        """Evaluate the actor for either `[B,D]` or `[K,B,D]` actions."""
        conditioned = observations
        while conditioned.ndim < actions.ndim:
            conditioned = conditioned.unsqueeze(0)
        conditioned = conditioned.expand(*actions.shape[:-1], observations.shape[-1])
        return self.mlp(torch.cat((conditioned, actions, times), dim=-1))

    def load_flax(self, tree: dict[str, Tensor], module: str) -> None:
        """Load one actor module from a full official parameter tree."""
        self.mlp.load_flax(tree, f"{module}/mlp")

    def flax_tensors(
        self,
        module: str,
        *,
        gradients: bool,
    ) -> dict[str, Tensor]:
        """Expose one actor module in official path/orientation."""
        root = f"{module}/mlp" if module else "mlp"
        return self.mlp.flax_tensors(
            root,
            gradients=gradients,
        )


class FlaxCritic(nn.Module):
    """Ten fully independent Flax-compatible critic MLPs."""

    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            FlaxMLP((7, 8, 8, 1), layer_norm=True) for _ in range(10)
        )

    def forward(self, observations: Tensor, actions: Tensor) -> Tensor:
        """Return official `[num_qs,batch]` ensemble values."""
        inputs = torch.cat((observations, actions), dim=-1)
        return torch.stack(
            [head(inputs).squeeze(-1) for head in self.heads],
            dim=0,
        )

    def load_flax(self, tree: dict[str, Tensor], module: str) -> None:
        """Load the ensemble axis into ten independent PyTorch MLPs."""
        root = f"{module}/value_net"
        with torch.no_grad():
            for head_index, head in enumerate(self.heads):
                for layer_index, layer in enumerate(head.layers):
                    layer.bias.copy_(
                        tree[f"{root}/Dense_{layer_index}/bias"][head_index]
                    )
                    layer.weight.copy_(
                        tree[f"{root}/Dense_{layer_index}/kernel"][
                            head_index
                        ].transpose(-1, -2)
                    )
                    if layer_index != len(head.layers) - 1:
                        head.norms[layer_index].bias.copy_(
                            tree[f"{root}/LayerNorm_{layer_index}/bias"][head_index]
                        )
                        head.norms[layer_index].weight.copy_(
                            tree[f"{root}/LayerNorm_{layer_index}/scale"][head_index]
                        )

    def flax_tensors(
        self,
        module: str,
        *,
        gradients: bool,
    ) -> dict[str, Tensor]:
        """Stack independent PyTorch heads back onto the Flax ensemble axis."""
        root = f"{module}/value_net" if module else "value_net"
        output: dict[str, Tensor] = {}
        for layer_index in range(3):
            biases = []
            kernels = []
            norm_biases = []
            norm_scales = []
            for head in self.heads:
                layer = head.layers[layer_index]
                bias = layer.bias.grad if gradients else layer.bias
                kernel = layer.weight.grad if gradients else layer.weight
                if bias is None or kernel is None:
                    raise AssertionError(
                        f"Missing critic gradient at Dense_{layer_index}"
                    )
                biases.append(bias)
                kernels.append(kernel.transpose(-1, -2))
                if layer_index != 2:
                    norm_bias = (
                        head.norms[layer_index].bias.grad
                        if gradients
                        else head.norms[layer_index].bias
                    )
                    norm_scale = (
                        head.norms[layer_index].weight.grad
                        if gradients
                        else head.norms[layer_index].weight
                    )
                    if norm_bias is None or norm_scale is None:
                        raise AssertionError(
                            f"Missing critic LayerNorm gradient at {layer_index}"
                        )
                    norm_biases.append(norm_bias)
                    norm_scales.append(norm_scale)
            output[f"{root}/Dense_{layer_index}/bias"] = torch.stack(biases)
            output[f"{root}/Dense_{layer_index}/kernel"] = torch.stack(kernels)
            if layer_index != 2:
                output[f"{root}/LayerNorm_{layer_index}/bias"] = torch.stack(
                    norm_biases
                )
                output[f"{root}/LayerNorm_{layer_index}/scale"] = torch.stack(
                    norm_scales
                )
        return output


def _actor(
    fixture: OfficialFixture,
    module: str,
    *,
    tree_prefix: str = "params_before",
    tree_root: str | None = None,
) -> FlaxActor:
    actor = FlaxActor()
    actor.load_flax(
        fixture.tree(tree_prefix),
        tree_root if tree_root is not None else module,
    )
    return actor


def _critic(
    fixture: OfficialFixture,
    module: str,
    *,
    tree_prefix: str = "params_before",
    tree_root: str | None = None,
) -> FlaxCritic:
    critic = FlaxCritic()
    critic.load_flax(
        fixture.tree(tree_prefix),
        tree_root if tree_root is not None else module,
    )
    return critic


def _assert_close(
    actual: Tensor,
    expected: Tensor,
    *,
    atol: float = FORWARD_ATOL,
    rtol: float = FORWARD_RTOL,
) -> None:
    torch.testing.assert_close(
        actual.detach(),
        expected.detach().to(dtype=actual.dtype),
        atol=atol,
        rtol=rtol,
    )


def _assert_flax_mapping(
    actual: dict[str, Tensor],
    expected: dict[str, Tensor],
    *,
    atol: float,
    rtol: float,
) -> None:
    for path, tensor in actual.items():
        assert path in expected
        _assert_close(tensor, expected[path], atol=atol, rtol=rtol)


def _online_parameters(
    fine: FlaxActor,
    behavior: FlaxActor,
    critic: FlaxCritic,
) -> list[nn.Parameter]:
    return [
        *fine.parameters(),
        *behavior.parameters(),
        *critic.parameters(),
    ]


def test_official_q_flow_matching_and_critic_path(
    official_fixture: OfficialFixture,
) -> None:
    """Match official behavior FM, fine ODE next action, Q target, and losses."""
    fixture = official_fixture
    observations = fixture.tensor("batch_observations")
    actions = fixture.tensor("batch_actions")[:, 0]
    next_observations = fixture.tensor("critic_next_observations")
    valid = fixture.tensor("batch_valid")[:, -1]

    behavior = _actor(fixture, "modules_actor_slow")
    fine = _actor(fixture, "modules_actor_fast")
    critic = _critic(fixture, "modules_critic")
    target_critic = _critic(fixture, "modules_target_critic")

    def behavior_velocity(action: Tensor, time: Tensor) -> Tensor:
        return behavior(observations, action, time)

    fm_prediction = behavior_velocity(
        fixture.tensor("fm_xt"),
        fixture.tensor("fm_t"),
    )
    _assert_close(fm_prediction, fixture.tensor("fm_prediction"))
    fm_loss = flow_matching_loss(
        behavior_velocity,
        actions,
        fixture.tensor("fm_x0"),
        fixture.tensor("fm_t"),
        valid,
    )
    _assert_close(fm_loss, fixture.tensor("loss_flow_matching"))

    next_action_noise = fixture.tensor("random_next_action_noise").squeeze(1)

    def next_fine_velocity(action: Tensor, time: Tensor) -> Tensor:
        return fine(next_observations, action, time)

    next_actions = flow_ode_sample(
        next_fine_velocity,
        next_action_noise,
        flow_steps=3,
    )
    _assert_close(next_actions, fixture.tensor("critic_next_actions"))

    current_qs = critic(observations, actions)
    next_target_qs = target_critic(next_observations, next_actions)
    _assert_close(current_qs, fixture.tensor("critic_current_qs"))
    _assert_close(next_target_qs, fixture.tensor("critic_next_target_qs"))
    pessimistic_next_q = pessimistic_ensemble_value(
        next_target_qs,
        rho=0.5,
    )
    _assert_close(
        pessimistic_next_q,
        fixture.tensor("critic_pessimistic_next_q"),
    )
    target = q_chunk_td_target(
        fixture.tensor("batch_rewards")[:, -1],
        fixture.tensor("batch_masks")[:, -1],
        next_target_qs,
        discount_h=0.99,
        rho=0.5,
    )
    _assert_close(target, fixture.tensor("critic_target"))
    critic_loss = ensemble_critic_mse(current_qs, target, valid)
    _assert_close(critic_loss, fixture.tensor("loss_critic"))


def test_official_am_path_endpoint_reverse_and_loss(
    official_fixture: OfficialFixture,
) -> None:
    """Match the auxiliary path, endpoint Q adjoint, reverse VJP, and AM loss."""
    fixture = official_fixture
    observations = fixture.tensor("batch_observations")
    fine = _actor(fixture, "modules_actor_fast")
    behavior = _actor(fixture, "modules_target_actor_slow")
    target_critic = _critic(fixture, "modules_target_critic")

    def fine_velocity(action: Tensor, time: Tensor) -> Tensor:
        return fine(observations, action, time)

    def behavior_velocity(action: Tensor, time: Tensor) -> Tensor:
        return behavior(observations, action, time)

    def target_q(action: Tensor) -> Tensor:
        return target_critic(observations, action)

    path = sample_memoryless_am_path(
        fine_velocity,
        behavior_velocity,
        fixture.tensor("random_adj_x0"),
        fixture.tensor("random_adj_step_noises"),
        flow_steps=3,
    )
    _assert_close(path.states, fixture.tensor("adj_xs_official"))
    _assert_close(path.times, fixture.tensor("adj_times_official"))
    _assert_close(path.sigmas, fixture.tensor("adj_forward_sigmas"))
    _assert_close(path.endpoint, fixture.tensor("adj_endpoint"))
    _assert_close(
        target_q(path.endpoint.clamp(-1.0, 1.0)),
        fixture.tensor("adj_endpoint_target_qs"),
    )

    terminal = terminal_mean_q_adjoint(
        target_q,
        path.endpoint,
        inv_temp=0.3,
        clip_action=True,
    )
    _assert_close(
        terminal,
        fixture.tensor("adj_terminal"),
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    _assert_close(
        -terminal / 0.3,
        fixture.tensor("adj_endpoint_q_gradient"),
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    reverse = reverse_behavior_adjoint(
        behavior_velocity,
        path,
        terminal,
    )
    _assert_close(
        reverse,
        fixture.tensor("adj_reverse_states"),
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    fine_values = fine_velocity(path.states, path.times)
    behavior_values = behavior_velocity(path.states, path.times)
    residual = (
        2.0 * (fine_values - behavior_values) / path.sigmas + path.sigmas * reverse
    )
    _assert_close(fine_values, fixture.tensor("adj_fine_velocity"))
    _assert_close(behavior_values, fixture.tensor("adj_base_velocity"))
    _assert_close(
        residual,
        fixture.tensor("adj_am_residual"),
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    am_loss = adjoint_matching_loss(
        fine_velocity,
        behavior_velocity,
        path,
        reverse,
    )
    _assert_close(
        am_loss,
        fixture.tensor("loss_adjoint_matching"),
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    decomposed_total = (
        fixture.tensor("loss_flow_matching") + fixture.tensor("loss_critic") + am_loss
    )
    _assert_close(
        decomposed_total,
        fixture.tensor("loss_total"),
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )


def test_official_gradients_adam_and_preupdate_ema(
    official_fixture: OfficialFixture,
) -> None:
    """Match all online gradients, one Adam step, and old-parameter EMA."""
    fixture = official_fixture
    observations = fixture.tensor("batch_observations")
    actions = fixture.tensor("batch_actions")[:, 0]
    next_observations = fixture.tensor("critic_next_observations")
    valid = fixture.tensor("batch_valid")[:, -1]

    fine = _actor(fixture, "modules_actor_fast")
    behavior = _actor(fixture, "modules_actor_slow")
    target_behavior = _actor(fixture, "modules_target_actor_slow")
    critic = _critic(fixture, "modules_critic")
    target_critic = _critic(fixture, "modules_target_critic")

    behavior_preupdate = clone_parameter_snapshot(behavior)
    critic_preupdate = clone_parameter_snapshot(critic)
    fine_before = {
        path: tensor.detach().clone()
        for path, tensor in fine.flax_tensors(
            "modules_actor_fast", gradients=False
        ).items()
    }
    behavior_before = {
        path: tensor.detach().clone()
        for path, tensor in behavior.flax_tensors(
            "modules_actor_slow", gradients=False
        ).items()
    }
    critic_before = {
        path: tensor.detach().clone()
        for path, tensor in critic.flax_tensors(
            "modules_critic", gradients=False
        ).items()
    }

    def fine_velocity(action: Tensor, time: Tensor) -> Tensor:
        return fine(observations, action, time)

    def behavior_velocity(action: Tensor, time: Tensor) -> Tensor:
        return behavior(observations, action, time)

    def target_behavior_velocity(action: Tensor, time: Tensor) -> Tensor:
        return target_behavior(observations, action, time)

    def target_q(action: Tensor) -> Tensor:
        return target_critic(observations, action)

    fm_loss = flow_matching_loss(
        behavior_velocity,
        actions,
        fixture.tensor("fm_x0"),
        fixture.tensor("fm_t"),
        valid,
    )
    path = sample_memoryless_am_path(
        fine_velocity,
        target_behavior_velocity,
        fixture.tensor("random_adj_x0"),
        fixture.tensor("random_adj_step_noises"),
        flow_steps=3,
    )
    terminal = terminal_mean_q_adjoint(
        target_q,
        path.endpoint,
        inv_temp=0.3,
    )
    reverse = reverse_behavior_adjoint(
        target_behavior_velocity,
        path,
        terminal,
    )
    am_loss = adjoint_matching_loss(
        fine_velocity,
        target_behavior_velocity,
        path,
        reverse,
    )

    next_action_noise = fixture.tensor("random_next_action_noise").squeeze(1)

    def next_fine_velocity(action: Tensor, time: Tensor) -> Tensor:
        return fine(next_observations, action, time)

    next_actions = flow_ode_sample(
        next_fine_velocity,
        next_action_noise,
        flow_steps=3,
    )
    next_target_qs = target_critic(next_observations, next_actions)
    target = q_chunk_td_target(
        fixture.tensor("batch_rewards")[:, -1],
        fixture.tensor("batch_masks")[:, -1],
        next_target_qs,
        discount_h=0.99,
        rho=0.5,
    )
    critic_loss = ensemble_critic_mse(
        critic(observations, actions),
        target,
        valid,
    )
    total_loss = fm_loss + am_loss + critic_loss
    _assert_close(
        total_loss,
        fixture.tensor("loss_total"),
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    total_loss.backward()

    raw_expected = fixture.tree("gradients_raw")
    _assert_flax_mapping(
        fine.flax_tensors("modules_actor_fast", gradients=True),
        raw_expected,
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    _assert_flax_mapping(
        behavior.flax_tensors("modules_actor_slow", gradients=True),
        raw_expected,
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    _assert_flax_mapping(
        critic.flax_tensors("modules_critic", gradients=True),
        raw_expected,
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    parameters = _online_parameters(fine, behavior, critic)
    raw_global_norm = torch.sqrt(
        sum(parameter.grad.square().sum() for parameter in parameters)
    )
    _assert_close(
        raw_global_norm,
        fixture.tensor("gradient_raw_global_norm"),
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
    clipped_expected = fixture.tree("gradients_clipped")
    clipped_actual = {
        **fine.flax_tensors("modules_actor_fast", gradients=True),
        **behavior.flax_tensors("modules_actor_slow", gradients=True),
        **critic.flax_tensors("modules_critic", gradients=True),
    }
    _assert_flax_mapping(
        {
            path: tensor
            for path, tensor in clipped_actual.items()
            if path.startswith("modules_actor_fast/")
        },
        clipped_expected,
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    _assert_flax_mapping(
        {
            path: tensor
            for path, tensor in clipped_actual.items()
            if path.startswith("modules_actor_slow/")
        },
        clipped_expected,
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    _assert_flax_mapping(
        {
            path: tensor
            for path, tensor in clipped_actual.items()
            if path.startswith("modules_critic/")
        },
        clipped_expected,
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    optimizer_before = fixture.tree("optimizer_state_before")
    optimizer_after = fixture.tree("optimizer_state_after")
    assert optimizer_before["1/0/count"].item() == 0
    assert optimizer_after["1/0/count"].item() == 1
    for path, gradient in clipped_actual.items():
        _assert_close(
            gradient * 0.1,
            optimizer_after[f"1/0/mu/{path}"],
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
        )
        _assert_close(
            gradient.square() * 0.001,
            optimizer_after[f"1/0/nu/{path}"],
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
        )

    optimizer = torch.optim.Adam(
        parameters,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    optimizer.step()
    after_adam = fixture.tree("params_after_adam")
    _assert_flax_mapping(
        fine.flax_tensors("modules_actor_fast", gradients=False),
        after_adam,
        atol=2e-6,
        rtol=2e-5,
    )
    _assert_flax_mapping(
        behavior.flax_tensors("modules_actor_slow", gradients=False),
        after_adam,
        atol=2e-6,
        rtol=2e-5,
    )
    _assert_flax_mapping(
        critic.flax_tensors("modules_critic", gradients=False),
        after_adam,
        atol=2e-6,
        rtol=2e-5,
    )

    adam_updates = fixture.tree("adam_updates")
    actual_updates = {
        **{
            path: tensor - fine_before[path]
            for path, tensor in fine.flax_tensors(
                "modules_actor_fast", gradients=False
            ).items()
        },
        **{
            path: tensor - behavior_before[path]
            for path, tensor in behavior.flax_tensors(
                "modules_actor_slow", gradients=False
            ).items()
        },
        **{
            path: tensor - critic_before[path]
            for path, tensor in critic.flax_tensors(
                "modules_critic", gradients=False
            ).items()
        },
    }
    _assert_flax_mapping(
        actual_updates,
        adam_updates,
        atol=2e-6,
        rtol=2e-5,
    )

    ema_from_preupdate_(
        target_behavior,
        behavior_preupdate,
        tau=0.005,
    )
    ema_from_preupdate_(
        target_critic,
        critic_preupdate,
        tau=0.005,
    )
    _assert_flax_mapping(
        target_behavior.flax_tensors("", gradients=False),
        fixture.tree("target_actor_slow_expected_preupdate_ema"),
        atol=2e-7,
        rtol=2e-6,
    )
    _assert_flax_mapping(
        target_critic.flax_tensors("", gradients=False),
        fixture.tree("target_critic_expected_preupdate_ema"),
        atol=2e-7,
        rtol=2e-6,
    )

    for prefix in (
        "modules_target_actor_fast",
        "modules_target_actor_slow",
        "modules_target_critic",
    ):
        target_gradients = [
            tensor
            for path, tensor in raw_expected.items()
            if path.startswith(prefix + "/")
        ]
        assert target_gradients
        assert all(torch.count_nonzero(tensor) == 0 for tensor in target_gradients)
