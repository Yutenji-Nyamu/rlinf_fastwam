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

from types import SimpleNamespace

import torch
from torch import nn

from rlinf.models.embodiment.modules.ogpo_modules import (
    OGPOEMAActionExpert,
    keep_tied_paligemma_weight_in_root_fsdp_unit_,
    ogpo_time_to_pi0_time,
    pi0_velocity_to_ogpo,
    project_ogpo_action_views,
    project_ogpo_canonical_action,
)
from rlinf.models.embodiment.openpi.openpi_ogpo import (
    clipped_ogpo_actor_loss,
    conservative_ogpo_advantages,
    flow_matching_success_bc_loss,
    openpi_velocity_as_ogpo,
    sample_ogpo_chains,
    sample_tapered_chains,
    score_ogpo_chains,
    score_tapered_chains,
)


class _ToyExpertModel(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Linear(width, width)
        self.config = SimpleNamespace(_attn_implementation=None)

    def forward(self, *, inputs_embeds, **_):
        return SimpleNamespace(last_hidden_state=self.proj(inputs_embeds))


class _ToyGemmaExpert(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.model = _ToyExpertModel(width)
        self.lm_head = nn.Linear(width, 99)


class _ToyPaliWithExpert(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gemma_expert = _ToyGemmaExpert(width)


class _ToyOnlinePi0(nn.Module):
    def __init__(self):
        super().__init__()
        width = 4
        action_dim = 2
        self.pi05 = False
        self.config = SimpleNamespace(action_horizon=2)
        self.paligemma_with_expert = _ToyPaliWithExpert(width)
        self.action_in_proj = nn.Linear(action_dim, width)
        self.action_out_proj = nn.Linear(width, action_dim)
        self.state_proj = nn.Linear(action_dim, width)
        self.action_time_mlp_in = nn.Linear(2 * width, width)
        self.action_time_mlp_out = nn.Linear(width, width)


class _ScaledOGPOVelocity(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x_t, time_ogpo):
        return self.scale * x_t + time_ogpo[..., None, None]


def test_ogpo_keeps_only_tied_paligemma_head_in_root_fsdp_unit():
    embedding = nn.Embedding(13, 5)
    tied_head = nn.Linear(5, 13, bias=False)
    tied_head.weight = embedding.weight
    embedding._fsdp_wrap_name = "embed_tokens"
    tied_head._fsdp_wrap_name = "lm_head"

    changed = keep_tied_paligemma_weight_in_root_fsdp_unit_(
        embedding, tied_head
    )

    assert changed
    assert embedding._fsdp_wrap_name == "embed_tokens"
    assert tied_head._fsdp_wrap_name == "ogpo_tied_paligemma_lm_head_root"
    assert tied_head.weight is embedding.weight

    independent_head = nn.Linear(5, 13, bias=False)
    independent_head._fsdp_wrap_name = "lm_head"
    changed = keep_tied_paligemma_weight_in_root_fsdp_unit_(
        embedding, independent_head
    )
    assert not changed
    assert independent_head._fsdp_wrap_name == "lm_head"


def test_ogpo_time_velocity_mapping_and_ema_copy_are_pure_ogpo():
    time_ogpo = torch.tensor([0.0, 0.25, 1.0])
    torch.testing.assert_close(
        ogpo_time_to_pi0_time(time_ogpo),
        torch.tensor([1.0, 0.75, 0.0]),
    )
    torch.testing.assert_close(
        pi0_velocity_to_ogpo(torch.tensor([1.0, -2.0])),
        torch.tensor([-1.0, 2.0]),
    )

    online = _ToyOnlinePi0()
    ema = OGPOEMAActionExpert(online)
    assert all(not parameter.requires_grad for parameter in ema.parameters())
    assert all("lm_head" not in name for name, _ in ema.named_parameters())
    ema.train()
    assert not ema.training

    ema_parameter = next(ema.parameters())
    online_parameter = next(
        online.paligemma_with_expert.gemma_expert.model.parameters()
    )
    with torch.no_grad():
        online_parameter.add_(2.0)
    ema.copy_from_online_(online)
    torch.testing.assert_close(ema_parameter, online_parameter)

    old_target = ema_parameter.detach().clone()
    with torch.no_grad():
        online_parameter.add_(4.0)
    ema.polyak_update_from_online_(online, tau=0.25)
    torch.testing.assert_close(ema_parameter, old_target + 1.0)


def test_low_precision_ema_uses_and_restores_fp32_shadow():
    online = _ToyOnlinePi0().to(dtype=torch.bfloat16)
    ema = OGPOEMAActionExpert(online)
    ema.match_online_dtypes_(online)
    ema.copy_from_online_(online)

    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(torch.full_like(parameter, 0.125))
    ema.polyak_update_from_online_(online, tau=0.25)

    shadow = ema.ema_shadow_state()
    assert shadow
    assert all(value.dtype == torch.float32 for value in shadow.values())

    restored = OGPOEMAActionExpert(online)
    restored.match_online_dtypes_(online)
    restored.copy_from_online_(online)
    restored.load_ema_shadow_state_(shadow, online)
    restored_shadow = restored.ema_shadow_state()
    assert restored_shadow.keys() == shadow.keys()
    for name, value in shadow.items():
        torch.testing.assert_close(restored_shadow[name], value)


def test_openpi_velocity_callback_reverses_both_time_and_velocity():
    observed = {}

    def velocity_pi0(x_t, time_pi0):
        observed["time_pi0"] = time_pi0
        return x_t + time_pi0[..., None, None]

    x_t = torch.ones(1, 2, 3, 2)
    time_ogpo = torch.tensor([[0.25, 0.75]])
    velocity = openpi_velocity_as_ogpo(velocity_pi0, x_t, time_ogpo)

    torch.testing.assert_close(
        observed["time_pi0"], torch.tensor([[0.75, 0.25]])
    )
    torch.testing.assert_close(
        velocity,
        -(x_t + observed["time_pi0"][..., None, None]),
    )


def test_success_bc_uses_ogpo_velocity_sign_and_executed_prefix_only():
    predicted_velocity = torch.cat(
        [torch.zeros(2, 2, 3), torch.full((2, 2, 3), 100.0)], dim=1
    ).requires_grad_()
    actions = torch.full((2, 4, 3), 2.0)
    noise = torch.full((2, 4, 3), 0.5)

    loss = flow_matching_success_bc_loss(
        predicted_velocity,
        actions,
        noise,
        execution_horizon=2,
    )

    torch.testing.assert_close(loss, torch.tensor(2.25))
    loss.backward()
    assert torch.count_nonzero(predicted_velocity.grad[:, :2]) > 0
    assert torch.count_nonzero(predicted_velocity.grad[:, 2:]) == 0


def test_tapered_raw_chain_same_policy_score_and_projection_contract():
    flow_steps = 4
    sigma_init = 0.1
    initial_noise = torch.tensor(
        [
            [
                [[1.4, -0.2, 0.3], [0.1, 0.2, 0.3]],
                [[-0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            ]
        ],
        dtype=torch.float32,
    )
    transition_noise = torch.zeros(1, 2, flow_steps, 2, 3)
    transition_noise[0, 0, 0, 0, 0] = 4.0  # must become +3 sigma
    target_velocity = _ScaledOGPOVelocity(0.2)

    sample = sample_ogpo_chains(
        target_velocity,
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        flow_steps=flow_steps,
        sigma_init=sigma_init,
        executed_horizon=1,
        active_action_dim=2,
    )

    assert sample.raw_chains.shape == (1, 2, 5, 2, 3)
    assert not sample.raw_chains.requires_grad
    assert sample.old_chain_score.shape == (1, 2)
    assert sample.canonical_action.shape == (1, 2, 1, 2)

    # Independent first-step oracle: t_ogpo=0, no intermediate/final clamp.
    x_0 = initial_noise
    velocity_0 = 0.2 * x_0
    correction_0 = 0.5 * sigma_init**2 * (-x_0)
    mean_0 = x_0 + (velocity_0 + correction_0) / flow_steps
    epsilon_0 = transition_noise[:, :, 0].clamp(-3.0, 3.0)
    expected_x_1 = mean_0 + sigma_init * epsilon_0
    torch.testing.assert_close(sample.raw_chains[:, :, 1], expected_x_1)

    current_score = score_ogpo_chains(
        target_velocity,
        sample.raw_chains,
        sigma_init=sigma_init,
    )
    torch.testing.assert_close(
        current_score,
        sample.old_chain_score,
        atol=1e-6,
        rtol=1e-6,
    )
    current_score.sum().backward()
    assert target_velocity.scale.grad is not None
    assert torch.isfinite(target_velocity.scale.grad)
    torch.testing.assert_close(
        sample.canonical_action,
        sample.raw_final_action[..., :1, :2],
    )

    # Canonical projection is an out-of-place slice and cannot mutate chain.
    raw_before = sample.raw_final_action.clone()
    projected = project_ogpo_canonical_action(
        sample.raw_final_action,
        executed_horizon=1,
        active_action_dim=2,
    )
    projected.add_(100.0)
    torch.testing.assert_close(sample.raw_final_action, raw_before)


def test_public_tapered_adapter_names_and_action_views():
    initial_noise = torch.zeros(1, 2, 2, 3)
    step_noise = torch.zeros(1, 2, 2, 2, 3)
    velocity = _ScaledOGPOVelocity(0.0)
    sample = sample_tapered_chains(
        velocity,
        initial_noise,
        step_noise,
        0.1,
        executed_horizon=1,
        active_action_dim=2,
    )
    current_score = score_tapered_chains(
        velocity,
        sample.raw_chains,
        0.1,
    )
    torch.testing.assert_close(current_score, sample.old_chain_score)

    raw, canonical = project_ogpo_action_views(
        sample.raw_final_action,
        executed_horizon=1,
        active_action_dim=2,
    )
    assert raw is sample.raw_final_action
    canonical.add_(1.0)
    torch.testing.assert_close(raw, sample.raw_final_action)


def test_ogpo_ca_and_whole_chain_ppo_keep_online_gradient_only():
    q_full = torch.tensor(
        [
            [
                [3.0, 2.0, 1.0],
                [-3.0, -2.0, -1.0],
                [1.0, -1.0, 0.5],
                [-1.0, 1.0, -0.5],
            ]
        ]
    )
    advantages = conservative_ogpo_advantages(q_full)
    torch.testing.assert_close(advantages, torch.tensor([[1.0, -1.0, 0.0, 0.0]]))
    shifted = q_full + torch.tensor([10.0, -7.0, 2.0])
    torch.testing.assert_close(conservative_ogpo_advantages(shifted), advantages)
    assert not advantages.requires_grad

    current = torch.tensor([[0.02, -0.03]], requires_grad=True)
    old = torch.zeros_like(current, requires_grad=True)
    ppo_advantages = torch.tensor([[1.0, -1.0]], requires_grad=True)
    loss, stats = clipped_ogpo_actor_loss(
        current,
        old,
        ppo_advantages,
        clip_epsilon=0.01,
    )
    loss.backward()

    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert old.grad is None
    assert ppo_advantages.grad is None
    assert stats["ratio"].ndim == 0
