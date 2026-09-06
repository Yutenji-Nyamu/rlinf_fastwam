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

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.qam_modules import (
    QAMFineActionExpert,
    build_qam_projection_fingerprint,
    canonicalize_qam_rollout_action,
    keep_tied_embedding_and_lm_head_in_root_fsdp_unit_,
    pool_qam_prefix_blocks,
    projection_fingerprint_tensor,
)
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
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


class _ToyBehavior(nn.Module):
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


def test_qam_prefix_pooling_uses_runtime_width_and_masks():
    prefix = torch.arange(2 * 8 * 5, dtype=torch.float32).reshape(2, 8, 5)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[0, 2:4] = False
    mask[1, 7] = False

    pooled, block_lengths = pool_qam_prefix_blocks(
        prefix,
        mask,
        language_token_count=2,
        num_image_blocks=3,
    )

    assert pooled.shape == (2, 4, 5)
    assert block_lengths == (2, 2, 2, 2)
    torch.testing.assert_close(pooled[0, 1], torch.zeros(5))
    torch.testing.assert_close(pooled[1, 3], prefix[1, 6])


def test_qam_keeps_tied_paligemma_weight_in_one_fsdp_unit():
    embedding = nn.Embedding(13, 5)
    tied_head = nn.Linear(5, 13, bias=False)
    tied_head.weight = embedding.weight
    embedding._fsdp_wrap_name = "embed_tokens"
    tied_head._fsdp_wrap_name = "lm_head"

    changed = keep_tied_embedding_and_lm_head_in_root_fsdp_unit_(embedding, tied_head)

    assert changed
    assert embedding._fsdp_wrap_name == "embed_tokens"
    assert tied_head._fsdp_wrap_name == "qam_tied_paligemma_lm_head_root"
    assert tied_head.weight is embedding.weight

    untied_head = nn.Linear(5, 13, bias=False)
    untied_head._fsdp_wrap_name = "lm_head"
    changed = keep_tied_embedding_and_lm_head_in_root_fsdp_unit_(embedding, untied_head)
    assert not changed
    assert untied_head._fsdp_wrap_name == "lm_head"


def test_qam_fine_copy_excludes_lm_head_and_refreshes_after_load():
    torch.manual_seed(5)
    behavior = _ToyBehavior()
    fine = QAMFineActionExpert(behavior)
    assert all("lm_head" not in name for name, _ in fine.named_parameters())

    with torch.no_grad():
        for parameter in behavior.parameters():
            parameter.add_(3.0)
    fine.copy_from_behavior_(behavior)

    behavior_state = behavior.paligemma_with_expert.gemma_expert.model.state_dict()
    fine_state = fine.expert_model.state_dict()
    assert behavior_state.keys() == fine_state.keys()
    for name in behavior_state:
        torch.testing.assert_close(fine_state[name], behavior_state[name])
    for name in fine._projection_names:  # noqa: SLF001
        fine_projection = getattr(fine, name).state_dict()
        behavior_projection = getattr(behavior, name).state_dict()
        for key in fine_projection:
            torch.testing.assert_close(
                fine_projection[key],
                behavior_projection[key],
            )


def test_qam_velocity_dispatch_flips_time_and_sign():
    observed = {}

    def fine_route(state, x_t, time_pi0, prefix_pad_masks, past_key_values):
        del state, prefix_pad_masks, past_key_values
        observed["time_pi0"] = time_pi0
        return x_t + time_pi0[:, None, None], torch.zeros_like(x_t)

    adapter = SimpleNamespace(
        config=SimpleNamespace(use_qam=True),
        _qam_fine_initialized=True,
        qam_fine=fine_route,
    )
    x_t = torch.ones(2, 3, 2)
    time_qam = torch.tensor([[0.25], [0.75]])
    velocity = OpenPi0ForRLActionPrediction._qam_velocity(
        adapter,
        state=torch.zeros(2, 2),
        x_t=x_t,
        time_qam=time_qam,
        prefix_pad_masks=torch.ones(2, 4, dtype=torch.bool),
        past_key_values=None,
        route="fine",
    )

    torch.testing.assert_close(
        observed["time_pi0"],
        torch.tensor([0.75, 0.25]),
    )
    expected = -(x_t + observed["time_pi0"][:, None, None])
    torch.testing.assert_close(velocity, expected)


def test_qam_projection_fingerprint_is_stack_safe_and_contract_sensitive():
    common = {
        "model_horizon": 50,
        "planned_horizon": 20,
        "model_action_dim": 32,
        "active_action_dim": 14,
        "projection_version": "pi0-fixed-prefix-active-v1",
        "data_fingerprint": "norm-and-output-transform",
    }
    fingerprint = build_qam_projection_fingerprint(**common)
    changed = build_qam_projection_fingerprint(**{**common, "planned_horizon": 19})

    assert fingerprint != changed
    encoded = projection_fingerprint_tensor(
        fingerprint,
        batch_size=3,
        device=torch.device("cpu"),
    )
    assert encoded.dtype == torch.uint8
    assert encoded.shape == (3, 32)
    torch.testing.assert_close(encoded[0], encoded[2])
    assert ForwardType.QAM_FLOW.value == "qam_flow"


def test_qam_rollout_clamps_only_active_block_and_reuses_canonical_action():
    raw_endpoint = torch.linspace(-2.0, 2.0, 4 * 5).reshape(1, 4, 5)
    original = raw_endpoint.clone()
    canonical, replay_planned = canonicalize_qam_rollout_action(
        raw_endpoint,
        planned_horizon=2,
        active_action_dim=3,
    )

    # Fine-flow endpoint is not mutated.
    torch.testing.assert_close(raw_endpoint, original)
    torch.testing.assert_close(
        canonical[:, :2, :3],
        original[:, :2, :3].clamp(-1.0, 1.0),
    )
    # Static padding and the unplanned horizon remain byte-for-byte unchanged.
    torch.testing.assert_close(canonical[:, :2, 3:], original[:, :2, 3:])
    torch.testing.assert_close(canonical[:, 2:, :], original[:, 2:, :])

    seen_by_output_transform = []

    def fake_output_transform(model_action):
        seen_by_output_transform.append(model_action)
        return model_action[:, :2, :3]

    env_action = fake_output_transform(canonical)
    assert seen_by_output_transform[0] is canonical
    torch.testing.assert_close(env_action, replay_planned)
