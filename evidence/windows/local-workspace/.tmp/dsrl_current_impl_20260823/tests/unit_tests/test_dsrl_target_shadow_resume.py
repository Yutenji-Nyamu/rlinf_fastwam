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

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class _DummyDSRLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic_image_encoder = nn.Linear(3, 4)
        self.critic_state_encoder = nn.Linear(2, 4)
        self.q_head = nn.Linear(8, 2)
        self.frozen_base = nn.Linear(5, 5)
        self.register_buffer(
            "dsrl_policy_phase",
            torch.tensor(1, dtype=torch.uint8),
            persistent=True,
        )
        self.to(dtype=torch.bfloat16)


def _make_worker() -> EmbodiedSACFSDPPolicy:
    worker = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    worker.model = _DummyDSRLModel()
    worker.target_model = copy.deepcopy(worker.model)
    worker.target_model.requires_grad_(False)
    worker.target_model_initialized = True
    worker.use_dsrl = True
    worker.use_dsrl_flat_replay = True
    worker.target_update_type = "all"
    worker.cfg = SimpleNamespace(algorithm=SimpleNamespace(tau=0.005))
    worker._rank = 0
    worker._world_size = 1
    worker.update_step = 0
    worker._local_new_transitions = 0
    worker._init_target_shadow()
    return worker


def _add_to_target_q_source(worker: EmbodiedSACFSDPPolicy, delta: float):
    with torch.no_grad():
        for name, parameter in worker.model.named_parameters():
            if worker._is_dsrl_target_q_parameter(name):
                parameter.add_(delta)


def test_target_shadow_resume_matches_continuous_ema(tmp_path):
    continuous = _make_worker()
    _add_to_target_q_source(continuous, 0.01)
    continuous.soft_update_target_model()
    continuous.update_step = 7
    continuous._local_new_transitions = 3

    checkpoint_online = copy.deepcopy(continuous.model.state_dict())
    checkpoint_target = copy.deepcopy(continuous.target_model.state_dict())
    continuous._save_dsrl_trainer_state(str(tmp_path))

    _add_to_target_q_source(continuous, 0.02)
    continuous.soft_update_target_model()

    resumed = _make_worker()
    resumed.model.load_state_dict(checkpoint_online)
    resumed.target_model.load_state_dict(checkpoint_target)
    resumed._load_dsrl_trainer_state(str(tmp_path))

    assert resumed.update_step == 7
    assert resumed._local_new_transitions == 3
    assert resumed._get_dsrl_policy_phase() == 1
    assert all(
        resumed._is_dsrl_target_q_parameter(name)
        for name in resumed._target_shadow_f32
    )

    frozen_before = copy.deepcopy(resumed.target_model.frozen_base.state_dict())
    _add_to_target_q_source(resumed, 0.02)
    resumed.soft_update_target_model()

    for name, expected in continuous._target_shadow_f32.items():
        assert torch.equal(resumed._target_shadow_f32[name], expected)
    for name, expected in continuous.target_model.state_dict().items():
        assert torch.equal(resumed.target_model.state_dict()[name], expected)
    for name, expected in frozen_before.items():
        assert torch.equal(resumed.target_model.frozen_base.state_dict()[name], expected)


def test_dsrl_actor_and_critic_clips_exclude_incidental_grads():
    worker = _make_worker()
    target_q_parameters = list(
        worker._named_dsrl_target_q_parameters(worker.model).values()
    )
    actor_parameter = worker.model.frozen_base.weight
    worker.qf_optimizer = torch.optim.SGD(target_q_parameters, lr=0.1)
    worker.optimizer = torch.optim.SGD([actor_parameter], lr=0.1)

    actor_parameter.grad = torch.ones_like(actor_parameter)
    for parameter in target_q_parameters:
        parameter.grad = torch.ones_like(parameter)
    worker._clear_dsrl_actor_grads_before_critic_clip()
    assert actor_parameter.grad is None
    assert all(parameter.grad is not None for parameter in target_q_parameters)

    actor_parameter.grad = torch.ones_like(actor_parameter)
    worker._clear_dsrl_critic_grads_before_actor_clip()
    assert actor_parameter.grad is not None
    assert all(parameter.grad is None for parameter in target_q_parameters)


def test_trainer_state_is_strict_and_phase_bound(tmp_path):
    worker = _make_worker()
    with pytest.raises(FileNotFoundError, match="requires trainer sidecar"):
        worker._load_dsrl_trainer_state(str(tmp_path))

    worker._save_dsrl_trainer_state(str(tmp_path))
    resumed = _make_worker()
    resumed.target_model.load_state_dict(worker.target_model.state_dict())
    resumed.target_model.dsrl_policy_phase.fill_(0)
    with pytest.raises(ValueError, match="phase mismatch"):
        resumed._load_dsrl_trainer_state(str(tmp_path))
