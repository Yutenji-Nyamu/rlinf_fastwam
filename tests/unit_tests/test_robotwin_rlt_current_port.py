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

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.algorithms.rlt.rollout import predict_rlt_actions
from rlinf.algorithms.rlt.route import (
    FullTaskRLTRoute,
    RLTRouteContext,
    build_rlt_route,
)
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer


def _route_cfg(*, route_type="full_task", transition_replay=True):
    return OmegaConf.create(
        {
            "algorithm": {
                "rlt_schedule": {
                    "enable": True,
                    "warmup_post_collect_updates": 3,
                },
                "rlt_route": {"type": route_type},
                "rlt_transition_replay": {"enable": transition_replay},
            },
            "env": {"train": {"env_type": "robotwin"}},
        }
    )


@pytest.mark.parametrize(
    ("mode", "version", "student", "record"),
    [
        ("train", 2, False, True),
        ("train", 3, True, True),
        ("eval", 2, True, False),
        ("eval", 3, True, False),
    ],
)
def test_full_task_route_truth_table(mode, version, student, record):
    student_actions = torch.full((2, 2, 3), 7.0)
    reference_actions = torch.full((2, 4, 3), -2.0)
    route = FullTaskRLTRoute(use_schedule=True, warmup_updates=3)

    output = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs={"ref_chunk": reference_actions},
            student_actions=student_actions,
            result={"forward_inputs": {}},
            mode=mode,
            version=version,
        )
    )

    expected = student_actions if student else reference_actions[:, :2]
    assert torch.equal(output.actions, expected)
    assert torch.equal(
        output.result["forward_inputs"]["action"], expected.reshape(2, -1)
    )
    assert output.result["forward_inputs"]["record_transition"].tolist() == [
        [record],
        [record],
    ]
    assert output.result["forward_inputs"]["actor_switch"].tolist() == [
        [student],
        [student],
    ]
    assert not output.result["intervene_flags"].any()


def test_robotwin_capabilities_are_explicit_opt_ins():
    cfg = _route_cfg()
    assert isinstance(build_rlt_route(cfg), FullTaskRLTRoute)
    assert use_simulator_transition_replay(cfg) is True
    cfg.algorithm.rlt_transition_replay.enable = False
    assert use_simulator_transition_replay(cfg) is False


class _TinyOpenPi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def _openpi_class():
    module = pytest.importorskip("rlinf.models.embodiment.openpi.openpi_action_model")
    return module.OpenPi0ForRLActionPrediction


def test_stage1_freeze_leaves_only_rlt_module_trainable():
    model = _TinyOpenPi(SimpleNamespace(use_rlt=True))
    model.backbone = nn.Linear(3, 4)
    model.rlt_module = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 2))

    _openpi_class().freeze_for_rlt_stage1(model)

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable == {
        name for name, _ in model.rlt_module.named_parameters(prefix="rlt_module")
    }


def test_canonical_decode_overwrites_once_and_preserves_template():
    model = _TinyOpenPi(
        SimpleNamespace(
            rlt_action_adapter="robotwin_aloha_canonical_v1",
            action_env_dim=3,
            action_chunk=2,
        )
    )
    captured = {}

    def output_transform(payload):
        captured["actions"] = payload["actions"].clone()
        return {"actions": payload["actions"] + 1.0}

    model.output_transform = output_transform
    raw_template = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5)
    original = raw_template.clone()
    canonical = torch.full((2, 2, 3), -5.0)

    decoded = _openpi_class().decode_rlt_action(
        model,
        canonical,
        {
            "raw_action_template": raw_template,
            "processed_state": torch.zeros(2, 1),
        },
    )

    assert torch.equal(decoded, torch.full((2, 2, 3), -4.0))
    assert torch.equal(raw_template, original)
    assert torch.equal(captured["actions"][:, 2:], original[:, 2:])
    assert torch.equal(captured["actions"][:, :2, 3:], original[:, :2, 3:])


class _FakeFeatureModel:
    config = SimpleNamespace(rlt_action_adapter="robotwin_aloha_canonical_v1")

    def __init__(self):
        self.decode_calls = 0

    def extract_rlt_obs(self, _obs, return_decode_context=False):
        rlt_obs = {
            "z_rl": torch.zeros(1, 2),
            "proprio": torch.zeros(1, 3),
            "ref_chunk": torch.zeros(1, 2, 3),
        }
        if return_decode_context:
            return rlt_obs, {"decode": True}
        return rlt_obs

    def decode_rlt_action(self, actions, _context):
        self.decode_calls += 1
        return actions + 10.0


class _FakePolicy:
    def predict_action_batch(self, **_kwargs):
        actions = torch.ones(1, 2, 3)
        return actions, {"forward_inputs": {}}


def test_rollout_keeps_canonical_replay_action_and_decodes_env_action_once():
    feature = _FakeFeatureModel()
    actions, result = predict_rlt_actions(
        policy_model=_FakePolicy(),
        feature_model=feature,
        rlt_route=FullTaskRLTRoute(use_schedule=True, warmup_updates=3),
        env_obs={},
        final_obs=None,
        mode="eval",
        version=0,
    )

    assert feature.decode_calls == 1
    assert torch.equal(actions, torch.full((1, 2, 3), 11.0))
    assert torch.equal(
        result["forward_inputs"]["action"], torch.ones(1, 6)
    )


def _worker(tmp_path, *, contract_id="accepted"):
    worker_cls = pytest.importorskip(
        "rlinf.workers.actor.fsdp_rlt_ac_policy_worker"
    ).RLTACFSDPPolicy
    worker = worker_cls.__new__(worker_cls)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "rlt_resume": {
                    "enable": True,
                    "contract": {
                        "stage1_manifest_id": contract_id,
                        "bootstrap": "pure_truncation_only",
                    },
                }
            }
        }
    )
    worker.rlt_resume_cfg = worker.cfg.algorithm.rlt_resume
    worker._rank = 0
    worker._world_size = 1
    worker.update_step = 17
    worker.total_transitions_added = 41
    worker.total_episodes_added = 6
    worker._warmup_ready_total_transitions = 24
    worker._warmup_ready_total_episodes = 4
    worker.transitions_since_train = 5
    worker.episodes_since_train = 2
    worker.pending_update_budget = 9
    tmp_path.mkdir(parents=True, exist_ok=True)
    return worker


def test_rlt_resume_sidecar_world1_roundtrip_and_contract_lock(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    source = _worker(tmp_path / "source")
    source._save_rlt_state(str(checkpoint), runner_step=11)

    resumed = _worker(tmp_path / "resumed")
    state = resumed._load_rlt_state(str(checkpoint))
    resumed._restore_rlt_state(state)
    assert resumed.update_step == 17
    assert resumed.total_transitions_added == 41
    assert resumed._warmup_ready_total_transitions == 24
    assert resumed.transitions_since_train == 0
    assert resumed.episodes_since_train == 0
    assert resumed.pending_update_budget == 0

    changed = _worker(tmp_path / "changed", contract_id="different")
    with pytest.raises(ValueError, match="contract mismatch"):
        changed._load_rlt_state(str(checkpoint))


def test_rlt_resume_rejects_unresolved_contract(tmp_path):
    worker = _worker(tmp_path / "source", contract_id="UNRESOLVED_STAGE1")
    with pytest.raises(ValueError, match="unresolved fields"):
        worker._rlt_contract()


class _FlattenOnlyReplay:
    def _flatten_trajectory(self, trajectory):
        return TrajectoryReplayBuffer._flatten_trajectory(self, trajectory)


def test_compact_replay_bootstraps_pure_truncation_from_real_next_obs():
    worker_cls = pytest.importorskip(
        "rlinf.workers.actor.fsdp_rlt_ac_policy_worker"
    ).RLTACFSDPPolicy
    worker = worker_cls.__new__(worker_cls)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "rlt_transition_replay": {
                    "compact": True,
                    "bootstrap_on_truncation": True,
                }
            },
            "env": {"train": {"auto_reset": False}},
        }
    )
    worker.replay_buffer = _FlattenOnlyReplay()
    curr_obs = {
        "z_rl": torch.tensor([[[1.0, 2.0]]]),
        "proprio": torch.tensor([[[3.0]]]),
        "ref_chunk": torch.tensor([[[4.0, 5.0]]]),
    }
    next_obs = {key: value + 10.0 for key, value in curr_obs.items()}
    trajectory = Trajectory(
        actions=torch.tensor([[[0.1, 0.2]]]),
        rewards=torch.tensor([[[0.0]]]),
        terminations=torch.tensor([[[False]], [[False]]]),
        truncations=torch.tensor([[[False]], [[True]]]),
        dones=torch.tensor([[[False]], [[True]]]),
        intervene_flags=torch.ones(1, 1, 2, dtype=torch.bool),
        prev_logprobs=torch.ones(1, 1, 1),
        prev_values=torch.ones(1, 1, 1),
        versions=torch.ones(1, 1, 1, dtype=torch.long),
        forward_inputs={
            "record_transition": torch.ones(1, 1, 1, dtype=torch.bool),
            "bulky_prefix_cache": torch.ones(1, 1, 32),
        },
        curr_obs=curr_obs,
        next_obs=next_obs,
    )

    transitions, completed = worker._transition_replay_trajectories(trajectory)

    assert completed == 1
    assert len(transitions) == 1
    transition = transitions[0]
    assert torch.equal(transition.next_obs["z_rl"], next_obs["z_rl"])
    assert transition.forward_inputs == {}
    assert transition.prev_logprobs is None
    assert transition.prev_values is None
    assert transition.versions is None
