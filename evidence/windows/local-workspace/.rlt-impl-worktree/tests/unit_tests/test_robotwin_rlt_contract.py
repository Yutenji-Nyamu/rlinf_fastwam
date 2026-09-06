# Copyright 2026 The RLinf Authors.
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

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.algorithms.rlt.route import (
    FullTaskRLTRoute,
    RealworldRLTRoute,
    RLTRouteContext,
    SimulatorRLTRoute,
    build_rlt_route,
)
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def _route_cfg(
    *,
    env_type: str,
    route_type: str | None = None,
    transition_replay: bool | None = None,
):
    algorithm = {
        "rlt_schedule": {
            "enable": True,
            "warmup_post_collect_updates": 3,
        }
    }
    if route_type is not None:
        algorithm["rlt_route"] = {"type": route_type}
    if transition_replay is not None:
        algorithm["rlt_transition_replay"] = {
            "enable": transition_replay,
        }
    return OmegaConf.create(
        {
            "algorithm": algorithm,
            "env": {"train": {"env_type": env_type}},
        }
    )


@pytest.mark.parametrize(
    ("mode", "version", "expected_student", "expected_record"),
    [
        ("train", 2, False, True),
        ("train", 3, True, True),
        ("eval", 2, True, False),
        ("eval", 3, True, False),
    ],
)
def test_full_task_route_truth_table(
    mode,
    version,
    expected_student,
    expected_record,
):
    student_actions = torch.full((2, 2, 3), 7.0)
    reference_actions = torch.full((2, 4, 3), -2.0)
    result = {"forward_inputs": {}}
    route = FullTaskRLTRoute(use_schedule=True, warmup_updates=3)

    output = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs={"ref_chunk": reference_actions},
            student_actions=student_actions,
            result=result,
            mode=mode,
            version=version,
        )
    )

    expected_actions = student_actions if expected_student else reference_actions[:, :2]
    assert torch.equal(output.actions, expected_actions)
    assert torch.equal(
        output.result["forward_inputs"]["action"],
        expected_actions.reshape(2, -1),
    )
    assert output.result["forward_inputs"]["record_transition"].tolist() == [
        [expected_record],
        [expected_record],
    ]
    assert output.result["forward_inputs"]["actor_switch"].tolist() == [
        [expected_student],
        [expected_student],
    ]
    assert not output.result["forward_inputs"]["intervention_requested"].any()
    assert not output.result["intervene_flags"].any()


def test_route_builder_keeps_legacy_routing_and_allows_full_task_opt_in():
    assert isinstance(
        build_rlt_route(_route_cfg(env_type="robotwin")),
        RealworldRLTRoute,
    )
    assert isinstance(
        build_rlt_route(_route_cfg(env_type="maniskill_rlt")),
        SimulatorRLTRoute,
    )
    assert isinstance(
        build_rlt_route(_route_cfg(env_type="robotwin", transition_replay=True)),
        SimulatorRLTRoute,
    )
    assert isinstance(
        build_rlt_route(_route_cfg(env_type="maniskill_rlt", transition_replay=False)),
        RealworldRLTRoute,
    )
    assert isinstance(
        build_rlt_route(
            _route_cfg(
                env_type="robotwin",
                route_type="full_task",
                transition_replay=True,
            )
        ),
        FullTaskRLTRoute,
    )

    with pytest.raises(ValueError, match="Unsupported RLT route type"):
        build_rlt_route(_route_cfg(env_type="robotwin", route_type="unknown"))


@pytest.mark.parametrize(
    ("env_type", "explicit_capability", "expected"),
    [
        ("robotwin", None, False),
        ("maniskill_rlt", None, True),
        ("robotwin", True, True),
        ("maniskill_rlt", False, False),
    ],
)
def test_transition_replay_capability_overrides_legacy_env_fallback(
    env_type,
    explicit_capability,
    expected,
):
    cfg = _route_cfg(
        env_type=env_type,
        transition_replay=explicit_capability,
    )
    assert use_simulator_transition_replay(cfg) is expected


def _openpi_action_model_class():
    module = pytest.importorskip("rlinf.models.embodiment.openpi.openpi_action_model")
    return module.OpenPi0ForRLActionPrediction


class _TinyOpenPi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def test_stage1_freeze_leaves_only_rlt_module_trainable():
    openpi_action_model = _openpi_action_model_class()
    model = _TinyOpenPi(SimpleNamespace(use_rlt=True))
    model.backbone = nn.Linear(3, 4)
    model.rlt_module = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 2))

    openpi_action_model.freeze_for_rlt_stage1(model)

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert trainable == {
        name for name, _ in model.rlt_module.named_parameters(prefix="rlt_module")
    }
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("rlt_module.")
    )


def test_canonical_decode_overwrites_only_slice_and_matches_transform():
    openpi_action_model = _openpi_action_model_class()
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
        captured["state"] = payload["state"].clone()
        return {
            "actions": payload["actions"]
            + payload["state"][:, None, :1].to(payload["actions"])
        }

    model.output_transform = output_transform
    raw_template = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5)
    original_template = raw_template.clone()
    canonical_actions = torch.full((2, 2, 3), -5.0)
    processed_state = torch.tensor([[0.25], [0.75]])
    expected_raw = raw_template.clone()
    expected_raw[:, :2, :3] = canonical_actions
    expected_decoded = expected_raw + processed_state[:, None, :1]

    decoded = openpi_action_model.decode_rlt_action(
        model,
        canonical_actions,
        {
            "raw_action_template": raw_template,
            "processed_state": processed_state,
        },
    )

    assert torch.equal(decoded, expected_decoded[:, :2, :3])
    assert torch.equal(captured["actions"], expected_raw)
    assert torch.equal(captured["state"], processed_state)
    assert torch.equal(raw_template, original_template)
    assert torch.equal(captured["actions"][:, 2:], original_template[:, 2:])
    assert torch.equal(captured["actions"][:, :2, 3:], original_template[:, :2, 3:])


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rlt_worker_module():
    return pytest.importorskip("rlinf.workers.actor.fsdp_rlt_ac_policy_worker")


def _make_resume_worker(
    tmp_path,
    *,
    gamma: float = 0.99,
    bootstrap_type: str = "standard",
):
    worker_class = _rlt_worker_module().RLTACFSDPPolicy
    tmp_path.mkdir(parents=True, exist_ok=True)
    stage1_manifest = tmp_path / "stage1_manifest.json"
    stage1_model = tmp_path / "stage1_model"
    norm_stats = tmp_path / "norm_stats.json"
    stage1_model.mkdir()
    norm_stats.write_text('{"state": {"mean": [0.0]}}', encoding="utf-8")
    stage1_manifest.write_text(
        json.dumps(
            {
                "accepted": True,
                "schema_version": 1,
                "manifest_id": "stage1-test",
                "stage1": {"model_path": str(stage1_model)},
                "model_contract": {
                    "norm_stats_sha256": _sha256(norm_stats),
                    "canonical_adapter_version": "robotwin_aloha_canonical_v1",
                    "action_horizon": 4,
                    "action_chunk": 2,
                    "action_dim": 3,
                    "z_rl_dim": 5,
                    "image_prefix_shape": [7, 11],
                },
            }
        ),
        encoding="utf-8",
    )
    contract = {
        "stage1_manifest_path": str(stage1_manifest),
        "stage1_manifest_id": "stage1-test",
        "stage1_manifest_sha256": _sha256(stage1_manifest),
        "norm_stats_sha256": _sha256(norm_stats),
        "canonical_adapter_version": "robotwin_aloha_canonical_v1",
    }
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "rlt_resume": {
                    "enable": True,
                    "schema_version": 1,
                    "contract": contract,
                },
                "rlt_schedule": {
                    "enable": True,
                    "warmup_post_collect_updates": 8,
                },
                "rlt_route": {"type": "full_task"},
                "rlt_transition_replay": {
                    "enable": True,
                    "compact": True,
                    "bootstrap_on_truncation": True,
                },
                "replay_buffer": {"sample_window_size": 64},
                "bootstrap_type": bootstrap_type,
                "gamma": gamma,
                "tau": 0.005,
                "update_epoch": 5,
                "critic_actor_ratio": 2,
                "reference_dropout_prob": 0.5,
                "actor_weight_schedule": {
                    "warmup_updates": 4,
                    "ramp_updates": 8,
                },
            },
            "actor": {
                "model": {
                    "model_type": "rlt_mlp",
                    "num_action_chunks": 2,
                    "ref_num_action_chunks": 2,
                    "action_dim": 3,
                    "z_dim": 5,
                }
            },
            "rollout": {
                "rlt_feature_model": {
                    "model_path": str(stage1_model),
                    "openpi": {
                        "rlt_action_adapter": "robotwin_aloha_canonical_v1",
                        "action_horizon": 4,
                        "action_chunk": 2,
                        "action_env_dim": 3,
                        "rlt_embed_dim": 5,
                        "rlt_prefix_seq_len": 7,
                        "rlt_input_dim": 11,
                    },
                    "openpi_data": {
                        "norm_stats_path": str(norm_stats),
                    },
                }
            },
            "runner": {"weight_sync_interval": 1},
            "weight_syncer": {"type": "collective"},
            "env": {"train": {"auto_reset": False}},
        }
    )
    worker = worker_class.__new__(worker_class)
    worker.cfg = cfg
    worker.rlt_resume_cfg = cfg.algorithm.rlt_resume
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
    return worker


def _bind_same_stage1_artifact(target, source) -> None:
    target.cfg.rollout.rlt_feature_model.model_path = (
        source.cfg.rollout.rlt_feature_model.model_path
    )
    target.cfg.rollout.rlt_feature_model.openpi_data.norm_stats_path = (
        source.cfg.rollout.rlt_feature_model.openpi_data.norm_stats_path
    )
    target.rlt_resume_cfg.contract = source.rlt_resume_cfg.contract


def test_rlt_resume_state_world1_roundtrip_restores_independent_state(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    source = _make_resume_worker(tmp_path / "source")
    source._save_rlt_trainer_state(str(checkpoint_dir), runner_step=11)

    resumed = _make_resume_worker(tmp_path / "resumed")
    _bind_same_stage1_artifact(resumed, source)
    state = resumed._preflight_rlt_trainer_state(str(checkpoint_dir))
    resumed._restore_rlt_trainer_state(state)

    assert resumed.update_step == 17
    assert resumed.total_transitions_added == 41
    assert resumed.total_episodes_added == 6
    assert resumed._warmup_ready_total_transitions == 24
    assert resumed._warmup_ready_total_episodes == 4
    assert resumed.transitions_since_train == 0
    assert resumed.episodes_since_train == 0
    assert resumed.pending_update_budget == 0


def test_rlt_resume_preflight_fails_closed_for_missing_checkpoint(tmp_path):
    worker = _make_resume_worker(tmp_path / "source")

    with pytest.raises(ValueError, match="missing RLT completion manifest"):
        worker._preflight_rlt_trainer_state(str(tmp_path / "missing_checkpoint"))


def test_rlt_resume_contract_rejects_unaccepted_stage1_manifest(tmp_path):
    worker = _make_resume_worker(tmp_path / "source")
    manifest_path = Path(worker.rlt_resume_cfg.contract.stage1_manifest_path)
    stage1_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage1_manifest["accepted"] = False
    manifest_path.write_text(json.dumps(stage1_manifest), encoding="utf-8")
    worker.rlt_resume_cfg.contract.stage1_manifest_sha256 = _sha256(
        manifest_path
    )

    with pytest.raises(ValueError, match="not marked accepted"):
        worker._rlt_resume_contract()


def test_rlt_resume_contract_rejects_stage1_model_path_mismatch(tmp_path):
    worker = _make_resume_worker(tmp_path / "source")
    other_model = tmp_path / "other_model"
    other_model.mkdir()
    worker.cfg.rollout.rlt_feature_model.model_path = str(other_model)

    with pytest.raises(ValueError, match="manifest/model path mismatch"):
        worker._rlt_resume_contract()


def test_rlt_resume_preflight_fails_closed_for_contract_change(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    source = _make_resume_worker(tmp_path / "source")
    source._save_rlt_trainer_state(str(checkpoint_dir), runner_step=11)
    changed = _make_resume_worker(tmp_path / "changed", gamma=0.98)
    _bind_same_stage1_artifact(changed, source)

    with pytest.raises(ValueError, match="contract fingerprint mismatch"):
        changed._preflight_rlt_trainer_state(str(checkpoint_dir))


def test_rlt_resume_preflight_locks_bootstrap_semantics(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    source = _make_resume_worker(tmp_path / "source")
    source._save_rlt_trainer_state(str(checkpoint_dir), runner_step=11)
    changed = _make_resume_worker(
        tmp_path / "changed",
        bootstrap_type="always",
    )
    _bind_same_stage1_artifact(changed, source)

    with pytest.raises(ValueError, match="contract fingerprint mismatch"):
        changed._preflight_rlt_trainer_state(str(checkpoint_dir))


def test_rlt_resume_invalidation_marker_blocks_stale_state(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    worker = _make_resume_worker(tmp_path / "source")
    worker._save_rlt_trainer_state(str(checkpoint_dir), runner_step=11)
    worker._mark_rlt_checkpoint_incomplete(
        str(checkpoint_dir),
        runner_step=11,
    )

    with pytest.raises(ValueError, match="not marked complete"):
        worker._preflight_rlt_trainer_state(str(checkpoint_dir))


class _FlattenOnlyReplay:
    def _flatten_trajectory(self, trajectory):
        return TrajectoryReplayBuffer._flatten_trajectory(self, trajectory)


def test_compact_replay_bootstraps_pure_truncation_from_linked_next_obs():
    worker_class = _rlt_worker_module().RLTACFSDPPolicy
    worker = worker_class.__new__(worker_class)
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
    assert torch.equal(transition.curr_obs["z_rl"], curr_obs["z_rl"])
    assert torch.equal(transition.next_obs["z_rl"], next_obs["z_rl"])
    assert transition.forward_inputs == {}
    assert torch.equal(transition.intervene_flags, trajectory.intervene_flags)
    assert transition.prev_logprobs is None
    assert transition.prev_values is None
    assert transition.versions is None
