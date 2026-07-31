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

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.qam.contracts import (
    QAMPolicyObservation,
    project_planned_action,
)
from rlinf.algorithms.qam.core import terminal_mean_q_adjoint
from rlinf.config import _validate_embodied_qam_contract
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.models.embodiment.modules.qam_critic import QAMCriticEnsemble
from rlinf.workers.actor.fsdp_qam_policy_worker import (
    QAMFSDPPolicy,
    QAMUpdateCredit,
    _am_is_enabled_for_next_update,
    _phase_transition_is_valid,
    _resume_update_credit,
    classify_qam_end,
    extract_qam_camera_triplet,
    validate_qam_prefix_block_lengths,
)


def _load_qam_source_config():
    return OmegaConf.load(
        Path(__file__).parents[2]
        / "examples"
        / "embodiment"
        / "config"
        / "robotwin_adjust_bottle_qam_openpi.yaml"
    )


def test_qam_critic_is_complete_independent_fp32_ensemble() -> None:
    torch.manual_seed(7)
    critic = QAMCriticEnsemble(
        feature_dim=8,
        num_q_heads=10,
        hidden_dims=(16, 16),
    )
    feature = torch.randn(3, 4, 8, dtype=torch.bfloat16)
    proprio = torch.randn(3, 14)
    action = torch.randn(3, 20, 14)

    output = critic(feature, proprio, action)
    assert output.shape == (10, 3)
    assert output.dtype == torch.float32
    first_weights = [q_function.network[0].weight for q_function in critic.q_functions]
    assert len({weight.data_ptr() for weight in first_weights}) == 10
    assert not torch.equal(first_weights[0], first_weights[1])

    output.sum().backward()
    assert all(weight.grad is not None for weight in first_weights)


def test_qam_critic_default_is_official_four_by_512_and_xavier_style() -> None:
    critic = QAMCriticEnsemble(feature_dim=1, num_q_heads=1)
    linear_layers = [
        module
        for module in critic.q_functions[0].network
        if isinstance(module, torch.nn.Linear)
    ]
    assert [layer.out_features for layer in linear_layers] == [
        512,
        512,
        512,
        512,
        1,
    ]
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in linear_layers)


def test_qam_prefix_block_lengths_are_stable_and_fail_closed() -> None:
    expected = (256, 256, 256, 48)
    assert (
        validate_qam_prefix_block_lengths(
            torch.tensor(expected),
            feature_blocks=4,
        )
        == expected
    )
    with pytest.raises(ValueError, match="four pooled prefix blocks"):
        validate_qam_prefix_block_lengths(
            torch.tensor(expected),
            feature_blocks=3,
        )
    with pytest.raises(ValueError, match="changed within a run"):
        validate_qam_prefix_block_lengths(
            torch.tensor((256, 256, 255, 49)),
            feature_blocks=4,
            expected=expected,
        )


def test_terminal_adjoint_uses_canonical_planned_action_gradient() -> None:
    endpoint = torch.zeros(1, 50, 32)
    endpoint[0, 0, 0] = 1.4
    endpoint[0, 0, 1] = 0.25
    endpoint[0, 0, 14] = 0.25
    endpoint[0, 20, 0] = 0.25
    endpoint = endpoint.reshape(1, -1)

    def target_q(full_action):
        planned = project_planned_action(full_action.reshape(-1, 50, 32))
        score = planned.sum(dim=(-2, -1))
        return torch.stack((score, score), dim=0)

    adjoint = terminal_mean_q_adjoint(
        target_q,
        endpoint,
        inv_temp=2.0,
        clip_action=True,
    ).reshape(1, 50, 32)
    assert adjoint[0, 0, 0] == 0
    assert adjoint[0, 0, 1] == -2
    assert adjoint[0, 0, 14] == 0
    assert adjoint[0, 20, 0] == 0


def test_global_insert_utd_credit_is_persistent_and_capped() -> None:
    credit = QAMUpdateCredit(utd_ratio=1.5)
    credit.add_global_inserts(3)
    assert credit.take(2) == 2
    assert credit.pending == pytest.approx(2.5)
    assert credit.take(10) == 2
    assert credit.pending == pytest.approx(0.5)
    credit.add_global_inserts(1)
    assert credit.take(10) == 2
    assert credit.pending == pytest.approx(0.0)


def test_am_starts_only_after_exact_critic_only_budget() -> None:
    for completed in (0, 510, 511):
        assert not _am_is_enabled_for_next_update(
            configured_phase="am_on",
            critic_updates=completed,
            q_only_updates_before_am=512,
        )
    assert _am_is_enabled_for_next_update(
        configured_phase="am_on",
        critic_updates=512,
        q_only_updates_before_am=512,
    )
    assert not _am_is_enabled_for_next_update(
        configured_phase="q_only",
        critic_updates=512,
        q_only_updates_before_am=512,
    )
    with pytest.raises(ValueError, match="non-negative"):
        _am_is_enabled_for_next_update(
            configured_phase="am_on",
            critic_updates=-1,
            q_only_updates_before_am=512,
        )


def test_q_only_utd_credit_starts_after_online_warmup() -> None:
    worker = object.__new__(QAMFSDPPolicy)
    worker.phase = "q_only"
    worker.qam_cfg = OmegaConf.create({"warmup_global_inserts": 5})
    worker.update_credit = QAMUpdateCredit(utd_ratio=2.0)
    worker.q_only_anchor_global_inserts = None

    worker.global_total_inserts = 4
    worker._accrue_update_credit(4)
    assert worker.q_only_anchor_global_inserts is None
    assert worker.update_credit.pending == 0

    worker.global_total_inserts = 5
    worker._accrue_update_credit(1)
    assert worker.q_only_anchor_global_inserts == 5
    assert worker.update_credit.pending == 0

    worker.global_total_inserts = 7
    worker._accrue_update_credit(2)
    assert worker.update_credit.pending == pytest.approx(4.0)

    crossing = object.__new__(QAMFSDPPolicy)
    crossing.phase = "q_only"
    crossing.qam_cfg = worker.qam_cfg
    crossing.update_credit = QAMUpdateCredit(utd_ratio=2.0)
    crossing.q_only_anchor_global_inserts = None
    crossing.global_total_inserts = 7
    crossing._accrue_update_credit(3)
    assert crossing.q_only_anchor_global_inserts == 5
    assert crossing.update_credit.pending == pytest.approx(4.0)


def test_collect_resume_establishes_q_only_credit_anchor() -> None:
    pending, anchor = _resume_update_credit(
        saved_phase="collect",
        requested_phase="q_only",
        saved_pending=512.0,
        saved_anchor=None,
        global_total_inserts=512,
        warmup_global_inserts=512,
    )
    assert pending == 0.0
    assert anchor == 512

    pending, anchor = _resume_update_credit(
        saved_phase="q_only",
        requested_phase="am_on",
        saved_pending=1.5,
        saved_anchor=512,
        global_total_inserts=640,
        warmup_global_inserts=512,
    )
    assert pending == pytest.approx(1.5)
    assert anchor == 512

    pending, anchor = _resume_update_credit(
        saved_phase="collect",
        requested_phase="collect",
        saved_pending=8.0,
        saved_anchor=None,
        global_total_inserts=128,
        warmup_global_inserts=512,
    )
    assert pending == 0.0
    assert anchor is None


def test_robotwin_time_limit_keeps_query_final_state_valid() -> None:
    assert classify_qam_end(terminated=False, truncated=False) == (
        False,
        False,
        False,
        True,
    )
    assert classify_qam_end(terminated=True, truncated=False) == (
        True,
        False,
        False,
        False,
    )
    assert classify_qam_end(terminated=False, truncated=True) == (
        False,
        True,
        False,
        True,
    )
    assert classify_qam_end(terminated=True, truncated=True) == (
        True,
        False,
        False,
        False,
    )


def test_robotwin_camera_triplet_is_main_plus_two_wrists() -> None:
    main = torch.randint(
        0,
        256,
        (2, 3, 8, 9, 3),
        dtype=torch.uint8,
    )
    wrists = torch.randint(
        0,
        256,
        (2, 3, 2, 8, 9, 3),
        dtype=torch.uint8,
    )
    cameras = extract_qam_camera_triplet(
        {
            "main_images": main,
            "wrist_images": wrists,
            "extra_view_images": None,
        },
        step=1,
        env=2,
    )
    assert cameras.shape == (3, 8, 9, 3)
    assert torch.equal(cameras[0], main[1, 2])
    assert torch.equal(cameras[1:], wrists[1, 2])


def test_qam_ingest_builds_one_causal_m2_replay_row(monkeypatch) -> None:
    worker = object.__new__(QAMFSDPPolicy)
    worker.qam_cfg = OmegaConf.create(
        {
            "task_prompt": "adjust the bottle",
            "replay_capacity": 4,
            "gamma_slot": 0.9,
        }
    )
    worker.cfg = OmegaConf.create(
        {
            "actor": {"seed": 5},
            "env": {
                "train": {
                    "task_config": {"task_name": "adjust_bottle"},
                }
            },
        }
    )
    worker._rank = 0
    worker._world_size = 1
    worker.runner_global_step = 11
    worker.fine_policy_version = 3
    worker.replay = None
    worker.contract_fingerprint = None
    worker.prefix_block_lengths = None

    captured_next = []

    def fake_condition(observations):
        captured_next.extend(observations)
        return (
            torch.full((len(observations), 4, 5), 9.0),
            torch.full((len(observations), 14), 8.0),
        )

    monkeypatch.setattr(worker, "_condition_observations", fake_condition)

    main = torch.arange(6 * 7 * 3, dtype=torch.uint8).reshape(
        1,
        1,
        6,
        7,
        3,
    )
    wrists = torch.stack((main + 1, main + 2), dim=2)
    next_main = main + 3
    next_wrists = torch.stack((main + 4, main + 5), dim=2)
    planned_normalized = torch.linspace(
        -0.8,
        0.8,
        20 * 14,
    ).reshape(1, 1, 20, 14)
    planned_env = torch.arange(
        20 * 14,
        dtype=torch.float32,
    ).reshape(1, 1, 20, 14)
    rewards = torch.zeros(1, 1, 20)
    rewards[..., -1] = 2.0
    obs_feature = torch.arange(
        4 * 5,
        dtype=torch.float32,
    ).reshape(1, 1, 4, 5)
    obs_proprio = torch.arange(14, dtype=torch.float32).reshape(1, 1, 14)

    trajectory = Trajectory(
        actions=planned_env,
        rewards=rewards,
        # The leading slot is true on purpose. Ingest must use slot t+1,
        # which is live, and therefore retain and condition next_obs.
        terminations=torch.tensor([[[True]], [[False]]]),
        truncations=torch.zeros(2, 1, 1, dtype=torch.bool),
        versions=torch.tensor([[[7]]]),
        forward_inputs={
            "qam_planned_action_normalized": planned_normalized,
            "qam_obs_feature": obs_feature,
            "qam_prefix_block_lengths": torch.tensor([[[[3, 3, 3, 2]]]]).reshape(
                1, 1, 4
            ),
            "qam_proprio_normalized": obs_proprio,
            "qam_projection_contract": torch.tensor([[[[50, 20, 32, 14]]]]).reshape(
                1, 1, 4
            ),
            "qam_projection_fingerprint": torch.arange(
                32,
                dtype=torch.uint8,
            ).reshape(1, 1, 32),
        },
        curr_obs={
            "main_images": main,
            "wrist_images": wrists,
            "extra_view_images": None,
            "states": obs_proprio,
        },
        next_obs={
            "main_images": next_main,
            "wrist_images": next_wrists,
            "extra_view_images": None,
            "states": obs_proprio + 1,
        },
    )

    assert worker._ingest_trajectory(trajectory) == 1
    assert worker.replay is not None
    sample = worker.replay.sample(1)[0]
    transition = sample.transition
    assert torch.equal(
        transition.planned_actions_normalized,
        planned_normalized[0, 0],
    )
    assert torch.equal(transition.planned_actions_env, planned_env[0, 0])
    assert torch.equal(transition.chunk_rewards_native, rewards[0, 0])
    assert transition.reward_macro_discounted == pytest.approx(2.0 * 0.9**19)
    assert transition.policy_version == 7
    assert transition.query_index == 0
    assert not transition.success_terminated
    assert transition.next_state_valid
    assert transition.bootstrap_mask == 1.0
    assert torch.equal(
        sample.observation.cameras_uint8,
        torch.stack((main[0, 0], wrists[0, 0, 0], wrists[0, 0, 1])),
    )
    assert sample.next_observation is not None
    assert torch.equal(
        sample.next_observation.cameras_uint8,
        torch.stack(
            (
                next_main[0, 0],
                next_wrists[0, 0, 0],
                next_wrists[0, 0, 1],
            )
        ),
    )
    assert len(captured_next) == 1
    assert torch.equal(transition.next_obs_feature, torch.full((4, 5), 9.0))
    assert torch.equal(transition.next_obs_proprio, torch.full((14,), 8.0))


def test_qam_ingest_skips_rows_after_first_episode_end(monkeypatch) -> None:
    worker = object.__new__(QAMFSDPPolicy)
    worker.qam_cfg = OmegaConf.create(
        {
            "task_prompt": "adjust the bottle",
            "replay_capacity": 8,
            "gamma_slot": 0.9,
        }
    )
    worker.cfg = OmegaConf.create(
        {
            "actor": {"seed": 5},
            "env": {"train": {"task_config": {"task_name": "adjust_bottle"}}},
        }
    )
    worker._rank = 0
    worker._world_size = 1
    worker.runner_global_step = 0
    worker.fine_policy_version = 0
    worker.replay = None
    worker.contract_fingerprint = None
    worker.prefix_block_lengths = None

    observation = QAMPolicyObservation(
        cameras_uint8=torch.zeros(3, 2, 2, 3, dtype=torch.uint8),
        proprio=torch.zeros(14),
        prompt="adjust the bottle",
        task_id="adjust_bottle",
        transform_fingerprint="test",
    )
    monkeypatch.setattr(
        worker,
        "_policy_observation",
        lambda *args, **kwargs: observation,
    )
    monkeypatch.setattr(
        worker,
        "_condition_observations",
        lambda observations: (
            torch.zeros(len(observations), 4, 5),
            torch.zeros(len(observations), 14),
        ),
    )

    trajectory_steps = 3
    trajectory = Trajectory(
        actions=torch.zeros(trajectory_steps, 1, 20, 14),
        rewards=torch.zeros(trajectory_steps, 1, 20),
        # step 1 is the first real terminal (fields have one bootstrap slot).
        terminations=torch.tensor([False, False, True, False]).reshape(4, 1, 1),
        truncations=torch.zeros(4, 1, 1, dtype=torch.bool),
        versions=torch.zeros(trajectory_steps, 1, 1),
        forward_inputs={
            "qam_planned_action_normalized": torch.zeros(
                trajectory_steps,
                1,
                20,
                14,
            ),
            "qam_obs_feature": torch.zeros(trajectory_steps, 1, 4, 5),
            "qam_prefix_block_lengths": torch.tensor([3, 3, 3, 2])
            .reshape(1, 1, 4)
            .repeat(trajectory_steps, 1, 1),
            "qam_proprio_normalized": torch.zeros(trajectory_steps, 1, 14),
            "qam_projection_contract": torch.tensor([50, 20, 32, 14])
            .reshape(1, 1, 4)
            .repeat(trajectory_steps, 1, 1),
            "qam_projection_fingerprint": torch.arange(32, dtype=torch.uint8)
            .reshape(1, 1, 32)
            .repeat(trajectory_steps, 1, 1),
        },
        curr_obs={"present": torch.ones(1)},
        next_obs={"present": torch.ones(1)},
    )

    assert worker._ingest_trajectory(trajectory) == 2
    assert worker.replay is not None
    query_indices = [
        transition.query_index
        for transition in worker.replay._slots
        if transition is not None
    ]
    assert query_indices == [0, 1]
    assert worker.replay._slots[1].success_terminated


def test_qam_resume_phase_is_monotonic() -> None:
    assert _phase_transition_is_valid("collect", "q_only")
    assert _phase_transition_is_valid("collect", "am_on")
    assert _phase_transition_is_valid("q_only", "am_on")
    assert _phase_transition_is_valid("am_on", "am_on")
    assert not _phase_transition_is_valid("am_on", "q_only")
    assert not _phase_transition_is_valid("unknown", "collect")


def _checkpoint_worker() -> QAMFSDPPolicy:
    worker = object.__new__(QAMFSDPPolicy)
    worker._rank = 0
    worker._world_size = 1
    worker.phase = "q_only"
    worker.replay = None
    worker.contract_fingerprint = None
    worker.prefix_block_lengths = (2, 2, 2, 1)
    worker.critic_feature_dim = None
    worker.critic = None
    worker.target_critic = None
    worker.critic_optimizer = None
    worker.runner_global_step = 2
    worker.fine_policy_version = 0
    worker.critic_updates = 4
    worker.fine_updates = 0
    worker.local_total_inserts = 7
    worker.global_total_inserts = 7
    worker.update_credit = QAMUpdateCredit(utd_ratio=1.0, pending=3.0)
    worker.q_only_anchor_global_inserts = 0
    worker.qam_cfg = OmegaConf.create(
        {
            "warmup_global_inserts": 512,
            "q_only_updates_before_am": 512,
            "utd_ratio": 1.0,
            "inv_temp": 1.0,
        }
    )
    return worker


def test_qam_checkpoint_snapshot_manifest_and_preflight(
    tmp_path,
    monkeypatch,
) -> None:
    worker = _checkpoint_worker()
    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_qam_policy_worker.get_rng_state",
        lambda: {"rank_local_marker": 17},
    )

    base_path = tmp_path / "global_step_3" / "actor"
    worker._write_qam_checkpoint_components(str(base_path), 3)
    manifest_path = worker._completion_manifest_path(str(base_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "complete",
        "schema_version",
        "checkpoint_step",
        "snapshot_id",
        "world_size",
    }

    sidecar = torch.load(
        worker._sidecar_path(str(base_path)),
        map_location="cpu",
        weights_only=False,
    )
    assert sidecar["snapshot"]["snapshot_id"] == manifest["snapshot_id"]
    assert sidecar["rng_state"] == {"rank_local_marker": 17}
    assert worker._preflight_qam_checkpoint(str(base_path))["rng_state"] == {
        "rank_local_marker": 17
    }

    worker.qam_cfg.inv_temp = 0.3
    with pytest.raises(ValueError, match="schedule contract mismatch"):
        worker._preflight_qam_checkpoint(str(base_path))
    worker.qam_cfg.inv_temp = 1.0

    manifest["snapshot_id"] = "wrong-snapshot"
    worker._atomic_json_dump(manifest, manifest_path)
    with pytest.raises(ValueError, match="snapshot mismatch"):
        worker._preflight_qam_checkpoint(str(base_path))


def test_qam_checkpoint_status_requires_every_rank() -> None:
    worker = _checkpoint_worker()
    worker._world_size = 2
    with pytest.raises(ValueError, match="rank set mismatch"):
        worker._gather_checkpoint_status(
            {
                "rank": 0,
                "error": None,
                "signature": ("snapshot",),
            }
        )


def test_qam_double_opt_in_is_fail_closed_and_legacy_is_untouched() -> None:
    legacy = OmegaConf.create(
        {
            "algorithm": {"loss_type": "actor"},
            "actor": {"model": {"openpi": {"use_qam": False}}},
        }
    )
    _validate_embodied_qam_contract(legacy, only_eval=False)

    half_enabled = OmegaConf.create(
        {
            "algorithm": {"loss_type": "embodied_qam"},
            "actor": {"model": {"openpi": {"use_qam": False}}},
        }
    )
    with pytest.raises(ValueError, match="requires both"):
        _validate_embodied_qam_contract(half_enabled, only_eval=False)


def test_qam_double_opt_in_allows_standalone_evaluation() -> None:
    cfg = _load_qam_source_config()
    cfg.actor.model.model_type = "openpi"
    _validate_embodied_qam_contract(cfg, only_eval=True)


def test_qam_requires_raw_transition_collection() -> None:
    cfg = _load_qam_source_config()
    cfg.actor.model.model_type = "openpi"
    cfg.rollout.collect_transitions = False
    with pytest.raises(ValueError, match="collect_transitions=true"):
        _validate_embodied_qam_contract(cfg, only_eval=False)


def test_qam_am_on_uses_counted_burn_in_without_manual_evidence_gate() -> None:
    cfg = _load_qam_source_config()
    cfg.actor.model.model_type = "openpi"
    cfg.algorithm.qam.phase = "am_on"
    cfg.algorithm.qam.inv_temp = 1.0
    _validate_embodied_qam_contract(cfg, only_eval=False)
