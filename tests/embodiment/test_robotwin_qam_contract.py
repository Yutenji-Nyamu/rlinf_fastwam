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

from pathlib import Path

import pytest
import torch

from rlinf.algorithms.qam.contracts import (
    ACTIVE_ACTION_DIM,
    MODEL_ACTION_DIM,
    MODEL_HORIZON,
    PLANNED_HORIZON,
    PREFIX_BLOCKS,
    QAMMacroTransition,
    QAMPolicyObservation,
    canonicalize_model_action,
    embed_planned_adjoint,
    fixed_slot_bootstrap_discount,
    fixed_slot_discounted_return,
    macro_bootstrap_mask,
    project_planned_action,
)
from rlinf.data.qam_transition_replay import QAMTransitionReplay

CONTRACT_FINGERPRINT = "base+norm+pooling+action-v1"
TEST_PREFIX_FEATURE_DIM = 8


def _observation(value: int) -> QAMPolicyObservation:
    return QAMPolicyObservation(
        cameras_uint8=torch.full((3, 2, 2, 3), value, dtype=torch.uint8),
        proprio=torch.full((ACTIVE_ACTION_DIM,), float(value)),
        prompt="adjust the bottle",
        task_id="adjust_bottle",
        transform_fingerprint="robotwin-transform-v1",
    )


def _transition(
    observation: QAMPolicyObservation,
    next_observation: QAMPolicyObservation | None,
    *,
    episode_id: str,
    query_index: int,
    reward_slot: int | None = None,
    success: bool = False,
    timeout: bool = False,
    other_truncated: bool = False,
    policy_version: int = 0,
    gamma_slot: float = 0.99,
) -> QAMMacroTransition:
    rewards = torch.zeros(PLANNED_HORIZON)
    if reward_slot is not None:
        rewards[reward_slot] = 1.0
    next_valid = next_observation is not None
    bootstrap = macro_bootstrap_mask(
        success_terminated=success,
        time_limit_truncated=timeout,
        other_truncated=other_truncated,
        next_state_valid=next_valid,
    )
    return QAMMacroTransition(
        obs_id=observation.content_id(),
        next_obs_id=(
            next_observation.content_id() if next_observation is not None else None
        ),
        obs_feature=torch.full(
            (PREFIX_BLOCKS, TEST_PREFIX_FEATURE_DIM),
            float(query_index),
            dtype=torch.bfloat16,
        ),
        obs_proprio=observation.proprio.clone(),
        next_obs_feature=(
            torch.full(
                (PREFIX_BLOCKS, TEST_PREFIX_FEATURE_DIM),
                float(query_index + 1),
                dtype=torch.bfloat16,
            )
            if next_valid
            else None
        ),
        next_obs_proprio=(
            next_observation.proprio.clone() if next_observation is not None else None
        ),
        next_state_valid=next_valid,
        planned_actions_normalized=torch.zeros(
            PLANNED_HORIZON,
            ACTIVE_ACTION_DIM,
        ),
        planned_actions_env=torch.full(
            (PLANNED_HORIZON, ACTIVE_ACTION_DIM),
            0.25,
        ),
        chunk_rewards_native=rewards,
        reward_macro_discounted=float(
            fixed_slot_discounted_return(
                rewards,
                gamma_slot=gamma_slot,
            ).item()
        ),
        success_terminated=success,
        time_limit_truncated=timeout,
        other_truncated=other_truncated,
        bootstrap_mask=bootstrap,
        policy_version=policy_version,
        episode_id=episode_id,
        query_index=query_index,
        contract_fingerprint=CONTRACT_FINGERPRINT,
    )


def _replay(*, capacity: int = 3, rank: int = 0) -> QAMTransitionReplay:
    return QAMTransitionReplay(
        capacity=capacity,
        rank=rank,
        world_size=2,
        seed=17,
        gamma_slot=0.99,
        contract_fingerprint=CONTRACT_FINGERPRINT,
    )


def test_model_projection_clamp_and_pullback() -> None:
    model_action = torch.zeros(MODEL_HORIZON, MODEL_ACTION_DIM)
    model_action[:PLANNED_HORIZON, :ACTIVE_ACTION_DIM] = 2.0
    model_action[PLANNED_HORIZON:, :ACTIVE_ACTION_DIM] = 3.0
    model_action[:PLANNED_HORIZON, ACTIVE_ACTION_DIM:] = 4.0

    canonical = canonicalize_model_action(model_action)
    planned = project_planned_action(model_action)
    assert canonical[:PLANNED_HORIZON, :ACTIVE_ACTION_DIM].max().item() == 1.0
    assert canonical[PLANNED_HORIZON:, :ACTIVE_ACTION_DIM].min().item() == 3.0
    assert canonical[:PLANNED_HORIZON, ACTIVE_ACTION_DIM:].min().item() == 4.0
    torch.testing.assert_close(
        planned,
        torch.ones(PLANNED_HORIZON, ACTIVE_ACTION_DIM),
    )

    embedded = embed_planned_adjoint(torch.ones_like(planned))
    torch.testing.assert_close(
        embedded[:PLANNED_HORIZON, :ACTIVE_ACTION_DIM],
        torch.ones_like(planned),
    )
    assert torch.count_nonzero(embedded[PLANNED_HORIZON:, :]).item() == 0
    assert (
        torch.count_nonzero(embedded[:PLANNED_HORIZON, ACTIVE_ACTION_DIM:]).item() == 0
    )

    differentiable = torch.zeros(
        MODEL_HORIZON,
        MODEL_ACTION_DIM,
        requires_grad=True,
    )
    project_planned_action(differentiable).sum().backward()
    expected_gradient = embed_planned_adjoint(torch.ones_like(planned))
    torch.testing.assert_close(differentiable.grad, expected_gradient)


def test_fixed_slot_return_and_bootstrap_rules() -> None:
    rewards = torch.zeros(PLANNED_HORIZON)
    rewards[0] = 1.0
    rewards[2] = 1.0
    reduced = fixed_slot_discounted_return(rewards, gamma_slot=0.5)
    torch.testing.assert_close(reduced, torch.tensor(1.25))
    integer_reduced = fixed_slot_discounted_return(
        rewards.to(torch.int64),
        gamma_slot=0.5,
    )
    torch.testing.assert_close(integer_reduced, torch.tensor(1.25))
    assert fixed_slot_bootstrap_discount(gamma_slot=0.5) == 0.5**20

    assert (
        macro_bootstrap_mask(
            success_terminated=True,
            time_limit_truncated=False,
            other_truncated=False,
            next_state_valid=False,
        )
        == 0.0
    )
    assert (
        macro_bootstrap_mask(
            success_terminated=False,
            time_limit_truncated=False,
            other_truncated=False,
            next_state_valid=True,
        )
        == 1.0
    )
    assert (
        macro_bootstrap_mask(
            success_terminated=False,
            time_limit_truncated=True,
            other_truncated=False,
            next_state_valid=True,
        )
        == 1.0
    )
    assert (
        macro_bootstrap_mask(
            success_terminated=False,
            time_limit_truncated=False,
            other_truncated=True,
            next_state_valid=False,
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="requires a valid next state"):
        macro_bootstrap_mask(
            success_terminated=False,
            time_limit_truncated=True,
            other_truncated=False,
            next_state_valid=False,
        )


def test_rank_local_ring_deduplicates_and_evicts_observations() -> None:
    observations = [_observation(index) for index in range(4)]
    replay = _replay(capacity=2)
    replay.add(
        _transition(observations[0], observations[1], episode_id="e0", query_index=0),
        observation=observations[0],
        next_observation=observations[1],
    )
    replay.add(
        _transition(observations[1], observations[2], episode_id="e0", query_index=1),
        observation=observations[1],
        next_observation=observations[2],
    )
    assert len(replay) == 2
    assert replay.observation_store_size == 3

    replay.add(
        _transition(observations[2], observations[3], episode_id="e0", query_index=2),
        observation=observations[2],
        next_observation=observations[3],
    )
    assert len(replay) == 2
    assert replay.cursor == 1
    assert replay.total_inserted == 3
    assert replay.observation_store_size == 3

    samples = replay.sample(8)
    assert {sample.transition.query_index for sample in samples} <= {1, 2}
    assert all(
        sample.observation.content_id() == sample.transition.obs_id
        for sample in samples
    )


def test_success_and_timeout_store_only_causal_next_views() -> None:
    current = _observation(1)
    final = _observation(2)
    replay = _replay()

    success = _transition(
        current,
        None,
        episode_id="success",
        query_index=0,
        reward_slot=PLANNED_HORIZON - 1,
        success=True,
    )
    replay.add(success, observation=current, next_observation=None)

    timeout = _transition(
        current,
        final,
        episode_id="timeout",
        query_index=0,
        timeout=True,
    )
    replay.add(timeout, observation=current, next_observation=final)
    assert timeout.bootstrap_mask == 1.0
    assert timeout.next_state_valid


def test_replay_checkpoint_restores_ring_rng_and_rejects_rank_mismatch(
    tmp_path: Path,
) -> None:
    observations = [_observation(index) for index in range(4)]
    replay = _replay(capacity=3)
    for index in range(3):
        replay.add(
            _transition(
                observations[index],
                observations[index + 1],
                episode_id="resume",
                query_index=index,
                policy_version=index,
            ),
            observation=observations[index],
            next_observation=observations[index + 1],
        )

    replay.sample(5)
    checkpoint = tmp_path / "rank0_qam_replay.pt"
    replay.save_checkpoint(checkpoint)
    assert checkpoint.is_file()
    assert not [path for path in tmp_path.iterdir() if ".tmp-" in path.name]

    resumed = _replay(capacity=3)
    resumed.load_checkpoint(checkpoint)
    assert len(resumed) == len(replay)
    assert resumed.cursor == replay.cursor
    assert resumed.total_inserted == replay.total_inserted
    assert resumed.observation_store_size == replay.observation_store_size
    expected = [sample.transition.query_index for sample in replay.sample(16)]
    actual = [sample.transition.query_index for sample in resumed.sample(16)]
    assert actual == expected

    wrong_rank = _replay(capacity=3, rank=1)
    with pytest.raises(ValueError, match="metadata mismatch"):
        wrong_rank.load_checkpoint(checkpoint)
