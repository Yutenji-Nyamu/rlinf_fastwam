# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.dvac_weighting import (
    FrozenGlobalZMoments,
    build_rlt_bc_targets_and_weights,
    centered_mean_one_weights,
    compute_endpoint_variances,
    episode_success_flags,
)
from rlinf.algorithms.rlt.transition import (
    core_rlt_obs,
    extract_rlt_obs_from_forward_inputs,
)


def test_endpoint_variances_use_population_tail_and_sum_action_dims():
    previews = torch.tensor(
        [[[[0.0], [0.0]], [[1.0], [2.0]], [[2.0], [4.0]], [[3.0], [6.0]]]]
    )
    result = compute_endpoint_variances(previews)

    torch.testing.assert_close(result[2], torch.tensor([[0.25, 1.0]]))
    torch.testing.assert_close(result[3], torch.tensor([[2.0 / 3.0, 8.0 / 3.0]]))
    torch.testing.assert_close(result[4], torch.tensor([[1.25, 5.0]]))


def test_pure04_weights_are_nonnegative_detached_and_mean_one():
    z_scores = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0, 2.0], [0.4, 0.4, 0.4, 0.4, 0.4]],
        requires_grad=True,
    )
    weights = centered_mean_one_weights(z_scores, strength=1.5)

    assert not weights.requires_grad
    assert torch.all(weights >= 0)
    torch.testing.assert_close(weights.mean(dim=-1), torch.ones(2))
    torch.testing.assert_close(weights[1], torch.ones(5))


def test_episode_success_uses_all_steps_but_returns_one_flag_per_env():
    rewards = torch.zeros(4, 3, 2)
    rewards[2, 1, 0] = 1.0
    rewards[0, 2, 1] = 0.5
    torch.testing.assert_close(
        episode_success_flags(rewards), torch.tensor([False, True, True])
    )


def test_disabled_weighting_preserves_original_reference_human_rule():
    executed = torch.tensor([[[2.0], [3.0]]])
    reference = torch.zeros_like(executed)
    human = torch.tensor([[False, True]])
    targets, weights, success = build_rlt_bc_targets_and_weights(
        executed, reference, human
    )
    torch.testing.assert_close(targets, torch.tensor([[[0.0], [3.0]]]))
    torch.testing.assert_close(weights, torch.ones(1, 2))
    assert not success.any()


def test_success_reference_bc_keeps_targets_and_reweights_entire_success_row():
    executed = torch.tensor([[[3.0], [3.0]], [[4.0], [4.0]]])
    reference = torch.ones_like(executed)
    human = torch.tensor([[True, False], [False, False]])
    success = torch.tensor([[True], [False]])
    success_weights = torch.tensor([[0.5, 1.5], [0.2, 1.8]])

    targets, weights, success_mask = build_rlt_bc_targets_and_weights(
        executed,
        reference,
        human,
        episode_success=success,
        success_weights=success_weights,
        apply_success_weights=True,
    )

    torch.testing.assert_close(targets[0, 0], executed[0, 0])
    torch.testing.assert_close(targets[0, 1], reference[0, 1])
    torch.testing.assert_close(targets[1], reference[1])
    torch.testing.assert_close(weights[0], success_weights[0])
    torch.testing.assert_close(weights[1], torch.ones(2))
    assert success_mask.tolist() == [[True, True], [False, False]]


def _loss_probe():
    module = pytest.importorskip("rlinf.workers.actor.fsdp_rlt_ac_policy_worker")

    class LossProbe(module.RLTACLossMixin):
        def __init__(self):
            self.cfg = OmegaConf.create(
                {"actor": {"model": {"num_action_chunks": 2, "action_dim": 1}}}
            )

    return LossProbe()


def test_weighted_bc_preserves_reference_and_human_override_math():
    probe = _loss_probe()
    pi = torch.zeros(2, 2, requires_grad=True)
    executed = torch.tensor([[3.0, 3.0], [4.0, 4.0]])
    reference = torch.ones(2, 2)
    intervene = torch.tensor([[True, False], [False, False]])

    loss, metrics = probe._bc_metrics(
        pi,
        executed,
        reference,
        intervene,
        episode_success=torch.tensor([[True], [False]]),
        success_weights=torch.tensor([[0.5, 1.5], [0.2, 1.8]]),
        apply_success_weights=True,
    )

    torch.testing.assert_close(loss, torch.tensor(2.0))
    assert metrics["rlt_dvac/success_query_ratio"] == pytest.approx(0.5)
    loss.backward()
    assert torch.isfinite(pi.grad).all()


def test_teacher_dvac_uses_typed_optional_path_but_next_obs_stays_core():
    forward_inputs = {
        "z_rl": torch.ones(1, 2),
        "proprio": torch.ones(1, 3),
        "ref_chunk": torch.ones(1, 2, 3),
        "teacher_dvac_v": torch.ones(1, 3, 50),
    }
    obs = extract_rlt_obs_from_forward_inputs(forward_inputs)
    assert set(obs) == {"z_rl", "proprio", "ref_chunk", "teacher_dvac_v"}

    obs["episode_success"] = torch.ones(1, 1, dtype=torch.bool)
    assert set(core_rlt_obs(obs)) == {"z_rl", "proprio", "ref_chunk"}


def _resume_probe(*, strength: float):
    worker_cls = pytest.importorskip(
        "rlinf.workers.actor.fsdp_rlt_ac_policy_worker"
    ).RLTACFSDPPolicy
    worker = worker_cls.__new__(worker_cls)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "rlt_resume": {
                    "enable": True,
                    "contract": {"stage1_manifest_id": "current-ar"},
                }
            }
        }
    )
    worker.rlt_resume_cfg = worker.cfg.algorithm.rlt_resume
    worker.rlt_dvac_mode = "apply"
    worker.rlt_dvac_cfg = {
        "mode": "apply",
        "application": "success_episode_bc",
        "success_target": "reference",
        "selected_l": 3,
        "strength": strength,
    }
    worker.rlt_dvac_stats = FrozenGlobalZMoments()
    worker.rlt_dvac_stats.freeze_from_statistics(2, 3.0, 5.0)
    worker._rank = 0
    worker._world_size = 1
    worker.update_step = 7
    worker.total_transitions_added = 20
    worker.total_episodes_added = 2
    worker._warmup_ready_total_transitions = 20
    worker._warmup_ready_total_episodes = 2
    worker.transitions_since_train = 1
    worker.episodes_since_train = 1
    worker.pending_update_budget = 3
    return worker


def test_strict_resume_restores_baseline_and_locks_pure_config():
    source = _resume_probe(strength=1.5)
    state = source._rlt_state_payload(runner_step=4)

    resumed = _resume_probe(strength=1.5)
    resumed.rlt_dvac_stats = FrozenGlobalZMoments()
    resumed._restore_rlt_state(state)
    assert resumed.rlt_dvac_stats.state_dict() == source.rlt_dvac_stats.state_dict()

    changed = _resume_probe(strength=1.0)
    with pytest.raises(ValueError, match="rlt_resume_contract"):
        changed._validate_rlt_state(state)
