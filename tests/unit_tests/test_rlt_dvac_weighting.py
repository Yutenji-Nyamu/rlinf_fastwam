import torch

from rlinf.algorithms.rlt.dvac_weighting import (
    FrozenGlobalZMoments,
    build_rlt_bc_targets_and_weights,
    centered_mean_one_weights,
    compute_endpoint_variances,
    episode_success_flags,
    global_z_weights,
    masked_weight_totals,
    straight_through_scale_actions,
)
from rlinf.algorithms.rlt.transition import (
    core_rlt_obs,
    extract_rlt_obs_from_forward_inputs,
)


def test_endpoint_variance_uses_population_tail_variance():
    endpoints = torch.tensor([0.0, 1.0, 2.0, 3.0]).reshape(1, 4, 1, 1)
    variances = compute_endpoint_variances(endpoints)
    torch.testing.assert_close(variances[2], torch.tensor([[0.25]]))
    torch.testing.assert_close(variances[3], torch.tensor([[2.0 / 3.0]]))
    torch.testing.assert_close(variances[4], torch.tensor([[1.25]]))


def test_global_z_mapping_and_frozen_state_roundtrip():
    moments = FrozenGlobalZMoments()
    moments.freeze_from_statistics(count=3, total=0.0, total_sq=3.0)
    weights, z_scores, _ = global_z_weights(
        torch.exp(torch.tensor([[-2.0, 0.0, 2.0]])),
        mean=moments.mean,
        std=moments.std,
        strength=0.5,
        z_clip=2.0,
    )
    torch.testing.assert_close(z_scores, torch.tensor([[-2.0, 0.0, 2.0]]))
    torch.testing.assert_close(weights, torch.tensor([[0.0, 1.0, 2.0]]))

    restored = FrozenGlobalZMoments()
    restored.load_state_dict(moments.state_dict())
    assert restored.state_dict() == moments.state_dict()


def test_action_straight_through_keeps_forward_and_scales_backward():
    actions = torch.arange(4.0).reshape(1, 2, 2).requires_grad_(True)
    weights = torch.tensor([[0.0, 2.0]])
    scaled = straight_through_scale_actions(actions, weights, action_dim=2)
    torch.testing.assert_close(scaled, actions)
    scaled.sum().backward()
    torch.testing.assert_close(actions.grad, torch.tensor([[[0.0, 0.0], [2.0, 2.0]]]))


def test_centered_mean_one_mapping_is_detached_and_bounded():
    z_scores = torch.tensor([[-2.0, -2.0, 2.0, 2.0]], requires_grad=True)
    weights = centered_mean_one_weights(z_scores, strength=0.25)
    torch.testing.assert_close(weights.mean(dim=-1), torch.ones(1))
    torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5, 1.5, 1.5]]))
    assert not weights.requires_grad
    flat = centered_mean_one_weights(torch.ones(2, 10), strength=0.25)
    torch.testing.assert_close(flat, torch.ones_like(flat))


def test_episode_success_flags_cover_all_rows_of_each_environment():
    rewards = torch.zeros(4, 3, 2)
    rewards[2, 1, 0] = 1.0
    rewards[0, 2, 1] = 0.5
    torch.testing.assert_close(
        episode_success_flags(rewards), torch.tensor([False, True, True])
    )


def test_success_episode_bc_targets_weights_and_action_gradients():
    executed = torch.tensor([[[2.0], [4.0]], [[3.0], [5.0]], [[7.0], [9.0]]])
    reference = torch.zeros_like(executed)
    human_mask = torch.tensor([[False, False], [False, False], [True, False]])
    episode_success = torch.tensor([True, False, False])
    success_weights = torch.tensor([[0.5, 1.5], [0.5, 1.5], [0.5, 1.5]])

    targets, weights, success_mask, executed_mask = build_rlt_bc_targets_and_weights(
        executed,
        reference,
        human_mask,
        episode_success=episode_success,
        success_weights=success_weights,
        success_episode_bc=True,
    )
    torch.testing.assert_close(targets[0], executed[0])
    torch.testing.assert_close(targets[1], reference[1])
    torch.testing.assert_close(targets[2, 0], executed[2, 0])
    torch.testing.assert_close(targets[2, 1], reference[2, 1])
    torch.testing.assert_close(weights[0], success_weights[0])
    torch.testing.assert_close(weights[1:], torch.ones_like(weights[1:]))
    assert success_mask[0].all() and not success_mask[1:].any()
    assert executed_mask[2, 0] and not executed_mask[2, 1]

    student = torch.zeros_like(executed, requires_grad=True)
    loss = (weights * (student - targets).square().mean(dim=-1)).mean()
    loss.backward()
    expected = 2.0 * weights[..., None] * (student.detach() - targets) / 6.0
    torch.testing.assert_close(student.grad, expected)


def test_disabled_success_episode_bc_preserves_original_human_rule():
    executed = torch.ones(1, 2, 1)
    reference = torch.zeros_like(executed)
    human_mask = torch.tensor([[False, True]])
    targets, weights, success_mask, executed_mask = build_rlt_bc_targets_and_weights(
        executed,
        reference,
        human_mask,
        success_episode_bc=False,
    )
    torch.testing.assert_close(targets, torch.tensor([[[0.0], [1.0]]]))
    torch.testing.assert_close(weights, torch.ones(1, 2))
    assert not success_mask.any()
    torch.testing.assert_close(executed_mask, human_mask)


def test_optional_teacher_dvac_fields_roundtrip_without_becoming_required():
    base = {
        "z_rl": torch.zeros(2, 4),
        "proprio": torch.zeros(2, 3),
        "ref_chunk": torch.zeros(2, 2, 3),
    }
    assert set(extract_rlt_obs_from_forward_inputs(base)) == set(base)

    with_dvac = {
        **base,
        "teacher_dvac_v": torch.zeros(2, 3, 50),
        "dvac_collection_version": torch.ones(2, 1, dtype=torch.long),
        "actor_switch": torch.ones(2, 1, dtype=torch.bool),
    }
    extracted = extract_rlt_obs_from_forward_inputs(with_dvac)
    assert set(extracted) == set(with_dvac)
    torch.testing.assert_close(extracted["teacher_dvac_v"], with_dvac["teacher_dvac_v"])

    transition_inputs = {
        f"rlt_transition_{key}": value for key, value in with_dvac.items()
    }
    transition_obs = extract_rlt_obs_from_forward_inputs(
        transition_inputs, transition=True
    )
    assert set(transition_obs) == set(with_dvac)
    torch.testing.assert_close(
        transition_obs["teacher_dvac_v"], with_dvac["teacher_dvac_v"]
    )


def test_core_rlt_obs_excludes_current_only_replay_metadata():
    obs = {
        "z_rl": torch.zeros(1, 4),
        "proprio": torch.zeros(1, 3),
        "ref_chunk": torch.zeros(1, 2, 3),
        "teacher_dvac_v": torch.zeros(1, 3, 50),
        "actor_switch": torch.ones(1, 1, dtype=torch.bool),
        "episode_success": torch.ones(1, 1, 1, dtype=torch.bool),
    }

    projected = core_rlt_obs(obs)

    assert set(projected) == {"z_rl", "proprio", "ref_chunk"}
    projected["z_rl"].add_(1)
    assert not torch.equal(projected["z_rl"], obs["z_rl"])


def test_masked_weight_totals_keep_empty_groups_in_the_metric_schema():
    weights = torch.tensor([0.5, 1.0, 2.0])
    selected_sum, selected_count = masked_weight_totals(
        weights, torch.tensor([True, False, True])
    )
    empty_sum, empty_count = masked_weight_totals(
        weights, torch.tensor([False, False, False])
    )

    assert selected_sum == 2.5
    assert selected_count == 2.0
    assert empty_sum == 0.0
    assert empty_count == 0.0
