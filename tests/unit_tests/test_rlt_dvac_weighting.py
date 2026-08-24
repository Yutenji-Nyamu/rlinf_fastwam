import torch

from rlinf.algorithms.rlt.dvac_weighting import (
    FrozenGlobalZMoments,
    compute_endpoint_variances,
    global_z_weights,
    masked_weight_totals,
    straight_through_scale_actions,
)
from rlinf.algorithms.rlt.transition import extract_rlt_obs_from_forward_inputs


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
