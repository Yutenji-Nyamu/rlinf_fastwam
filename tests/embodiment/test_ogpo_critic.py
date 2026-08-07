from __future__ import annotations

import inspect

import torch

from rlinf.models.embodiment.modules.ogpo_critic import (
    OGPOCriticEnsemble,
    pool_ogpo_prefix_blocks,
)


def test_prefix_pooling_produces_three_image_blocks_and_language_block() -> None:
    prefix = torch.arange(2 * 8 * 2, dtype=torch.float32).reshape(2, 8, 2)
    masks = torch.ones(2, 8, dtype=torch.bool)
    masks[0, -2:] = False

    pooled, block_lengths = pool_ogpo_prefix_blocks(
        prefix,
        masks,
        language_token_count=2,
    )

    assert pooled.shape == (2, 4, 2)
    assert block_lengths == (2, 2, 2, 2)
    assert torch.equal(pooled[0, 0], prefix[0, :2].mean(dim=0))
    assert torch.equal(pooled[0, -1], torch.zeros(2))
    assert torch.equal(pooled[1, -1], prefix[1, -2:].mean(dim=0))


def test_critic_returns_batch_by_head_and_zero_pads_short_actions() -> None:
    torch.manual_seed(5)
    critic = OGPOCriticEnsemble(
        feature_dim=2,
        proprio_dim=3,
        action_horizon=4,
        action_dim=2,
        num_q_heads=3,
        hidden_dims=(8, 8, 8, 8, 8),
    )
    feature = torch.randn(2, 4, 2, dtype=torch.bfloat16)
    proprio = torch.randn(2, 3, dtype=torch.bfloat16)
    short_action = torch.randn(2, 2, 2, dtype=torch.bfloat16)
    lengths = torch.tensor([2, 1])

    short_output = critic(feature, proprio, short_action, lengths)
    full_action = torch.zeros(2, 4, 2, dtype=torch.bfloat16)
    full_action[:, :2] = short_action
    full_action[1, 1] = 0
    full_output = critic(feature, proprio, full_action)

    assert short_output.shape == (2, 3)
    assert short_output.dtype == torch.float32
    assert torch.allclose(short_output, full_output)
    assert all(parameter.dtype == torch.float32 for parameter in critic.parameters())
    assert (
        critic.q_functions[0].network[0].weight
        is not critic.q_functions[1].network[0].weight
    )

    clone = OGPOCriticEnsemble(
        feature_dim=2,
        proprio_dim=3,
        action_horizon=4,
        action_dim=2,
        num_q_heads=3,
        hidden_dims=(8, 8, 8, 8, 8),
    )
    clone.load_state_dict(critic.state_dict())
    assert torch.equal(short_output, clone(feature, proprio, short_action, lengths))


def test_planned_default_ensemble_and_network_depth_are_explicit() -> None:
    signature = inspect.signature(OGPOCriticEnsemble.__init__)
    assert signature.parameters["action_horizon"].default == 10
    assert signature.parameters["num_q_heads"].default == 10
    assert signature.parameters["hidden_dims"].default == (
        512,
        512,
        512,
        512,
        512,
    )
