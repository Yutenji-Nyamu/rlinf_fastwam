# Copyright 2025 The RLinf Authors.
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

"""Prism reward and DVAC action-weight composition; run on the server CPU."""

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from test_dvac_adv_new_loss import native_loss, sample_inputs

from rlinf.algorithms.advantages import compute_prism_rloo_advantages
from rlinf.algorithms.dvac_rank_reward import trajectory_dvac_quality
from rlinf.algorithms.dvac_two_level import compute_dvac_two_level_weights
from rlinf.config import validate_prism_dvac_cfg

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "examples/embodiment/config"


def control_config() -> DictConfig:
    """Return the fields read by the real global Prism contract validator."""
    model = {
        "model_type": "openpi",
        "num_action_chunks": 50,
        "openpi": {"num_steps": 10, "action_chunk": 50},
    }
    return OmegaConf.create(
        {
            "runner": {
                "task_type": "embodied",
                "logger": {"log_path": "/tmp/combination", "experiment_name": "test"},
            },
            "actor": {"model": model},
            "rollout": {"model": model},
            "algorithm": {
                "adv_type": "grpo",
                "normalize_advantages": True,
                "filter_rewards": True,
                "reward_type": "chunk_level",
                "logprob_type": "chunk_level",
                "reward_coef": 1.0,
                "group_size": 8,
                "dvac_gradient_weighting": {"mode": "off"},
            },
            "env": {
                "train": {
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "use_custom_reward": True,
                    "use_rel_reward": True,
                    "reward_coef": 1.0,
                    "group_size": 8,
                }
            },
        }
    )


def combined_config() -> DictConfig:
    """Apply the actual global overlay to explicit Control method fields."""
    return OmegaConf.merge(
        control_config(),
        OmegaConf.load(CONFIG_ROOT / "prism_dvac/rank_rloo_adv_new.yaml"),
    )


def composed_inputs(
    outcomes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one complete G8 scene with a local ramp and trajectory offsets."""
    log_variance = torch.arange(8).reshape(1, 8, 1).float() + torch.tensor(
        [0.0, 0.3, 1.0]
    ).reshape(1, 1, 3)
    variance = log_variance.exp().requires_grad_()
    action_mask = torch.ones_like(variance, dtype=torch.bool)
    _, quality = trajectory_dvac_quality(
        variance, action_mask, group_size=8, log_eps=1e-12
    )
    advantages, _ = compute_prism_rloo_advantages(
        outcomes.reshape(1, 8), action_mask.any(-1), 8, quality, 0.2
    )
    return variance, action_mask, quality, advantages.unsqueeze(-1)


def test_joint_preset_overrides_control_and_preserves_single_modes() -> None:
    config = combined_config()
    validate_prism_dvac_cfg(config)
    assert config.algorithm.adv_type == "prism_rloo"
    assert not config.algorithm.normalize_advantages
    assert not config.algorithm.filter_rewards
    assert config.algorithm.prism_dvac.quality_lambda == 0.2
    dvac = config.algorithm.dvac_gradient_weighting
    assert dvac.mode == "apply" and dvac.scope == "both"
    assert dvac.alpha_local == dvac.alpha_chunk == 1.0
    assert config.algorithm.prism_dvac.selected_l == dvac.selected_l == 3
    assert config.algorithm.prism_dvac.log_eps == dvac.log_eps == 1e-12
    assert dvac.output_dir == "/tmp/combination/test/dvac_train"

    prism_only = OmegaConf.merge(
        control_config(), OmegaConf.load(CONFIG_ROOT / "prism_dvac/rank_rloo.yaml")
    )
    validate_prism_dvac_cfg(prism_only)
    assert prism_only.algorithm.dvac_gradient_weighting.mode == "off"
    new_only = control_config()
    new_only.algorithm.dvac_gradient_weighting = OmegaConf.load(
        CONFIG_ROOT / "dvac_grpo/adv_new.yaml"
    )
    validate_prism_dvac_cfg(new_only)
    assert new_only.algorithm.adv_type == "grpo"
    assert new_only.algorithm.filter_rewards
    validate_prism_dvac_cfg(control_config())


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("dvac_gradient_weighting.mode", "shadow", "two-level"),
        ("dvac_gradient_weighting.application", "action_advantage", "two-level"),
        ("dvac_gradient_weighting.normalization", "recent_stats", "two-level"),
        ("dvac_gradient_weighting.selected_l", 2, "selected_l"),
        ("dvac_gradient_weighting.log_eps", 1e-8, "log_eps"),
        ("filter_rewards", True, "all-zero"),
        ("normalize_advantages", True, "standardize"),
        ("adv_type", "grpo", "must agree"),
    ],
)
def test_joint_config_rejects_unreviewed_contracts(
    key: str, value: object, message: str
) -> None:
    config = combined_config()
    OmegaConf.update(config, f"algorithm.{key}", value)
    with pytest.raises(ValueError, match=message):
        validate_prism_dvac_cfg(config)


@pytest.mark.parametrize("outcome", [0.0, 1.0])
def test_same_outcome_quality_signal_survives_weighting_without_std(
    outcome: float,
) -> None:
    variance, mask, quality, advantages = composed_inputs(torch.full((8,), outcome))
    expected = 0.2 * (8 * quality - quality.sum()) / 7
    torch.testing.assert_close(advantages.flatten(), expected, atol=1e-7, rtol=1e-6)
    assert advantages.abs().max() < 0.115
    weights, _ = compute_dvac_two_level_weights(
        variance, mask.any(-1), torch.zeros(8, dtype=torch.long), advantages
    )
    logprobs = torch.zeros(8, 3, 14, requires_grad=True)
    loss, _ = native_loss(
        logprobs, torch.zeros_like(logprobs), advantages[0], weights[0]
    )
    loss.backward()
    expected_grad = -(advantages[0] * weights[0])[..., None].expand_as(logprobs) / 8
    torch.testing.assert_close(logprobs.grad, expected_grad)
    assert torch.count_nonzero(logprobs.grad) > 0
    assert variance.grad is None and not weights.requires_grad


def test_weights_multiply_full_prism_advantage_not_just_success_component() -> None:
    outcomes = torch.tensor([0.0, 1, 0, 1, 0, 1, 0, 1])
    variance, mask, quality, advantages = composed_inputs(outcomes)
    binary_adv = (8 * outcomes - outcomes.sum()) / 7
    quality_adv = 0.2 * (8 * quality - quality.sum()) / 7
    torch.testing.assert_close(advantages.flatten(), binary_adv + quality_adv)
    weights, _ = compute_dvac_two_level_weights(
        variance, mask.any(-1), torch.zeros(8, dtype=torch.long), advantages
    )
    logprobs = torch.zeros(8, 3, 14, requires_grad=True)
    loss, _ = native_loss(
        logprobs, torch.zeros_like(logprobs), advantages[0], weights[0]
    )
    loss.backward()
    effective = (binary_adv + quality_adv)[:, None] * weights[0]
    torch.testing.assert_close(
        logprobs.grad, -effective[..., None].expand_as(logprobs) / 8
    )
    wrong = binary_adv[:, None] * weights[0] + quality_adv[:, None]
    assert not torch.allclose(effective, wrong)
    assert torch.equal(
        torch.sign(effective), torch.sign(advantages[0]).expand_as(effective)
    )


@pytest.mark.parametrize("masked", [False, True])
def test_identity_weights_recover_prism_loss_clip_metrics_and_gradient(
    masked: bool,
) -> None:
    _, _, _, advantages = composed_inputs(torch.tensor([1.0, 0, 1, 0, 1, 0, 1, 0]))
    weighted, _ = sample_inputs(50)
    prism_only = weighted.detach().clone().requires_grad_()
    options = {"masked": masked, "numeric_clamp": True, "dual_clip": True}
    loss, metrics = native_loss(
        weighted,
        torch.zeros_like(weighted),
        advantages[0],
        torch.ones(8, 50),
        **options,
    )
    reference, reference_metrics = native_loss(
        prism_only, torch.zeros_like(prism_only), advantages[0], **options
    )
    torch.testing.assert_close(loss, reference, atol=2e-6, rtol=2e-6)
    assert metrics.keys() == reference_metrics.keys()
    for key in metrics:
        torch.testing.assert_close(
            metrics[key], reference_metrics[key], equal_nan=True, atol=2e-6, rtol=2e-6
        )
    loss.backward()
    reference.backward()
    torch.testing.assert_close(weighted.grad, prism_only.grad, atol=2e-6, rtol=2e-6)


def test_terminal_action_mask_and_actor_chunk_mask_keep_distinct_domains() -> None:
    variance = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])
    action_mask = torch.ones_like(variance, dtype=torch.bool)
    action_mask[0, 0, 1] = (
        False  # Synthetic reported terminal tail, not physical timing.
    )
    changed = variance.clone()
    changed[0, 0, 1] = 1e4
    _, quality = trajectory_dvac_quality(
        variance, action_mask, group_size=4, log_eps=1e-12
    )
    _, changed_quality = trajectory_dvac_quality(
        changed, action_mask, group_size=4, log_eps=1e-12
    )
    torch.testing.assert_close(quality, changed_quality)
    chunk_mask = action_mask.any(-1)
    advantages, _ = compute_prism_rloo_advantages(
        torch.zeros(1, 4), chunk_mask, 4, quality, 0.2
    )
    args = (chunk_mask, torch.zeros(4, dtype=torch.long), advantages.unsqueeze(-1))
    weights, _ = compute_dvac_two_level_weights(variance, *args)
    changed_weights, _ = compute_dvac_two_level_weights(changed, *args)
    assert not torch.allclose(weights, changed_weights)
    with pytest.raises(ValueError, match="chunk mask"):
        compute_dvac_two_level_weights(variance, action_mask, args[1], args[2])


@pytest.mark.parametrize("scope", ["both", "positive", "negative"])
def test_scope_uses_final_prism_sign_in_same_outcome_groups(scope: str) -> None:
    variance, mask, _, advantages = composed_inputs(torch.zeros(8))
    weights, data = compute_dvac_two_level_weights(
        variance,
        mask.any(-1),
        torch.zeros(8, dtype=torch.long),
        advantages,
        scope=scope,
    )
    selected = torch.ones_like(advantages, dtype=torch.bool)
    if scope == "positive":
        selected = advantages > 0
    elif scope == "negative":
        selected = advantages < 0
    assert torch.equal(data["eligible_mask"], selected)
    excluded = ~selected.expand_as(weights)
    assert torch.equal(weights[excluded], torch.ones_like(weights[excluded]))
    assert (weights >= 0).all() and torch.isfinite(weights).all()


def test_tied_quality_and_zero_advantage_cannot_create_learning_signal() -> None:
    # Each trajectory has a different action pattern but the same mean log V.
    variance = torch.tensor([[[1.0, 4.0], [4.0, 1.0], [2.0, 2.0], [2.0, 2.0]]])
    mask = torch.ones_like(variance, dtype=torch.bool)
    _, quality = trajectory_dvac_quality(variance, mask, group_size=4, log_eps=1e-12)
    torch.testing.assert_close(quality, torch.full((4,), 0.5))
    advantages, _ = compute_prism_rloo_advantages(
        torch.zeros(1, 4), mask.any(-1), 4, quality, 0.2
    )
    weights, _ = compute_dvac_two_level_weights(
        variance,
        mask.any(-1),
        torch.zeros(4, dtype=torch.long),
        advantages.unsqueeze(-1),
    )
    assert not torch.allclose(weights, torch.ones_like(weights))
    logprobs = torch.zeros(4, 2, 14, requires_grad=True)
    loss, _ = native_loss(
        logprobs, torch.zeros_like(logprobs), advantages[0, :, None], weights[0]
    )
    loss.backward()
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)
    torch.testing.assert_close(
        logprobs.grad, torch.zeros_like(logprobs), atol=1e-7, rtol=0
    )
