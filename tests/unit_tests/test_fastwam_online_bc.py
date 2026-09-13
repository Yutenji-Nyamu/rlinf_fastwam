# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Small CPU contracts; model/loading and one-update checks run on GPU separately."""
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.online_bc import SuccessEpisodeCollector
from rlinf.models.embodiment.fastwam.online_bc import (
    native_action_fm_loss,
    prepare_command_targets,
)
from rlinf.models.embodiment.fastwam.fastwam_rl import prepare_initial_action_latents


def processor():
    mean = torch.arange(14).float() / 10
    std = torch.arange(1, 15).float() / 7
    norm = SimpleNamespace(
        forward=lambda a: (a - mean) / std,
        backward=lambda a: a * std + mean,
    )
    return SimpleNamespace(normalizer=SimpleNamespace(
        normalizers={"action": {"default": norm}}
    ))


def test_submitted_prefix_roundtrip_and_unsubmitted_tail_mask():
    commands = torch.randn(2, 24, 14)
    p = processor()
    target, valid = prepare_command_targets(commands, torch.ones_like(commands).bool(), p)
    recovered = p.normalizer.normalizers["action"]["default"].backward(target[:, :24])
    torch.testing.assert_close(recovered, commands)
    assert target.shape == valid.shape == (2, 32, 14)
    assert valid[:, :24].all() and not valid[:, 24:].any()
    assert target[:, 24:].eq(0).all()


def test_fm_matches_official_valid_action_mean_and_timestep_weight():
    pred = torch.randn(3, 32, 14, requires_grad=True)
    target = torch.randn_like(pred)
    valid = torch.zeros_like(pred).bool()
    valid[:, :24] = True
    weight = torch.tensor([0.2, 1.0, 1.7])
    scheduler = SimpleNamespace(training_weight=lambda t: weight)
    got = native_action_fm_loss(pred, target, valid, torch.ones(3), scheduler)
    expected = (((pred[:, :24] - target[:, :24]) ** 2).mean((1, 2)) * weight).mean()
    torch.testing.assert_close(got, expected)
    got.backward()
    assert pred.grad[:, 24:].eq(0).all()
    assert pred.grad[:, :24].abs().sum() > 0


def test_invalid_tail_values_do_not_change_loss():
    pred = torch.zeros(1, 32, 14)
    target = torch.ones_like(pred)
    valid = torch.zeros_like(pred).bool()
    valid[:, :24] = True
    scheduler = SimpleNamespace(training_weight=lambda t: torch.ones(1))
    before = native_action_fm_loss(pred, target, valid, torch.ones(1), scheduler)
    pred[:, 24:] = 10000
    after = native_action_fm_loss(pred, target, valid, torch.ones(1), scheduler)
    torch.testing.assert_close(before, after)


def test_missing_labels_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        prepare_command_targets(torch.ones(1, 24, 14), torch.zeros(1, 24, 14).bool(), processor())
    with pytest.raises(ValueError, match="submitted"):
        prepare_command_targets(torch.ones(1, 32, 14), torch.ones(1, 32, 14).bool(), processor())


def test_success_collector_keeps_fast_observations_and_actual_commands():
    collector = SuccessEpisodeCollector(2)
    inputs = {"image": torch.ones(2, 3, 2, 2), "text_context": torch.ones(2, 4, 8),
              "text_context_mask": torch.ones(2, 4).bool(), "proprio": torch.ones(2, 14),
              "chains": torch.full((2, 11, 32, 14), 90.0),
              "model_action": torch.full((2, 32 * 14), 99.0)}
    commands = torch.full((2, 24, 14), 7.0)
    collector.append(inputs, commands, torch.tensor([True, False]), torch.ones(2).bool())
    commands.zero_()
    inputs["image"].zero_()
    episodes = collector.drain()
    assert len(episodes) == 1 and len(episodes[0]) == 1
    record = episodes[0][0]
    assert record["action"].eq(7).all() and record["image"].eq(1).all()
    assert record["action_valid_mask"].shape == (24, 14)
    assert "chains" not in record and "model_action" not in record


def test_continuous_bc_noise_and_eval_rng_restore():
    def draw():
        return prepare_initial_action_latents(batch_size=2, action_horizon=32,
            action_dim=14, device="cpu", dtype=torch.float32)
    torch.manual_seed(42)
    first = draw()
    state = torch.get_rng_state()
    expected_next = draw()
    torch.set_rng_state(state)
    saved = torch.get_rng_state()
    torch.manual_seed(42)
    eval_first = draw()
    draw()
    torch.set_rng_state(saved)
    actual_next = draw()
    assert not torch.equal(first[0], first[1])
    assert not torch.equal(first, actual_next)
    torch.testing.assert_close(eval_first, first)
    torch.testing.assert_close(actual_next, expected_next)
