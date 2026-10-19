# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Sign-direction, conservation and actual actor wiring regressions."""

import pytest
import torch
from test_dvac_adv_new_actor import fake_all_gather, make_actor

from rlinf.algorithms.dvac_linear_controls import apply_chunk_dropout
from rlinf.algorithms.dvac_two_level import (
    canonicalize_dvac_two_level_contract,
    compute_dvac_two_level_weights,
    dvac_mapping_contract,
)


def sample():
    """Identical increasing signals for matched positive and negative pools."""
    values = torch.tensor(
        [
            [
                [1.0, 2.0, 4.0],
                [2.0, 4.0, 8.0],
                [1.0, 2.0, 4.0],
                [2.0, 4.0, 8.0],
                [9.0, 2.0, 1.0],
            ]
        ]
    )
    adv = torch.tensor([[[1.0], [1.0], [-1.0], [-1.0], [0.0]]])
    return (
        values,
        torch.ones(1, 5, 1, dtype=torch.bool),
        torch.zeros(5, dtype=torch.long),
        adv,
    )


def compute(*args, **options):
    config = {
        "mapping": "exp_mean",
        "direction": "advantage_sign",
        "temperature_local": 2.0,
        "temperature_chunk": 2.0,
    }
    config.update(options)
    return compute_dvac_two_level_weights(*args, **config)


@pytest.mark.parametrize("scope", ["both", "positive"])
def test_direction_and_separate_sign_mass(scope):
    values, mask, ids, adv = sample()
    mass = torch.tensor([[1.0, 3.0, 2.0, 5.0, 4.0]])
    weights, data = compute(
        values, mask, ids, adv, scope=scope, chunk_contributions=mass
    )
    assert (weights[0, :2, 2] > weights[0, :2, 0]).all()
    assert data["chunk_factors"][0, 1] > data["chunk_factors"][0, 0]
    if scope == "both":
        assert (weights[0, 2:4, 0] > weights[0, 2:4, 2]).all()
        assert data["chunk_factors"][0, 2] > data["chunk_factors"][0, 3]
    else:
        assert torch.equal(weights[0, 2:], torch.ones_like(weights[0, 2:]))
    assert torch.equal(weights[0, 4], torch.ones_like(weights[0, 4]))
    for sign in (1, -1):
        selected = adv.squeeze(-1) * sign > 0
        mean = (weights.mean(-1)[selected] * mass[selected]).sum() / mass[
            selected
        ].sum()
        torch.testing.assert_close(mean, torch.tensor(1.0))
    # Multiplication changes magnitude, never advantage sign; zero remains zero.
    assert torch.equal((adv * weights).sign(), adv.sign().expand_as(weights))


def test_success_weights_identical_in_both_variants_and_legacy_positive_scope():
    args = sample()
    both, _ = compute(*args)
    positive, _ = compute(*args, scope="positive")
    legacy, _ = compute(*args, scope="positive", direction="increasing")
    assert torch.equal(both[:, :2], positive[:, :2])
    assert torch.equal(positive, legacy)
    changed = args[0].clone()
    changed[:, 2:] *= 1e20
    again, _ = compute(changed, *args[1:])
    assert torch.equal(both[:, :2], again[:, :2])


def test_constant_zero_mask_shuffle_tau_and_tricks():
    args = sample()
    constant, _ = compute(torch.ones_like(args[0]), *args[1:])
    assert torch.equal(constant, torch.ones_like(constant))
    zero, _ = compute(*args[:3], torch.zeros_like(args[3]))
    assert torch.equal(zero, torch.ones_like(zero))
    warm, details = compute(*args)
    cold, _ = compute(*args, temperature_local=0.25, temperature_chunk=0.25)
    assert cold.max() > warm.max()
    annealed, _ = compute(*args, alpha_local=0.0, alpha_chunk=0.0)
    assert torch.equal(annealed, torch.ones_like(annealed))
    order = torch.tensor([4, 2, 0, 3, 1])
    permuted, _ = compute(
        args[0][:, order], args[1][:, order], args[2][order], args[3][:, order]
    )
    torch.testing.assert_close(permuted, warm[:, order])
    dropped, mask = apply_chunk_dropout(
        warm,
        details["eligible_mask"],
        {"chunk_dropout": {"probability": 1.0, "seed": 42}},
        runner_step=0,
    )
    assert mask.sum() == 4
    assert torch.equal(dropped, torch.ones_like(dropped))
    args[1][:, 3] = False
    args[0][:, 3] = float("nan")
    masked, _ = compute(*args)
    assert torch.isfinite(masked).all()
    assert torch.equal(masked[:, 3], torch.ones_like(masked[:, 3]))


@pytest.mark.parametrize("scope", ["both", "positive"])
@pytest.mark.parametrize("rank", [0, 1])
def test_actual_actor_freezes_signed_u_and_locks_resume(
    monkeypatch, tmp_path, scope, rank
):
    actor = make_actor(rank, scope, tmp_path, mapping="exp_mean")
    actor.dvac_train_cfg.update(
        direction="advantage_sign",
        signal_source="ugrow_10_5",
        mode="apply",
        application="chunk_clipped_action_advantage",
    )
    inputs = actor.rollout_batch["forward_inputs"]
    inputs["grpo_action_signal"] = inputs.pop("dvac_v_l3")
    fake_all_gather(monkeypatch, actor)
    actor._prepare_dvac_train_step()
    pending = actor._dvac_pending_step
    assert pending["config"]["direction"] == "advantage_sign"
    weights = inputs["dvac_weights"]
    assert torch.isfinite(weights).all() and not weights.requires_grad
    excluded = (
        pending["advantages"].squeeze(-1) <= 0
        if scope == "positive"
        else pending["advantages"].squeeze(-1) == 0
    )
    assert torch.equal(weights[excluded], torch.ones_like(weights[excluded]))
    actor._dvac_pending_step = None
    actor.save_checkpoint(str(tmp_path / "checkpoint"), 7)
    actor.load_checkpoint(str(tmp_path / "checkpoint"))
    actor.dvac_train_cfg["direction"] = "increasing"
    with pytest.raises(ValueError, match="two-level resume configuration mismatch"):
        actor.load_checkpoint(str(tmp_path / "checkpoint"))


def test_legacy_contract_unchanged_and_unknown_direction_rejected():
    legacy = {"scope": "both", "alpha_local": 1.0}
    assert (
        canonicalize_dvac_two_level_contract(dict(legacy, direction="increasing"))
        == legacy
    )
    assert dvac_mapping_contract(direction="advantage_sign") == {
        "direction": "advantage_sign"
    }
    with pytest.raises(ValueError, match="direction"):
        compute(*sample(), direction="unknown")
