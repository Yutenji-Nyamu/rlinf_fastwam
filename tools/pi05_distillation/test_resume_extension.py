# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Protect full-state resume from accidentally accepting a different experiment."""

import copy

import pytest

from rlinf.algorithms.pi05_distillation.resume_extension import validate_resume


def configs(method="endpoint"):
    batch = 448 if method == "endpoint" else 160
    per_epoch = 11 if method == "endpoint" else 30
    old = {
        "method": method,
        "epochs": 100,
        "batch_size": batch,
        "warmup_steps": per_epoch * 5,
        "lr": 2.5e-5,
        "model": {"base": "native"},
        "data_manifest": "unchanged",
        "checkpoint_steps": 100 * per_epoch + 1,
        "checkpoint_keep_steps": [100 * per_epoch],
    }
    new = copy.deepcopy(old)
    new.update(
        epochs=200,
        checkpoint_steps=200 * per_epoch + 1,
        checkpoint_keep_steps=[e * per_epoch for e in [120, 140, 160, 180, 200]],
    )
    ck = {"config": old, "windows": 4738, "world_size": 2, "step": 100 * per_epoch}
    return ck, new


@pytest.mark.parametrize("method", ["endpoint", "cp"])
def test_opt_in_completed_extension_then_strict_resume(method):
    ck, new = configs(method)
    with pytest.raises(ValueError):
        validate_resume(ck, new, 4738, 2)
    audit = validate_resume(ck, new, 4738, 2, extend_completed=True)
    assert audit["to_step"] == ck["step"] * 2
    assert 1.2e-5 < audit["first_extension_lr"] < 1.4e-5
    # The next segment is an ordinary exact-config resume, without the opt-in.
    assert validate_resume(dict(ck, config=new), new, 4738, 2) is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("lr", 1e-3),
        ("model", {"base": "different"}),
        ("batch_size", 64),
        ("data_manifest", "different"),
        ("warmup_steps", 110),
    ],
)
def test_extension_rejects_scientific_changes(key, value):
    ck, new = configs()
    new[key] = value
    with pytest.raises(ValueError):
        validate_resume(ck, new, 4738, 2, extend_completed=True)


def test_rejects_incomplete_checkpoint_bad_points_and_data_change():
    ck, new = configs()
    for checkpoint, config, windows, world in [
        (dict(ck, step=880), new, 4738, 2),
        (ck, dict(new, checkpoint_keep_steps=[1100, 2200]), 4738, 2),
        (ck, new, 4739, 2),
        (ck, new, 4738, 1),
    ]:
        with pytest.raises(ValueError):
            validate_resume(checkpoint, config, windows, world, extend_completed=True)


def dmd_configs():
    old = {
        "method": "onedp",
        "epochs": 100,
        "batch_size": 384,
        "lr": 1e-5,
        "aux_lr": 2e-4,
        "auxiliary_updates_per_student": 5,
        "onedp_loss_action_dim": 14,
        "checkpoint_steps": 1301,
        "checkpoint_keep_steps": [1300],
    }
    new = dict(
        old,
        epochs=200,
        checkpoint_steps=2601,
        checkpoint_keep_steps=[1560, 1820, 2080, 2340, 2600],
    )
    return {"config": old, "windows": 4738, "world_size": 2, "step": 1300}, new


def test_dmd_extension_keeps_fixed_rates_and_auxiliary_ratio():
    ck, new = dmd_configs()
    audit = validate_resume(ck, new, 4738, 2, extend_completed=True)
    assert (audit["from_step"], audit["to_step"]) == (1300, 2600)
    assert audit["first_extension_lr"] == 1e-5
    assert audit["aux_lr"] == 2e-4
    assert audit["auxiliary_updates_per_student"] == 5
    assert validate_resume(dict(ck, config=new), new, 4738, 2) is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("aux_lr", 1e-3),
        ("auxiliary_updates_per_student", 1),
        ("onedp_loss_action_dim", 32),
    ],
)
def test_dmd_extension_rejects_method_changes(key, value):
    ck, new = dmd_configs()
    new[key] = value
    with pytest.raises(ValueError):
        validate_resume(ck, new, 4738, 2, extend_completed=True)


def test_mfd_extension_keeps_full_fixed_rate_method():
    ck, new = dmd_configs()
    for c in (ck["config"], new):
        c.update(method="mfd", mfd_step_size=0.025, mfd_loss_action_dim=14)
    audit = validate_resume(ck, new, 4738, 2, extend_completed=True)
    assert audit["to_step"] == 2600 and audit["first_extension_lr"] == 1e-5
    assert audit["aux_lr"] == 2e-4 and audit["auxiliary_updates_per_student"] == 5
    for key, value in [("mfd_step_size", 0.1), ("mfd_loss_action_dim", 32)]:
        with pytest.raises(ValueError):
            validate_resume(
                ck, dict(new, **{key: value}), 4738, 2, extend_completed=True
            )
