# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Physical v2 targets, gradient ownership, legacy compatibility and resume."""

import copy

import pytest
import torch

from rlinf.algorithms.pi05_distillation import mfd
from rlinf.algorithms.pi05_distillation.losses import cp_distance, cp_loss, fake_loss
from rlinf.algorithms.pi05_distillation.resume_extension import validate_resume
from rlinf.algorithms.pi05_distillation.variants import (
    V2_METHODS,
    resolve_method,
    smoke_config,
)


class Coordinates(torch.nn.Module):
    def __init__(self, physical=0.2, padding=5.0, trainable=True):
        super().__init__()
        self.values = torch.nn.Parameter(
            torch.tensor([physical] * 14 + [padding] * 18), requires_grad=trainable
        )

    def forward(self, x, t, context, stop=None):
        return self.values.expand_as(x)


def test_cp_final_distance_ignores_padding_and_uses_valid_dimension_delta():
    pred = torch.full((2, 50, 32), 0.1, requires_grad=True)
    target = torch.zeros_like(pred)
    target[..., 14:] = 100
    actual = cp_distance(pred, target, action_dim=14)
    reference = cp_distance(pred[..., :14], target[..., :14])
    torch.testing.assert_close(actual, reference)
    actual.backward()
    assert pred.grad[..., :14].abs().sum() > 0
    assert not pred.grad[..., 14:].any()
    assert (50 * 14) ** 0.5 * 0.00054 != (50 * 32) ** 0.5 * 0.00054


def test_cp_v2_changes_both_targets_but_freezes_teacher_and_ema():
    actions = torch.zeros(8, 50, 32)
    student, teacher, ema = (
        Coordinates(),
        Coordinates(0.4, 0.4, False),
        Coordinates(0.3, 0.3, False),
    )
    torch.manual_seed(91)
    loss, parts = cp_loss(student, teacher, ema, actions, None, action_dim=14)
    loss.backward()
    assert student.values.grad[:14].abs().sum() > 0
    assert not student.values.grad[14:].any()  # Coordinate-independent toy only.
    assert teacher.values.grad is None and ema.values.grad is None
    assert float(parts["cp_distance_delta"]) == pytest.approx(
        (50 * 14) ** 0.5 * 0.00054
    )
    with torch.no_grad():
        student.values[14:] = 1000
    torch.manual_seed(91)
    other, other_parts = cp_loss(student, teacher, ema, actions, None, action_dim=14)
    torch.testing.assert_close(loss, other)
    torch.testing.assert_close(parts["fm"], other_parts["fm"])


def test_cp_must_keep_indirect_padding_derivative_through_ema():
    class CoupledEMA:
        def __call__(self, x, t, context, stop=None):
            result = torch.zeros_like(x)
            result[..., 0] = x[..., 14]
            return result

    student = Coordinates()
    torch.manual_seed(7)
    _, parts = cp_loss(
        student,
        Coordinates(0.4, 0.4, False),
        CoupledEMA(),
        torch.zeros(8, 50, 32),
        None,
        action_dim=14,
    )
    gradient = torch.autograd.grad(parts["consistency"], student.values)[0]
    assert gradient[14].abs() > 0


@pytest.mark.parametrize("method", ["dmd", "mfd"])
def test_auxiliary_v2_selects_physical_targets_and_detaches_student(method):
    generated = torch.zeros(3, 50, 32, requires_grad=True)
    fake = Coordinates()

    def compute(dim):
        torch.manual_seed(5)
        if method == "dmd":
            return fake_loss(
                fake, generated, None, return_parts=True, loss_action_dim=dim
            )
        return mfd.auxiliary_loss(fake, generated, None, loss_action_dim=dim)

    legacy, legacy_parts = compute(None)
    valid, parts = compute(14)
    torch.testing.assert_close(
        legacy.detach(),
        (legacy_parts["fake_physical_mse"] * 14 + legacy_parts["fake_padding_mse"] * 18)
        / 32,
    )
    torch.testing.assert_close(valid.detach(), parts["fake_physical_mse"])
    assert float(valid) < float(legacy)
    valid.backward()
    assert fake.values.grad[:14].abs().sum() > 0
    assert not fake.values.grad[14:].any()
    assert generated.grad is None


def test_mfd_diagnostics_observe_full_path_without_changing_integration():
    x = torch.ones(2, 50, 32)
    start, end = torch.ones(2), torch.zeros(2)
    model = Coordinates(0.2, 2.0, False)
    trace = {}
    plain = mfd.average_velocity(model, x, start, end, None, steps=4)
    observed = mfd.average_velocity(
        model, x, start, end, None, steps=4, diagnostics=trace
    )
    torch.testing.assert_close(plain, observed)
    assert float(trace["padding_position_abs_max"]) == 1
    assert float(trace["padding_velocity_abs_max"]) == 2
    assert (
        float(trace["padding_start_rms"]) == 1 and float(trace["padding_end_rms"]) == 1
    )
    assert not observed.requires_grad


@pytest.mark.parametrize("variant", V2_METHODS)
def test_v2_configuration_is_explicit_isolated_and_resumable(variant):
    base = {
        "method": V2_METHODS[variant],
        "model": {"action_dim": 14, "openpi": {"action_env_dim": 14}},
        "lr": 1e-5,
        "aux_lr": 2e-4,
        "epochs": 100,
        "batch_size": 384,
        "microbatch_per_rank": 192,
        "seed": 99,
    }
    before = copy.deepcopy(base)
    config = smoke_config(base, variant)
    assert base == before and resolve_method(base) == (base["method"], None)
    assert resolve_method(config) == (base["method"], 14)
    for key in ("lr", "aux_lr", "epochs", "batch_size", "microbatch_per_rank", "seed"):
        assert config[key] == base[key]
    if variant == "dmd_v2":
        assert config["cuda_allocator_conf"] == "expandable_segments:True"
    checkpoint = {"config": config, "windows": 4738, "world_size": 2}
    assert validate_resume(checkpoint, config, 4738, 2) is None
    changed = dict(config, loss_coordinate_scope="full")
    with pytest.raises(ValueError):
        resolve_method(changed)
    with pytest.raises(ValueError):
        validate_resume(checkpoint, changed, 4738, 2, extend_completed=True)
    with pytest.raises(ValueError):
        resolve_method(dict(config, onedp_loss_action_dim=32))


def test_legacy_scope_cannot_silently_change():
    with pytest.raises(ValueError):
        resolve_method({"method": "cp", "loss_coordinate_scope": "physical"})
