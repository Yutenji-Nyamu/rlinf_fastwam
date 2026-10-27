# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Failure continuation and exposure/seed boundaries for the formal queue."""

import pytest
from single_formal_plan import METHODS, batch_indices, formal_config, run_stages


def test_balanced_exposure_and_preserved_method_parameters():
    samples = []
    for method in METHODS[:-1]:
        reference = {
            "method": method,
            "lr": 1e-5,
            "aux_lr": 2e-4,
            "warmup_steps": 100,
            "model": {"native": True},
        }
        c = formal_config(reference)
        steps = c["checkpoint_keep_steps"]
        assert len(steps) == 12 and steps == sorted(set(steps))
        assert c["model"] == reference["model"] and c["lr"] == reference["lr"]
        assert "epochs" not in reference
        assert steps[0] < steps[1] < steps[2] < steps[-1]
        assert c["checkpoint_keep_last"] == 2
        samples.append(steps[-1] * c["batch_size"])
    assert max(samples) / min(samples) < 1.05


def test_small_fql_replay_does_not_become_twenty_smoke_updates():
    c = formal_config(
        {"method": "fql_fixed_teacher", "warmup_steps": 1}, transitions=400
    )
    assert c["checkpoint_keep_steps"][-1] == 1100
    assert c["warmup_steps"] == 55 and c["budget_resolved"]
    assert not formal_config({"method": "fql_fixed_teacher"})["budget_resolved"]


def test_two_cards_cover_each_heldout_batch_once():
    left, right = batch_indices(0), batch_indices(1)
    assert not set(left) & set(right)
    assert sorted(left + right) == list(range(4))
    with pytest.raises(ValueError):
        batch_indices(2)


def test_method_failure_advances_but_cancellation_does_not():
    visited, failures = [], []

    def run(method):
        visited.append(method)
        if method == "cp":
            raise RuntimeError("OOM")
        return method

    result = run_stages(
        ["endpoint", "cp", "mfd"], run, lambda method, exc: failures.append(method)
    )
    assert visited == ["endpoint", "cp", "mfd"] and failures == ["cp"]
    assert result["mfd"]["status"] == "complete"

    def cancelled(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_stages(["endpoint", "mfd"], cancelled, lambda *_: None)
