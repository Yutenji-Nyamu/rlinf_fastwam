# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Failure continuation and exposure/seed boundaries for the formal queue."""

import pytest
from single_formal_plan import (
    METHODS,
    batch_indices,
    formal_config,
    fql_fix_formal_config,
    inference_method,
    planned_methods,
    run_stages,
    validate_post_eval_retention,
    v2_formal_config,
    validate_reused_fql_replay,
)


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


def test_pending_queue_edit_does_not_restart_current_or_repeat_stages():
    plan = {"methods": ["endpoint", "fql_fixed_teacher"]}
    stages = planned_methods(lambda: plan)
    assert next(stages) == "endpoint"
    # Move pending FQL elsewhere and append CP; the current method is untouched.
    plan["methods"] = ["endpoint", "cp", "cp"]
    assert list(stages) == ["cp"]


@pytest.mark.parametrize(
    "variant,method,total,micro",
    [
        ("cp_v2", "cp", 6000, 80),
        ("dmd_v2", "onedp", 2600, 192),
        ("mfd_v2", "mfd", 2600, 192),
    ],
)
def test_v2_keeps_all_evaluations_separate_from_retention(
    variant, method, total, micro
):
    reference = {
        "method": variant,
        "loss_coordinate_scope": "physical",
        "lr": 1e-5,
        "aux_lr": 2e-4,
        "warmup_steps": 1,
        "auxiliary_updates_per_student": 5,
    }
    c = v2_formal_config(reference)
    old = formal_config(dict(reference, method=method))
    assert c["evaluation_steps"][:12] == old["checkpoint_keep_steps"]
    assert c["evaluation_steps"][12:] == [
        total * e // 200 for e in (120, 140, 160, 180, 200)
    ]
    assert len(c["evaluation_steps"]) == 17 and c["epochs"] == 200
    assert c["batch_size"] == micro * 2 and c["microbatch_per_rank"] == micro
    assert c["checkpoint_keep_last"] is None and c["checkpoint_keep_steps"] == []
    assert c["warmup_steps"] == old["warmup_steps"]
    assert inference_method(variant) == method
    assert (c["lr"], c["aux_lr"], c["auxiliary_updates_per_student"]) == (1e-5, 2e-4, 5)


def test_extension_retention_preserves_prior_complete_run(tmp_path):
    output = tmp_path / "new" / "endpoint"
    c = dict(
        retention_mode="latest2_best",
        checkpoint_keep_last=None,
        checkpoint_keep_steps=[],
    )
    plan = dict(
        retention_scope="new_extension_only",
        initial_receipts={
            "endpoint": {"path": str(tmp_path / "old" / "checkpoint-002200.pt")}
        },
    )
    validate_post_eval_retention("endpoint", c, plan, output)
    with pytest.raises(ValueError):
        validate_post_eval_retention("endpoint", c, {}, output)
    plan["initial_receipts"]["endpoint"]["path"] = str(output / "checkpoint-002200.pt")
    with pytest.raises(ValueError):
        validate_post_eval_retention("endpoint", c, plan, output)


def test_fql_fix_200_reuses_formal_replay_and_keeps_dense_early_curve():
    reference = dict(method="fql_fixed_teacher", q_diagnostics=True, alpha=30,
                     warmup_steps=55, lr=2.5e-5, q_lr=3e-4,
                     experiment_name="fql-fix")
    c = fql_fix_formal_config(reference, 405)
    assert c["epochs"] == 2200 and c["evaluation_steps"][-1] == 2200
    assert c["evaluation_steps"][:4] == [14, 28, 55, 110]
    assert len(c["evaluation_steps"]) == 17 and c["warmup_steps"] == 55
    assert (c["alpha"], c["lr"], c["q_lr"]) == (30, 2.5e-5, 3e-4)
    validate_post_eval_retention("fql_fixed_teacher", c, {"fresh_fql_fix": True}, "/new/fix")
    with pytest.raises(ValueError):
        validate_post_eval_retention("fql_fixed_teacher", c, {}, "/new/fix")
    manifest = {"episodes": [dict(actual_seed=i, success=bool(i % 2), task="adjust_bottle", transitions=3) for i in range(128)]}
    assert validate_reused_fql_replay(manifest, [1000])["transitions"] == 384
    with pytest.raises(ValueError):
        validate_reused_fql_replay(manifest, [1000, 5])
