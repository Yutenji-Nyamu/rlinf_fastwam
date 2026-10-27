# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Native teacher, macro-action and resume preparation contracts (CPU only)."""

import copy
import json

import numpy as np
import pytest
import torch
from single_task_prepare import evaluation_config, training_config
from test_fql import observation

from rlinf.algorithms.pi05_distillation.fql import TwinQ, prefix_mask
from rlinf.algorithms.pi05_distillation.losses import endpoint_loss
from rlinf.algorithms.pi05_distillation.transitions import (
    Transitions,
    TransitionWriter,
    build_manifest,
)
from tools.pi05_policy_eval.protocol import validate_config


def test_single_smoke_retention_cannot_relax_formal_retention(tmp_path):
    from rlinf.algorithms.pi05_distillation.checkpoint_retention import prune_checkpoints

    out = tmp_path / "smoke-endpoint"
    out.mkdir()
    for step in (12, 14):
        for prefix in ("checkpoint", "policy"):
            (out / f"{prefix}-{step:06d}.pt").write_bytes(b"test")
    (out / "checkpoint.json").write_text(json.dumps({
        "step": 14, "path": str(out / "checkpoint-000014.pt"),
        "policy": str(out / "policy-000014.pt"),
    }))
    with pytest.raises(ValueError):
        prune_checkpoints(out, 1, [])
    deleted = prune_checkpoints(out, 1, [], single_task_smoke=True)
    assert len(deleted) == 2 and (out / "checkpoint-000014.pt").exists()
    with pytest.raises(ValueError):
        prune_checkpoints(tmp_path / "formal", 1, [], single_task_smoke=True)


def test_native_n5_endpoint_keeps_same_noise_and_frozen_teacher():
    class Teacher:
        def endpoint(self, noise, context, steps):
            assert steps == 5
            self.noise = noise
            return noise + 0.5

    teacher = Teacher()
    p = torch.tensor(0.1, requires_grad=True)
    z = torch.randn(2, 50, 32)
    loss, _ = endpoint_loss(
        lambda x, t, c: p.expand_as(x), teacher, z, None, teacher_steps=5
    )
    loss.backward()
    assert teacher.noise is z and p.grad > 0
    with pytest.raises(ValueError):
        endpoint_loss(None, None, z, None, teacher_steps=0)


def test_c50_real_transition_mask_terminal_and_coordinate_guard(tmp_path):
    config = {
        "task": "adjust_bottle",
        "global_task_id": 0,
        "step_limit": 200,
        "execute_chunk": 50,
        "model": {"model_path": "rlinf-native"},
    }
    seeds = [{"slot": 0, "actual": 123, "requested": 123, "native_retry": False}]
    writer = TransitionWriter(tmp_path, config, seeds, observation(0, 1))
    a = np.zeros((1, 50, 14), np.float32)
    for i in range(4):
        writer.step(
            observation(i + 1, 1),
            a,
            a,
            np.array([True]),
            np.array([50 * i]),
            np.array([50 * (i + 1)]),
            np.array([False]),
            np.array([False]),
            ([0], [False], [i == 3]),
            i,
            100 + i,
        )
    writer.close()
    build_manifest([tmp_path], tmp_path / "manifest.json")
    data = Transitions(tmp_path / "manifest.json")
    data.validate_coordinates("rlinf-native", 50, 200)
    with pytest.raises(ValueError, match="coordinates"):
        data.validate_coordinates("sidney", 10, 400)
    b = data.batch([0, 3])
    assert b["action"].shape == (2, 50, 14)
    assert b["bootstrap_mask"].tolist() == [1, 0]
    assert b["remaining_after"].tolist() == [150, 0]
    q = TwinQ(4, chunk=50, width=8, depth=1)
    pred = q(
        torch.zeros(2, 4),
        b["action"],
        prefix_mask(b["remaining_before"], 50),
        torch.ones(2),
    )
    assert pred.shape == (2, 2)


def test_preparation_preserves_native_protocol_and_method_settings(tmp_path):
    model = {
        "model_path": "native",
        "num_steps": 5,
        "num_action_chunks": 50,
        "openpi": {
            "config_name": "pi05_aloha_robotwin",
            "action_chunk": 50,
            "num_steps": 5,
        },
    }
    manifest = tmp_path / "data.json"
    manifest.write_text(json.dumps({"total_windows": 4738}))
    ref = {"lr": 1e-5, "aux_lr": 2e-4, "epochs": 20, "auxiliary_updates_per_student": 5}
    original = copy.deepcopy(ref)
    c = training_config(ref, model, manifest, "onedp", 192)
    assert (
        ref == original
        and c["batch_size"] == 384
        and c["auxiliary_updates_per_student"] == 5
    )
    template = {
        "task": "adjust_bottle",
        "step_limit": 200,
        "policy": "weak",
        "model": model,
        "env": {
            "task_config": {
                "task_name": "adjust_bottle",
                "step_lim": 200,
                "domain_randomization": {"random_background": False},
            },
            "video_cfg": {},
        },
    }
    for method in ("cp", "endpoint", "onedp", "mfd", "fql_fixed_teacher"):
        e = evaluation_config(
            template,
            tmp_path / method,
            tmp_path / "seeds",
            7,
            64,
            weights="policy.pt",
            method=method,
        )
        validate_config(e)
        assert e["model"]["openpi"]["distill_target_time"] == (method == "cp")
        assert e["step_limit"] == 200 and e["execute_chunk"] == 50
        assert not e["collect_transitions"]
