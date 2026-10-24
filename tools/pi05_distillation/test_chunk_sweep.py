# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Protect paired noise, native budgets and existing training evaluation."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from multitask_eval import make_config

spec = importlib.util.spec_from_file_location(
    "protocol", Path(__file__).parents[1] / "pi05_policy_eval/protocol.py"
)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def config(n, c, batch=0):
    template = {
        "policy": "weak",
        "noise_seed": 721000,
        "execute_chunk": 50,
        "model": {"num_action_chunks": 50, "openpi": {"action_chunk": 50}},
        "env": {"task_config": {}, "video_cfg": {}},
    }
    return make_config(
        template,
        {"task": "open_microwave", "step_limit": 1500},
        6,
        Path("/tmp/result"),
        Path("/tmp/seeds.json"),
        "test",
        steps=n,
        options={
            "episodes_per_task": 8,
            "recording_mode": "none",
            "record_decisions": False,
            "chunk_sweep": True,
            "execute_chunk": c,
            "seed_batch": batch,
        },
    )


def test_paired_noise_and_independent_batches():
    for c in (10, 20, 30, 40):
        a, b = config(1, c), config(10, c)
        protocol.validate_config(a)
        protocol.validate_config(b)
        assert a["noise_seed"] == b["noise_seed"] != config(1, c, 1)["noise_seed"]
        assert a["env"] == b["env"]
        assert a["step_limit"] == a["env"]["task_config"]["step_lim"] == 1500


def test_old_distillation_protocol_stays_c50_and_sweep_is_raw_only():
    c = config(1, 20)
    old = copy.deepcopy(c)
    old["suite"] = "pi05_distillation_multitask"
    with pytest.raises(AssertionError):
        protocol.validate_config(old)
    old["execute_chunk"] = 50
    protocol.validate_config(old)
    c["distill_weights"] = "/tmp/student.pt"
    with pytest.raises(AssertionError):
        protocol.validate_config(c)


def test_prefix_respects_native_remaining_budget_and_finished_slots():
    lengths = protocol.execution_lengths(
        np.zeros(3, bool),
        np.array([1, 1, 0], bool),
        np.array([0, 1497, 20]),
        chunk=20,
        limit=1500,
    )
    assert lengths.tolist() == [20, 3, 0]
