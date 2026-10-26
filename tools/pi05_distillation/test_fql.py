# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""CPU contract tests; synthetic data never masquerades as rollout evidence."""

import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from fql_prepare import prepare

from rlinf.algorithms.pi05_distillation.fql import (
    TwinQ,
    actor_objective,
    bellman_target,
    prefix_mask,
    update_target,
)
from rlinf.algorithms.pi05_distillation.transitions import (
    Transitions,
    TransitionWriter,
    build_manifest,
)


def observation(value, batch=2):
    return {
        "main_images": torch.full((batch, 8, 8, 3), value, dtype=torch.uint8),
        "wrist_images": torch.full((batch, 2, 8, 8, 3), value, dtype=torch.uint8),
        "states": torch.full((batch, 14), float(value)),
        "task_descriptions": ["press bell"] * batch,
    }


def test_q_mask_and_action_gradient():
    torch.manual_seed(3)
    q = TwinQ(9, width=16, depth=2).requires_grad_(False)
    a = torch.randn(2, 50, 32, requires_grad=True)
    mask = prefix_mask(torch.tensor([10, 3]))
    f = torch.randn(2, 9)
    out = q(f, a, mask, torch.ones(2))
    out.mean().backward()
    assert a.grad[:, :3, :14].abs().sum() > 0
    assert not a.grad[:, 10:].any() and not a.grad[:, :, 14:].any()
    assert not a.grad[1, 3:].any() and all(p.grad is None for p in q.parameters())


def test_bellman_terminal_and_target_update():
    y = bellman_target(
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([[9.0, 9.0], [9.0, 9.0], [2.0, 4.0]]),
    )
    torch.testing.assert_close(y, torch.tensor([1.0, 0.0, 2.97]))
    q = TwinQ(2, width=8, depth=1)
    target = copy.deepcopy(q).requires_grad_(False)
    before = [p.clone() for p in target.parameters()]
    with torch.no_grad():
        for p in q.parameters():
            p.add_(1)
    update_target(target, q, 0.005)
    for p, b in zip(target.parameters(), before):
        torch.testing.assert_close(p, b + 0.005)


def test_actor_anchor_uses_h50_q_uses_c10():
    torch.manual_seed(2)
    q = TwinQ(5, width=16, depth=1).requires_grad_(False)
    a = torch.randn(2, 50, 32, requires_grad=True)
    teacher = torch.zeros_like(a, requires_grad=False)
    loss, parts = actor_objective(
        a, teacher, q(torch.zeros(2, 5), a, torch.ones(2, 10), torch.ones(2))
    )
    torch.testing.assert_close(
        loss, 10 * a[:, :, :14].square().mean() + parts["q_actor_loss"]
    )
    loss.backward()
    assert a.grad[:, 10:, :14].abs().sum() > 0 and not a.grad[:, :, 14:].any()
    assert teacher.grad is None


def create_episodes(tmp_path):
    config = {
        "task": "click_bell",
        "global_task_id": 7,
        "step_limit": 13,
        "forbidden_actual_seeds": [],
    }
    seeds = [
        {"slot": i, "actual": 100 + i, "requested": 100 + i, "native_retry": False}
        for i in range(2)
    ]
    writer = TransitionWriter(tmp_path, config, seeds, observation(0))
    a = np.zeros((2, 50, 14), np.float32)
    writer.step(
        observation(1),
        a,
        a,
        np.array([1, 1], bool),
        np.array([0, 0]),
        np.array([10, 10]),
        np.array([0, 0], bool),
        np.array([1, 0], bool),
        ([1, 0], [True, False], [False, False]),
        0,
        10,
    )
    writer.step(
        observation(2),
        a,
        a,
        np.array([0, 1], bool),
        np.array([10, 10]),
        np.array([10, 13]),
        np.array([1, 0], bool),
        np.array([1, 0], bool),
        ([1, 0], [True, False], [False, True]),
        1,
        11,
    )
    writer.close()
    return writer


def test_real_transition_contract_synthetic_boundaries(tmp_path):
    writer = create_episodes(tmp_path)
    assert len(writer.rows) == 2
    m = build_manifest([tmp_path], tmp_path / "index.json")
    assert sum(r["success"] for r in m["episodes"]) == 1
    assert sum(r["transitions"] for r in m["episodes"]) == 3
    data = Transitions(tmp_path / "index.json")
    b = data.batch([0, 1, 2])
    assert b["obs"]["states"][:, 0].tolist() == [0.0, 0.0, 1.0]
    assert b["next_obs"]["states"][:, 0].tolist() == [1.0, 1.0, 2.0]
    assert b["reward"].tolist() == [1.0, 0.0, 0.0] and b["bootstrap_mask"].tolist() == [
        0.0,
        1.0,
        0.0,
    ]
    assert b["mask"].sum(1).tolist() == [10.0, 10.0, 3.0]
    np.testing.assert_array_equal(
        data.indices(2, 12, 55), Transitions(tmp_path / "index.json").indices(2, 12, 55)
    )


def test_invalid_partial_excluded_and_seed_overlap_rejected(tmp_path):
    create_episodes(tmp_path)
    partial = tmp_path / "transitions/episode-999.partial.h5"
    with h5py.File(partial, "w") as f:
        f.attrs["complete"] = False
    assert len(build_manifest([tmp_path], tmp_path / "index.json")["episodes"]) == 2
    other = tmp_path / "bad"
    other.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        TransitionWriter(
            other,
            {
                "task": "x",
                "global_task_id": 1,
                "step_limit": 10,
                "forbidden_actual_seeds": [1],
            },
            [{"slot": 0, "actual": 1, "requested": 1, "native_retry": False}],
            observation(0, 1),
        )


def test_native_early_end_is_not_a_failure_label(tmp_path):
    writer = TransitionWriter(
        tmp_path,
        {"task": "x", "global_task_id": 1, "step_limit": 100},
        [{"slot": 0, "actual": 3, "requested": 3, "native_retry": False}],
        observation(0, 1),
    )
    a = np.zeros((1, 50, 14), np.float32)
    with pytest.raises(ValueError, match="Unexpected native end"):
        writer.step(
            observation(1, 1),
            a,
            a,
            np.array([True]),
            np.array([0]),
            np.array([10]),
            np.array([False]),
            np.array([False]),
            ([0], [False], [True]),
            0,
            1,
        )
    writer.close()
    assert not list((tmp_path / "transitions").glob("episode-3.h5"))


def test_prepare_uses_native_seed_schema_and_disjoint_actual_seeds(tmp_path):
    ref = tmp_path / "reference"
    ref.mkdir()
    (ref / "eval-seeds").mkdir()
    (ref / "task-list.json").write_text(
        json.dumps({"tasks": [{"task": "click_bell", "step_limit": 200}]})
    )
    (ref / "prepared.json").write_text(json.dumps({"output": str(tmp_path / "old")}))
    (ref / "eval-seeds/click_bell.json").write_text(
        json.dumps({"click_bell": {"success_seeds": [11, 12]}})
    )
    (ref / "endpoint.json").write_text(
        json.dumps({"model": {"openpi": {}}, "data_manifest": "unused"})
    )
    (ref / "eval-template.json").write_text(json.dumps({"env": {"task_config": {}}}))
    paths = prepare(ref, tmp_path / "new", 3, batches=1)
    c = json.loads(Path(paths[0]).read_text())
    seeds = json.loads(Path(c["env"]["seeds_path"]).read_text())["click_bell"][
        "success_seeds"
    ]
    assert len(seeds) == 8 and not set(seeds) & set(c["forbidden_actual_seeds"])
    assert (
        c["collect_transitions"]
        and c["recording_mode"] == "none"
        and not c["record_signals"]
    )
    fql = json.loads((tmp_path / "new/fql.json").read_text())
    assert not fql["budget_resolved"]
