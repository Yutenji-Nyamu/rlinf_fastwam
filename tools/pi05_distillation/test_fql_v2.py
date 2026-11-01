# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""FQL v2 formula, DDP scaling and unchanged old-default contracts."""

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fql_v2_plan import formal_config

from rlinf.algorithms.pi05_distillation.fql import (
    actor_objective,
    bellman_target,
    q_normalization,
)


def test_min_target_preserves_terminal_reward():
    reward = torch.tensor([1.0, 0.0, 0.0])
    boot = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([[100.0, 2.0], [8.0, 9.0], [2.0, 4.0]])
    torch.testing.assert_close(
        bellman_target(reward, boot, values), torch.tensor([1.0, 0.0, 2.97])
    )
    torch.testing.assert_close(
        bellman_target(reward, boot, values, q_agg="min"),
        torch.tensor([1.0, 0.0, 1.98]),
    )


def test_normalization_uses_per_sample_head_mean_and_detached_scale():
    values = torch.tensor([[-3.0, -1.0], [1.0, 5.0]], requires_grad=True)
    actions = torch.zeros(2, 50, 32, requires_grad=True)
    loss, parts = actor_objective(
        actions, torch.zeros_like(actions), values, 30, normalize_q_loss=True
    )
    torch.testing.assert_close(parts["q_global_mean_abs"], torch.tensor(2.5))
    torch.testing.assert_close(parts["q_normalization_scale"], torch.tensor(0.4))
    assert not parts["q_normalization_scale"].requires_grad
    loss.backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, -0.1))
    assert not actions.grad[..., 14:].any()


def test_zero_q_floor_and_legacy_objective():
    values = torch.zeros(2, 2, requires_grad=True)
    scale, mean_abs = q_normalization(values)
    assert scale.item() == 1e6 and mean_abs.item() == 0
    pred = torch.ones(2, 50, 32, requires_grad=True)
    target = torch.zeros_like(pred)
    loss, parts = actor_objective(pred, target, values, 30)
    torch.testing.assert_close(loss, torch.tensor(30.0))
    assert set(parts) == {"endpoint_mse", "q_actor_loss"}
    normalized, parts = actor_objective(pred, target, values, 30, normalize_q_loss=True)
    normalized.backward()
    assert torch.isfinite(values.grad).all()
    assert parts["q_normalization_floor_hit"].item() == 1


def _distributed_scale(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    values = torch.tensor(
        [[-3.0, -1.0]] if rank == 0 else [[1.0, 5.0]], requires_grad=True
    )
    pred = torch.zeros(1, 50, 32)
    loss, parts = actor_objective(pred, pred, values, normalize_q_loss=True)
    loss.backward()
    Path(output, f"rank{rank}.json").write_text(
        json.dumps(
            {
                "scale": parts["q_normalization_scale"].item(),
                "gradient": values.grad.tolist(),
            }
        )
    )
    dist.destroy_process_group()


def test_two_ranks_equal_the_unsharded_objective(tmp_path):
    mp.spawn(
        _distributed_scale,
        args=(f"file://{tmp_path}/rendezvous", str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        row = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert row["scale"] == pytest.approx(0.4)
        # DDP averages rank parameter gradients: /2 matches the global batch.
        torch.testing.assert_close(
            torch.tensor(row["gradient"]) / 2, torch.tensor([[-0.1, -0.1]])
        )


def test_fresh_200_budget_reuses_actual_warmup_and_17_evaluations():
    ref = {
        "method": "fql_fixed_teacher",
        "experiment_name": "fql-fix",
        "alpha": 30,
        "q_diagnostics": True,
        "warmup_steps": 55,
        "lr": 2.5e-5,
        "q_lr": 3e-4,
    }
    c = formal_config(ref, 405)
    assert c["epochs"] == 2200 and c["budget_epochs"] == 200
    assert c["microbatch_per_rank"] == 224 and c["batch_size"] == 448
    assert c["warmup_steps"] == 55 and c["q_agg"] == "min"
    assert c["normalize_q_loss"] and c["checkpoint_keep_last"] == 1
    assert c["evaluation_steps"] == [
        14,
        28,
        55,
        110,
        165,
        220,
        330,
        440,
        550,
        715,
        880,
        1100,
        1320,
        1540,
        1760,
        1980,
        2200,
    ]
    assert ref["warmup_steps"] == 55 and "normalize_q_loss" not in ref
