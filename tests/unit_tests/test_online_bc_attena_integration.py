# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Replay and actual SFT-path checks; execute in the server project environment."""

import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.online_bc import SuccessReplay, masked_fm_loss


def _record(index):
    return {
        "action": torch.full((700,), float(index), dtype=torch.float64),
        "observation/state": torch.arange(14, dtype=torch.float64),
        "action_valid_mask": torch.ones(50, 14, dtype=torch.bool),
        "query_idx": torch.tensor(index),
        "episode_id": torch.tensor([0, 0]),
    }


def _annotate(record):
    record["attena_fk_motion"] = torch.full(
        (50, 2, 2), float(record["query_idx"]), dtype=torch.float32
    )
    record["attena_fk_geometry_id"] = torch.arange(32, dtype=torch.uint8)
    return record


def _assert_batch_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_unit_weights_match_clean_loss_gradient_and_detach():
    target = torch.linspace(-0.7, 1.2, 2 * 50 * 14, dtype=torch.float64).reshape(2, 50, 14)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 35:] = False
    scale = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    errors = (target - scale).square()
    clean = ((errors * mask).sum((1, 2)) / mask.sum((1, 2))).mean()
    clean_grad = torch.autograd.grad(clean, scale, retain_graph=True)[0]
    weights = torch.ones(2, 50, dtype=torch.float64, requires_grad=True)
    weighted = masked_fm_loss(errors, mask, weights)
    scale_grad, weight_grad = torch.autograd.grad(
        weighted, (scale, weights), allow_unused=True
    )
    torch.testing.assert_close(weighted, clean, rtol=0, atol=0)
    torch.testing.assert_close(scale_grad, clean_grad, rtol=0, atol=0)
    assert weight_grad is None


def test_weighted_loss_preserves_valid_count_denominator():
    errors = torch.tensor([[[2.0, 4.0], [10.0, 20.0]], [[3.0, 5.0], [7.0, 9.0]]])
    mask = torch.tensor([[[1, 1], [0, 0]], [[1, 1], [1, 1]]], dtype=torch.bool)
    weights = torch.tensor([[2.0, 0.5], [0.5, 2.0]])
    # Per-query values: 6; ((3+5)*0.5+(7+9)*2)/4 = 9.
    assert masked_fm_loss(errors, mask, weights).item() == pytest.approx(7.5)


@pytest.mark.parametrize("damage", ["shape", "nan", "inf", "negative"])
def test_bad_action_weights_fail(damage):
    weights = torch.ones(2, 50)
    if damage == "shape":
        weights = weights.unsqueeze(-1)
    else:
        weights[0, 0] = {"nan": float("nan"), "inf": float("inf"), "negative": -0.1}[damage]
    with pytest.raises(ValueError, match="Action weights"):
        masked_fm_loss(torch.ones(2, 50, 14), torch.ones(2, 50, 14), weights)


def test_disabled_replay_preserves_records_and_sampling(tmp_path):
    episodes = [[_record(0), _record(1)], [_record(2)]]
    replay = SuccessReplay(42, str(tmp_path / "default"))
    explicit = SuccessReplay(42, str(tmp_path / "explicit"), prepare_record=None)
    replay.add_episodes(episodes)
    explicit.add_episodes(episodes)
    assert replay.records[0] is episodes[0][0]
    assert all(not any(key.startswith("attena_") for key in row) for row in replay.records)
    assert replay.get_stats() == {"success_episodes": 2, "query_records": 3}
    indices = torch.randint(3, (32,), generator=torch.Generator().manual_seed(42))
    actual = replay.sample(32)["forward_inputs"]
    torch.testing.assert_close(actual["query_idx"], indices)
    _assert_batch_equal(actual, explicit.sample(32)["forward_inputs"])


def test_callback_runs_only_after_length_filter_and_before_archive(tmp_path):
    called = []

    def prepare(record):
        called.append(int(record["query_idx"]))
        return _annotate(record)

    episodes = [[_record(0), _record(1)], [_record(i) for i in range(10, 14)]]
    replay = SuccessReplay(7, str(tmp_path / "pool"), 3, prepare_record=prepare)
    replay.add_episodes(episodes)
    assert called == [0, 1]
    assert replay.get_stats() == {
        "success_episodes": 1, "query_records": 2, "filtered_success_episodes": 1
    }
    assert all("attena_fk_motion" not in row for episode in episodes for row in episode)
    saved = torch.load(tmp_path / "pool/batch_000000.pt", weights_only=True)
    assert saved[0][0]["attena_fk_motion"].shape == (50, 2, 2)
    batch = replay.sample(8)["forward_inputs"]
    assert batch["attena_fk_motion"].shape == (8, 50, 2, 2)
    assert batch["attena_fk_geometry_id"].shape == (8, 32)
    assert batch["attena_fk_geometry_id"].dtype == torch.uint8


def test_all_filtered_does_not_compute_or_create_archive(tmp_path):
    replay = SuccessReplay(
        7, str(tmp_path / "pool"), 1,
        prepare_record=lambda record: pytest.fail("filtered records must not reach FK"),
    )
    replay.add_episodes([[_record(0), _record(1)]])
    assert replay.filtered_success_episodes == 1
    assert replay.episodes == replay.archive_id == len(replay) == 0
    assert not (tmp_path / "pool").exists()


def test_callback_failure_preserves_pool_counters_rng_and_archive(tmp_path):
    def prepare(record):
        if int(record["query_idx"]) == 99:
            raise ValueError("FK query 99 failed")
        return _annotate(record)

    replay = SuccessReplay(42, str(tmp_path / "pool"), 3, prepare_record=prepare)
    replay.add_episodes([[_record(0)]])
    before_stats = replay.get_stats()
    before_rng = replay.rng.get_state().clone()
    before_record = replay.records[0]
    before_archive = (tmp_path / "pool/batch_000000.pt").read_bytes()
    rejected = [_record(i) for i in range(4)]
    pending = [[_record(1)], [_record(99)], rejected]
    with pytest.raises(ValueError, match="FK query 99 failed"):
        replay.add_episodes(pending)
    assert replay.get_stats() == before_stats
    assert replay.archive_id == 1
    assert len(replay.records) == 1 and replay.records[0] is before_record
    torch.testing.assert_close(replay.rng.get_state(), before_rng)
    assert (tmp_path / "pool/batch_000000.pt").read_bytes() == before_archive
    assert not (tmp_path / "pool/batch_000001.pt").exists()
    assert all("attena_fk_motion" not in row for episode in pending for row in episode)


@pytest.mark.parametrize("prepared", [None, {}, {"bad": "not a tensor"}])
def test_bad_callback_result_is_rejected_before_admission(tmp_path, prepared):
    replay = SuccessReplay(
        7, str(tmp_path / "pool"), prepare_record=lambda record: prepared
    )
    with pytest.raises(ValueError, match="tensor record"):
        replay.add_episodes([[_record(0)]])
    assert len(replay) == replay.episodes == replay.archive_id == 0
    assert not (tmp_path / "pool").exists()


def test_checkpoint_preserves_features_rng_and_live_callback(tmp_path):
    original = SuccessReplay(17, str(tmp_path / "pool"), 3, prepare_record=_annotate)
    original.add_episodes([[_record(0), _record(1)], [_record(2)]])
    original.sample(11)
    original.save_checkpoint(tmp_path / "checkpoint")
    expected = original.sample(32)["forward_inputs"]
    restored = SuccessReplay(99, str(tmp_path / "newpool"), 3, prepare_record=_annotate)
    restored.load_checkpoint(tmp_path / "checkpoint")
    assert restored.prepare_record is _annotate
    assert restored.get_stats() == original.get_stats()
    _assert_batch_equal(restored.sample(32)["forward_inputs"], expected)
    restored.add_episodes([[_record(3)]])
    assert restored.records[-1]["attena_fk_motion"].shape == (50, 2, 2)
    assert restored.archive_id == 2
    assert (tmp_path / "newpool/batch_000001.pt").is_file()


def test_archive_collision_does_not_commit_pool_or_filter_count(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    existing = pool / "batch_000000.pt"
    existing.write_bytes(b"existing archive")
    replay = SuccessReplay(7, str(pool), 1, prepare_record=_annotate)
    with pytest.raises(FileExistsError):
        replay.add_episodes([[_record(0)], [_record(1), _record(2)]])
    assert replay.get_stats() == {
        "success_episodes": 0, "query_records": 0, "filtered_success_episodes": 0
    }
    assert replay.archive_id == 0
    assert existing.read_bytes() == b"existing archive"


def _load_actual_sft_method():
    source = (
        Path(inspect.getfile(SuccessReplay)).resolve().parents[2]
        / "rlinf/models/embodiment/openpi/openpi_action_model.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "sft_forward"
    )
    method = copy.deepcopy(method)
    method.decorator_list = []
    wrapper = ast.ClassDef(
        name="ActualSFT", bases=[ast.Name(id="_LossBase", ctx=ast.Load())],
        keywords=[], body=[method], decorator_list=[],
    )

    class LossBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25))
            self.config = SimpleNamespace(use_rlt=False, action_chunk=50, action_env_dim=14)

        def forward(self, observation, actions):
            self.last_actions = actions.detach().clone()
            return (actions - self.scale).square()

    def tree_map(function, value):
        if isinstance(value, dict):
            return {key: tree_map(function, item) for key, item in value.items()}
        return function(value)

    namespace = {
        "torch": torch, "_LossBase": LossBase, "tree_map": tree_map,
        "register_pytree_dataclasses": lambda observation: None,
    }
    module = ast.Module(body=[wrapper], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["ActualSFT"]


@pytest.mark.parametrize("weighted", [False, True])
def test_actual_sft_routes_weights_after_crop_without_changing_labels(weighted):
    model = _load_actual_sft_method()()
    actions = torch.linspace(-1.0, 1.0, 2 * 50 * 32).reshape(2, 50, 32)
    actions[..., 14:] = 1000  # Padded dimensions must contribute no BC loss.
    mask = torch.ones(2, 50, 14, dtype=torch.bool)
    mask[0, 30:] = False
    weights = torch.linspace(0.5, 2.0, 100).reshape(2, 50) if weighted else None
    actual = model.sft_forward(
        {"observation": {}, "actions": actions}, use_action_chunk_loss=True,
        action_valid_mask=mask, action_weights=weights,
    )
    errors = (actions[..., :14] - model.scale).square()
    expected = masked_fm_loss(errors, mask, weights)
    actual_grad = torch.autograd.grad(actual, model.scale, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, model.scale)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(model.last_actions, actions, rtol=0, atol=0)
