# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Execute actual actor checkpoint methods without constructing FSDP/VLA models."""

import ast
import copy
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rlinf.algorithms.online_bc_attena_fk import (
    AGGREGATION,
    GEOMETRY_KEY,
    MOTION_KEY,
    OnlineBCAttenaFK,
)
from rlinf.data.online_bc import SuccessReplay


@pytest.fixture
def actor_type():
    source = (
        Path(inspect.getfile(SuccessReplay)).resolve().parents[2]
        / "rlinf/workers/actor/fsdp_online_bc_policy_worker.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = copy.deepcopy(next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EmbodiedOnlineBCFSDPPolicy"
    ))
    node.bases = [ast.Name(id="object", ctx=ast.Load())]
    for method in node.body:
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method.decorator_list = [
                decorator for decorator in method.decorator_list
                if isinstance(decorator, ast.Name)
                and decorator.id in ("staticmethod", "classmethod")
            ]
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        node,
    ], type_ignores=[])
    namespace = {"torch": torch, "Path": Path, "json": json, "time": time}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["EmbodiedOnlineBCFSDPPolicy"]


def _method():
    # Geometry/FK itself has separate CPU tests. Here use the production method
    # identity and replay validators, with a fixed synthetic geometry identity.
    geometry_id = "11" * 32
    geometry = SimpleNamespace(
        geometry_id=geometry_id,
        geometry_id_tensor=torch.tensor(list(bytes.fromhex(geometry_id)), dtype=torch.uint8),
        identity={"schema_version": 1, "test_geometry": "resume-only"},
    )
    calibration = {
        "schema_version": 1,
        "geometry_id": geometry_id,
        "aggregation": AGGREGATION,
        "ell": 0.1,
        "c_m": 0.01,
        "positive_tolerance_m": 1e-8,
        "reference": {"source_sha256": "22" * 32, "valid_positions": 100},
    }
    return OnlineBCAttenaFK(geometry, calibration)


def _record(method, index=0):
    record = {
        "action": torch.full((700,), float(index), dtype=torch.float64),
        "observation/state": torch.zeros(14, dtype=torch.float64),
        "action_valid_mask": torch.ones(50, 14, dtype=torch.bool),
        "query_idx": torch.tensor(index),
        "episode_id": torch.tensor([0, 0]),
    }
    if method is not None:
        record[MOTION_KEY] = torch.full((50, 2, 2), 0.002 * (index + 1))
        record[GEOMETRY_KEY] = method.geometry.geometry_id_tensor.clone()
    return record


def _actor(actor_type, tmp_path, method):
    actor = actor_type.__new__(actor_type)
    actor.attena = method
    actor._rank = 0
    actor.device = "cpu"
    actor.checkpoint_format = "local_shard"
    actor.is_weight_offloaded = False
    actor.is_optimizer_offloaded = False
    actor.model = actor.optimizer = actor.lr_scheduler = None
    actor.update_step = 37
    actor.replay_buffer = SuccessReplay(19, str(tmp_path / "pool"), max_success_chunks=3)
    actor._strategy = SimpleNamespace(
        save_checkpoint=lambda **kwargs: None,
        load_checkpoint=lambda **kwargs: None,
    )
    return actor


def _save_replay_with_method(actor, path):
    actor.replay_buffer.add_episodes([[_record(actor.attena, 0), _record(actor.attena, 1)]])
    actor.replay_buffer.sample(7)
    actor.save_checkpoint(path, step=5)
    return path / "online_bc/rank_0"


def test_actual_actor_roundtrip_keeps_method_raw_features_and_next_draw(actor_type, tmp_path):
    events = []
    method = _method()
    actor = _actor(actor_type, tmp_path / "original", method)
    actor._strategy.save_checkpoint = lambda **kwargs: events.append("save_model")
    checkpoint = tmp_path / "checkpoint"
    target = _save_replay_with_method(actor, checkpoint)
    saved_state = json.loads((target / "attena_fk.json").read_text())
    assert saved_state == method.method_state()
    expected = actor.replay_buffer.sample(32)["forward_inputs"]

    restored = _actor(actor_type, tmp_path / "restored", _method())
    restored.update_step = 0
    restored._strategy.load_checkpoint = lambda **kwargs: events.append("load_model")
    restored.load_checkpoint(checkpoint)
    assert events == ["save_model", "load_model"]
    assert restored.update_step == 37
    assert restored.attena.method_state() == saved_state
    assert restored.replay_buffer.get_stats() == actor.replay_buffer.get_stats()
    actual = restored.replay_buffer.sample(32)["forward_inputs"]
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert "action_weights" not in restored.replay_buffer.records[0]


@pytest.mark.parametrize("damage", ["missing", "geometry", "calibration", "version"])
def test_incompatible_method_fails_before_model_or_replay_load(actor_type, tmp_path, damage):
    actor = _actor(actor_type, tmp_path, _method())
    actor._strategy.load_checkpoint = lambda **kwargs: pytest.fail("model loaded before method validation")
    actor.replay_buffer.load_checkpoint = lambda path: pytest.fail("replay loaded before method validation")
    target = tmp_path / "checkpoint/online_bc/rank_0"
    target.mkdir(parents=True)
    state = actor.attena.method_state()
    if damage == "geometry":
        state["geometry_id"] = "33" * 32
    elif damage == "calibration":
        state["calibration"]["c_m"] *= 2
    elif damage == "version":
        state["method_version"] = "different-method-version"
    if damage != "missing":
        (target / "attena_fk.json").write_text(json.dumps(state))
    with pytest.raises(ValueError):
        actor.load_checkpoint(tmp_path / "checkpoint")


@pytest.mark.parametrize("damage", ["missing_motion", "stale_geometry", "nonfinite_raw_action"])
def test_invalid_replay_fails_before_model_load(actor_type, tmp_path, damage):
    actor = _actor(actor_type, tmp_path, _method())
    checkpoint = tmp_path / "checkpoint"
    target = _save_replay_with_method(actor, checkpoint)
    replay_state = torch.load(target / "success_replay.pt", weights_only=True)
    row = replay_state["records"][0]
    if damage == "missing_motion":
        del row[MOTION_KEY]
    elif damage == "stale_geometry":
        row[GEOMETRY_KEY][0] = 255
    else:
        row["action"][0] = float("nan")
    torch.save(replay_state, target / "success_replay.pt")
    actor._strategy.load_checkpoint = lambda **kwargs: pytest.fail("model loaded before replay validation")
    with pytest.raises(ValueError, match="Invalid FK replay record"):
        actor.load_checkpoint(checkpoint)


def test_disabled_actor_rejects_attena_checkpoint_before_model_load(actor_type, tmp_path):
    actor = _actor(actor_type, tmp_path, None)
    actor._strategy.load_checkpoint = lambda **kwargs: pytest.fail("weighted checkpoint loaded as clean BC")
    target = tmp_path / "checkpoint/online_bc/rank_0"
    target.mkdir(parents=True)
    (target / "attena_fk.json").write_text(json.dumps(_method().method_state()))
    with pytest.raises(ValueError, match="cannot resume as unweighted"):
        actor.load_checkpoint(tmp_path / "checkpoint")


def test_disabled_actor_keeps_clean_checkpoint_path(actor_type, tmp_path):
    actor = _actor(actor_type, tmp_path / "original", None)
    checkpoint = tmp_path / "checkpoint"
    target = _save_replay_with_method(actor, checkpoint)
    assert not (target / "attena_fk.json").exists()
    expected = actor.replay_buffer.sample(8)["forward_inputs"]
    restored = _actor(actor_type, tmp_path / "restored", None)
    restored.load_checkpoint(checkpoint)
    assert restored.update_step == actor.update_step
    actual = restored.replay_buffer.sample(8)["forward_inputs"]
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
