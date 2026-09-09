"""Targeted CPU tests for the production FK and sum-of-arm-motion contract."""

import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from rlinf.algorithms.online_bc_attena_fk import (
    AGGREGATION,
    GEOMETRY_KEY,
    MOTION_KEY,
    Geometry,
    OnlineBCAttenaFK,
    motion_to_magnitude,
    motion_to_weights,
)


@pytest.fixture
def geometry_config(tmp_path):
    # Actual Pinocchio FK, with analytically simple two-arm kinematics.
    nodes = ['<robot name="two_arms"><link name="root"/>']
    arms = []
    for arm, prefix, indices in (("left", "fl", list(range(6))),
                                 ("right", "fr", list(range(7, 13)))):
        names = []
        for index in range(1, 7):
            name = f"{prefix}_joint{index}"
            names.append(name)
            child = f"{prefix}_link{index}"
            parent = "root" if index == 1 else f"{prefix}_link{index-1}"
            kind = "revolute" if index == 6 else "prismatic"
            axis = "0 0 1" if index == 6 else ("1 0 0" if index == 1 else "0 1 0")
            nodes.append(f'<link name="{child}"/><joint name="{name}" type="{kind}">'
                         f'<parent link="{parent}"/><child link="{child}"/>'
                         f'<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="{axis}"/>'
                         '<limit lower="-1" upper="1" effort="10" velocity="10"/></joint>')
        arms.append({"name": arm, "action_indices": indices, "joint_names": names,
                     "link": f"{prefix}_link6", "T_child": np.eye(4).tolist()})
    # An irrelevant gripper branch must not enter the twelve-dimensional model.
    nodes.append('<link name="finger"/><joint name="finger_joint" type="prismatic">'
                 '<parent link="fl_link6"/><child link="finger"/><axis xyz="1 0 0"/>'
                 '<limit lower="0" upper="1" effort="1" velocity="1"/></joint></robot>')
    path = tmp_path / "robot.urdf"
    path.write_text("".join(nodes), encoding="utf-8")
    return {
        "schema_version": 1, "feature_version": "tcp-motion-v1",
        "urdf_path": str(path), "urdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "robot_config_sha256": "a" * 64, "backend": "mplib.pinocchio",
        "backend_version": importlib.metadata.version("mplib"), "arms": arms,
        "G": np.eye(3).tolist(), "D": np.eye(3).tolist(), "bias": 0.12,
        "horizon": 50, "action_dim": 14,
    }


@pytest.fixture
def geometry(geometry_config):
    return Geometry(geometry_config)


def record(valid=50):
    mask = torch.zeros(50, 14, dtype=torch.bool)
    mask[:valid] = True
    return {"action": torch.zeros(700, dtype=torch.float64),
            "observation/state": torch.zeros(14, dtype=torch.float64),
            "action_valid_mask": mask, "episode_id": torch.tensor([2, 9]),
            "query_idx": torch.tensor(0)}


def calibration(geometry):
    return {"schema_version": 1, "geometry_id": geometry.geometry_id,
            "aggregation": AGGREGATION, "ell": 0.1, "c_m": 0.01,
            "positive_tolerance_m": 1e-8,
            "reference": {"source_sha256": "b" * 64, "checkpoint": "R100",
                          "max_success_chunks": 3, "valid_positions": 100}}


def batch(records):
    return {key: torch.stack([r[key] for r in records]) for key in records[0]}


def test_magnitude_is_sum_of_individual_arm_norms():
    values = torch.tensor([[[[0.003, 0.04], [0.006, 0.08]]]], dtype=torch.float64)
    result = motion_to_magnitude(values, 0.1)
    assert result.item() == pytest.approx(0.015, abs=1e-14)
    assert result.item() != pytest.approx(math.sqrt(0.005**2 + 0.01**2))
    values[..., 1, :] = 0
    assert motion_to_magnitude(values, 0.1).item() == pytest.approx(0.005, abs=1e-14)
    assert motion_to_magnitude(torch.zeros_like(values), 0.1).item() == 0


def test_mapping_clips_without_mean_normalization_and_detaches():
    motion = torch.zeros(1, 4, 2, 2, dtype=torch.float64)
    motion[0, :, 0, 0] = torch.tensor([0, 0.005, 0.01, 0.02], dtype=torch.float64)
    motion.requires_grad_(True)
    weights, stats = motion_to_weights(motion, c_m=0.01)
    torch.testing.assert_close(weights, torch.tensor([[2.0, 2.0, 1.0, 0.5]]))
    assert not weights.requires_grad
    assert weights.dtype == torch.float32
    assert stats["attena_fk/weight_mean"] == pytest.approx(1.375)
    assert stats["attena_fk/upper_clip_fraction"] == 0.5
    assert stats["attena_fk/lower_clip_fraction"] == 0.25
    assert stats["attena_fk/ess_fraction"] == pytest.approx(5.5**2 / (4 * 9.25))


def test_weights_are_independent_of_batch_and_microbatch():
    generator = torch.Generator().manual_seed(9)
    motion = torch.rand(65, 50, 2, 2, generator=generator) * 0.02
    all_weights, _ = motion_to_weights(motion, 0.012)
    pieces = [motion_to_weights(part, 0.012)[0] for part in motion.split(32)]
    assert torch.equal(all_weights, torch.cat(pieces))
    assert torch.equal(all_weights[3:4], motion_to_weights(motion[3:4], 0.012)[0])


@pytest.mark.parametrize("field,bad", [("c_m", 0), ("c_m", float("nan")),
                                      ("ell", -1), ("epsilon", 0), ("clip_max", 0.5)])
def test_invalid_mapping_settings_fail(field, bad):
    options = {"c_m": 0.01, "ell": 0.1, "epsilon": 1e-3, "clip_max": 2}
    options[field] = bad
    with pytest.raises(ValueError):
        motion_to_weights(torch.zeros(1, 50, 2, 2), **options)


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_invalid_motion_fails(bad):
    motion = torch.zeros(1, 50, 2, 2)
    motion[0, 0, 0, 0] = bad
    with pytest.raises(ValueError):
        motion_to_weights(motion, 0.01)


def test_actual_fk_first_state_right_arm_and_input_immutability(geometry):
    raw = record()
    raw["observation/state"][0] = 0.2
    action = raw["action"].reshape(50, 14)
    action[:, 0] = 0.3
    action[:, 7] = 0.4
    before = {key: value.clone() for key, value in raw.items()}
    rng = torch.get_rng_state().clone()
    prepared = geometry.prepare_record(raw)
    assert set(prepared) == set(raw) | {MOTION_KEY, GEOMETRY_KEY}
    for key in raw:
        assert torch.equal(raw[key], before[key])
        assert prepared[key] is raw[key]
    assert torch.equal(torch.get_rng_state(), rng)
    feature = prepared[MOTION_KEY]
    assert feature.shape == (50, 2, 2) and feature.dtype == torch.float32
    assert feature[0, 0, 0].item() == pytest.approx(0.1, abs=1e-7)
    assert feature[0, 1, 0].item() == pytest.approx(0.4, abs=1e-7)
    assert torch.equal(feature[1:], torch.zeros_like(feature[1:]))
    assert prepared[GEOMETRY_KEY].shape == (32,)
    assert prepared[GEOMETRY_KEY].dtype == torch.uint8


def test_actual_fk_rotation_includes_tcp_offset_and_principal_angle(geometry):
    raw = record()
    angle = 0.5
    raw["action"].reshape(50, 14)[:, 5] = angle
    prepared = geometry.prepare_record(raw)
    first = prepared[MOTION_KEY][0, 0]
    assert first[0].item() == pytest.approx(2 * 0.12 * math.sin(angle / 2), abs=1e-7)
    assert first[1].item() == pytest.approx(angle, abs=1e-7)
    assert torch.equal(prepared[MOTION_KEY][1:], torch.zeros(49, 2, 2))


def test_child_frame_pose_contract(geometry_config):
    flip = np.diag([1.0, -1.0, -1.0])
    frame = np.eye(4)
    frame[:3, :3] = flip
    frame[:3, 3] = [0.02, 0.01, 0]
    geometry_config["arms"][0]["T_child"] = frame.tolist()
    geometry_config["G"] = flip.tolist()
    left_p, left_r = Geometry(geometry_config).poses(np.zeros(14))[0]
    np.testing.assert_allclose(left_p, [0.14, 0.01, 0], atol=1e-10)
    np.testing.assert_allclose(left_r, np.eye(3), atol=1e-10)


def test_grippers_do_not_enter_fk_and_limits_do_not_clip_labels(geometry):
    raw = record()
    raw["action"].reshape(50, 14)[:, 0] = 2.0  # Deliberately outside toy URDF limit.
    original = geometry.prepare_record(raw)
    raw["action"].reshape(50, 14)[:, 6] = 999
    raw["action"].reshape(50, 14)[:, 13] = -999
    raw["observation/state"][[6, 13]] = 42
    changed = geometry.prepare_record(raw)
    assert torch.equal(original[MOTION_KEY], changed[MOTION_KEY])
    assert changed[MOTION_KEY][0, 0, 0].item() == pytest.approx(2.0)


def test_tail_padding_is_neutral_without_fk_of_padding(geometry):
    raw = record(valid=3)
    raw["action"].reshape(50, 14)[3:, :6] = 20
    prepared = geometry.prepare_record(raw)
    assert torch.equal(prepared[MOTION_KEY], torch.zeros(50, 2, 2))
    method = OnlineBCAttenaFK(geometry, calibration(geometry))
    weights, metrics = method.weights(batch([prepared]))
    assert torch.equal(weights[:, :3], torch.full((1, 3), 2.0))
    assert torch.equal(weights[:, 3:], torch.ones(1, 47))
    assert metrics["attena_fk/weight_mean"] == 2


@pytest.mark.parametrize("mode", ["hole", "per_dimension", "empty", "nonbinary"])
def test_undefined_masks_fail(geometry, mode):
    raw = record()
    if mode == "hole":
        raw["action_valid_mask"][3] = False
    elif mode == "per_dimension":
        raw["action_valid_mask"][3, 7] = False
    elif mode == "empty":
        raw["action_valid_mask"][:] = False
    else:
        raw["action_valid_mask"] = raw["action_valid_mask"].float()
        raw["action_valid_mask"][0, 0] = 0.5
    with pytest.raises(ValueError):
        geometry.prepare_record(raw)


def test_geometry_identity_ignores_path_but_rejects_geometry_changes(geometry_config, tmp_path):
    first = Geometry(geometry_config)
    moved = tmp_path / "moved.urdf"
    moved.write_bytes(Path(geometry_config["urdf_path"]).read_bytes())
    geometry_config["urdf_path"] = str(moved)
    assert Geometry(geometry_config).geometry_id == first.geometry_id
    geometry_config["bias"] = 0.13
    assert Geometry(geometry_config).geometry_id != first.geometry_id
    geometry_config["urdf_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="URDF content hash"):
        Geometry(geometry_config)


def test_method_state_and_cached_records_are_strict(geometry):
    method = OnlineBCAttenaFK(geometry, calibration(geometry))
    prepared = method.prepare_record(record())
    method.validate_records([prepared])
    method.validate_state(method.method_state())
    state = method.method_state()
    state["calibration"]["reference"]["checkpoint"] = "R90"
    with pytest.raises(ValueError, match="state mismatch"):
        method.validate_state(state)
    changed = copy.deepcopy(prepared)
    changed[GEOMETRY_KEY][0] ^= 1
    with pytest.raises(ValueError, match="geometry"):
        method.weights(batch([prepared, changed]))
    with pytest.raises(ValueError, match="record"):
        method.validate_records([record()])
    with pytest.raises(ValueError, match="overwrite"):
        method.prepare_record(prepared)


@pytest.mark.parametrize("change", ["aggregation", "geometry", "ell", "zero_scale", "provenance"])
def test_old_or_incompatible_calibration_is_rejected(geometry, change):
    value = calibration(geometry)
    if change == "aggregation":
        value["aggregation"] = "combined_arm_l2"
    elif change == "geometry":
        value["geometry_id"] = "0" * 64
    elif change == "ell":
        value["ell"] = 0.2
    elif change == "zero_scale":
        value["c_m"] = 0
    else:
        value["reference"] = {}
    with pytest.raises(ValueError):
        OnlineBCAttenaFK(geometry, value)


def test_disabled_never_builds_geometry(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled method attempted to load geometry")
    monkeypatch.setattr(Geometry, "from_file", unexpected)
    assert OnlineBCAttenaFK.from_config({"enabled": False}) is None


def test_enabled_config_loads_locked_artifacts(geometry_config, tmp_path):
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps(geometry_config))
    geometry = Geometry.from_file(geometry_path)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration(geometry)))
    method = OnlineBCAttenaFK.from_config({"enabled": True, "geometry_path": str(geometry_path),
                                         "calibration_path": str(calibration_path),
                                         "epsilon": 1e-3, "clip_max": 2, "ell": 0.1})
    assert method.method_state()["aggregation"] == "sum_arm_l2"
    assert method.method_state()["c_m"] == 0.01
