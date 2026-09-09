"""Frozen target-TCP-motion weights for online BC; no simulation in training.

Geometry uses raw environment joint targets.  The two arms are combined as the
SUM of their individual pose-motion norms, never a norm over both arms.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np
import torch


MOTION_KEY = "attena_fk_motion"
GEOMETRY_KEY = "attena_fk_geometry_id"
AGGREGATION = "sum_arm_l2"
METHOD_VERSION = "online-bc-attena-fk-v1"
FEATURE_VERSION = "tcp-motion-v1"
EXTRACTION_VERSION = "urdf-ancestor-kinematics-v1"
HORIZON = 50
ACTION_DIM = 14


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_json(path: str | Path) -> dict:
    def reject(value: str):
        raise ValueError(f"Non-finite JSON value: {value}")

    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    _canonical(value)
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}") from exc
    return value.lower()


def _rotation(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 rotation")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError(f"{name} is not orthogonal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} is not a proper rotation")
    return matrix


def _quaternion_rotation(value: Any) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("FK returned an invalid quaternion")
    w, x, y, z = q / np.linalg.norm(q)  # MPlib Pose uses wxyz.
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _angle(current: np.ndarray, previous: np.ndarray) -> float:
    if np.array_equal(current, previous):
        return 0.0
    relative = current @ previous.T
    skew = np.asarray([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ])
    sine = np.linalg.norm(skew) * 0.5
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arctan2(sine, cosine))


def _valid_prefix(mask: torch.Tensor, batch: int, horizon: int) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.shape != (batch, horizon, ACTION_DIM):
        raise ValueError(f"action_valid_mask must have shape [{batch},{horizon},14]")
    mask = mask.detach()
    if mask.dtype != torch.bool:
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("action_valid_mask must be binary and finite")
        mask = mask.bool()
    valid = mask[..., 0]
    if not torch.equal(mask, valid.unsqueeze(-1).expand_as(mask)):
        raise ValueError("Per-dimension masks are not supported by TCP motion")
    if not valid.any(dim=-1).all():
        raise ValueError("Every query needs a nonempty valid action prefix")
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("Only contiguous valid prefixes are supported")
    return valid


def motion_to_magnitude(motion: torch.Tensor, ell: float) -> torch.Tensor:
    """[..., arm=2, (translation_m, angle_rad)] -> [...] in equivalent metres.

    Uses float64 over the stored features both for calibration and training.
    A stationary arm contributes exactly zero; the output has no gradient.
    """
    ell = _positive(ell, "ell")
    if not isinstance(motion, torch.Tensor) or motion.ndim < 2 or motion.shape[-2:] != (2, 2):
        raise ValueError("motion must end in [2 arms, 2 components]")
    motion = motion.detach().to(dtype=torch.float64)
    if not torch.isfinite(motion).all() or (motion < 0).any():
        raise ValueError("motion must be finite and nonnegative")
    if (motion[..., 1] > math.pi + 1e-6).any():
        raise ValueError("Rotation magnitude exceeds the SO(3) principal angle")
    scaled = torch.stack((motion[..., 0], ell * motion[..., 1]), dim=-1)
    return torch.linalg.vector_norm(scaled, dim=-1).sum(dim=-1)


def motion_to_weights(
    motion: torch.Tensor,
    c_m: float,
    ell: float = 0.1,
    epsilon: float = 1e-3,
    clip_max: float = 2.0,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Inverse-square clipped weights; no mean normalization or batch statistics."""
    c_m = _positive(c_m, "c_m")
    epsilon = _positive(epsilon, "epsilon")
    clip_max = _positive(clip_max, "clip_max")
    if clip_max < 1:
        raise ValueError("clip_max must be at least 1")
    if not isinstance(motion, torch.Tensor) or motion.ndim != 4:
        raise ValueError("Batched motion must have shape [B,H,2,2]")
    magnitude = motion_to_magnitude(motion, ell)
    batch, horizon = magnitude.shape
    if batch == 0 or horizon == 0:
        raise ValueError("Empty batches are not supported")
    if valid_mask is None:
        valid = torch.ones_like(magnitude, dtype=torch.bool)
    else:
        valid = _valid_prefix(valid_mask, batch, horizon).to(magnitude.device)
    scaled = magnitude / c_m
    unmasked = scaled.clamp_min(epsilon).square().reciprocal().clamp(1 / clip_max, clip_max)
    weights = torch.where(valid, unmasked, 1.0).to(torch.float32).detach()
    selected = weights[valid].to(torch.float64)
    selected_u = magnitude[valid]
    total = selected.sum()
    metrics = {
        "attena_fk/weight_mean": float(selected.mean().item()),
        "attena_fk/weight_std": float(selected.std(unbiased=False).item()),
        "attena_fk/weight_min": float(selected.min().item()),
        "attena_fk/weight_max": float(selected.max().item()),
        "attena_fk/lower_clip_fraction": float((selected == float(torch.tensor(1 / clip_max, dtype=torch.float32))).double().mean().item()),
        "attena_fk/upper_clip_fraction": float((selected == float(torch.tensor(clip_max, dtype=torch.float32))).double().mean().item()),
        "attena_fk/ess_fraction": float((total.square() / (selected.square().sum() * selected.numel())).item()),
        "attena_fk/magnitude_mean_m": float(selected_u.mean().item()),
    }
    for arm, name in enumerate(("left", "right")):
        for component, unit in enumerate(("translation_m", "rotation_rad")):
            values = motion.detach()[..., arm, component][valid]
            metrics[f"attena_fk/{name}_{unit}"] = float(values.double().mean().item())
    return weights, metrics


def _extract_urdf(raw: bytes, target_links: list[str]) -> str:
    tree = ET.fromstring(raw)
    links = {node.get("name") for node in tree.findall("link")}
    joints = tree.findall("joint")
    by_child = {}
    for joint in joints:
        child = joint.find("child").get("link")
        if child in by_child:
            raise ValueError("URDF contains multiple parents for a link")
        by_child[child] = joint
    keep_links, keep_joints = set(), set()
    for target in target_links:
        if target not in links:
            raise ValueError(f"Missing target link: {target}")
        seen = set()
        link = target
        while True:
            if link in seen:
                raise ValueError("Cycle in URDF ancestor chain")
            seen.add(link)
            keep_links.add(link)
            if link not in by_child:
                break
            joint = by_child[link]
            keep_joints.add(joint.get("name"))
            link = joint.find("parent").get("link")
    root = ET.Element("robot", {"name": "online_bc_attena_fk_kinematics_v1"})
    for node in tree:
        if not ((node.tag == "link" and node.get("name") in keep_links)
                or (node.tag == "joint" and node.get("name") in keep_joints)):
            continue
        node = copy.deepcopy(node)
        for child in list(node):
            if child.tag in ("visual", "collision", "inertial"):
                node.remove(child)
        root.append(node)
    return ET.tostring(root, encoding="unicode")


class Geometry:
    """CPU-only immutable geometry and record annotation, independent of scale."""

    @classmethod
    def from_file(cls, path: str | Path) -> "Geometry":
        return cls(_read_json(path))

    def __init__(self, config: Mapping[str, Any]):
        config = copy.deepcopy(dict(config))
        _canonical(config)
        if config.get("schema_version") != 1 or config.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Unsupported FK geometry schema/feature version")
        if config.get("horizon") != HORIZON or config.get("action_dim") != ACTION_DIM:
            raise ValueError("FK geometry requires horizon=50 and action_dim=14")
        if config.get("backend") != "mplib.pinocchio":
            raise ValueError("Only mplib.pinocchio is supported")
        version = importlib.metadata.version("mplib")
        if config.get("backend_version") != version:
            raise ValueError("FK backend version mismatch")
        _sha(config.get("robot_config_sha256"), "robot_config_sha256")
        raw = Path(config["urdf_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != _sha(config.get("urdf_sha256"), "urdf_sha256"):
            raise ValueError("URDF content hash mismatch")
        arms = config.get("arms")
        if not isinstance(arms, list) or len(arms) != 2:
            raise ValueError("Geometry requires exactly left and right arms")
        self.arms = arms
        self.child_frames = []
        for arm, name, expected_indices in zip(arms, ("left", "right"), (list(range(6)), list(range(7, 13)))):
            if arm.get("name") != name or arm.get("action_indices") != expected_indices:
                raise ValueError("Arm order/indices must exclude both grippers")
            if len(arm.get("joint_names", [])) != 6 or len(set(arm["joint_names"])) != 6:
                raise ValueError("Each arm needs six unique named joints")
            frame = np.asarray(arm.get("T_child"), dtype=np.float64)
            if frame.shape != (4, 4) or not np.isfinite(frame).all() or not np.array_equal(frame[3], [0, 0, 0, 1]):
                raise ValueError("Invalid T_child homogeneous transform")
            _rotation(frame[:3, :3], "T_child rotation")
            self.child_frames.append(frame)
        self.rotation_fix = _rotation(config.get("G"), "G") @ _rotation(config.get("D"), "D")
        self.bias = _positive(config.get("bias"), "bias")
        extracted = _extract_urdf(raw, [arm["link"] for arm in arms])
        # Lazy import: disabled methods and pure mapping never load an FK backend.
        from mplib.kinematics.pinocchio import PinocchioModel

        self.model = PinocchioModel.create_from_urdf_string(extracted)
        names = list(self.model.get_joint_names())
        dims = list(self.model.get_joint_dims())
        expected_names = [name for arm in arms for name in arm["joint_names"]]
        if len(set(expected_names)) != 12 or set(names) != set(expected_names) or len(names) != 12:
            raise ValueError("Extracted URDF joints differ from configured twelve joints")
        if any(int(d) != 1 for d in dims) or len(dims) != 12:
            raise ValueError("FK expects twelve one-DoF joints")
        offsets = np.cumsum([0] + [int(d) for d in dims])
        self.q_indices = [[int(offsets[names.index(n)]) for n in arm["joint_names"]] for arm in arms]
        link_names = list(self.model.get_link_names())
        self.link_ids = [link_names.index(arm["link"]) for arm in arms]
        identity = {key: value for key, value in config.items() if key != "urdf_path"}
        identity.update({
            "extraction_version": EXTRACTION_VERSION,
            "extracted_urdf_sha256": hashlib.sha256(extracted.encode()).hexdigest(),
            "actual_backend_version": version,
        })
        self.identity = identity
        self.geometry_id = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        self.geometry_id_tensor = torch.tensor(list(bytes.fromhex(self.geometry_id)), dtype=torch.uint8)

    def poses(self, environment_action: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return (TCP position, TCP rotation) for each arm in a fixed root frame."""
        values = np.asarray(environment_action, dtype=np.float64)
        if values.shape != (ACTION_DIM,) or not np.isfinite(values).all():
            raise ValueError("Expected one finite raw 14-dimensional action/state")
        q = np.zeros(12, dtype=np.float64)
        for arm, q_indices in zip(self.arms, self.q_indices):
            q[q_indices] = values[arm["action_indices"]]
        self.model.compute_forward_kinematics(q)
        result = []
        for link_id, child in zip(self.link_ids, self.child_frames):
            pose = self.model.get_link_pose(link_id)
            link_r = _quaternion_rotation(pose.q)
            joint_p = np.asarray(pose.p, dtype=np.float64) + link_r @ child[:3, 3]
            tcp_r = link_r @ child[:3, :3] @ self.rotation_fix
            tcp_p = joint_p + tcp_r @ np.asarray([self.bias, 0.0, 0.0])
            if not np.isfinite(tcp_p).all() or not np.isfinite(tcp_r).all():
                raise ValueError("FK produced non-finite TCP geometry")
            result.append((tcp_p, tcp_r))
        return result

    def prepare_record(self, record: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if MOTION_KEY in record or GEOMETRY_KEY in record:
            raise ValueError("Refusing to overwrite an existing FK cache")
        try:
            action = record["action"]
            state = record["observation/state"]
            mask = record["action_valid_mask"]
            if not all(isinstance(value, torch.Tensor) for value in (action, state, mask)):
                raise ValueError("Action, state and mask must be tensors")
            if action.shape != (HORIZON * ACTION_DIM,) or state.shape != (ACTION_DIM,):
                raise ValueError("Record must have action[700] and state[14]")
            valid = _valid_prefix(mask.unsqueeze(0), 1, HORIZON)[0].cpu()
            actions = action.detach().to(device="cpu", dtype=torch.float64).reshape(HORIZON, ACTION_DIM).numpy()
            state = state.detach().to(device="cpu", dtype=torch.float64).numpy()
            # Reject non-finite labels even in padding: they still enter model forward.
            if not np.isfinite(actions).all() or not np.isfinite(state).all():
                raise ValueError("Raw action/state contains non-finite values")
            motion = np.zeros((HORIZON, 2, 2), dtype=np.float64)
            previous = self.poses(state)
            for h in range(int(valid.sum().item())):
                current = self.poses(actions[h])
                for arm in range(2):
                    p, r = current[arm]
                    p0, r0 = previous[arm]
                    motion[h, arm] = (np.linalg.norm(p - p0), _angle(r, r0))
                previous = current
            features = torch.from_numpy(motion).to(torch.float32)
            motion_to_magnitude(features, ell=0.1)  # Cache shape/finite/SO(3) validation.
            output = dict(record)
            output[MOTION_KEY] = features
            output[GEOMETRY_KEY] = self.geometry_id_tensor.clone()
            return output
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            raise ValueError(f"FK record failed (episode={record.get('episode_id')}, query={record.get('query_idx')}): {exc}") from exc


class OnlineBCAttenaFK:
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "OnlineBCAttenaFK | None":
        if not bool(cfg.get("enabled", False)):
            return None
        geometry = Geometry.from_file(cfg["geometry_path"])
        calibration = _read_json(cfg["calibration_path"])
        return cls(geometry, calibration, ell=cfg.get("ell", 0.1),
                   epsilon=cfg.get("epsilon", 1e-3), clip_max=cfg.get("clip_max", 2.0))

    def __init__(self, geometry: Geometry, calibration: Mapping[str, Any], *, ell=0.1, epsilon=1e-3, clip_max=2.0):
        self.geometry = geometry
        self.ell = _positive(ell, "ell")
        self.epsilon = _positive(epsilon, "epsilon")
        self.clip_max = _positive(clip_max, "clip_max")
        if self.clip_max < 1:
            raise ValueError("clip_max must be at least 1")
        calibration = copy.deepcopy(dict(calibration))
        _canonical(calibration)
        if calibration.get("schema_version") != 1 or calibration.get("aggregation") != AGGREGATION:
            raise ValueError("Calibration must use schema 1 and sum_arm_l2; old combined-norm scale is invalid")
        if calibration.get("geometry_id") != geometry.geometry_id:
            raise ValueError("Calibration geometry identity mismatch")
        if _positive(calibration.get("ell"), "calibration ell") != self.ell:
            raise ValueError("Calibration ell mismatch")
        self.c_m = _positive(calibration.get("c_m"), "c_m")
        tolerance = _positive(calibration.get("positive_tolerance_m"), "positive_tolerance_m")
        if self.c_m <= tolerance:
            raise ValueError("Calibration c_m must exceed its numerical positive tolerance")
        if not isinstance(calibration.get("reference"), dict) or not calibration["reference"]:
            raise ValueError("Calibration requires its reference provenance")
        self.calibration = calibration
        self._state = {
            "schema_version": 1, "method_version": METHOD_VERSION,
            "geometry_id": geometry.geometry_id, "geometry": geometry.identity,
            "calibration": calibration, "calibration_sha256": hashlib.sha256(_canonical(calibration).encode()).hexdigest(),
            "aggregation": AGGREGATION, "ell": self.ell, "c_m": self.c_m,
            "epsilon": self.epsilon, "clip_max": self.clip_max,
            "mapping": "inverse_squared_clip", "weight_rescaling": "none",
        }
        _canonical(self._state)

    def prepare_record(self, record):
        return self.geometry.prepare_record(record)

    def _validate_batch(self, batch):
        motion = batch[MOTION_KEY]
        ids = batch[GEOMETRY_KEY]
        if not isinstance(motion, torch.Tensor) or motion.ndim != 4 or motion.shape[1:] != (HORIZON, 2, 2):
            raise ValueError("Cached motion must have shape [B,50,2,2]")
        if motion.dtype != torch.float32:
            raise ValueError("Cached motion must be float32")
        count = motion.shape[0]
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.uint8 or ids.shape != (count, 32):
            raise ValueError("Cached geometry identity must be uint8[B,32]")
        expected = self.geometry.geometry_id_tensor.to(ids.device).expand(count, -1)
        if not torch.equal(ids, expected):
            raise ValueError("Mixed or stale FK geometry identity")
        valid = _valid_prefix(batch["action_valid_mask"], count, HORIZON).to(motion.device)
        motion_to_magnitude(motion, self.ell)
        if (motion[~valid] != 0).any():
            raise ValueError("Padded cached motion must be zero")

    def weights(self, batch):
        self._validate_batch(batch)
        return motion_to_weights(batch[MOTION_KEY], self.c_m, self.ell, self.epsilon,
                                 self.clip_max, batch["action_valid_mask"])

    def method_state(self) -> dict:
        return copy.deepcopy(self._state)

    def validate_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or _canonical(dict(state)) != _canonical(self._state):
            raise ValueError("AttenA FK method/geometry/calibration state mismatch")

    def validate_records(self, records) -> None:
        for index, record in enumerate(records):
            try:
                for key, shape in (("action", (HORIZON * ACTION_DIM,)), ("observation/state", (ACTION_DIM,))):
                    value = record[key]
                    if not isinstance(value, torch.Tensor) or value.shape != shape or not torch.isfinite(value).all():
                        raise ValueError(f"Invalid raw replay field {key}")
                self._validate_batch({key: record[key].unsqueeze(0) for key in
                                      (MOTION_KEY, GEOMETRY_KEY, "action_valid_mask")})
            except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                raise ValueError(f"Invalid FK replay record {index}: {exc}") from exc
