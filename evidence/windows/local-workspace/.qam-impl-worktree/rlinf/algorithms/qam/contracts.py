# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data and action contracts for the fixed-N RoboTwin QAM adaptation."""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

MODEL_HORIZON = 50
MODEL_ACTION_DIM = 32
PLANNED_HORIZON = 20
ACTIVE_ACTION_DIM = 14
PREFIX_BLOCKS = 4
QAM_REPLAY_SCHEMA_VERSION = 1
QAM_PROMPT_BYTES = 256


def encode_qam_prompt_batch(
    prompts: Sequence[str],
    *,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Encode exact rollout prompts as fixed-width tensors for Trajectory."""
    if isinstance(prompts, (str, bytes)) or len(prompts) == 0:
        raise ValueError("QAM prompts must be a non-empty sequence of strings")
    encoded = torch.zeros(
        len(prompts),
        QAM_PROMPT_BYTES,
        dtype=torch.uint8,
        device=device,
    )
    lengths = torch.zeros(len(prompts), dtype=torch.long, device=device)
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("each QAM prompt must be a non-empty string")
        payload = prompt.encode("utf-8")
        if len(payload) > QAM_PROMPT_BYTES:
            raise ValueError(
                f"QAM prompt uses {len(payload)} UTF-8 bytes; "
                f"the fixed trajectory limit is {QAM_PROMPT_BYTES}"
            )
        encoded[index, : len(payload)] = torch.tensor(
            list(payload),
            dtype=torch.uint8,
            device=device,
        )
        lengths[index] = len(payload)
    return encoded, lengths


def decode_qam_prompt(encoded: Tensor, length: Tensor | int) -> str:
    """Decode one exact prompt row received through Trajectory."""
    row = encoded.detach().cpu().to(dtype=torch.uint8).reshape(-1)
    size = int(length.detach().cpu().reshape(-1)[0].item()) if isinstance(
        length, Tensor
    ) else int(length)
    if row.numel() != QAM_PROMPT_BYTES:
        raise ValueError(
            f"QAM prompt row must contain {QAM_PROMPT_BYTES} bytes, "
            f"got {row.numel()}"
        )
    if size <= 0 or size > QAM_PROMPT_BYTES:
        raise ValueError(f"invalid QAM prompt byte length: {size}")
    try:
        return bytes(row[:size].tolist()).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("QAM prompt tensor is not valid UTF-8") from exc


def _expect_shape(tensor: Tensor, shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def _expect_cpu(tensor: Tensor, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor, got {tensor.device}")


def _expect_floating_finite(tensor: Tensor, name: str) -> None:
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")


def canonicalize_model_action(model_action: Tensor) -> Tensor:
    """Clamp the active planned coordinates used by both Q and the environment.

    The returned model-space tensor preserves the unexecuted model suffix and
    static padding coordinates. Callers must pass this returned tensor, rather
    than the unclamped input, into the existing output transform.
    """
    if model_action.shape[-2:] != (MODEL_HORIZON, MODEL_ACTION_DIM):
        raise ValueError(
            "model_action must end in "
            f"[{MODEL_HORIZON}, {MODEL_ACTION_DIM}], got {tuple(model_action.shape)}"
        )
    planned = torch.cat(
        (
            model_action[..., :PLANNED_HORIZON, :ACTIVE_ACTION_DIM].clamp(
                -1.0,
                1.0,
            ),
            model_action[..., :PLANNED_HORIZON, ACTIVE_ACTION_DIM:],
        ),
        dim=-1,
    )
    return torch.cat(
        (
            planned,
            model_action[..., PLANNED_HORIZON:, :],
        ),
        dim=-2,
    )


def project_planned_action(model_action: Tensor) -> Tensor:
    """Apply the canonical clamp and project model action to ``[N, 14]``."""
    canonical = canonicalize_model_action(model_action)
    return canonical[..., :PLANNED_HORIZON, :ACTIVE_ACTION_DIM].contiguous()


def embed_planned_adjoint(planned_adjoint: Tensor) -> Tensor:
    """Apply ``P_N^T`` and zero the model suffix and padding coordinates."""
    if planned_adjoint.shape[-2:] != (PLANNED_HORIZON, ACTIVE_ACTION_DIM):
        raise ValueError(
            "planned_adjoint must end in "
            f"[{PLANNED_HORIZON}, {ACTIVE_ACTION_DIM}], "
            f"got {tuple(planned_adjoint.shape)}"
        )
    embedded = planned_adjoint.new_zeros(
        *planned_adjoint.shape[:-2],
        MODEL_HORIZON,
        MODEL_ACTION_DIM,
    )
    embedded[..., :PLANNED_HORIZON, :ACTIVE_ACTION_DIM] = planned_adjoint
    return embedded


def fixed_slot_discounted_return(
    slot_rewards: Tensor,
    *,
    gamma_slot: float,
) -> Tensor:
    """Reduce a fixed N-slot native reward vector without inventing duration."""
    if slot_rewards.shape[-1] != PLANNED_HORIZON:
        raise ValueError(
            f"slot_rewards must end in {PLANNED_HORIZON}, "
            f"got {tuple(slot_rewards.shape)}"
        )
    if not math.isfinite(gamma_slot) or not 0.0 <= gamma_slot <= 1.0:
        raise ValueError("gamma_slot must be finite and in [0, 1]")
    work_dtype = torch.float64 if slot_rewards.dtype == torch.float64 else torch.float32
    rewards = slot_rewards.to(dtype=work_dtype)
    powers = torch.arange(
        PLANNED_HORIZON,
        device=rewards.device,
        dtype=rewards.dtype,
    )
    discounts = torch.as_tensor(
        gamma_slot,
        device=rewards.device,
        dtype=rewards.dtype,
    ).pow(powers)
    return (rewards * discounts).sum(dim=-1)


def fixed_slot_bootstrap_discount(*, gamma_slot: float) -> float:
    """Return the M2 macro discount ``gamma_slot ** N``."""
    if not math.isfinite(gamma_slot) or not 0.0 <= gamma_slot <= 1.0:
        raise ValueError("gamma_slot must be finite and in [0, 1]")
    return float(gamma_slot) ** PLANNED_HORIZON


def macro_bootstrap_mask(
    *,
    success_terminated: bool,
    time_limit_truncated: bool,
    other_truncated: bool,
    next_state_valid: bool,
) -> float:
    """Map causal end semantics to the v1 bootstrap mask.

    Success and non-time-limit truncation do not bootstrap. A live transition
    and a time-limit truncation both require a true next/final state and do
    bootstrap.
    """
    end_count = sum(
        (
            bool(success_terminated),
            bool(time_limit_truncated),
            bool(other_truncated),
        )
    )
    if end_count > 1:
        raise ValueError("QAM end flags must be mutually exclusive")
    if success_terminated or other_truncated:
        return 0.0
    if not next_state_valid:
        kind = "time-limit" if time_limit_truncated else "live"
        raise ValueError(f"{kind} QAM transition requires a valid next state")
    return 1.0


def _update_hash_text(hasher: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "little"))
    hasher.update(encoded)


def _update_hash_tensor(hasher: Any, tensor: Tensor) -> None:
    canonical = tensor.detach().cpu().contiguous()
    _update_hash_text(hasher, str(canonical.dtype))
    _update_hash_text(hasher, repr(tuple(canonical.shape)))
    hasher.update(canonical.view(torch.uint8).reshape(-1).numpy().tobytes())


@dataclass(frozen=True)
class QAMPolicyObservation:
    """Canonical raw policy conditioning stored once and referenced by ID."""

    cameras_uint8: Tensor
    proprio: Tensor
    prompt: str
    task_id: str
    transform_fingerprint: str

    def __post_init__(self) -> None:
        _expect_cpu(self.cameras_uint8, "cameras_uint8")
        _expect_cpu(self.proprio, "proprio")
        if self.cameras_uint8.dtype != torch.uint8:
            raise ValueError("cameras_uint8 must use torch.uint8")
        if (
            self.cameras_uint8.ndim != 4
            or self.cameras_uint8.shape[0] != 3
            or self.cameras_uint8.shape[-1] != 3
            or min(self.cameras_uint8.shape[1:3]) <= 0
        ):
            raise ValueError(
                "cameras_uint8 must have shape [3, H, W, 3], got "
                f"{tuple(self.cameras_uint8.shape)}"
            )
        _expect_shape(self.proprio, (ACTIVE_ACTION_DIM,), "proprio")
        _expect_floating_finite(self.proprio, "proprio")
        if not self.task_id or not self.transform_fingerprint:
            raise ValueError("task_id and transform_fingerprint must be non-empty")

    def content_id(self) -> str:
        """Return a stable content ID for exact observation deduplication."""
        hasher = hashlib.sha256()
        _update_hash_text(hasher, "qam-policy-observation-v1")
        _update_hash_tensor(hasher, self.cameras_uint8)
        _update_hash_tensor(hasher, self.proprio)
        _update_hash_text(hasher, self.prompt)
        _update_hash_text(hasher, self.task_id)
        _update_hash_text(hasher, self.transform_fingerprint)
        return hasher.hexdigest()


@dataclass(frozen=True)
class QAMMacroTransition:
    """One fixed-N query-level transition with critic and policy views."""

    obs_id: str
    next_obs_id: str | None
    obs_feature: Tensor
    obs_proprio: Tensor
    next_obs_feature: Tensor | None
    next_obs_proprio: Tensor | None
    next_state_valid: bool
    planned_actions_normalized: Tensor
    planned_actions_env: Tensor
    chunk_rewards_native: Tensor
    reward_macro_discounted: float
    success_terminated: bool
    time_limit_truncated: bool
    other_truncated: bool
    bootstrap_mask: float
    policy_version: int
    episode_id: str
    query_index: int
    contract_fingerprint: str

    def __post_init__(self) -> None:
        tensor_fields = {
            "obs_feature": self.obs_feature,
            "obs_proprio": self.obs_proprio,
            "planned_actions_normalized": self.planned_actions_normalized,
            "planned_actions_env": self.planned_actions_env,
            "chunk_rewards_native": self.chunk_rewards_native,
        }
        for name, tensor in tensor_fields.items():
            _expect_cpu(tensor, name)
        if (
            self.obs_feature.ndim != 2
            or self.obs_feature.shape[0] != PREFIX_BLOCKS
            or self.obs_feature.shape[1] <= 0
        ):
            raise ValueError(
                "obs_feature must have shape [4, D] with D > 0, got "
                f"{tuple(self.obs_feature.shape)}"
            )
        _expect_shape(self.obs_proprio, (ACTIVE_ACTION_DIM,), "obs_proprio")
        _expect_shape(
            self.planned_actions_normalized,
            (PLANNED_HORIZON, ACTIVE_ACTION_DIM),
            "planned_actions_normalized",
        )
        _expect_shape(
            self.planned_actions_env,
            (PLANNED_HORIZON, ACTIVE_ACTION_DIM),
            "planned_actions_env",
        )
        _expect_shape(
            self.chunk_rewards_native,
            (PLANNED_HORIZON,),
            "chunk_rewards_native",
        )
        for name, tensor in tensor_fields.items():
            _expect_floating_finite(tensor, name)
        if self.planned_actions_normalized.abs().max().item() > 1.0 + 1e-6:
            raise ValueError("planned_actions_normalized violates canonical clamp")
        if not math.isfinite(float(self.reward_macro_discounted)):
            raise ValueError("reward_macro_discounted must be finite")

        if self.next_state_valid:
            if (
                self.next_obs_id is None
                or self.next_obs_feature is None
                or self.next_obs_proprio is None
            ):
                raise ValueError(
                    "next_state_valid requires next obs ID, feature, and proprio"
                )
            _expect_cpu(self.next_obs_feature, "next_obs_feature")
            _expect_cpu(self.next_obs_proprio, "next_obs_proprio")
            if self.next_obs_feature.shape != self.obs_feature.shape:
                raise ValueError(
                    "next_obs_feature must match obs_feature shape "
                    f"{tuple(self.obs_feature.shape)}, got "
                    f"{tuple(self.next_obs_feature.shape)}"
                )
            _expect_shape(
                self.next_obs_proprio,
                (ACTIVE_ACTION_DIM,),
                "next_obs_proprio",
            )
            _expect_floating_finite(
                self.next_obs_feature,
                "next_obs_feature",
            )
            _expect_floating_finite(
                self.next_obs_proprio,
                "next_obs_proprio",
            )
        elif any(
            value is not None
            for value in (
                self.next_obs_id,
                self.next_obs_feature,
                self.next_obs_proprio,
            )
        ):
            raise ValueError("invalid next state must not carry next-view fields")

        expected_mask = macro_bootstrap_mask(
            success_terminated=self.success_terminated,
            time_limit_truncated=self.time_limit_truncated,
            other_truncated=self.other_truncated,
            next_state_valid=self.next_state_valid,
        )
        if abs(float(self.bootstrap_mask) - expected_mask) > 1e-8:
            raise ValueError(
                f"bootstrap_mask must be {expected_mask}, got {self.bootstrap_mask}"
            )
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if self.query_index < 0:
            raise ValueError("query_index must be non-negative")
        if not self.obs_id or not self.episode_id or not self.contract_fingerprint:
            raise ValueError(
                "obs_id, episode_id, and contract_fingerprint are required"
            )
