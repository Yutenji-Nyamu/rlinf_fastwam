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

"""Strict RoboTwin observation adapter for Fast-WAM.

The image, state, action, and prompt transforms mirror the pinned official
Fast-WAM RoboTwin deploy policy. This module intentionally owns no rollout,
probability, reward, or action-queue behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

ACTION_DIM = 14
ACTION_HORIZON = 32
HEAD_SIZE_WH = (320, 256)
WRIST_SIZE_WH = (160, 128)
COMPOSITE_SHAPE_HWC = (384, 320, 3)

# Exact literal from Fast-WAM 45d8e145:
# src/fastwam/datasets/lerobot/robot_video_dataset.py::DEFAULT_PROMPT.
# Keeping the pinned literal here avoids importing the heavyweight dataset
# stack merely to use the lightweight adapter.
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


@dataclass(frozen=True)
class ValidatedRobotWinBatch:
    """Validated RLinf RoboTwin observation batch."""

    main_images: torch.Tensor
    wrist_images: torch.Tensor
    states: torch.Tensor
    task_descriptions: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        """Return the common observation batch size."""

        return int(self.main_images.shape[0])


def _require_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _validate_image_batches(
    main_images: torch.Tensor, wrist_images: torch.Tensor
) -> None:
    if main_images.dtype != torch.uint8:
        raise TypeError(f"main_images must have dtype uint8, got {main_images.dtype}")
    if main_images.ndim != 4 or main_images.shape[-1] != 3:
        raise ValueError(
            "main_images must have shape [B,H,W,3], "
            f"got {tuple(main_images.shape)}"
        )
    if main_images.shape[0] < 1 or min(main_images.shape[1:3]) < 1:
        raise ValueError("main_images must have non-empty batch and spatial dimensions")

    if wrist_images.dtype != torch.uint8:
        raise TypeError(
            f"wrist_images must have dtype uint8, got {wrist_images.dtype}"
        )
    if (
        wrist_images.ndim != 5
        or wrist_images.shape[1] != 2
        or wrist_images.shape[-1] != 3
    ):
        raise ValueError(
            "wrist_images must have shape [B,2,H,W,3] with left wrist at "
            f"index 0 and right wrist at index 1, got {tuple(wrist_images.shape)}"
        )
    if wrist_images.shape[0] < 1 or min(wrist_images.shape[2:4]) < 1:
        raise ValueError(
            "wrist_images must have non-empty batch and spatial dimensions"
        )
    if wrist_images.shape[0] != main_images.shape[0]:
        raise ValueError(
            "main_images and wrist_images batch sizes differ: "
            f"{main_images.shape[0]} != {wrist_images.shape[0]}"
        )


def _validate_states(states: torch.Tensor, batch_size: int) -> None:
    if not states.dtype.is_floating_point:
        raise TypeError(f"states must have floating dtype, got {states.dtype}")
    if states.ndim != 2 or tuple(states.shape) != (batch_size, ACTION_DIM):
        raise ValueError(
            f"states must have shape [B,{ACTION_DIM}], got {tuple(states.shape)}"
        )
    if not bool(torch.isfinite(states).all().item()):
        raise ValueError("states must contain only finite values")


def _validate_task_descriptions(
    task_descriptions: Any, batch_size: int
) -> tuple[str, ...]:
    if not isinstance(task_descriptions, list):
        raise TypeError(
            "task_descriptions must be a list[str], "
            f"got {type(task_descriptions).__name__}"
        )
    if len(task_descriptions) != batch_size:
        raise ValueError(
            "task_descriptions length must equal batch size: "
            f"{len(task_descriptions)} != {batch_size}"
        )
    for index, description in enumerate(task_descriptions):
        if not isinstance(description, str):
            raise TypeError(
                f"task_descriptions[{index}] must be str, "
                f"got {type(description).__name__}"
            )
        if not description.strip():
            raise ValueError(f"task_descriptions[{index}] must be non-empty")
    return tuple(task_descriptions)


def validate_env_obs(env_obs: Mapping[str, Any]) -> ValidatedRobotWinBatch:
    """Validate the exact RLinf RoboTwin batch contract.

    Args:
        env_obs: Mapping containing ``main_images``, ``wrist_images``,
            ``states``, and ``task_descriptions``.

    Returns:
        A typed view of the validated batch.

    Raises:
        KeyError: If a required field is missing.
        TypeError: If a field has the wrong container or tensor dtype.
        ValueError: If batch, shape, finite-value, or prompt checks fail.
    """

    if not isinstance(env_obs, Mapping):
        raise TypeError(f"env_obs must be a mapping, got {type(env_obs).__name__}")
    required = ("main_images", "wrist_images", "states", "task_descriptions")
    missing = [key for key in required if key not in env_obs]
    if missing:
        raise KeyError(f"env_obs is missing required fields: {missing}")

    main_images = _require_tensor("main_images", env_obs["main_images"])
    wrist_images = _require_tensor("wrist_images", env_obs["wrist_images"])
    states = _require_tensor("states", env_obs["states"])
    _validate_image_batches(main_images, wrist_images)
    batch_size = int(main_images.shape[0])
    _validate_states(states, batch_size)
    task_descriptions = _validate_task_descriptions(
        env_obs["task_descriptions"], batch_size
    )

    return ValidatedRobotWinBatch(
        main_images=main_images,
        wrist_images=wrist_images,
        states=states,
        task_descriptions=task_descriptions,
    )


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Mirror official deploy ``_resize_rgb`` exactly."""

    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def compose_three_camera_image(
    main_images: torch.Tensor,
    wrist_images: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compose head, left-wrist, and right-wrist images like official deploy.

    The CPU PIL loop is deliberate: replacing it with tensor interpolation would
    change the pinned deploy oracle.
    """

    main_images = _require_tensor("main_images", main_images)
    wrist_images = _require_tensor("wrist_images", wrist_images)
    _validate_image_batches(main_images, wrist_images)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError(f"dtype must be a floating torch.dtype, got {dtype!r}")

    main_numpy = main_images.detach().cpu().numpy()
    wrist_numpy = wrist_images.detach().cpu().numpy()
    composites: list[np.ndarray] = []
    for batch_index in range(main_numpy.shape[0]):
        head = _resize_rgb(main_numpy[batch_index], HEAD_SIZE_WH)
        left = _resize_rgb(wrist_numpy[batch_index, 0], WRIST_SIZE_WH)
        right = _resize_rgb(wrist_numpy[batch_index, 1], WRIST_SIZE_WH)
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)
        if image.shape != COMPOSITE_SHAPE_HWC:
            raise RuntimeError(
                "official RoboTwin camera composition produced unexpected shape "
                f"{image.shape}, expected {COMPOSITE_SHAPE_HWC}"
            )
        composites.append(image)

    image_tensor = torch.from_numpy(np.stack(composites, axis=0)).permute(0, 3, 1, 2)
    image_tensor = image_tensor.to(device=device, dtype=dtype)
    return image_tensor * (2.0 / 255.0) - 1.0


def _default_shape_meta_key(processor: Any, field: str) -> str:
    try:
        entries = processor.shape_meta[field]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"processor.shape_meta[{field!r}] is required") from exc
    if len(entries) != 1:
        raise ValueError(
            f"Expected exactly one merged {field} key in shape_meta[{field!r}]"
        )
    entry = entries[0]
    try:
        key = entry["key"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"shape_meta[{field!r}][0] must contain 'key'") from exc
    if key != "default":
        raise ValueError(
            f"Fast-WAM RoboTwin requires shape_meta[{field!r}] key 'default', "
            f"got {key!r}"
        )
    return key


def normalize_proprio(states: torch.Tensor, processor: Any) -> torch.Tensor:
    """Normalize a physical absolute-qpos batch with the official processor."""

    states = _require_tensor("states", states)
    if states.ndim != 2:
        raise ValueError(f"states must have shape [B,{ACTION_DIM}], got {states.shape}")
    _validate_states(states, int(states.shape[0]))
    state_key = _default_shape_meta_key(processor, "state")

    state_batch = {
        "state": {
            state_key: states.detach().to(device="cpu", dtype=torch.float32)
        }
    }
    try:
        state_batch = processor.action_state_transform(state_batch)
        state_batch = processor.normalizer.forward(state_batch)
        normalized = state_batch["state"][state_key]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "processor must expose official action_state_transform and "
            "normalizer.forward state semantics"
        ) from exc
    if not isinstance(normalized, torch.Tensor):
        raise TypeError("processor normalized state must be a torch.Tensor")
    if tuple(normalized.shape) != tuple(states.shape):
        raise ValueError(
            "processor changed normalized state shape: "
            f"{tuple(normalized.shape)} != {tuple(states.shape)}"
        )
    normalized = normalized.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(normalized).all().item()):
        raise ValueError("normalized proprio contains non-finite values")
    return normalized


def denormalize_actions(model_actions: torch.Tensor, processor: Any) -> torch.Tensor:
    """Convert normalized Fast-WAM actions to physical absolute qpos."""

    model_actions = _require_tensor("model_actions", model_actions)
    expected_tail = (ACTION_HORIZON, ACTION_DIM)
    if model_actions.ndim != 3 or tuple(model_actions.shape[1:]) != expected_tail:
        raise ValueError(
            "model_actions must have shape "
            f"[B,{ACTION_HORIZON},{ACTION_DIM}], got {tuple(model_actions.shape)}"
        )
    if not model_actions.dtype.is_floating_point:
        raise TypeError(
            f"model_actions must have floating dtype, got {model_actions.dtype}"
        )
    if not bool(torch.isfinite(model_actions).all().item()):
        raise ValueError("model_actions must contain only finite values")
    action_key = _default_shape_meta_key(processor, "action")

    try:
        normalizer = processor.normalizer.normalizers["action"][action_key]
        physical_actions = normalizer.backward(
            model_actions.detach().to(device="cpu", dtype=torch.float32)
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "processor must expose normalizer.normalizers['action']['default'].backward"
        ) from exc
    if not isinstance(physical_actions, torch.Tensor):
        raise TypeError("denormalized actions must be a torch.Tensor")
    if tuple(physical_actions.shape) != tuple(model_actions.shape):
        raise ValueError(
            "action normalizer changed action shape: "
            f"{tuple(physical_actions.shape)} != {tuple(model_actions.shape)}"
        )
    physical_actions = physical_actions.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    if not bool(torch.isfinite(physical_actions).all().item()):
        raise ValueError("denormalized actions contain non-finite values")
    return physical_actions


def build_prompts(task_descriptions: Sequence[str]) -> list[str]:
    """Wrap task descriptions with the pinned official Fast-WAM prompt."""

    if isinstance(task_descriptions, (str, bytes)) or not isinstance(
        task_descriptions, Sequence
    ):
        raise TypeError("task_descriptions must be a sequence of strings")
    if len(task_descriptions) < 1:
        raise ValueError("task_descriptions must be non-empty")
    prompts: list[str] = []
    for index, description in enumerate(task_descriptions):
        if not isinstance(description, str):
            raise TypeError(
                f"task_descriptions[{index}] must be str, "
                f"got {type(description).__name__}"
            )
        if not description.strip():
            raise ValueError(f"task_descriptions[{index}] must be non-empty")
        prompts.append(DEFAULT_PROMPT.format(task=description))
    return prompts


def adapt_robotwin_observation(
    env_obs: Mapping[str, Any],
    processor: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Validate and adapt one RLinf RoboTwin observation batch."""

    batch = validate_env_obs(env_obs)
    image = compose_three_camera_image(
        batch.main_images,
        batch.wrist_images,
        device=device,
        dtype=dtype,
    )
    proprio = normalize_proprio(batch.states, processor)
    prompts = build_prompts(batch.task_descriptions)
    return image, proprio, prompts


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "DEFAULT_PROMPT",
    "ValidatedRobotWinBatch",
    "adapt_robotwin_observation",
    "build_prompts",
    "compose_three_camera_image",
    "denormalize_actions",
    "normalize_proprio",
    "validate_env_obs",
]
