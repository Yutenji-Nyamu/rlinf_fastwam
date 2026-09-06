from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

import cv2 as cv
import numpy as np
import torch
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.modules.value_head import ValueHead


def _as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _to_hwc_uint8(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _resize_rgb(img: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Match LaWAM official RoboTwin adapter image resize semantics."""
    return cv.resize(np.asarray(img, dtype=np.uint8), tuple(size_wh), interpolation=cv.INTER_AREA)


class LaWAMPolicy(nn.Module, BasePolicy):
    """RLinf policy wrapper for LaWAM RoboTwin PPO/eval.

    The wrapper owns the RL interface (rollout trace, logprob replay, ValueHead,
    checkpoint/FSDP visibility). LaWAM model internals are loaded from a vendored
    or external LaWAM tree and are exposed as ``self.policy``.
    """

    def __init__(self, cfg, torch_dtype=None):
        super().__init__()
        self.cfg = cfg
        self.torch_dtype = torch_dtype
        self.model_type = "lawam"

        lawam_cfg = cfg.get("lawam", {})
        self.repo_path = Path(str(lawam_cfg.get("repo_path", "/root/autodl-tmp/RLinf/third_party/LaWAM"))).expanduser()
        self.policy_ckpt_path = Path(str(lawam_cfg.get("policy_ckpt_path", cfg.model_path))).expanduser()
        if not self.repo_path.exists():
            raise FileNotFoundError(f"LaWAM repo_path not found: {self.repo_path}")
        if not self.policy_ckpt_path.exists():
            raise FileNotFoundError(f"LaWAM policy_ckpt_path/model_path not found: {self.policy_ckpt_path}")

        self._insert_lawam_paths()
        self._import_lawam_symbols()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = cfg.get("precision", "bf16")
        self.cast_dtype = self._resolve_precision_dtype(self.precision)

        self.num_action_chunks = int(cfg.get("num_action_chunks", lawam_cfg.get("replan_steps", 8)))
        self.action_dim = int(cfg.get("action_dim", lawam_cfg.get("env_action_dim", 16)))
        self.internal_action_horizon = int(lawam_cfg.get("internal_action_horizon", 50))
        self.internal_action_dim = int(lawam_cfg.get("internal_action_dim", 32))
        self.num_inference_steps = int(lawam_cfg.get("num_inference_steps", cfg.get("num_steps", 10)))
        self.allow_batch_size = int(lawam_cfg.get("allow_batch_size", 8))
        self.batch_inference = _as_bool(lawam_cfg.get("batch_inference", True), default=True)
        self.vlm_max_seq_len = int(lawam_cfg.get("vlm_max_seq_len", 512))
        if self.vlm_max_seq_len <= 0:
            raise ValueError(f"lawam.vlm_max_seq_len must be positive, got {self.vlm_max_seq_len}")

        self.noise_method = str(lawam_cfg.get("noise_method", "flow_sde")).lower()
        self.noise_level = float(lawam_cfg.get("noise_level", lawam_cfg.get("logprob_sigma", 0.5)))
        self.collect_denoise_step = str(lawam_cfg.get("collect_denoise_step", "random")).lower()
        self.ignore_first = _as_bool(lawam_cfg.get("ignore_first", True), default=True)
        self.ignore_last = _as_bool(lawam_cfg.get("ignore_last", True), default=True)
        self.guidance_scale = lawam_cfg.get("guidance_scale", None)
        self.guidance_scale = None if self.guidance_scale in [None, "null", "None"] else float(self.guidance_scale)

        self.add_value_head = _as_bool(cfg.get("add_value_head", False), default=False)
        self.detach_critic_input = _as_bool(lawam_cfg.get("detach_critic_input", True), default=True)
        self.value_after_vlm = _as_bool(lawam_cfg.get("value_after_vlm", True), default=True)
        self.value_feature = str(lawam_cfg.get("value_feature", "flow_query_tokens")).lower()
        self.value_vlm_mode = str(lawam_cfg.get("value_vlm_mode", "mean_query")).lower()
        if self.value_feature not in {"flow_query_tokens", "act_query_tokens", "masked_vlm_mean"}:
            raise ValueError(
                f"Unsupported lawam.value_feature={self.value_feature!r}. "
                "Use flow_query_tokens, act_query_tokens, or masked_vlm_mean."
            )

        self.trainable = str(lawam_cfg.get("trainable", "flow")).lower()
        self.train_flow_action_query = _as_bool(lawam_cfg.get("train_flow_action_query", False), default=False)
        self.freeze_vlm = _as_bool(lawam_cfg.get("freeze_vlm", True), default=True)
        self.freeze_lam = _as_bool(lawam_cfg.get("freeze_lam", True), default=True)
        self.freeze_dino = _as_bool(lawam_cfg.get("freeze_dino", True), default=True)
        self.train_vlm_to_lam = _as_bool(lawam_cfg.get("train_vlm_to_lam", False), default=False)

        self.model_config, self.norm_stats = self.read_mode_config(str(self.policy_ckpt_path))
        self.data_mix = str(lawam_cfg.get("data_mix", self._extract_robotwin_data_mix(self.model_config)))
        self.control_spec = self.resolve_robotwin_control_from_data_mix(self.data_mix)
        self.robotwin_mode = self.control_spec.mode
        self.env_action_type = self.control_spec.env_action_type
        self.action_hz = float(lawam_cfg.get("action_hz", self.control_spec.action_hz))
        self.state_gripper_indices = self.control_spec.state_gripper_indices
        self.passthrough_indices = self.control_spec.passthrough_indices
        self.gripper_is_binary = bool(self.control_spec.gripper_is_binary)
        self.unnorm_key = str(lawam_cfg.get("unnorm_key", "")) or self._check_unnorm_key(self.norm_stats, None)
        if self.unnorm_key not in self.norm_stats:
            self.unnorm_key = self._check_unnorm_key(self.norm_stats, None)
        self.dataset_stats = self.norm_stats[self.unnorm_key]
        self.action_norm_stats = self.dataset_stats["action"]
        self.state_norm_stats = self.dataset_stats.get("state", None)
        self.action_binary_indices = self.LocalStarVLARobotwinPolicy._resolve_action_binary_indices(
            self.action_norm_stats,
            self.robotwin_mode,
            self.state_gripper_indices,
            gripper_is_binary=self.gripper_is_binary,
        )
        self.state_binary_indices = self.LocalStarVLARobotwinPolicy._resolve_state_binary_indices(
            self.state_norm_stats,
            self.robotwin_mode,
            self.state_gripper_indices,
            gripper_is_binary=self.gripper_is_binary,
        )
        self.state_invert_indices = self.LocalStarVLARobotwinPolicy._resolve_state_invert_indices(
            self.robotwin_mode,
            self.state_gripper_indices,
        )
        self.action_invert_indices = self.LocalStarVLARobotwinPolicy._resolve_action_invert_indices(
            self.robotwin_mode,
            self.state_gripper_indices,
        )
        self.image_size = tuple(int(x) for x in lawam_cfg.get("image_size", [256, 256]))
        if len(self.image_size) != 2:
            raise ValueError(f"lawam.image_size must have length 2, got {self.image_size}")

        self.policy = self.baseframework.from_pretrained(str(self.policy_ckpt_path))
        if self.cast_dtype is not None:
            self.policy = self.policy.to(dtype=self.cast_dtype)
        self.policy = self.policy.to(self.device)
        self.policy.eval()
        self.policy_backend = self.policy.policy_backend
        self.flow = self.policy_backend.flow
        self.use_state = bool(getattr(self.flow.config, "use_state", False))
        self._action_debug_count = 0
        self.model = self.policy_backend  # convenient alias for FSDP/optimizer/debug

        if self.add_value_head:
            value_dim = self._get_value_input_dim()
            self.value_head = ValueHead(
                input_dim=value_dim,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            ).to(self.device)
            print(
                f"[RLinf-LaWAM] value_head feature={self.value_feature}, input_dim={value_dim}, "
                f"value_after_vlm={self.value_after_vlm}",
                flush=True,
            )

        self._configure_trainable_parameters()
        self._print_parameter_summary()

    def _insert_lawam_paths(self) -> None:
        root = str(self.repo_path.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    def _import_lawam_symbols(self) -> None:
        from examples.Robotwin.starvla_policy.deploy_policy import (  # type: ignore
            LocalStarVLARobotwinPolicy,
            build_robotwin_example,
            resolve_robotwin_control_from_data_mix,
            _extract_robotwin_data_mix,
        )
        from starVLA.model.framework.base_framework import baseframework  # type: ignore
        from starVLA.model.framework.vlas.flowmatching_expert import build_time_grid  # type: ignore
        from starVLA.model.tools import read_mode_config  # type: ignore

        self.LocalStarVLARobotwinPolicy = LocalStarVLARobotwinPolicy
        self.build_robotwin_example = build_robotwin_example
        self.resolve_robotwin_control_from_data_mix = resolve_robotwin_control_from_data_mix
        self._extract_robotwin_data_mix = _extract_robotwin_data_mix
        self.baseframework = baseframework
        self.build_time_grid = build_time_grid
        self.read_mode_config = read_mode_config

    @staticmethod
    def _resolve_precision_dtype(precision: Any) -> Optional[torch.dtype]:
        p = str(precision or "bf16").lower()
        if p in {"bf16", "bf16-mixed"}:
            return torch.bfloat16
        if p in {"fp16", "16", "16-mixed"}:
            return torch.float16
        if p in {"fp32", "32", "32-true"}:
            return torch.float32
        if p in {"none", "null"}:
            return None
        return torch.bfloat16

    @staticmethod
    def _check_unnorm_key(norm_stats: dict, unnorm_key: Optional[str]) -> str:
        if unnorm_key is None or unnorm_key not in norm_stats:
            return next(iter(norm_stats.keys()))
        return str(unnorm_key)

    def _get_value_input_dim(self) -> int:
        cfg = getattr(self.policy_backend.vlm, "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        hidden = getattr(text_cfg, "hidden_size", None)
        if hidden is None:
            hidden = getattr(cfg, "hidden_size", None)
        if hidden is None:
            hidden = int(getattr(self.flow.config, "vlm_dim", 2048))
        return int(hidden)

    def _configure_trainable_parameters(self) -> None:
        for p in self.policy.parameters():
            p.requires_grad_(False)
        if hasattr(self, "value_head"):
            for p in self.value_head.parameters():
                p.requires_grad_(True)
        if self.trainable in {"none", "eval", "false"}:
            return
        if self.trainable not in {"flow", "flow_query", "all"}:
            raise ValueError("lawam.trainable must be one of: none, flow, flow_query, all")
        if self.trainable == "all":
            for p in self.policy.parameters():
                p.requires_grad_(True)
            return
        for p in self.flow.parameters():
            p.requires_grad_(True)
        if self.trainable == "flow_query" or self.train_flow_action_query:
            self.policy_backend.flow_action_query.requires_grad_(True)
        if self.train_vlm_to_lam:
            for p in self.policy_backend.vlm_to_lam.parameters():
                p.requires_grad_(True)

    def _print_parameter_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[RLinf-LaWAM] trainable={self.trainable}, total_params={total/1e6:.2f}M, "
            f"trainable_params={trainable/1e6:.2f}M",
            flush=True,
        )

    def get_fsdp_ignored_modules(self):
        modules = []
        if self.freeze_vlm and hasattr(self.policy_backend, "vlm"):
            modules.append(self.policy_backend.vlm)
        if self.freeze_lam and hasattr(self.policy_backend, "lam"):
            modules.append(self.policy_backend.lam)
        out, seen = [], set()
        for module in modules:
            if module is not None and id(module) not in seen:
                seen.add(id(module))
                out.append(module)
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        if not hasattr(self, "policy_backend"):
            return self
        if not mode:
            self.policy.eval()
            return self
        self.policy.train()
        if self.freeze_vlm and hasattr(self.policy_backend, "vlm"):
            self.policy_backend.vlm.eval()
        if self.freeze_lam and hasattr(self.policy_backend, "lam"):
            self.policy_backend.lam.eval()
        return self

    def eval(self):
        super().eval()
        if hasattr(self, "policy"):
            self.policy.eval()
        return self

    def forward(self, *args, **kwargs):
        return self.default_forward(*args, **kwargs)

    def default_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs) -> dict[str, Any]:
        if forward_inputs is None:
            raise ValueError("LaWAM default_forward requires forward_inputs.")
        device = next(self.parameters()).device
        batch = self._batch_from_forward_inputs(forward_inputs, device=device)
        action_chains = forward_inputs["lawam_action_chains"].to(device=device, dtype=self._flow_dtype())
        denoise_inds = forward_inputs["lawam_denoise_inds"].to(device=device, dtype=torch.long).view(-1)
        logprobs, values, entropy = self.get_log_prob_value(
            batch=batch,
            action_chains=action_chains,
            denoise_inds=denoise_inds,
            compute_values=self.add_value_head,
            compute_entropy=bool(kwargs.get("compute_entropy", False)),
        )
        return {"logprobs": logprobs.float(), "values": values, "entropy": entropy}

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict[str, Any], mode: str = "eval", **kwargs):
        batch_size = self._infer_batch_size(env_obs)
        if batch_size > self.allow_batch_size:
            raise ValueError(
                f"LaWAM adapter supports batch_size <= {self.allow_batch_size}, got {batch_size}."
            )
        examples = self._build_examples_from_env_obs(env_obs, batch_size)
        if mode == "train":
            return self._predict_train(examples, batch_size)
        return self._predict_eval(examples, batch_size)

    def _infer_batch_size(self, env_obs: dict[str, Any]) -> int:
        for key in ("states", "main_images", "task_descriptions"):
            value = env_obs.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
            if isinstance(value, np.ndarray):
                return int(value.shape[0])
            if isinstance(value, list):
                return len(value)
        raise ValueError(f"Cannot infer batch size from env_obs keys={list(env_obs.keys())}")

    @staticmethod
    def _take_batch(value: Any, idx: int) -> Any:
        if isinstance(value, torch.Tensor):
            return value[idx]
        if isinstance(value, np.ndarray):
            return value[idx]
        if isinstance(value, list):
            return value[idx]
        return value

    def _get_instruction(self, env_obs: dict[str, Any], idx: int) -> str:
        desc = env_obs.get("task_descriptions", "")
        if isinstance(desc, str):
            return desc
        if isinstance(desc, list):
            return str(desc[idx])
        return str(self._take_batch(desc, idx))

    def _extract_wrist_pair(self, env_obs: dict[str, Any], idx: int) -> tuple[np.ndarray, np.ndarray]:
        wrist = env_obs.get("wrist_images", None)
        if wrist is None:
            raise ValueError("LaWAM RoboTwin requires wrist_images; set collect_wrist_camera=true.")
        wrist_i = self._take_batch(wrist, idx)
        if isinstance(wrist_i, (list, tuple)):
            if len(wrist_i) == 0:
                raise ValueError("wrist_images list is empty")
            left = wrist_i[0]
            right = wrist_i[1] if len(wrist_i) > 1 else wrist_i[0]
            return _to_hwc_uint8(left), _to_hwc_uint8(right)
        arr = _to_numpy(wrist_i)
        if arr.ndim == 4:
            left = arr[0]
            right = arr[1] if arr.shape[0] > 1 else arr[0]
            return _to_hwc_uint8(left), _to_hwc_uint8(right)
        if arr.ndim == 3:
            img = _to_hwc_uint8(arr)
            return img, img
        raise ValueError(f"Unsupported wrist_images shape after batch indexing: {arr.shape}")

    def _build_examples_from_env_obs(self, env_obs: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
        """Convert RLinf batched obs to LaWAM infer examples via official adapter semantics.

        Official LaWAM RoboTwin inference first reconstructs a Robotwin observation,
        calls ``build_robotwin_example()``, then applies ``_prepare_example`` and
        ``_build_infer_example``.  Keep RLinf train/eval on the same path so image
        order, resize, state/endpose handling, and prompt construction stay aligned
        with the successful native eval pipeline.
        """
        examples: list[dict[str, Any]] = []
        for i in range(batch_size):
            official_obs = self._build_official_robotwin_observation(env_obs, i)
            robotwin_example = self.build_robotwin_example(
                self._get_instruction(env_obs, i),
                official_obs,
                robotwin_mode=self.robotwin_mode,
            )
            prepared = self._prepare_official_example(robotwin_example)
            examples.append(self._build_infer_example_from_prepared(prepared))
        return examples

    def _build_official_robotwin_observation(self, env_obs: dict[str, Any], idx: int) -> dict[str, Any]:
        main = _to_hwc_uint8(self._take_batch(env_obs["main_images"], idx))
        left, right = self._extract_wrist_pair(env_obs, idx)
        obs: dict[str, Any] = {
            "observation": {
                "head_camera": {"rgb": main},
                "left_camera": {"rgb": left},
                "right_camera": {"rgb": right},
            },
            "joint_action": {},
            "endpose": {},
        }
        if "states" in env_obs and env_obs.get("states") is not None:
            obs["joint_action"]["vector"] = _to_numpy(self._take_batch(env_obs["states"], idx)).astype(np.float32).reshape(-1)
        if "endpose_states" in env_obs and env_obs.get("endpose_states") is not None:
            endpose_state = _to_numpy(self._take_batch(env_obs["endpose_states"], idx)).astype(np.float32).reshape(-1)
            if endpose_state.shape[0] >= 16:
                obs["endpose"] = {
                    "left_endpose": endpose_state[:7],
                    "left_gripper": float(endpose_state[7]),
                    "right_endpose": endpose_state[8:15],
                    "right_gripper": float(endpose_state[15]),
                }
        return obs

    def _prepare_official_state(self, state: Any) -> Optional[np.ndarray]:
        if state is None or not self.use_state:
            return None
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim == 2 and state_array.shape[0] == 1:
            state_array = state_array[0]
        if state_array.ndim != 1:
            raise ValueError(f"`state` must have shape [D], got {state_array.shape}.")
        if self.state_norm_stats is None:
            return state_array
        return self.LocalStarVLARobotwinPolicy.normalize_state(
            state=state_array,
            state_norm_stats=self.state_norm_stats,
            binary_indices=self.state_binary_indices,
            passthrough_indices=self.passthrough_indices,
            invert_indices=self.state_invert_indices,
        )

    def _prepare_official_example(self, example: dict[str, Any]) -> dict[str, Any]:
        images = example.get("image", None)
        if not isinstance(images, (list, tuple)) or len(images) == 0:
            raise ValueError("Robotwin example must contain non-empty `image` list.")
        return {
            "lang": str(example.get("lang", "")),
            "image": [_resize_rgb(np.asarray(image), (int(self.image_size[0]), int(self.image_size[1]))) for image in images],
            "state": self._prepare_official_state(example.get("state", None)),
        }

    def _build_infer_example_from_prepared(self, example: dict[str, Any]) -> dict[str, Any]:
        images = list(example["image"])
        infer_example: dict[str, Any] = {
            "lang": str(example["lang"]),
            "primary_image": [images[0]],
            "embodiment_id": 1,
            "action_hz": float(self.action_hz),
        }
        wrist_images = list(images[1:])
        if wrist_images:
            infer_example["wrist_image"] = wrist_images
        if example.get("state", None) is not None:
            infer_example["state"] = np.asarray(example["state"], dtype=np.float32)
        return infer_example

    def _build_batch(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.policy.policy_infer_batch_builder.build_infer_batch(examples)
        return self._pad_token_fields(batch)

    def _pad_token_fields(self, batch: dict[str, Any]) -> dict[str, Any]:
        keys = ["input_ids", "attention_mask", "act_placeholder_mask", "flow_placeholder_mask"]
        cur_len = int(batch["input_ids"].shape[1])
        if cur_len > self.vlm_max_seq_len:
            raise ValueError(
                f"LaWAM VLM input length {cur_len} exceeds vlm_max_seq_len={self.vlm_max_seq_len}."
            )
        if cur_len == self.vlm_max_seq_len:
            return batch
        pad = self.vlm_max_seq_len - cur_len
        for key in keys:
            value = batch[key]
            if key == "input_ids":
                pad_value = int(getattr(self.policy_backend.tokenizer, "pad_token_id", 0) or 0)
            else:
                pad_value = 0
            batch[key] = torch.nn.functional.pad(value, (0, pad), value=pad_value)
        return batch

    def _flow_dtype(self) -> torch.dtype:
        try:
            return next(self.flow.parameters()).dtype
        except Exception:
            return torch.float32

    def _predict_eval(self, examples: list[dict[str, Any]], batch_size: int):
        predict_kwargs = {"examples": examples}
        if self.guidance_scale is not None:
            predict_kwargs["guidance_scale"] = float(self.guidance_scale)
        predict_kwargs["num_inference_steps"] = int(self.num_inference_steps)
        with torch.inference_mode():
            output = self.policy.predict_action(**predict_kwargs)
        normalized = np.asarray(output["normalized_actions"], dtype=np.float32)
        if normalized.ndim != 3:
            raise ValueError(f"LaWAM eval normalized_actions must be [B,T,D], got {normalized.shape}")
        actions = self._unnormalize_and_truncate(normalized)
        if actions.shape[0] != batch_size:
            raise ValueError(f"LaWAM eval batch mismatch: expected {batch_size}, got {actions.shape[0]}")
        return torch.from_numpy(actions).float().cpu().contiguous(), {"forward_inputs": {}}

    def _unnormalize_and_truncate(self, normalized: np.ndarray) -> np.ndarray:
        chunks = []
        for i in range(int(normalized.shape[0])):
            raw = self.LocalStarVLARobotwinPolicy.unnormalize_actions(
                normalized_actions=normalized[i],
                action_norm_stats=self.action_norm_stats,
                gripper_indices=self.action_binary_indices,
                passthrough_indices=self.passthrough_indices,
                invert_indices=self.action_invert_indices,
            )
            raw = raw[: self.num_action_chunks, : self.action_dim]
            self._maybe_print_action_debug(raw, sample_idx=i)
            if raw.shape != (self.num_action_chunks, self.action_dim):
                raise ValueError(
                    f"LaWAM env action shape mismatch: expected {(self.num_action_chunks, self.action_dim)}, got {raw.shape}"
                )
            chunks.append(raw.astype(np.float32))
        return np.stack(chunks, axis=0)


    def _maybe_print_action_debug(self, raw: np.ndarray, *, sample_idx: int) -> None:
        if not _as_bool(os.environ.get("LAWAM_ACTION_DEBUG", "0"), default=False):
            return
        limit = int(os.environ.get("LAWAM_ACTION_DEBUG_LIMIT", "8"))
        if self._action_debug_count >= limit:
            return
        self._action_debug_count += 1
        arr = np.asarray(raw, dtype=np.float32)
        parts = [
            f"sample={sample_idx}",
            f"shape={tuple(arr.shape)}",
            f"min={float(np.nanmin(arr)):.4f}",
            f"max={float(np.nanmax(arr)):.4f}",
        ]
        if arr.ndim == 2 and arr.shape[1] >= 16:
            left_q = arr[:, 3:7]
            right_q = arr[:, 11:15]
            parts.extend([
                f"left_xyz=({float(np.nanmin(arr[:,0:3])):.4f},{float(np.nanmax(arr[:,0:3])):.4f})",
                f"right_xyz=({float(np.nanmin(arr[:,8:11])):.4f},{float(np.nanmax(arr[:,8:11])):.4f})",
                f"left_q_norm=({float(np.nanmin(np.linalg.norm(left_q, axis=1))):.4f},{float(np.nanmax(np.linalg.norm(left_q, axis=1))):.4f})",
                f"right_q_norm=({float(np.nanmin(np.linalg.norm(right_q, axis=1))):.4f},{float(np.nanmax(np.linalg.norm(right_q, axis=1))):.4f})",
                f"gripper=({float(np.nanmin(arr[:,[7,15]])):.4f},{float(np.nanmax(arr[:,[7,15]])):.4f})",
            ])
        print("[RLinf-LaWAM][action_debug] " + " | ".join(parts), flush=True)

    def _predict_train(self, examples: list[dict[str, Any]], batch_size: int):
        batch = self._build_batch(examples)
        action_chains, denoise_inds, shared = self.sample_actions_with_trace(batch)
        normalized = action_chains[:, -1].detach().float().cpu().numpy()
        actions = self._unnormalize_and_truncate(normalized)
        action_tensor = torch.from_numpy(actions).float().cpu().contiguous()

        prev_logprobs, prev_values, _ = self.get_log_prob_value(
            batch=batch,
            action_chains=action_chains,
            denoise_inds=denoise_inds,
            shared=shared,
            compute_values=self.add_value_head,
            compute_entropy=False,
        )

        forward_inputs = self._forward_inputs_from_batch(
            batch=batch,
            action_chains=action_chains,
            denoise_inds=denoise_inds,
            env_action=action_tensor,
            model_action=action_chains[:, -1].detach(),
        )
        return action_tensor, {
            "prev_logprobs": prev_logprobs.detach().float().cpu().contiguous(),
            "prev_values": None if prev_values is None else prev_values.detach().float().cpu().contiguous(),
            "forward_inputs": forward_inputs,
        }

    def _select_trace_step(self, num_steps: int, device: torch.device, batch_size: int) -> torch.Tensor:
        low = 1 if self.ignore_first and num_steps > 2 else 0
        high_exclusive = num_steps - 1 if self.ignore_last and num_steps > 2 else num_steps
        high_exclusive = max(low + 1, high_exclusive)
        if self.collect_denoise_step == "random":
            return torch.randint(low=low, high=high_exclusive, size=(batch_size,), device=device)
        k = int(self.collect_denoise_step)
        k = max(low, min(k, high_exclusive - 1))
        return torch.full((batch_size,), k, dtype=torch.long, device=device)

    def _run_shared_encoding(self, batch: dict[str, torch.Tensor]):
        return self.policy_backend._run_shared_encoding_infer(
            prepared_batch=batch,
            source="RLinf-LaWAM",
            lam_features_with_no_grad=bool(self.freeze_lam),
        )

    def sample_actions_with_trace(self, batch: dict[str, torch.Tensor]):
        flow = self.flow
        device = batch["input_ids"].device
        model_dtype = self._flow_dtype()
        shared = self._run_shared_encoding(batch)
        h_t = shared.h_t.to(dtype=model_dtype)
        h_t1 = shared.h_t1_pred.to(dtype=model_dtype)
        h_vlm = shared.h_vlm.to(dtype=model_dtype)
        state = batch["state"].to(dtype=model_dtype)
        state_mask = batch["state_mask"]
        action_hz = batch["action_hz"]
        embodiment_id = batch["embodiment_id"]
        attention_mask = batch["attention_mask"] == 1
        batch_size = int(h_t.shape[0])
        action_horizon = int(getattr(flow, "action_horizon", self.internal_action_horizon) or self.internal_action_horizon)
        action_dim = int(flow.config.action_dim)
        t_grid, time_valid, _ = self.build_time_grid(
            horizon_sec=float(flow.config.horizon_sec),
            hz=action_hz.to(device=device, dtype=torch.float32),
            seq_len=action_horizon,
        )
        x_t = flow.sample_noise(
            shape=(batch_size, action_horizon, action_dim),
            device=device,
            dtype=model_dtype,
        )
        x_t = x_t * time_valid.unsqueeze(-1).to(dtype=x_t.dtype)
        denoise_inds = self._select_trace_step(self.num_inference_steps, device, batch_size)
        chains = [x_t.detach().clone()]
        dt = 1.0 / float(self.num_inference_steps)
        for step in range(self.num_inference_steps):
            tau = torch.full((batch_size,), step / float(self.num_inference_steps), device=device, dtype=model_dtype)
            velocity = self._flow_velocity(
                h_t=h_t,
                h_t1_star=h_t1,
                h_vlm=h_vlm,
                state=state,
                state_mask=state_mask,
                action_hz=action_hz,
                embodiment_id=embodiment_id,
                x_t=x_t,
                tau=tau,
                t_grid=t_grid,
                time_valid=time_valid,
                attention_mask=attention_mask,
            )
            mean_ode = x_t + dt * velocity
            if self.noise_method in {"flow_sde", "transition_sde"}:
                mean_sde, std_sde = self._flow_sde_mean_std_lawam(x_t, velocity, tau, dt)
                eps = torch.randn_like(mean_sde)
                proposal = mean_sde + std_sde * eps
                use_sde = denoise_inds == int(step)
                while use_sde.dim() < proposal.dim():
                    use_sde = use_sde.unsqueeze(-1)
                x_next = torch.where(use_sde, proposal, mean_ode)
            else:
                x_next = mean_ode
            x_t = x_next * time_valid.unsqueeze(-1).to(dtype=x_next.dtype)
            chains.append(x_t.detach().clone())
        return torch.stack(chains, dim=1), denoise_inds, shared

    def _flow_velocity(
        self,
        *,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        action_hz: torch.Tensor,
        embodiment_id: torch.Tensor,
        x_t: torch.Tensor,
        tau: torch.Tensor,
        t_grid: torch.Tensor,
        time_valid: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        flow = self.flow
        device = x_t.device
        model_dtype = self._flow_dtype()
        x_t = x_t.to(dtype=model_dtype)
        h_t = h_t.to(dtype=model_dtype)
        h_t1_star = h_t1_star.to(dtype=model_dtype)
        h_vlm = h_vlm.to(dtype=model_dtype)
        batch_size, action_horizon, _ = x_t.shape
        tau_bucket = torch.clamp(
            (tau.float() * int(flow.config.num_timestep_buckets)).long(),
            min=0,
            max=int(flow.config.num_timestep_buckets) - 1,
        )
        action_features = flow.action_encoder(x_t, embodiment_id)
        time_emb = flow.time_encoder(t_grid).to(dtype=action_features.dtype)
        action_time_emb = time_emb * time_valid.unsqueeze(-1).to(dtype=action_features.dtype)
        action_features = action_features * time_valid.unsqueeze(-1).to(dtype=action_features.dtype)
        cond_vlm = flow.enc_vlm(h_vlm)
        cond_state = flow._prepare_state_condition(
            state=state,
            state_mask=state_mask,
            embodiment_id=embodiment_id,
            model_dtype=model_dtype,
        )
        cond_encoder_hidden = torch.cat((h_t, h_t1_star, cond_vlm), dim=1)
        num_vision = h_t.shape[1] + h_t1_star.shape[1]
        if attention_mask is not None:
            vlm_mask_bool = attention_mask.to(device=device, dtype=torch.bool)
            vision_mask_bool = torch.ones(batch_size, num_vision, dtype=torch.bool, device=device)
            encoder_attention_mask = torch.cat([vision_mask_bool, vlm_mask_bool], dim=1)
        else:
            encoder_attention_mask = None
        future_tokens, future_token_valid = flow._expand_future_tokens(
            batch_size=batch_size,
            device=device,
            dtype=model_dtype,
        )
        future_token_count = 0 if future_tokens is None else int(future_tokens.shape[1])
        hidden_positional_embeddings = None
        if flow.config.use_action_positional_embeddings:
            hidden_positional_embeddings = flow._build_hidden_positional_embeddings(
                action_time_emb=action_time_emb,
                batch_size=batch_size,
                device=device,
                dtype=action_features.dtype,
                has_state_token=bool(flow.config.use_state),
                future_token_count=future_token_count,
            )
        if flow.config.use_state:
            state_token_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            if future_tokens is not None and future_token_valid is not None:
                hidden_states = torch.cat((cond_state, future_tokens, action_features), dim=1)
                hidden_attention_mask = torch.cat([state_token_valid, future_token_valid, time_valid], dim=1)
            else:
                hidden_states = torch.cat((cond_state, action_features), dim=1)
                hidden_attention_mask = torch.cat([state_token_valid, time_valid], dim=1)
        else:
            if future_tokens is not None and future_token_valid is not None:
                hidden_states = torch.cat((future_tokens, action_features), dim=1)
                hidden_attention_mask = torch.cat([future_token_valid, time_valid], dim=1)
            else:
                hidden_states = action_features
                hidden_attention_mask = time_valid
        if flow.config.use_alternate_vldit:
            num_h_t = h_t.shape[1]
            num_h_t1 = h_t1_star.shape[1]
            num_vlm = cond_vlm.shape[1]
            image_mask = torch.cat([
                torch.ones(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.zeros(batch_size, num_vlm, dtype=torch.bool, device=device),
            ], dim=1)
            vlm_mask = torch.cat([
                torch.zeros(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.ones(batch_size, num_vlm, dtype=torch.bool, device=device),
            ], dim=1)
            model_output = flow.DiT(
                hidden_states=hidden_states,
                encoder_hidden_states=cond_encoder_hidden,
                timestep=tau_bucket,
                hidden_attention_mask=hidden_attention_mask,
                image_mask=image_mask,
                vlm_mask=vlm_mask,
                encoder_attention_mask=encoder_attention_mask,
                hidden_positional_embeddings=hidden_positional_embeddings,
            )
        else:
            model_output = flow.DiT(
                hidden_states=hidden_states,
                encoder_hidden_states=cond_encoder_hidden,
                timestep=tau_bucket,
                hidden_attention_mask=hidden_attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                hidden_positional_embeddings=hidden_positional_embeddings,
            )
        pred_velocity_all = flow.action_decoder(model_output, embodiment_id)
        pred_velocity = pred_velocity_all[:, -action_horizon:, :]
        return pred_velocity * time_valid.unsqueeze(-1).to(dtype=pred_velocity.dtype)

    def _flow_sde_mean_std_lawam(self, x_t: torch.Tensor, velocity: torch.Tensor, tau: torch.Tensor, dt: float):
        # LaWAM integrates tau: 0(noise) -> 1(action).  This is the OpenPI
        # flow_sde branch under t_pi = 1 - tau, expressed in LaWAM coordinates.
        while tau.dim() < x_t.dim():
            tau = tau.unsqueeze(-1)
        delta = torch.as_tensor(float(dt), device=x_t.device, dtype=torch.float32)
        t_pi = (1.0 - tau.float()).clamp(1e-6, 1.0)
        noise_pred = x_t.float() - velocity.float() * tau.float()
        action_pred = x_t.float() + velocity.float() * (1.0 - tau.float())
        t_next = (t_pi - delta).clamp_min(1e-6)
        denom = torch.where(t_pi >= 1.0 - 1e-6, t_next, t_pi).clamp_min(1e-6)
        sigma_ratio = t_pi / (1.0 - denom).clamp_min(1e-6)
        sigma_i = float(self.noise_level) * torch.sqrt(sigma_ratio)
        action_weight = 1.0 - (t_pi - delta)
        noise_weight = (t_pi - delta) - sigma_i.pow(2) * delta / (2.0 * t_pi.clamp_min(1e-6))
        mean = action_pred * action_weight + noise_pred * noise_weight
        std = (torch.sqrt(delta) * sigma_i).clamp_min(1e-6)
        return mean.to(dtype=x_t.dtype), std.to(dtype=x_t.dtype)

    def get_log_prob_value(
        self,
        *,
        batch: dict[str, torch.Tensor],
        action_chains: torch.Tensor,
        denoise_inds: torch.Tensor,
        shared: Any | None = None,
        compute_values: bool = True,
        compute_entropy: bool = False,
    ):
        if shared is None:
            shared = self._run_shared_encoding(batch)
        device = action_chains.device
        bsz = int(action_chains.shape[0])
        gather = denoise_inds.view(bsz, 1, 1, 1).expand(-1, 1, action_chains.shape[2], action_chains.shape[3])
        x_t = action_chains.gather(1, gather).squeeze(1)
        x_next = action_chains.gather(1, gather + 1).squeeze(1)
        tau = denoise_inds.to(device=device, dtype=self._flow_dtype()) / float(self.num_inference_steps)
        t_grid, time_valid, _ = self.build_time_grid(
            horizon_sec=float(self.flow.config.horizon_sec),
            hz=batch["action_hz"].to(device=device, dtype=torch.float32),
            seq_len=int(x_t.shape[1]),
        )
        velocity = self._flow_velocity(
            h_t=shared.h_t,
            h_t1_star=shared.h_t1_pred,
            h_vlm=shared.h_vlm,
            state=batch["state"],
            state_mask=batch["state_mask"],
            action_hz=batch["action_hz"],
            embodiment_id=batch["embodiment_id"],
            x_t=x_t,
            tau=tau,
            t_grid=t_grid,
            time_valid=time_valid,
            attention_mask=batch["attention_mask"] == 1,
        )
        dt = 1.0 / float(self.num_inference_steps)
        if self.noise_method in {"flow_sde", "transition_sde"}:
            mean, std = self._flow_sde_mean_std_lawam(x_t, velocity, tau, dt)
            diff = (x_next.float() - mean.float()) / std.float()
            log_norm = torch.log(std.float()) + 0.5 * math.log(2.0 * math.pi)
            logprobs = -0.5 * diff.pow(2) - log_norm
        else:
            target_velocity = (x_next - x_t) / dt
            sigma = max(float(self.noise_level), 1e-6)
            diff = (velocity.float() - target_velocity.float()) / sigma
            log_norm = math.log(sigma) + 0.5 * math.log(2.0 * math.pi)
            logprobs = -0.5 * diff.pow(2) - log_norm
        logprobs = logprobs[:, : self.num_action_chunks, : self.action_dim].float()
        values = None
        if compute_values:
            values = self._compute_values_from_shared(shared, batch).float()
        entropy = None
        if compute_entropy:
            entropy = torch.zeros((bsz, 1), dtype=torch.float32, device=device)
        return logprobs, values, entropy

    def _compute_values_from_shared(self, shared: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.add_value_head or not hasattr(self, "value_head"):
            raise ValueError("LaWAM PPO requires add_value_head=True for value computation.")
        h_vlm = shared.h_vlm
        if self.value_feature == "flow_query_tokens":
            mask = batch["flow_placeholder_mask"].to(device=h_vlm.device, dtype=torch.bool)
            n_q = int(self.policy_backend.flow_action_query.shape[0])
            tokens = h_vlm[mask].view(h_vlm.shape[0], n_q, h_vlm.shape[-1])
            feat = tokens.mean(dim=1)
        elif self.value_feature == "act_query_tokens":
            mask = batch["act_placeholder_mask"].to(device=h_vlm.device, dtype=torch.bool)
            n_q = int(self.policy_backend.num_action_queries)
            tokens = h_vlm[mask].view(h_vlm.shape[0], n_q, h_vlm.shape[-1])
            feat = tokens.mean(dim=1)
        else:
            mask = batch["attention_mask"].to(device=h_vlm.device, dtype=torch.bool)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=h_vlm.dtype)
            feat = (h_vlm * mask.unsqueeze(-1).to(dtype=h_vlm.dtype)).sum(dim=1) / denom
        if self.detach_critic_input:
            feat = feat.detach()
        value_param = next(self.value_head.parameters())
        return self.value_head(feat.to(dtype=value_param.dtype)).reshape(feat.shape[0], 1)

    def _pack_first_dim_by_batch(self, tensor: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            return None
        if int(tensor.shape[0]) % int(batch_size) != 0:
            raise ValueError(f"Cannot pack tensor with shape {tuple(tensor.shape)} by batch_size={batch_size}")
        return tensor.detach().cpu().reshape(batch_size, -1, *tensor.shape[1:]).contiguous()

    @staticmethod
    def _unpack_first_dim(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.reshape(-1, *tensor.shape[2:]).contiguous()

    def _forward_inputs_from_batch(
        self,
        *,
        batch: dict[str, torch.Tensor],
        action_chains: torch.Tensor,
        denoise_inds: torch.Tensor,
        env_action: torch.Tensor,
        model_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz = int(batch["input_ids"].shape[0])
        image_grid = batch.get("image_grid_thw")
        return {
            "action": env_action.reshape(bsz, -1).detach().cpu().contiguous(),
            "model_action": model_action.reshape(bsz, -1).detach().float().cpu().contiguous(),
            "lawam_input_ids": batch["input_ids"].detach().cpu().contiguous(),
            "lawam_attention_mask": batch["attention_mask"].detach().cpu().contiguous(),
            "lawam_act_placeholder_mask": batch["act_placeholder_mask"].detach().cpu().contiguous(),
            "lawam_flow_placeholder_mask": batch["flow_placeholder_mask"].detach().cpu().contiguous(),
            "lawam_primary_image": batch["primary_image"].detach().float().cpu().contiguous(),
            "lawam_state": batch["state"].detach().float().cpu().contiguous(),
            "lawam_state_mask": batch["state_mask"].detach().cpu().contiguous(),
            "lawam_embodiment_id": batch["embodiment_id"].detach().cpu().contiguous(),
            "lawam_action_hz": batch["action_hz"].detach().float().cpu().contiguous(),
            "lawam_pixel_values": self._pack_first_dim_by_batch(batch["pixel_values"], bsz),
            **({"lawam_image_grid_thw": self._pack_first_dim_by_batch(image_grid, bsz)} if torch.is_tensor(image_grid) else {}),
            "lawam_action_chains": action_chains.detach().float().cpu().contiguous(),
            "lawam_denoise_inds": denoise_inds.detach().cpu().contiguous(),
        }

    def _batch_from_forward_inputs(self, forward_inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        pixel_values = self._unpack_first_dim(forward_inputs["lawam_pixel_values"].to(device))
        image_grid = forward_inputs.get("lawam_image_grid_thw")
        if image_grid is not None:
            image_grid = self._unpack_first_dim(image_grid.to(device))
        return {
            "pixel_values": pixel_values,
            "input_ids": forward_inputs["lawam_input_ids"].to(device),
            "attention_mask": forward_inputs["lawam_attention_mask"].to(device),
            "act_placeholder_mask": forward_inputs["lawam_act_placeholder_mask"].to(device),
            "flow_placeholder_mask": forward_inputs["lawam_flow_placeholder_mask"].to(device),
            "primary_image": forward_inputs["lawam_primary_image"].to(device),
            "state": forward_inputs["lawam_state"].to(device),
            "state_mask": forward_inputs["lawam_state_mask"].to(device),
            "embodiment_id": forward_inputs["lawam_embodiment_id"].to(device),
            "action_hz": forward_inputs["lawam_action_hz"].to(device),
            "image_grid_thw": image_grid,
        }
