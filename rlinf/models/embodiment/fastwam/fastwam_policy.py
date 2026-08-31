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

"""RLinf policy wrapper for the pinned Fast-WAM RoboTwin model.

The wrapper vectorizes the official Fast-WAM action-only deployment path without
changing its image, prompt, proprio, scheduler, or action-normalization semantics.
Evaluation, stochastic rollout, and actor replay all use the shared primitives in
``fastwam_rl``.  Resource chunking is deliberately outside those primitives so it
cannot change the logical-batch random variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .fastwam_rl import (
    build_action_conditioning,
    flow_sde_rollout,
    prepare_initial_action_latents,
    prepare_rollout_randomness,
    recompute_logprob,
)
from .robotwin_adapter import (
    ACTION_DIM,
    ACTION_HORIZON,
    adapt_robotwin_observation,
    denormalize_actions,
)


@dataclass(frozen=True)
class FastWAMPolicyConfig:
    """Resolved inference and Flow-SDE settings for the first integration."""

    action_dim: int = ACTION_DIM
    action_horizon: int = ACTION_HORIZON
    num_action_chunks: int = 24
    num_inference_steps: int = 10
    sigma_shift: float | None = None
    text_cfg_scale: float = 1.0
    negative_prompt: str = ""
    rand_device: str = "cpu"
    tiled: bool = False
    model_forward_batch_size: int = 2
    eval_seed: int = 0
    noise_level: float = 0.1


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


def _cpu_detached(tensor: torch.Tensor, *, dtype=None) -> torch.Tensor:
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.detach().to(device="cpu").contiguous()


class FastWAMPolicy(nn.Module, BasePolicy):
    """Expose official Fast-WAM action inference through RLinf's policy contract."""

    rlinf_accepts_rollout_mode = True
    rlinf_dvac_endpoint_capable = True

    def __init__(
        self,
        *,
        model: nn.Module,
        processor: Any,
        config: FastWAMPolicyConfig,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.config = config
        self._validate_config()
        self.train(False)

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.action_dim != ACTION_DIM:
            raise ValueError(f"Fast-WAM RoboTwin action_dim must be {ACTION_DIM}")
        if cfg.action_horizon != ACTION_HORIZON:
            raise ValueError(
                f"Fast-WAM RoboTwin action_horizon must be {ACTION_HORIZON}"
            )
        if cfg.num_action_chunks != 24:
            raise ValueError("The first Fast-WAM RoboTwin integration requires N=24")
        if cfg.num_inference_steps != 10:
            raise ValueError("The released RoboTwin checkpoint requires S=10")
        if cfg.model_forward_batch_size <= 0:
            raise ValueError("model_forward_batch_size must be positive")
        if cfg.noise_level <= 0:
            raise ValueError("Flow-SDE noise_level must be positive")
        if cfg.rand_device != "cpu":
            raise ValueError(
                "Pinned official Fast-WAM action noise is generated on CPU"
            )
        if cfg.tiled:
            raise ValueError(
                "Pinned official WanVideoVAE does not support tiled encoding"
            )
        if cfg.text_cfg_scale != 1.0 or cfg.negative_prompt:
            raise ValueError(
                "Pinned action-only Fast-WAM uses text_cfg_scale=1 and no negative prompt"
            )

    @property
    def device(self) -> torch.device:
        return _module_device_dtype(self.model)[0]

    @property
    def model_dtype(self) -> torch.dtype:
        dtype = getattr(self.model, "torch_dtype", None)
        return dtype if dtype is not None else _module_device_dtype(self.model)[1]

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        """Explicitly dispatch around the ``nn.Module, BasePolicy`` MRO."""

        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"FastWAMPolicy does not support {forward_type}")

    def train(self, mode: bool = True):
        """Keep frozen conditioners in eval mode while the action expert trains."""

        if not isinstance(mode, bool):
            raise ValueError("training mode must be a bool")
        nn.Module.train(self, mode)
        if mode:
            self.model.training = True
            self.model.mot.training = True
            self.model.mot.mixtures["action"].train(True)
            self.model.mot.mixtures["video"].train(False)
            for component_name in ("vae", "text_encoder", "proprio_encoder"):
                component = getattr(self.model, component_name, None)
                if component is not None and hasattr(component, "train"):
                    component.train(False)
        return self

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        return_dvac_telemetry: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return a physical RoboTwin qpos chunk and optional behavior replay."""

        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported Fast-WAM rollout mode: {mode!r}")
        if not isinstance(return_dvac_telemetry, bool):
            raise TypeError("return_dvac_telemetry must be a bool")
        image, proprio, prompts = adapt_robotwin_observation(
            env_obs,
            self.processor,
            device=self.device,
            dtype=self.model_dtype,
        )
        batch_size = int(image.shape[0])
        text_context, text_context_mask = self.model.encode_prompt(prompts)
        if text_context.ndim != 3 or tuple(text_context.shape) != (
            batch_size,
            128,
            4096,
        ):
            raise ValueError(
                "Official encode_prompt must return text_context [B,128,4096]"
            )
        if text_context_mask.shape != text_context.shape[:2]:
            raise ValueError(
                "Official encode_prompt must return text_context_mask [B,128]"
            )
        if text_context.dtype != self.model_dtype:
            raise TypeError(
                "Official encode_prompt context dtype must match Fast-WAM model dtype"
            )
        if text_context_mask.dtype != torch.bool:
            raise TypeError("Official encode_prompt mask must have dtype bool")
        if not bool(torch.isfinite(text_context).all()):
            raise ValueError("Official encode_prompt returned non-finite context")

        deterministic = mode == "eval"
        if deterministic:
            initial_latents = prepare_initial_action_latents(
                batch_size=batch_size,
                action_horizon=self.config.action_horizon,
                action_dim=self.config.action_dim,
                device=self.device,
                dtype=self.model_dtype,
                rand_device=self.config.rand_device,
                seed=self.config.eval_seed,
                broadcast_singleton=True,
            )
            rollout_randomness = None
        else:
            rollout_randomness = prepare_rollout_randomness(
                batch_size=batch_size,
                action_horizon=self.config.action_horizon,
                action_dim=self.config.action_dim,
                num_inference_steps=self.config.num_inference_steps,
                device=self.device,
                rand_device=self.config.rand_device,
            )
            initial_latents = rollout_randomness.initial_latents

        model_actions: list[torch.Tensor] = []
        chains: list[torch.Tensor] = []
        old_logprobs: list[torch.Tensor] = []
        denoise_inds: list[torch.Tensor] = []
        endpoint_traces: list[torch.Tensor] = []
        chunk_size = self.config.model_forward_batch_size
        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            item = slice(start, stop)
            conditioning = build_action_conditioning(
                self.model,
                input_image=image[item],
                text_context=text_context[item],
                text_context_mask=text_context_mask[item],
                proprio=proprio[item],
                action_horizon=self.config.action_horizon,
                tiled=self.config.tiled,
            )
            rollout = flow_sde_rollout(
                self.model,
                conditioning=conditioning,
                initial_latents=initial_latents[item],
                num_inference_steps=self.config.num_inference_steps,
                sigma_shift=self.config.sigma_shift,
                noise_level=self.config.noise_level,
                deterministic=deterministic,
                denoise_inds=(
                    None
                    if rollout_randomness is None
                    else rollout_randomness.denoise_inds[item]
                ),
                sde_epsilon=(
                    None
                    if rollout_randomness is None
                    else rollout_randomness.sde_epsilon[item]
                ),
                return_endpoint_trace=return_dvac_telemetry,
            )
            model_actions.append(_cpu_detached(rollout.actions, dtype=torch.float32))
            if return_dvac_telemetry:
                if rollout.endpoint_trace is None:
                    raise RuntimeError("Fast-WAM rollout omitted DVAC endpoint trace")
                endpoint_traces.append(
                    _cpu_detached(
                        rollout.endpoint_trace[:, :, : self.config.num_action_chunks],
                        dtype=torch.float32,
                    )
                )
            if not deterministic:
                if (
                    rollout.chains is None
                    or rollout.prev_logprobs is None
                    or rollout.denoise_inds is None
                ):
                    raise RuntimeError(
                        "Stochastic Fast-WAM rollout omitted replay data"
                    )
                chains.append(_cpu_detached(rollout.chains, dtype=torch.float32))
                old_logprobs.append(
                    _cpu_detached(
                        rollout.prev_logprobs[:, : self.config.num_action_chunks],
                        dtype=torch.float32,
                    )
                )
                denoise_inds.append(
                    _cpu_detached(rollout.denoise_inds, dtype=torch.long)
                )

        normalized_actions = torch.cat(model_actions, dim=0)
        physical_actions = denormalize_actions(normalized_actions, self.processor)
        executed_actions = physical_actions[:, : self.config.num_action_chunks]
        if deterministic:
            # Evaluation ignores replay, but current rollout workers still read
            # these three result keys before forwarding the physical actions.
            result = {
                "prev_logprobs": torch.zeros(
                    batch_size,
                    self.config.num_action_chunks,
                    self.config.action_dim,
                    dtype=torch.float32,
                ),
                "prev_values": None,
                "forward_inputs": {},
            }
            if return_dvac_telemetry:
                result["dvac_telemetry"] = {
                    "z_endpoint": torch.cat(endpoint_traces, dim=0)
                }
            return executed_actions, result

        replay = {
            "chains": torch.cat(chains, dim=0),
            "denoise_inds": torch.cat(denoise_inds, dim=0),
            "image": _cpu_detached(image),
            "text_context": _cpu_detached(text_context),
            "text_context_mask": _cpu_detached(text_context_mask, dtype=torch.bool),
            "proprio": _cpu_detached(proprio, dtype=torch.float32),
            "action": executed_actions.reshape(batch_size, -1).contiguous(),
            "model_action": normalized_actions.reshape(batch_size, -1).contiguous(),
        }
        result = {
            "prev_logprobs": torch.cat(old_logprobs, dim=0),
            "prev_values": None,
            "forward_inputs": replay,
        }
        if return_dvac_telemetry:
            result["dvac_telemetry"] = {
                "z_endpoint": torch.cat(endpoint_traces, dim=0)
            }
        return executed_actions, result

    def _validate_replay(self, forward_inputs: dict[str, torch.Tensor]) -> int:
        if not isinstance(forward_inputs, dict):
            raise TypeError("forward_inputs must be a dict[str, Tensor]")
        required = {
            "chains",
            "denoise_inds",
            "image",
            "text_context",
            "text_context_mask",
            "proprio",
            "action",
            "model_action",
        }
        missing = required.difference(forward_inputs)
        unexpected = set(forward_inputs).difference(required)
        if missing or unexpected:
            raise ValueError(
                f"Fast-WAM replay schema mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        if not all(torch.is_tensor(value) for value in forward_inputs.values()):
            raise TypeError("Every Fast-WAM forward_inputs value must be a Tensor")

        cfg = self.config
        chains = forward_inputs["chains"]
        if chains.ndim != 4:
            raise ValueError("chains must be [B,S+1,H,D]")
        batch_size = int(chains.shape[0])
        expected_shapes = {
            "chains": (
                batch_size,
                cfg.num_inference_steps + 1,
                cfg.action_horizon,
                cfg.action_dim,
            ),
            "denoise_inds": (batch_size,),
            "image": (batch_size, 3, 384, 320),
            "text_context_mask": (batch_size, 128),
            "proprio": (batch_size, cfg.action_dim),
            "action": (batch_size, cfg.num_action_chunks * cfg.action_dim),
            "model_action": (batch_size, cfg.action_horizon * cfg.action_dim),
        }
        for key, expected in expected_shapes.items():
            if tuple(forward_inputs[key].shape) != expected:
                raise ValueError(
                    f"{key} must have shape {expected}, got "
                    f"{tuple(forward_inputs[key].shape)}"
                )
        text_context = forward_inputs["text_context"]
        if (
            text_context.ndim != 3
            or tuple(text_context.shape[:2])
            != (
                batch_size,
                128,
            )
            or text_context.shape[2] != 4096
        ):
            raise ValueError("text_context must be [B,128,4096]")
        if forward_inputs["denoise_inds"].dtype != torch.long:
            raise TypeError("denoise_inds must have dtype int64")
        if forward_inputs["text_context_mask"].dtype != torch.bool:
            raise TypeError("text_context_mask must have dtype bool")
        expected_dtypes = {
            "chains": torch.float32,
            "image": self.model_dtype,
            "text_context": self.model_dtype,
            "proprio": torch.float32,
            "action": torch.float32,
            "model_action": torch.float32,
        }
        for key, expected_dtype in expected_dtypes.items():
            value = forward_inputs[key]
            if value.dtype != expected_dtype:
                raise TypeError(
                    f"{key} must have dtype {expected_dtype}, got {value.dtype}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{key} contains non-finite values")
        return batch_size

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_values: bool = False,
        compute_entropy: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Recompute the real behavior transition under current action weights."""

        if not compute_logprobs:
            raise ValueError("Fast-WAM actor replay requires compute_logprobs=True")
        if compute_values:
            raise ValueError("The first current Fast-WAM integration is GRPO-only")
        batch_size = self._validate_replay(forward_inputs)

        logprobs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        chunk_size = self.config.model_forward_batch_size
        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            item = slice(start, stop)
            replay_output = recompute_logprob(
                self.model,
                input_image=forward_inputs["image"][item],
                text_context=forward_inputs["text_context"][item],
                text_context_mask=forward_inputs["text_context_mask"][item],
                proprio=forward_inputs["proprio"][item],
                chains=forward_inputs["chains"][item],
                denoise_inds=forward_inputs["denoise_inds"][item],
                action_horizon=self.config.action_horizon,
                num_inference_steps=self.config.num_inference_steps,
                sigma_shift=self.config.sigma_shift,
                noise_level=self.config.noise_level,
                tiled=self.config.tiled,
            )
            logprob, entropy = replay_output
            logprobs.append(logprob[:, : self.config.num_action_chunks].float())
            if compute_entropy:
                entropies.append(entropy[:, : self.config.num_action_chunks].float())

        output = {"logprobs": torch.cat(logprobs, dim=0)}
        if compute_entropy:
            output["entropy"] = torch.cat(entropies, dim=0)
        return output


__all__ = ["FastWAMPolicy", "FastWAMPolicyConfig"]
