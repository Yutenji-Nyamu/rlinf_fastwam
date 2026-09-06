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

"""Narrow pi0-side modules used by the OGPO actor adapter.

The online action expert remains the normal OpenPI action expert.  The only
additional registered model state is a frozen exponential-moving-average
(EMA) copy of that suffix expert.  Critic, replay, and optimizer state do not
belong in this module.
"""

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from torch import nn


def keep_tied_paligemma_weight_in_root_fsdp_unit_(
    embedding: nn.Embedding,
    lm_head: nn.Linear,
) -> bool:
    """Keep PaliGemma's tied token weight inside one FSDP ownership unit.

    RLinf normally wraps every module named ``lm_head``.  Under FSDP1 with
    ``use_orig_params=True``, separately wrapping PaliGemma's tied language
    head can leave the shared embedding view one-dimensional when prefix
    conditioning calls ``embed_tokens``.  OGPO never evaluates that language
    head, so it stays in the root unit beside the embedding.  Independent
    action-expert language heads keep their normal wrap names.
    """
    if embedding.weight is not lm_head.weight:
        return False
    lm_head._fsdp_wrap_name = "ogpo_tied_paligemma_lm_head_root"
    return True


def ogpo_time_to_pi0_time(time_ogpo: torch.Tensor) -> torch.Tensor:
    """Map OGPO noise-to-action time to pi0 action-to-noise time."""
    return torch.ones_like(time_ogpo) - time_ogpo


def pi0_velocity_to_ogpo(velocity_pi0: torch.Tensor) -> torch.Tensor:
    """Map ``dx/dt_pi0`` to ``dx/dt_ogpo`` under ``t_ogpo=1-t_pi0``."""
    return -velocity_pi0


def project_ogpo_canonical_action(
    raw_final_action: torch.Tensor,
    *,
    executed_horizon: int,
    active_action_dim: int,
) -> torch.Tensor:
    """Return the normalized canonical action consumed by Q and replay.

    Projection is deliberately out-of-place and does not clamp.  The complete
    raw endpoint remains part of the likelihood chain, while Q consumes only
    the actually executed prefix and active robot coordinates.
    """
    if raw_final_action.ndim < 2:
        raise ValueError(
            "raw_final_action must end in horizon and action dimensions"
        )
    if not 0 < executed_horizon <= raw_final_action.shape[-2]:
        raise ValueError(
            "executed_horizon must be within the model horizon, got "
            f"{executed_horizon} and {raw_final_action.shape[-2]}"
        )
    if not 0 < active_action_dim <= raw_final_action.shape[-1]:
        raise ValueError(
            "active_action_dim must be within the model action dimension, got "
            f"{active_action_dim} and {raw_final_action.shape[-1]}"
        )
    return raw_final_action[
        ..., :executed_horizon, :active_action_dim
    ].clone()


def project_ogpo_action_views(
    raw_final_action: torch.Tensor,
    *,
    executed_horizon: int = 10,
    active_action_dim: int = 14,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the unchanged raw endpoint and an out-of-place canonical view.

    The raw tensor is returned by identity so likelihood always refers to the
    exact sampled endpoint.  The canonical ``C x active`` tensor is a clone,
    making downstream clipping or environment conversion unable to mutate the
    raw chain.
    """
    canonical_action = project_ogpo_canonical_action(
        raw_final_action,
        executed_horizon=executed_horizon,
        active_action_dim=active_action_dim,
    )
    return raw_final_action, canonical_action


def _sinusoidal_time_embedding(
    time: torch.Tensor,
    dimension: int,
    *,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    if dimension % 2:
        raise ValueError(f"Time embedding dimension must be even, got {dimension}")
    if time.ndim != 1:
        raise ValueError(f"Expected time shape [batch], got {tuple(time.shape)}")
    compute_dtype = torch.float64
    fraction = torch.linspace(
        0.0,
        1.0,
        dimension // 2,
        dtype=compute_dtype,
        device=time.device,
    )
    period = min_period * (max_period / min_period) ** fraction
    phase = (2.0 * math.pi / period)[None, :] * time.to(compute_dtype)[:, None]
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)


class OGPOEMAActionExpert(nn.Module):
    """Frozen EMA copy of pi0's suffix action expert and projections.

    The full PaliGemma VLM and unused causal-LM head are intentionally not
    copied.  Prefix KV conditioning is supplied by the frozen online model.
    ``forward`` accepts OGPO time and returns OGPO-oriented velocity, keeping
    the time/velocity convention conversion in one source-locked place.
    """

    def __init__(self, online_model: Any):
        super().__init__()
        self.pi05 = bool(online_model.pi05)
        self.action_horizon = int(online_model.config.action_horizon)

        self.expert_model = copy.deepcopy(
            online_model.paligemma_with_expert.gemma_expert.model
        )
        self.action_in_proj = copy.deepcopy(online_model.action_in_proj)
        self.action_out_proj = copy.deepcopy(online_model.action_out_proj)

        if self.pi05:
            self.time_mlp_in = copy.deepcopy(online_model.time_mlp_in)
            self.time_mlp_out = copy.deepcopy(online_model.time_mlp_out)
            self._projection_names = (
                "action_in_proj",
                "action_out_proj",
                "time_mlp_in",
                "time_mlp_out",
            )
        else:
            self.state_proj = copy.deepcopy(online_model.state_proj)
            self.action_time_mlp_in = copy.deepcopy(
                online_model.action_time_mlp_in
            )
            self.action_time_mlp_out = copy.deepcopy(
                online_model.action_time_mlp_out
            )
            self._projection_names = (
                "action_in_proj",
                "action_out_proj",
                "state_proj",
                "action_time_mlp_in",
                "action_time_mlp_out",
            )

        self.requires_grad_(False)
        self.eval()
        # Low-precision EMA targets need an unregistered FP32 accumulator.
        # It is created lazily after FSDP has placed/unsharded the parameters;
        # the actor worker persists it in its rank sidecar, while rollout
        # replicas only receive the visible target parameters.
        self._ema_shadow_f32: dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        """Keep the EMA target deterministic when its parent enters train mode."""
        del mode
        return super().train(False)

    def _module_pairs(self, online_model: Any):
        yield self.expert_model, online_model.paligemma_with_expert.gemma_expert.model
        for name in self._projection_names:
            yield getattr(self, name), getattr(online_model, name)

    @torch.no_grad()
    def match_online_dtypes_(self, online_model: Any) -> None:
        """Mirror online expert dtypes before FSDP wrapping.

        OpenPI converts most Gemma weights to BF16 only after model
        construction.  The EMA copy is constructed earlier in FP32, so it must
        be converted field-by-field to preserve identical suffix computation
        and KV-cache compatibility.
        """
        for target_module, source_module in self._module_pairs(online_model):
            target_parameters = dict(target_module.named_parameters())
            source_parameters = dict(source_module.named_parameters())
            if target_parameters.keys() != source_parameters.keys():
                raise ValueError("EMA and online action-expert parameters differ")
            for name, target_value in target_parameters.items():
                source_value = source_parameters[name]
                target_value.data = target_value.data.to(dtype=source_value.dtype)

            target_buffers = dict(target_module.named_buffers())
            source_buffers = dict(source_module.named_buffers())
            if target_buffers.keys() != source_buffers.keys():
                raise ValueError("EMA and online action-expert buffers differ")
            for name, target_value in target_buffers.items():
                source_value = source_buffers[name]
                target_value.data = target_value.data.to(dtype=source_value.dtype)
        self._ema_shadow_f32.clear()

    @torch.no_grad()
    def copy_from_online_(self, online_model: Any) -> None:
        """Initialize EMA weights after the SFT checkpoint has been loaded."""
        for target, source in self._module_pairs(online_model):
            target.load_state_dict(source.state_dict(), strict=True)
        self.requires_grad_(False)
        self.eval()
        self._ema_shadow_f32.clear()

    def _named_parameter_pairs(self, online_model: Any):
        for module_index, (target_module, source_module) in enumerate(
            self._module_pairs(online_model)
        ):
            target_parameters = dict(target_module.named_parameters())
            source_parameters = dict(source_module.named_parameters())
            if target_parameters.keys() != source_parameters.keys():
                raise ValueError("EMA and online action-expert parameters differ")
            for name, target_value in target_parameters.items():
                source_value = source_parameters[name]
                if target_value.shape != source_value.shape:
                    raise ValueError(
                        f"EMA and online parameter shard shapes differ: {name}"
                    )
                if (
                    target_value.device != source_value.device
                    or target_value.dtype != source_value.dtype
                ):
                    raise ValueError(
                        f"EMA and online parameter shard placement differs: {name}"
                    )
                yield (
                    f"{module_index}:{name}",
                    target_value,
                    source_value,
                )

    @torch.no_grad()
    def ema_shadow_state(self) -> dict[str, torch.Tensor]:
        """Return the persistent FP32 low-precision EMA accumulators on CPU."""
        return {
            name: value.detach().cpu().clone()
            for name, value in self._ema_shadow_f32.items()
        }

    @torch.no_grad()
    def load_ema_shadow_state_(
        self, state: dict[str, torch.Tensor], online_model: Any
    ) -> None:
        if not state:
            self._ema_shadow_f32.clear()
            return
        expected = {
            name: target
            for name, target, _ in self._named_parameter_pairs(online_model)
            if target.is_floating_point() and target.dtype != torch.float32
        }
        if state.keys() != expected.keys():
            raise ValueError("OGPO EMA shadow schema differs from target parameters")
        restored = {}
        for name, target in expected.items():
            value = state[name]
            if value.shape != target.shape or value.dtype != torch.float32:
                raise ValueError(f"Invalid OGPO EMA shadow tensor: {name}")
            restored[name] = value.to(device=target.device).clone()
        self._ema_shadow_f32 = restored

    @torch.no_grad()
    def polyak_update_from_online_(self, online_model: Any, tau: float) -> None:
        """Apply ``target <- (1-tau)*target + tau*online`` in-place."""
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        for name, target_value, source_parameter in self._named_parameter_pairs(
            online_model
        ):
            source_value = source_parameter.detach().to(
                device=target_value.device, dtype=torch.float32
            )
            if target_value.is_floating_point() and target_value.dtype != torch.float32:
                shadow = self._ema_shadow_f32.get(name)
                if shadow is None:
                    shadow = target_value.detach().to(dtype=torch.float32).clone()
                    self._ema_shadow_f32[name] = shadow
                shadow.mul_(1.0 - tau).add_(source_value, alpha=tau)
                target_value.copy_(shadow.to(dtype=target_value.dtype))
            else:
                target_value.mul_(1.0 - tau).add_(
                    source_value.to(dtype=target_value.dtype), alpha=tau
                )

        for target_module, source_module in self._module_pairs(online_model):
            target_buffers = dict(target_module.named_buffers())
            source_buffers = dict(source_module.named_buffers())
            if target_buffers.keys() != source_buffers.keys():
                raise ValueError("EMA and online action-expert buffers differ")
            for name, target_value in target_buffers.items():
                target_value.copy_(
                    source_buffers[name].detach().to(
                        device=target_value.device, dtype=target_value.dtype
                    )
                )

    def _embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep_pi0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        embeddings = []
        pad_masks = []
        attention_masks: list[int] = []

        if not self.pi05:
            state_embedding = self.state_proj(
                state.to(dtype=self.state_proj.weight.dtype)
            )
            embeddings.append(state_embedding[:, None, :])
            pad_masks.append(
                torch.ones(
                    state_embedding.shape[0],
                    1,
                    dtype=torch.bool,
                    device=state_embedding.device,
                )
            )
            attention_masks.append(1)

        time_embedding = _sinusoidal_time_embedding(
            timestep_pi0,
            self.action_in_proj.out_features,
        ).to(dtype=timestep_pi0.dtype)
        action_embedding = self.action_in_proj(
            noisy_actions.to(dtype=self.action_in_proj.weight.dtype)
        )

        if self.pi05:
            time_embedding = self.time_mlp_in(time_embedding)
            time_embedding = F.silu(time_embedding)
            time_embedding = self.time_mlp_out(time_embedding)
            adarms_cond = F.silu(time_embedding)
            action_time_embedding = action_embedding
        else:
            expanded_time = time_embedding[:, None, :].expand_as(action_embedding)
            action_time_embedding = torch.cat(
                [action_embedding, expanded_time], dim=-1
            )
            action_time_embedding = self.action_time_mlp_in(action_time_embedding)
            action_time_embedding = F.silu(action_time_embedding)
            action_time_embedding = self.action_time_mlp_out(action_time_embedding)
            adarms_cond = None

        embeddings.append(action_time_embedding)
        pad_masks.append(
            torch.ones(
                action_time_embedding.shape[:2],
                dtype=torch.bool,
                device=action_time_embedding.device,
            )
        )
        attention_masks.extend([1] + [0] * (self.action_horizon - 1))

        suffix_embeddings = torch.cat(embeddings, dim=1)
        suffix_pad_masks = torch.cat(pad_masks, dim=1)
        suffix_attention_masks = torch.tensor(
            attention_masks,
            dtype=suffix_embeddings.dtype,
            device=suffix_embeddings.device,
        ).unsqueeze(0).expand(suffix_embeddings.shape[0], -1)
        return (
            suffix_embeddings,
            suffix_pad_masks,
            suffix_attention_masks,
            adarms_cond,
        )

    @torch.no_grad()
    def forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep_ogpo: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate EMA expert and return ``(velocity_ogpo, suffix_output)``."""
        batch_size = x_t.shape[0]
        if timestep_ogpo.numel() == 1:
            timestep_ogpo = timestep_ogpo.reshape(1).expand(batch_size)
        else:
            timestep_ogpo = timestep_ogpo.reshape(batch_size)
        timestep_ogpo = timestep_ogpo.to(device=x_t.device, dtype=torch.float32)
        timestep_pi0 = ogpo_time_to_pi0_time(timestep_ogpo)

        (
            suffix_embeddings,
            suffix_pad_masks,
            suffix_attention_masks,
            adarms_cond,
        ) = self._embed_suffix(state, x_t, timestep_pi0)

        suffix_length = suffix_pad_masks.shape[1]
        prefix_length = prefix_pad_masks.shape[1]
        prefix_attention = prefix_pad_masks[:, None, :].expand(
            state.shape[0], suffix_length, prefix_length
        )
        suffix_attention = make_att_2d_masks(
            suffix_pad_masks, suffix_attention_masks
        )
        full_attention = torch.cat([prefix_attention, suffix_attention], dim=2)
        attention_mask = torch.where(
            full_attention[:, None, :, :],
            0.0,
            -2.3819763e38,
        )
        prefix_offsets = prefix_pad_masks.sum(dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        self.expert_model.config._attn_implementation = "eager"  # noqa: SLF001
        expert_output = self.expert_model.forward(
            inputs_embeds=suffix_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            adarms_cond=adarms_cond,
        )
        suffix_output = (
            expert_output.last_hidden_state
            if hasattr(expert_output, "last_hidden_state")
            else expert_output[0]
        )
        suffix_output = suffix_output[:, -self.action_horizon :].to(
            dtype=torch.float32
        )
        velocity_pi0 = self.action_out_proj(
            suffix_output.to(dtype=self.action_out_proj.weight.dtype)
        ).to(dtype=torch.float32)
        return pi0_velocity_to_ogpo(velocity_pi0), suffix_output
