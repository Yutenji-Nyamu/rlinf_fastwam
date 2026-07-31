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

"""π0-side modules and structural contracts for the QAM integration.

This module intentionally contains no critic, replay, or optimizer state.  The
registered ``QAMFineActionExpert`` is the only QAM state that must be present
in both actor and rollout policy models.
"""

import copy
import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from torch import nn


def qam_time_to_pi0_time(time_qam: torch.Tensor) -> torch.Tensor:
    """Map QAM's noise-to-action time to π0's action-to-noise time."""
    return torch.ones_like(time_qam) - time_qam


def pi0_velocity_to_qam(velocity_pi0: torch.Tensor) -> torch.Tensor:
    """Map π0's ``noise - action`` velocity to QAM's opposite convention."""
    return -velocity_pi0


def keep_tied_embedding_and_lm_head_in_root_fsdp_unit_(
    embedding: nn.Embedding,
    lm_head: nn.Linear,
) -> bool:
    """Keep PaliGemma's tied token weight inside one FSDP ownership unit.

    RLinf's default OpenPI policy wraps modules whose ``_fsdp_wrap_name`` is
    ``lm_head``. With ``use_orig_params=True``, separately wrapping a tied
    language-model head leaves its shared embedding weight exposed as a
    one-dimensional shard when prefix conditioning calls ``embed_tokens``.
    QAM never evaluates the language-model head, so only that tied PaliGemma
    head is kept at the root alongside the embedding. The independent action
    expert's unused ``lm_head`` retains the legacy wrap name.
    """
    if embedding.weight is not lm_head.weight:
        return False
    lm_head._fsdp_wrap_name = "qam_tied_paligemma_lm_head_root"
    return True


def canonicalize_qam_rollout_action(
    model_action: torch.Tensor,
    *,
    planned_horizon: int,
    active_action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp only the active planned block used by Q and the environment.

    The raw fine-flow endpoint is not mutated.  The first returned tensor is a
    model-space copy whose unplanned horizon and static padding coordinates are
    unchanged; callers must pass this exact tensor into ``output_transform``.
    The second return is its ``[N, active_dim]`` view for replay.
    """
    if model_action.ndim < 2:
        raise ValueError("model_action must end in horizon and action dimensions")
    if not 0 < planned_horizon <= model_action.shape[-2]:
        raise ValueError(
            "planned_horizon must be within the model horizon, got "
            f"{planned_horizon} and {model_action.shape[-2]}"
        )
    if not 0 < active_action_dim <= model_action.shape[-1]:
        raise ValueError(
            "active_action_dim must be within the model action dimension, got "
            f"{active_action_dim} and {model_action.shape[-1]}"
        )

    canonical = model_action.clone()
    canonical[..., :planned_horizon, :active_action_dim] = model_action[
        ..., :planned_horizon, :active_action_dim
    ].clamp(-1.0, 1.0)
    planned = canonical[..., :planned_horizon, :active_action_dim]
    return canonical, planned


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=tokens.device, dtype=torch.bool)
    weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    denominator = weights.sum(dim=1).clamp_min_(1.0)
    pooled = (tokens * weights).sum(dim=1) / denominator
    valid = mask.any(dim=1, keepdim=True)
    return torch.where(valid, pooled, torch.zeros_like(pooled))


def pool_qam_prefix_blocks(
    prefix_output: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    *,
    language_token_count: int,
    num_image_blocks: int = 3,
    image_token_counts: Sequence[int] | None = None,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Pool three image blocks and one language block without fixing width.

    OpenPI concatenates image position blocks followed by the padded language
    block.  The deployed π0 dependency does not expose those block boundaries,
    so the default contract derives equal image-block lengths from the runtime
    prefix length.  A future dependency that exposes exact per-image lengths
    can pass ``image_token_counts`` without changing the replay schema.

    Returns:
        A ``[batch, num_image_blocks + 1, d_prefix]`` tensor and the exact
        runtime block lengths used to construct it.
    """
    if prefix_output.ndim != 3:
        raise ValueError(
            "prefix_output must have shape [batch, tokens, width], got "
            f"{tuple(prefix_output.shape)}"
        )
    if prefix_pad_masks.shape != prefix_output.shape[:2]:
        raise ValueError(
            "prefix_pad_masks must match prefix_output's first two dimensions; "
            f"got {tuple(prefix_pad_masks.shape)} and "
            f"{tuple(prefix_output.shape)}"
        )
    if num_image_blocks <= 0:
        raise ValueError("num_image_blocks must be positive")
    if language_token_count <= 0:
        raise ValueError("language_token_count must be positive")

    prefix_tokens = prefix_output.shape[1]
    image_tokens = prefix_tokens - language_token_count
    if image_tokens <= 0:
        raise ValueError(
            "language_token_count leaves no image tokens in the prefix: "
            f"prefix={prefix_tokens}, language={language_token_count}"
        )

    if image_token_counts is None:
        if image_tokens % num_image_blocks != 0:
            raise ValueError(
                "Runtime prefix cannot be partitioned into equal image blocks. "
                "Pass exact image_token_counts after probing the OpenPI "
                f"dependency; image_tokens={image_tokens}, "
                f"num_image_blocks={num_image_blocks}."
            )
        per_image = image_tokens // num_image_blocks
        image_token_counts = (per_image,) * num_image_blocks
    else:
        image_token_counts = tuple(int(count) for count in image_token_counts)
        if len(image_token_counts) != num_image_blocks:
            raise ValueError(
                "image_token_counts length must equal num_image_blocks; "
                f"got {len(image_token_counts)} and {num_image_blocks}"
            )
        if any(count <= 0 for count in image_token_counts):
            raise ValueError("Every image token count must be positive")
        if sum(image_token_counts) != image_tokens:
            raise ValueError(
                "image_token_counts do not cover the runtime image prefix; "
                f"sum={sum(image_token_counts)}, expected={image_tokens}"
            )

    block_lengths = (*image_token_counts, language_token_count)
    pooled_blocks = []
    start = 0
    for block_length in block_lengths:
        stop = start + block_length
        pooled_blocks.append(
            _masked_mean(
                prefix_output[:, start:stop],
                prefix_pad_masks[:, start:stop],
            )
        )
        start = stop

    return torch.stack(pooled_blocks, dim=1), tuple(block_lengths)


def build_qam_projection_fingerprint(
    *,
    model_horizon: int,
    planned_horizon: int,
    model_action_dim: int,
    active_action_dim: int,
    projection_version: str,
    data_fingerprint: str = "",
) -> bytes:
    """Return a stable SHA-256 digest for the normalized action contract."""
    payload = {
        "active_action_dim": int(active_action_dim),
        "data_fingerprint": str(data_fingerprint),
        "model_action_dim": int(model_action_dim),
        "model_horizon": int(model_horizon),
        "planned_horizon": int(planned_horizon),
        "projection_version": str(projection_version),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).digest()


def projection_fingerprint_tensor(
    fingerprint: bytes,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode a policy fingerprint as a stack-safe ``uint8[batch, 32]``."""
    if len(fingerprint) != hashlib.sha256().digest_size:
        raise ValueError(
            "QAM projection fingerprint must be a SHA-256 digest, got "
            f"{len(fingerprint)} bytes"
        )
    digest = torch.tensor(
        list(fingerprint),
        dtype=torch.uint8,
        device=device,
    )
    return digest.unsqueeze(0).expand(batch_size, -1).clone()


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


class QAMFineActionExpert(nn.Module):
    """Independent trainable copy of π0's action suffix expert.

    Only ``gemma_expert.model`` and the action/state/time projections are
    copied.  The unused causal-LM head and the full PaliGemma VLM are excluded.
    Prefix KV conditioning is supplied by the frozen behavior model.
    """

    def __init__(self, behavior_model: Any):
        super().__init__()
        self.pi05 = bool(behavior_model.pi05)
        self.action_horizon = int(behavior_model.config.action_horizon)

        self.expert_model = copy.deepcopy(
            behavior_model.paligemma_with_expert.gemma_expert.model
        )
        self.action_in_proj = copy.deepcopy(behavior_model.action_in_proj)
        self.action_out_proj = copy.deepcopy(behavior_model.action_out_proj)

        if self.pi05:
            self.time_mlp_in = copy.deepcopy(behavior_model.time_mlp_in)
            self.time_mlp_out = copy.deepcopy(behavior_model.time_mlp_out)
            self._projection_names = (
                "action_in_proj",
                "action_out_proj",
                "time_mlp_in",
                "time_mlp_out",
            )
        else:
            self.state_proj = copy.deepcopy(behavior_model.state_proj)
            self.action_time_mlp_in = copy.deepcopy(behavior_model.action_time_mlp_in)
            self.action_time_mlp_out = copy.deepcopy(behavior_model.action_time_mlp_out)
            self._projection_names = (
                "action_in_proj",
                "action_out_proj",
                "state_proj",
                "action_time_mlp_in",
                "action_time_mlp_out",
            )

    @torch.no_grad()
    def copy_from_behavior_(self, behavior_model: Any) -> None:
        """Initialize F1 after the behavior SFT checkpoint has been loaded."""
        self.expert_model.load_state_dict(
            behavior_model.paligemma_with_expert.gemma_expert.model.state_dict(),
            strict=True,
        )
        for name in self._projection_names:
            getattr(self, name).load_state_dict(
                getattr(behavior_model, name).state_dict(),
                strict=True,
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
            state = state.to(dtype=self.state_proj.weight.dtype)
            state_embedding = self.state_proj(state)
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
                [action_embedding, expanded_time],
                dim=-1,
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
        )
        suffix_attention_masks = suffix_attention_masks.unsqueeze(0).expand(
            suffix_embeddings.shape[0],
            -1,
        )
        return (
            suffix_embeddings,
            suffix_pad_masks,
            suffix_attention_masks,
            adarms_cond,
        )

    def forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep_pi0: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the copied fine expert with frozen prefix conditioning."""
        (
            suffix_embeddings,
            suffix_pad_masks,
            suffix_attention_masks,
            adarms_cond,
        ) = self._embed_suffix(state, x_t, timestep_pi0)

        suffix_length = suffix_pad_masks.shape[1]
        prefix_length = prefix_pad_masks.shape[1]
        prefix_attention = prefix_pad_masks[:, None, :].expand(
            state.shape[0],
            suffix_length,
            prefix_length,
        )
        suffix_attention = make_att_2d_masks(
            suffix_pad_masks,
            suffix_attention_masks,
        )
        full_attention = torch.cat(
            [prefix_attention, suffix_attention],
            dim=2,
        )
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
        suffix_output = suffix_output[:, -self.action_horizon :].to(dtype=torch.float32)
        velocity = self.action_out_proj(
            suffix_output.to(dtype=self.action_out_proj.weight.dtype)
        ).to(dtype=torch.float32)
        return velocity, suffix_output
