"""Fast-WAM's current-frame post-block feature, without editing upstream.

The prefill arithmetic below follows FastWAM 7faa711 mot.py:474-526 exactly.
Only the return value changes; Stage 1 can omit retaining each layer's K/V.
"""

from __future__ import annotations

import torch

from .fastwam_rl import ActionConditioning, encode_first_frame_latents


@torch.no_grad()
def prefill_video_features(mot, *, retain_cache=True, **inputs):
    from fastwam.models.wan22.mot import flash_attention

    expert = mot.mixtures["video"]
    x = inputs["video_tokens"]
    cache_k, cache_v = [], []
    for layer_idx in range(mot.num_layers):
        block = expert.blocks[layer_idx]
        q, k, v, residual, gate_msa, shift_mlp, scale_mlp, gate_mlp, _ = (
            mot._build_expert_attention_io(
                expert=expert, block=block, x=x,
                freqs=inputs["video_freqs"], t_mod=inputs["video_t_mod"],
            )
        )
        mixed = flash_attention(
            q=q, k=k, v=v, num_heads=mot.num_heads,
            ctx_mask=inputs["video_attention_mask"].to(device=q.device),
        )
        x = mot._apply_expert_post_block_tensor(
            block=block, residual_x=residual, mixed_attn_out=mixed,
            gate_msa=gate_msa, shift_mlp=shift_mlp, scale_mlp=scale_mlp,
            gate_mlp=gate_mlp, context=inputs["video_context"],
            context_mask=inputs["video_context_mask"],
        )
        if retain_cache:
            cache_k.append(k)
            cache_v.append(v)
    return cache_k, cache_v, x


@torch.no_grad()
def build_rlt_conditioning(
    model, *, input_image, text_context, text_context_mask, proprio,
    action_horizon=32, tiled=False, retain_cache=True, verify_oracle=False,
):
    """Return original teacher conditioning and final contextual video tokens."""
    device = next(model.parameters()).device
    dtype = model.torch_dtype
    context, context_mask = model._append_proprio_to_context(
        context=text_context.to(device=device, dtype=dtype),
        context_mask=text_context_mask.to(device=device, dtype=torch.bool),
        proprio=proprio.to(device=device, dtype=torch.float32),
    )
    latents = encode_first_frame_latents(
        model, input_image.to(device=device, dtype=dtype), tiled=tiled,
    )
    prepared = model.video_expert.prepare(
        x=latents, timestep=torch.zeros(latents.shape[0], device=device, dtype=latents.dtype),
        context=context, context_mask=context_mask, action=None,
        fuse_vae_embedding_in_latents=bool(
            getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
        ),
    )
    tokens, _, t_mod, video_context, video_mask, freqs, _, _, _, tokens_per_frame = prepared
    length = tokens.shape[1]
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=length, action_seq_len=action_horizon,
        video_tokens_per_frame=int(tokens_per_frame), device=tokens.device,
    )
    inputs = dict(
        video_tokens=tokens, video_freqs=freqs, video_t_mod=t_mod,
        video_context=video_context, video_context_mask=video_mask,
        video_attention_mask=attention_mask[:length, :length],
    )
    keys, values, hidden = prefill_video_features(
        model.mot, retain_cache=retain_cache, **inputs,
    )
    if verify_oracle:
        if not retain_cache:
            raise ValueError("The oracle comparison requires retain_cache=True")
        oracle_k, oracle_v = model.mot.prefill_video_cache_tensor(**inputs)
        for observed, expected in zip(keys + values, oracle_k + oracle_v, strict=True):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    conditioning = None
    if retain_cache:
        conditioning = ActionConditioning(
            context=context, context_mask=context_mask,
            video_cache_k=keys, video_cache_v=values,
            action_attention_mask=attention_mask[length:, :],
        )
    return conditioning, hidden
