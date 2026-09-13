"""Frozen Fast-WAM + current causal-AR RLT encoder for Stage 2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenEncoder

from .fastwam_policy import FastWAMPolicy, FastWAMPolicyConfig
from .fastwam_rl import flow_sde_rollout, prepare_initial_action_latents
from .rlt_features import build_rlt_conditioning
from .robotwin_adapter import adapt_robotwin_observation, denormalize_actions


@dataclass(frozen=True)
class FastWAMRLTConfig(FastWAMPolicyConfig):
    rlt_action_adapter: str = "fastwam_robotwin_zscore_v1"


def feature_identity(checkpoint_path, stats_path, token_kwargs):
    """Cheap identity of pinned teacher artifact plus exact normalization bytes."""
    checkpoint_path = Path(checkpoint_path).resolve()
    return {
        "schema": 1,
        "teacher_checkpoint_path": str(checkpoint_path),
        "teacher_checkpoint_bytes": checkpoint_path.stat().st_size,
        "official_revision": "7faa71108368fbb3b6885649f112af607427a2d4",
        "stats_sha256": hashlib.sha256(Path(stats_path).read_bytes()).hexdigest(),
        "feature": "first_frame_video_final_post_block",
        "reconstruction": "current_causal_ar_teacher_forcing",
        "token_kwargs": dict(token_kwargs),
        "action_horizon": 32,
        "action_chunk": 24,
        "action_dim": 14,
        "action_adapter": "fastwam_robotwin_zscore_v1",
        "student_output_activation": "identity",
    }


class FastWAMRLTPolicy(FastWAMPolicy):
    def __init__(self, *, model, processor, config, stage1_checkpoint, identity):
        super().__init__(model=model, processor=processor, config=config)
        stage1 = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
        if stage1["identity"] != identity:
            raise ValueError("Fast-WAM RLT Stage1 teacher/features/normalization identity mismatch")
        self.rlt_encoder = RLTTokenEncoder(**identity["token_kwargs"])
        self.rlt_encoder.load_state_dict(stage1["encoder"], strict=True)
        self.rlt_encoder.to(device=self.device, dtype=torch.float32)
        self.stage1_steps = int(stage1["step"])
        self.stage1_identity = identity
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def extract_rlt_obs(self, env_obs, return_decode_context=False):
        images, proprio, prompts = adapt_robotwin_observation(
            env_obs, self.processor, device=self.device, dtype=self.model_dtype,
        )
        text_context, text_mask = self.model.encode_prompt(prompts)
        batch = images.shape[0]
        initial = prepare_initial_action_latents(
            batch_size=batch, action_horizon=32, action_dim=14,
            device=self.device, dtype=self.model_dtype, rand_device=self.config.rand_device,
            seed=self.config.eval_seed, broadcast_singleton=True,
        )
        zs, refs = [], []
        for start in range(0, batch, self.config.model_forward_batch_size):
            item = slice(start, min(start + self.config.model_forward_batch_size, batch))
            condition, hidden = build_rlt_conditioning(
                self.model, input_image=images[item], text_context=text_context[item],
                text_context_mask=text_mask[item], proprio=proprio[item],
            )
            expected = self.stage1_identity["token_kwargs"]
            if hidden.shape[1:] != (expected["prefix_seq_len"], expected["input_dim"]):
                raise ValueError(f"Unexpected Fast RLT feature shape {tuple(hidden.shape)}")
            z = self.rlt_encoder(hidden.float()).flatten(1)
            teacher = flow_sde_rollout(
                self.model, conditioning=condition, initial_latents=initial[item],
                num_inference_steps=self.config.num_inference_steps,
                sigma_shift=self.config.sigma_shift, noise_level=self.config.noise_level,
                deterministic=True,
            )
            zs.append(z.float().cpu())
            refs.append(teacher.actions.float().cpu())
            # Cache lifetimes end at this microbatch, never enter replay.
            del condition, hidden, teacher
        template = torch.cat(refs)
        obs = {
            "z_rl": torch.cat(zs).contiguous(),
            "proprio": proprio.float().cpu().contiguous(),
            "ref_chunk": template[:, :24].contiguous(),
        }
        if return_decode_context:
            return obs, {"teacher_template": template}
        return obs

    def decode_rlt_action(self, actions, context):
        template = context["teacher_template"].clone()
        prefix = torch.as_tensor(actions).detach().cpu().float().reshape(-1, 24, 14)
        if prefix.shape[0] != template.shape[0]:
            raise ValueError("RLT action/decode batch mismatch")
        template[:, :24] = prefix
        return denormalize_actions(template, self.processor)[:, :24].contiguous()
