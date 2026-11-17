# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Frozen-VLA decoding and action-space critics for DSRL-NA."""
import torch
from rlinf.models.embodiment.modules.compact_encoders import CompactMultiQHead, CompactStateEncoder, LightweightImageEncoder64


def last_valid_prefix(prefix, mask):
    """Match arayabrain/openpi_pub Pi0.get_prefix_rep: last unmasked token."""
    if not mask.any(-1).all():
        raise ValueError("VLA prefix has no valid token")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    index = positions.masked_fill(~mask.bool(), -1).max(-1).values
    return prefix[torch.arange(len(prefix), device=prefix.device), index].float().detach()


def append_vla_features(model, obs, state):
    if model.config.dsrl_vla_features:
        features = obs['vla_features'].to(device=state.device, dtype=state.dtype)
        if features.shape != (state.shape[0], 2048):
            raise ValueError('Expected one frozen 2048-D VLA prefix per observation')
        state = torch.cat([state, features], dim=-1)
    return state


def initialize_action_critic(model, dtype):
    c = model.config
    model.action_critic_image_encoder = LightweightImageEncoder64(1, c.dsrl_image_latent_dim, 64).to(dtype=dtype)
    model.action_critic_state_encoder = CompactStateEncoder(c.dsrl_state_dim, c.dsrl_state_latent_dim).to(dtype=dtype)
    model.action_q_head = CompactMultiQHead(state_dim=c.dsrl_state_latent_dim + (2048 if c.dsrl_vla_features else 0), image_dim=c.dsrl_image_latent_dim,
        action_dim=c.action_chunk * c.action_env_dim, hidden_dims=c.dsrl_hidden_dims,
        num_q_heads=c.dsrl_num_q_heads, output_dim=1).to(dtype=dtype)
    model.action_critic_image_encoder.channels_last = c.dsrl_channels_last
    model.action_q_head.vectorized = c.dsrl_vectorized_q
    model.action_q_head.compile_vectorized = c.dsrl_compile_q
    model.action_q_head.compile_mode = c.dsrl_compile_q_mode


def action_q_forward(model, obs, actions, detach_encoder=False, **kwargs):
    images = model._preprocess_dsrl_images(obs['images'] if 'images' in obs else [obs['main_images']])
    states = model._preprocess_states(obs['states'])
    parameter = next(model.action_critic_image_encoder.parameters())
    im = model.action_critic_image_encoder(images.to(device=parameter.device, dtype=parameter.dtype))
    st = model.action_critic_state_encoder(states.to(device=parameter.device, dtype=parameter.dtype))
    st = append_vla_features(model, obs, st)
    if detach_encoder: im, st = im.detach(), st.detach()
    actions = actions.reshape(actions.shape[0], -1).to(device=parameter.device, dtype=parameter.dtype)
    if actions.shape[1] != model.config.action_chunk * model.config.action_env_dim:
        raise ValueError('Action Q requires exactly the executed C x environment-action coordinates')
    return model.action_q_head(st, im, actions)


@torch.no_grad()
def model_operation(model, na_operation, obs, actions=None, tokens=None, **kwargs):
    if not model.config.dsrl_na_enabled:
        raise ValueError('DSRL-NA operation requires explicit model opt-in')
    if na_operation == 'pack':
        raw = dict(obs)
        raw.setdefault('wrist_images', None); raw.setdefault('extra_view_images', None)
        has_prompt = 'task_descriptions' in raw
        raw.setdefault('task_descriptions', [''] * len(raw['states']))
        inputs = model.obs_processor(raw)
        if not has_prompt:
            inputs.pop('prompt')
            inputs.update(tokens)
        # pi0.5 encodes normalized state into the prompt. Retokenize each
        # observation using the original instruction, especially next_obs.
        processed = model.input_transform(inputs, transpose=False)
        # Preserve the exact transformed image values; the production transform
        # emits uint8 RGB, which Observation.from_dict converts during decoding.
        for image in processed['image'].values():
            if image.dtype != torch.uint8:
                raise ValueError(f'NA replay expects lossless uint8 transformed RGB, got {image.dtype}')
        return processed
    if na_operation == 'features':
        from openpi.models import model as observation_module
        observation = observation_module.Observation.from_dict(model.precision_processor(dict(obs)))
        prefix, mask, _, _, _ = model._build_rlt_prefix_cache(observation, train=False)
        return last_valid_prefix(prefix, mask)
    if na_operation == 'decode':
        from openpi.models import model as observation_module
        processed = model.precision_processor(dict(obs))
        observation = observation_module.Observation.from_dict(processed)
        noise = actions.reshape(-1, model.config.action_horizon, model.config.action_dim)
        output = model.sample_actions(observation, noise=noise, mode='eval', compute_values=False)
        real = model.output_transform({'actions': output['actions'], 'state': observation.state})['actions']
        return real.reshape(real.shape[0], -1).float()
    raise ValueError(na_operation)
