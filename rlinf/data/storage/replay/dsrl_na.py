# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Aligned raw action and VLA observation fields for DSRL-NA replay."""
import torch


def project_na_fields(trajectory, keep, chunk_length, pack):
    fields = trajectory.forward_inputs
    if 'dsrl_env_actions' not in fields:
        raise ValueError('NA requires the physical actions sent to the environment')
    env_actions = fields['dsrl_env_actions']
    if env_actions.shape[:2] != keep.shape:
        raise ValueError('NA action and trajectory axes differ')
    selected = env_actions[keep]
    if selected.ndim != 3 or selected.shape[1] != chunk_length:
        raise ValueError('NA actions must retain exactly the executed action chunk')
    tokens = {name: fields[name][keep] for name in ['tokenized_prompt', 'tokenized_prompt_mask']}
    packed = {}
    for source, target in [('curr_obs', 'curr_vla'), ('next_obs', 'next_vla')]:
        obs = getattr(trajectory, source)
        raw = {key: obs[key][keep] for key in ['main_images', 'wrist_images', 'extra_view_images', 'states']
               if isinstance(obs.get(key), torch.Tensor)}
        if any(value.shape[0] != selected.shape[0] for value in raw.values()):
            raise ValueError('NA observation rows do not match selected macro transitions')
        packed[target] = pack(raw, tokens)
    packed['env_actions'] = selected.float().reshape(selected.shape[0], -1).cpu().contiguous()
    return packed
