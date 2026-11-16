# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""DSRL-NA losses: Bellman learning in action space, supervised noise Q."""
import torch
import torch.nn.functional as F


def reduce_q(q, reduction='mean'):
    if reduction == 'mean': return q.mean(-1, keepdim=True)
    if reduction == 'min': return q.min(-1, keepdim=True).values
    raise ValueError(reduction)


def action_td_loss(q, rewards, continuations, discounts, next_q):
    target = (rewards + continuations.to(rewards.dtype) * discounts * next_q).detach()
    return F.mse_loss(q, target.expand_as(q)), target


def noise_distillation_loss(noise_q, action_q, reduction='mean'):
    target = reduce_q(action_q, reduction).detach()
    return F.mse_loss(noise_q, target.expand_as(noise_q)), target


def index_tree(tree, index):
    if isinstance(tree, dict): return {k: index_tree(v, index) for k, v in tree.items()}
    if isinstance(tree, torch.Tensor): return tree.index_select(0, index.to(tree.device))
    raise TypeError(type(tree))


def detach_cpu(tree):
    if isinstance(tree, dict): return {k: detach_cpu(v) for k, v in tree.items()}
    return tree.detach().cpu().contiguous()
