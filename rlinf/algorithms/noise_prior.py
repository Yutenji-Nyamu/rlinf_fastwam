# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Gaussian behavior prior in the physical DSRL noise coordinates.

Inspired by BRAC policy regularization. This is an online noise-space adaptation,
not a claim that BRAC evaluated this policy or benchmark. log_prob must include
all tanh and scale Jacobians. Average over dimensions to keep the coefficient
independent of the latent dimension; do not regress to random Gaussian samples.
"""
import math


def gaussian_prior_kl(noise, log_prob):
    z = noise.float().reshape(noise.shape[0], -1)
    lp = log_prob.float().reshape(z.shape[0], -1).sum(-1, keepdim=True)
    log_prior = -0.5 * (z.square() + math.log(2.0 * math.pi)).sum(-1, keepdim=True)
    return (lp - log_prior) / z.shape[-1]
