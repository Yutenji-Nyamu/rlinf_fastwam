# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the deployed worker RNG boundary, without loading a policy/env."""
import asyncio
import random
from inspect import unwrap
from types import SimpleNamespace, MethodType

import numpy as np
import pytest
import torch

from rlinf.utils.utils import seed_everything
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

def worker(seed=42, rank=0):
    w = SimpleNamespace(rollout_seed=seed, _rank=rank)
    w._seed_rollout = MethodType(MultiStepRolloutWorker._seed_rollout, w)
    return w

def draw(device):
    # The deployed ODE path consumes an initial normal and ten zero-scaled draws.
    result = torch.normal(0., 1., (2, 50, 32), device=device)
    for _ in range(10):
        torch.normal(0., 1., result.shape, device=device)
    return result

@pytest.mark.parametrize('device', DEVICES)
def test_same_seed_pairs_fresh_query_sequence(device):
    a=worker();a._seed_rollout();first=[draw(device) for _ in range(4)]
    b=worker();b._seed_rollout();second=[draw(device) for _ in range(4)]
    assert all(torch.equal(x,y) for x,y in zip(first,second))
    assert not torch.equal(first[0],first[1])

@pytest.mark.parametrize('device', DEVICES)
def test_eval_repeats_without_consuming_train_rng(device):
    w=worker();w._seed_rollout();draw(device)
    expected=(draw(device), random.random(), np.random.random())
    w._seed_rollout();draw(device)
    async def evaluate(*_):
        values=[draw(device) for _ in range(16)]
        random.random();np.random.random()
        return values
    w._evaluate=evaluate
    first=asyncio.run(unwrap(MultiStepRolloutWorker.evaluate)(w,None,None))
    second=asyncio.run(unwrap(MultiStepRolloutWorker.evaluate)(w,None,None))
    actual=(draw(device),random.random(),np.random.random())
    assert all(torch.equal(x,y) for x,y in zip(first,second))
    assert torch.equal(actual[0],expected[0]) and actual[1:]==expected[1:]

@pytest.mark.parametrize('device', DEVICES)
def test_exception_restores_rng(device):
    w=worker();w._seed_rollout();expected=draw(device)
    w._seed_rollout()
    async def fail(*_):
        draw(device)
        raise RuntimeError('expected')
    w._evaluate=fail
    with pytest.raises(RuntimeError,match='expected'):
        asyncio.run(unwrap(MultiStepRolloutWorker.evaluate)(w,None,None))
    assert torch.equal(draw(device),expected)

def test_unconfigured_worker_keeps_existing_rng():
    w=worker(None)
    seed_everything(17);expected=torch.randn(4)
    seed_everything(17);w._seed_rollout()
    async def evaluate(*_):return torch.randn(4)
    w._evaluate=evaluate
    assert torch.equal(asyncio.run(unwrap(MultiStepRolloutWorker.evaluate)(w,None,None)),expected)

def test_logical_rank_not_physical_gpu():
    w=worker(42,0);w._seed_rollout();a=torch.randn(4)
    w=worker(42,1);w._seed_rollout();b=torch.randn(4)
    assert not torch.equal(a,b)
