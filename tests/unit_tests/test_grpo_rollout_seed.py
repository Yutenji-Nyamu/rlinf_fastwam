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

def sampler(device, mode="train"):
    # Execute the production GRPO sampler and its actual Python SDE-step draw;
    # replace expensive neural dynamics with deterministic tensor arithmetic.
    from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0ForRLActionPrediction
    fake=SimpleNamespace(config=SimpleNamespace(num_steps=10,action_horizon=50,action_dim=32,
        joint_logprob=False,is_nft=False,ignore_last=False,noise_method="flow_sde",action_chunk=50,action_env_dim=14),use_vlm_value=False)
    fake.sample_noise=lambda shape,dev: torch.normal(0.,1.,shape,device=dev)
    fake._init_nft_state=lambda *_: None
    fake._update_nft_state=lambda *_: None
    fake.get_logprob_norm=lambda x,*_: torch.zeros_like(x)
    def dynamics(x,idx,state,prefix,past,method,steps,compute):
        return x*.9,torch.full_like(x,.1 if method=="flow_sde" else 0.),torch.zeros((x.shape[0],1),device=x.device),torch.zeros_like(x)
    fake.sample_mean_var_val=dynamics
    return OpenPi0ForRLActionPrediction._sample_actions_with_prefix_cache(fake,
        torch.zeros((2,14),device=device),None,None,None,mode=mode)

@pytest.mark.parametrize('device', DEVICES)
def test_actual_grpo_sampler_initial_noise_and_sde_indices_pair(device):
    w=worker(42,0);w._seed_rollout();a=[sampler(device) for _ in range(12)]
    w._seed_rollout();b=[sampler(device) for _ in range(12)]
    assert all(torch.equal(x['chains'],y['chains']) and torch.equal(x['denoise_inds'],y['denoise_inds']) for x,y in zip(a,b))
    assert len({int(x['denoise_inds'][0,0]) for x in a})>1
    assert not torch.equal(a[0]['chains'][:,0],a[1]['chains'][:,0])
    assert all(x['denoise_inds'].unique().numel()==1 for x in a)

@pytest.mark.parametrize('device', DEVICES)
def test_eval_preserves_next_actual_grpo_noise_and_python_sde_step(device):
    w=worker(42,1);w._seed_rollout();sampler(device);expected=sampler(device)
    w._seed_rollout();sampler(device)
    async def evaluate(*_):
        return [sampler(device,mode="eval") for _ in range(4)]
    w._evaluate=evaluate
    a=asyncio.run(unwrap(MultiStepRolloutWorker.evaluate)(w,None,None))
    b=asyncio.run(unwrap(MultiStepRolloutWorker.evaluate)(w,None,None))
    actual=sampler(device)
    assert all(torch.equal(x['chains'],y['chains']) for x,y in zip(a,b))
    assert torch.equal(expected['chains'],actual['chains'])
    assert torch.equal(expected['denoise_inds'],actual['denoise_inds'])

def test_real_cuda_coverage_required():
    assert torch.cuda.is_available(), "Run this targeted test with the approved free CUDA device."
