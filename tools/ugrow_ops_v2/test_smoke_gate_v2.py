"""CPU acceptance checks using the just-finished real BC/RLT scalar streams."""
import copy
from pathlib import Path
from unittest.mock import patch
import pytest
import runtime_v2 as rt

ROOT=Path('/data/chenyiteng/deployment-20261006/ugrow-bc-rlt-g45-v1')

@pytest.fixture
def bc_stream():
    p=rt.read(ROOT/'bc/plan.json')
    return rt.scalars(p['runs']['smoke']['run'])

def test_bc_accepts_actual_optimizer_stream_without_update_step(bc_stream):
    assert not any('update_step' in k for k in bc_stream)
    with patch.object(rt,'scalars',return_value=bc_stream):
        receipt=rt.smoke_gate({'lane':'bc'},{'run':'unused'})
    assert receipt['updates']['completed_rounds']==2
    assert receipt['updates']['updates_derived_from_original_U5']==10
    assert receipt['finite_train_metrics']

def test_bc_requires_completed_gradient_evidence(bc_stream):
    stream=copy.deepcopy(bc_stream);stream.pop('train/actor/grad_norm')
    with patch.object(rt,'scalars',return_value=stream),pytest.raises(AssertionError):
        rt.smoke_gate({'lane':'bc'},{'run':'unused'})

def test_finite_and_successful_weighting_are_not_relaxed(bc_stream):
    for bad in ('nan','all_unit'):
        stream=copy.deepcopy(bc_stream)
        if bad=='nan':stream['train/bc/actor_loss'][-1]['value']=float('nan')
        else:
            for row in stream['train/ugrow/weight_nonunit_fraction']:row['value']=0
        with patch.object(rt,'scalars',return_value=stream),pytest.raises(AssertionError):
            rt.smoke_gate({'lane':'bc'},{'run':'unused'})

def test_previous_rlt_zero_success_still_fails_gate():
    p=rt.read(ROOT/'rlt/plan.json')
    with pytest.raises(AssertionError,match='No nontrivial U weighting'):
        rt.smoke_gate({'lane':'rlt'},p['runs']['smoke'])

def test_formal_bc_requires_finite_same_round_optimizer_evidence(bc_stream):
    assert rt.formal_metrics({'lane':'bc'},bc_stream,0)
    for bad in ('missing','nan','wrong_step','stale','zero_grad'):
        stream=copy.deepcopy(bc_stream)
        if bad=='missing':stream.pop('train/actor/grad_norm')
        elif bad=='nan':stream['train/bc/actor_loss'][-1]['value']=float('nan')
        elif bad=='wrong_step':stream['train/actor/grad_norm'][-1]['step']+=1
        elif bad=='stale':stream['train/actor/grad_norm'][-1]['time']=-1
        elif bad=='zero_grad':stream['train/actor/grad_norm'][-1]['value']=0
        assert rt.formal_metrics({'lane':'bc'},stream,0) is None,bad
