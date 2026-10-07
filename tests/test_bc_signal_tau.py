"""CPU checks for the requested U/Norm temperature and control contracts."""
import copy
import pytest
import torch
from rlinf.algorithms.online_bc_signal_tau import signal_tau_contract, apply_signal_tau
from rlinf.algorithms.online_bc_dvac_two_level import compute_two_level_bc_weights
from rlinf.algorithms.ugrow_signal import UGROW_SIGNAL_SPEC
from rlinf.algorithms.norm_signal import NORM_SIGNAL_SPEC


def config(kind, tau=1., tricks=True):
    return dict(enabled=True, signal_kind=kind, normalization='two_level_batch',
        factor_mapping='exp_mean', temperature_local=tau, temperature_chunk=tau,
        alpha_local=1., alpha_chunk=1., log_eps=1e-12, range_eps=1e-6,
        chunk_dropout=dict(enabled=tricks, probability=.2, seed=42),
        alpha_schedule=dict(enabled=tricks, **{k:dict(enabled=True,start_step=1,end_step=200,end_alpha=0.) for k in ['local','chunk']}))


@pytest.mark.parametrize('spec', [UGROW_SIGNAL_SPEC, NORM_SIGNAL_SPEC])
def test_raw_signal_matches_existing_dvca_mapping_and_tau_strength(spec):
    kind=spec['kind'];v=torch.linspace(.01,.99,64*50,dtype=torch.float64).reshape(64,50)
    if kind.startswith('norm'):v=v*800
    mask=torch.ones(64,50,14);mask[0,30:]=0
    deviations=[]
    for tau in [1.,2.,3.,4.]:
        c=signal_tau_contract(config(kind,tau,False),spec)
        batch={c['raw_key']:v.clone(),'action_valid_mask':mask}
        w,_=apply_signal_tau(batch,c,runner_step=0,update_step=0)
        ref,_=compute_two_level_bc_weights(v,mask,**c['settings'])
        torch.testing.assert_close(w,ref,rtol=0,atol=0)
        assert torch.equal(batch[c['raw_key']],v) and not w.requires_grad
        assert torch.equal(w[0,30:],torch.ones(20,dtype=w.dtype))
        deviations.append(float((w-1).square().mean()))
    assert all(a>b for a,b in zip(deviations,deviations[1:]))


@pytest.mark.parametrize('spec', [UGROW_SIGNAL_SPEC, NORM_SIGNAL_SPEC])
def test_dropout_rng_and_anneal_endpoints(spec):
    c=signal_tau_contract(config(spec['kind']),spec)
    batch={c['raw_key']:torch.linspace(.01,.99,1024*50).reshape(1024,50),'action_valid_mask':torch.ones(1024,50,14)}
    state=torch.random.get_rng_state().clone()
    w,m=apply_signal_tau(batch,c,runner_step=0,update_step=5)
    assert torch.equal(state,torch.random.get_rng_state())
    assert .15<m['dropout_fraction']<.25 and m['alpha_local']==m['alpha_chunk']==1
    assert torch.equal(w,apply_signal_tau(batch,c,runner_step=0,update_step=5)[0])
    assert not torch.equal(w,apply_signal_tau(batch,c,runner_step=0,update_step=6)[0])
    last,metrics=apply_signal_tau(batch,c,runner_step=199,update_step=7)
    assert torch.equal(last,torch.ones_like(last)) and metrics['alpha_local']==metrics['alpha_chunk']==0


def test_contract_rejects_legacy_and_wrong_signal(tmp_path):
    cfg=config(UGROW_SIGNAL_SPEC['kind']);c=signal_tau_contract(cfg,UGROW_SIGNAL_SPEC)
    p=tmp_path/'signal_tau.pt';torch.save(c,p);assert torch.load(p,weights_only=True)==c
    other=signal_tau_contract(config(UGROW_SIGNAL_SPEC['kind'],2),UGROW_SIGNAL_SPEC)
    assert c!=other
    with pytest.raises(ValueError):signal_tau_contract(dict(cfg,window=5),UGROW_SIGNAL_SPEC)
    with pytest.raises(ValueError):signal_tau_contract(cfg,NORM_SIGNAL_SPEC)
    with pytest.raises(ValueError):signal_tau_contract(dict(cfg,temperature_local=0),UGROW_SIGNAL_SPEC)
