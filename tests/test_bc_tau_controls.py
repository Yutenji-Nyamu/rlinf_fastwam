import importlib.util
from pathlib import Path
import torch
R = Path(__file__).resolve().parents[1]

def module(rel):
    s = importlib.util.spec_from_file_location(rel.replace('/', '_'), R / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

w = module('rlinf/algorithms/online_bc_dvac_two_level.py')
c = module('rlinf/algorithms/online_bc_dvac_controls.py')

def test_exp_matches_rlt_full_mask():
    s = importlib.util.spec_from_file_location('rlt_reference', '/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-rlt-notricks-four-20260920/rlinf/algorithms/rlt/dvac_two_level.py')
    r = importlib.util.module_from_spec(s)
    s.loader.exec_module(r)
    v = torch.arange(1, 2049, dtype=torch.float64).reshape(128, 16)
    out, _ = w.compute_two_level_bc_weights(v, torch.ones(128,16,14), factor_mapping='exp_mean', temperature_local=2.5, temperature_chunk=2.5)
    lv = torch.log(v + 1e-12)
    ref = r._exp_minmax(lv, alpha=1., eps=1e-6, temperature=2.5) * r._exp_minmax(lv.mean(-1), alpha=1., eps=1e-6, temperature=2.5)[:,None]
    torch.testing.assert_close(out, ref)

def test_neutral_mask():
    v = torch.tensor([[1.,2.,3.],[5.,5.,5.]])
    mask = torch.ones(2,3,4)
    mask[0,2] = 0
    out, _ = w.compute_two_level_bc_weights(v, mask, alpha_local=0, alpha_chunk=0, factor_mapping='exp_mean')
    assert torch.equal(out, torch.ones_like(v))

def test_dropout_private_rng():
    weights = torch.full((1024,50),1.3)
    eligible = torch.ones(1024,dtype=torch.bool)
    cfg = {'chunk_dropout': {'probability':.2,'seed':42}}
    state = torch.random.get_rng_state()
    out, drop = c.apply_chunk_dropout(weights, eligible, cfg, update_step=5)
    assert torch.equal(state, torch.random.get_rng_state())
    assert .15 < float(drop.float().mean()) < .25
    assert (out[drop] == 1).all() and (out[~drop] == weights[~drop]).all()
    assert torch.equal(out, c.apply_chunk_dropout(weights, eligible, cfg, update_step=5)[0])

def test_schedule_and_gradient():
    ctrl = {'alpha_schedule': {'enabled':True, **{k: {'start_step':1,'end_step':200,'end_alpha':0.} for k in ['local','chunk']}}}
    assert c.effective_alphas(1,1,ctrl,runner_step=0) == (1,1)
    assert c.effective_alphas(1,1,ctrl,runner_step=199) == (0,0)
    assert c.effective_alphas(1,1,ctrl,runner_step=250) == (0,0)
    x = torch.tensor(2.,requires_grad=True)
    v = torch.arange(1.,513.).reshape(128,4)
    weights, _ = w.compute_two_level_bc_weights(v, torch.ones(128,4,2), alpha_local=0, alpha_chunk=0, factor_mapping='exp_mean')
    ((x*weights)**2).mean().backward()
    assert x.grad == 4
