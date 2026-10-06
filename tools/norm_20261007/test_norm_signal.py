"""CPU contract tests using the actual sampler methods and replay interfaces."""
import ast,copy,importlib.util,random,sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
def load(name,rel):
    spec=importlib.util.spec_from_file_location(name,ROOT/rel)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
norm=load('rlinf.algorithms.norm_signal','rlinf/algorithms/norm_signal.py')
DSRL=(ROOT/'rlinf/algorithms/dsrl_ugrow.py').exists()
# BC's frozen baseline can contain unrelated DSRL files; select by actual signature.
src=ROOT/'rlinf/models/embodiment/openpi/openpi_action_model.py'
DSRL='collect_dsrl_u: bool' in src.read_text()
u=load('norm_test_dsrl','rlinf/algorithms/dsrl_ugrow.py') if DSRL else None

def method(name):
    tree=ast.parse(src.read_text());node=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name))
    for arg in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]:arg.annotation=None
    node.returns=None
    scope={'torch':torch,'random':random,'nullcontext':nullcontext,
           'capture_expert_norm':norm.capture_expert_norm,'reduce_expert_norm':norm.reduce_expert_norm}
    if DSRL:scope['relative_disagreement']=u.relative_disagreement
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(src),'exec'),scope)
    return scope[name]

class Block(torch.nn.Module):
    def __init__(self,index):super().__init__();self.index=index
    def forward(self,x):return x + (self.index+1)*torch.tensor([1.,-2.,3.])

class Sampler:
    sample_actions=method('sample_actions')
    _sample_actions_with_prefix_cache=method('_sample_actions_with_prefix_cache')
    def __init__(self):
        self.config=SimpleNamespace(use_dsrl=True,dsrl_u_enabled=True,num_steps=10,action_horizon=50,action_chunk=20,action_dim=32,action_env_dim=14,joint_logprob=False,is_nft=False,ignore_last=False)
        if DSRL:self.dsrl_u_spec=u.make_signal_spec(signal_kind='norm_residual_t5_l3',chunk_length=20)
        self.action_in_proj=SimpleNamespace(weight=torch.empty(1))
        self.layers=torch.nn.ModuleList(Block(i) for i in range(4))
        self.paligemma_with_expert=SimpleNamespace(gemma_expert=SimpleNamespace(model=SimpleNamespace(layers=self.layers)))
        self.use_vlm_value=False;self.observed=[];self.calls=0
    def sample_noise(self,shape,device):return torch.randn(shape,device=device)
    def _preprocess_observation(self,obs,train=False):return None,None,None,None,obs.state
    def _build_prefix_cache(self,*args):return None,None,None
    def _init_nft_state(self,*args):return {}
    def _update_nft_state(self,*args):pass
    def get_logprob_norm(self,x,*args):return torch.zeros_like(x)
    def sample_mean_var_val(self,x,idx,*args):
        self.calls+=1
        h=torch.arange(51,dtype=torch.float32)[None,:,None].expand(x.shape[0],51,3)*(idx+1)
        for i,layer in enumerate(self.layers):
            h=layer(h)
            if idx>=5 and i>=1:self.observed.append(torch.linalg.vector_norm(h[:,-50:].float(),dim=-1))
        # Final norm deliberately destroys magnitude, after the observed blocks.
        h=torch.nn.functional.layer_norm(h,(3,))
        return x*.9,torch.zeros_like(x),torch.zeros(x.shape[0],1),x*.1

def test_actual_sampler_norm_is_passive_and_uses_fifteen_pre_final_norms():
    obs=SimpleNamespace(state=torch.zeros(2,14));noise=torch.randn(2,50,32)
    rng=torch.random.get_rng_state();py_rng=random.getstate()
    base=Sampler();a=base.sample_actions(obs,noise=noise,mode='eval',compute_values=False)
    after=torch.random.get_rng_state();torch.random.set_rng_state(rng)
    model=Sampler();kwargs={'collect_dsrl_u':True} if DSRL else {'norm_enabled':True}
    b=model.sample_actions(obs,noise=noise,mode='eval',compute_values=False,**kwargs)
    key='dsrl_u' if DSRL else 'norm_raw'
    assert model.calls==base.calls==10 and len(model.observed)==15
    assert torch.equal(a['actions'],b['actions']) and torch.equal(a['chains'],b['chains'])
    assert torch.equal(after,torch.random.get_rng_state()) and py_rng==random.getstate()
    torch.testing.assert_close(b[key],torch.stack(model.observed).mean(0)[:,:20])
    assert b[key].shape==(2,20) and not b[key].requires_grad
    assert all(not layer._forward_hooks for layer in model.layers)
    assert (b[key]>1).all(), 'Norm must not inherit U upper bound'

def test_hook_cleanup_on_exception_and_missing_layer():
    model=Sampler()
    with pytest.raises(RuntimeError):
        with norm.capture_expert_norm(model,50):raise RuntimeError('expected')
    assert all(not layer._forward_hooks for layer in model.layers)
    with pytest.raises(ValueError,match='all three'):
        with norm.capture_expert_norm(model,50):pass
    assert all(not layer._forward_hooks for layer in model.layers)
    with pytest.raises(ValueError,match='exactly'):
        norm.reduce_expert_norm([torch.zeros(2,50)])

def test_norm_metadata_rejects_old_signal_and_keeps_existing_weight_mapping(tmp_path):
    if DSRL:
        spec=u.make_signal_spec(signal_kind='norm_residual_t5_l3',chunk_length=20)
        assert u.validate_signal_spec(spec)==spec and spec!=u.make_signal_spec(chunk_length=20)
        with pytest.raises(ValueError):u.validate_signal_spec(dict(spec,deep_layers=2))
        weights,metrics=u.build_dsrl_u_weights(torch.arange(1.,161.).reshape(8,20),torch.ones(8,20,dtype=torch.bool))
        torch.testing.assert_close(weights.mean(),torch.tensor(1.))
        assert metrics['weight_std']>0
        return
    data=load('rlinf.data.online_bc','rlinf/data/online_bc.py')
    mapping=load('norm_test_mapping','rlinf/algorithms/online_bc_dvac.py')
    sys.modules['rlinf.algorithms.online_bc_dvac']=mapping
    collector=data.SuccessEpisodeCollector(1,dvac_log_eps=1e-12,signal_kind='norm_residual_t5_l3')
    def collect(raw):
        collector.reset();collector.append({'observation/state':torch.zeros(1,14),'norm_raw':raw[None]},torch.ones(1,50,14),torch.tensor([True]),torch.tensor([True]))
        return collector.drain(),collector.drain_dvac_moments()
    mapper=mapping.OnlineBCDvac(signal_kind='norm_residual_t5_l3',mapping='bounded_linear',weight_min=0,weight_max=5)
    episodes,moments=collect(torch.linspace(10,100,50));mapper.annotate(episodes,moments)
    assert torch.equal(episodes[0][0]['action_weights'],torch.ones(50))
    episodes2,moments2=collect(torch.linspace(20,200,50));metrics=mapper.annotate(episodes2,moments2)
    assert metrics['norm/weight_nonunit_fraction']>0
    replay=data.SuccessReplay(42,str(tmp_path/'pool'),3,dict(norm.NORM_SIGNAL_SPEC));replay.add_episodes(episodes+episodes2)
    replay.add_episodes([episodes[0]*4]);assert replay.filtered_success_episodes==1
    replay.save_checkpoint(tmp_path/'cp');restored=data.SuccessReplay(42,str(tmp_path/'pool'),3,dict(norm.NORM_SIGNAL_SPEC));restored.load_checkpoint(tmp_path/'cp')
    assert torch.equal(restored.sample(8)['forward_inputs']['action_weights'],replay.sample(8)['forward_inputs']['action_weights'])
    with pytest.raises(ValueError):data.SuccessReplay(42,str(tmp_path/'old'),3,{'kind':'ugrow_10_5'}).load_checkpoint(tmp_path/'cp')
    changed=dict(episodes[0][0],norm_deep_layers=torch.tensor(2))
    with pytest.raises(ValueError):data.validate_norm_record(changed)
