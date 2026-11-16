"""CPU tests for NA value supervision, trajectory alignment and exact resumes."""
import tempfile
from types import SimpleNamespace
import torch
from rlinf.algorithms.dsrl_na import action_td_loss, noise_distillation_loss, index_tree
from rlinf.data.storage.replay.dsrl_na import project_na_fields
from rlinf.data.storage.replay.dsrl_transition import DSRLTransitionReplayBuffer
from rlinf.models.embodiment.modules.gaussian_shift_policy import GaussianShiftPolicy
from rlinf.models.embodiment.modules.dsrl_na import model_operation


def test_loss_contract():
    q=torch.zeros(3,2,requires_grad=True);n=torch.tensor([[-8.],[-4.],[-3.]],requires_grad=True)
    rewards=torch.tensor([[0.],[-1.],[-1.]])
    loss,target=action_td_loss(q,rewards,torch.tensor([[False],[True],[True]]),torch.full((3,1),.9),n)
    torch.testing.assert_close(target,torch.tensor([[0.],[-4.6],[-3.7]]))
    loss.backward();assert q.grad is not None and n.grad is None
    nq=torch.zeros(3,2,requires_grad=True);aq=torch.tensor([[1.,3.],[4.,6.],[-3.,1.]],requires_grad=True)
    loss,target=noise_distillation_loss(nq,aq)
    torch.testing.assert_close(target,torch.tensor([[2.],[5.],[-1.]]))
    loss.backward();assert nq.grad is not None and aq.grad is None


def test_projection_alignment():
    t,b,c=3,2,10;v=torch.arange(t*b).reshape(t,b,1)
    obs={'main_images':v[...,None,None].expand(t,b,3,4,4).byte(),'states':v.expand(t,b,14).float()}
    nxt={k:x+1 for k,x in obs.items()}
    actions=v[...,None].expand(t,b,c,14).float()
    fields={'dsrl_env_actions':actions,'tokenized_prompt':v.expand(t,b,5),'tokenized_prompt_mask':torch.ones(t,b,5,dtype=torch.bool)}
    keep=torch.tensor([[True,True],[False,True],[False,False]])
    def pack(raw,tokens):return {'image':raw['main_images'],'state':raw['states'],**tokens}
    out=project_na_fields(SimpleNamespace(curr_obs=obs,next_obs=nxt,forward_inputs=fields),keep,c,pack)
    assert out['env_actions'].shape==(3,140)
    torch.testing.assert_close(out['env_actions'][:,0],torch.tensor([0.,1.,3.]))
    torch.testing.assert_close(out['next_vla']['state'][:,0],torch.tensor([1.,2.,4.]))
    assert out['curr_vla']['tokenized_prompt'][:,0].tolist()==[0,1,3]


def test_replay_resume():
    n=5;obs={'main_images':torch.rand(n,3,4,4).bfloat16(),'states':torch.rand(n,14)}
    batch={'curr_obs':obs,'next_obs':obs,'actions':torch.rand(n,1600).bfloat16(),
      'rewards':-torch.ones(n,1),'continuations':torch.ones(n,1,dtype=torch.bool),
      'terminations':torch.zeros(n,1,dtype=torch.bool),'truncations':torch.zeros(n,1,dtype=torch.bool),'discounts':torch.full((n,1),.99),
      'env_actions':torch.rand(n,140),'curr_vla':{'images':{'base':torch.zeros(n,3,4,4,dtype=torch.uint8)}},
      'next_vla':{'images':{'base':torch.ones(n,3,4,4,dtype=torch.uint8)}}}
    a=DSRLTransitionReplayBuffer(capacity=6,seed=1,rank=0,world_size=1,schema_version=5);a.add_batch(batch);a.add_batch(batch)
    with tempfile.TemporaryDirectory() as d:
        a.save_checkpoint(d);b=DSRLTransitionReplayBuffer(capacity=6,seed=9,rank=0,world_size=1,schema_version=5);b.load_checkpoint(d)
        aa,bb=a.sample(9),b.sample(9)
        torch.testing.assert_close(aa['actions'],bb['actions']);torch.testing.assert_close(aa['env_actions'],bb['env_actions'])
        assert b.total_inserted==10 and b.resident_size==6
    old={k:v for k,v in batch.items()if k not in ['env_actions','curr_vla','next_vla']}
    legacy=DSRLTransitionReplayBuffer(capacity=6,seed=1,rank=0,world_size=1,schema_version=3)
    legacy.add_batch(old);assert len(legacy)==5


def test_total_shift_bound():
    p=GaussianShiftPolicy(4,1600,hidden_dims=(8,),shift_bound=.5,l2_bound=4.)
    with torch.no_grad():p.mean_layer.bias.fill_(8)
    d=p(torch.zeros(4,4));assert (d.mean.norm(dim=-1)<=4.00001).all()
    torch.testing.assert_close(d.stddev,torch.ones_like(d.stddev))
    d.mean.sum().backward();assert p.mean_layer.bias.grad.isfinite().all()


def test_openpi_observation_contract():
    from openpi.models.model import Observation
    image=torch.zeros(2,4,4,3,dtype=torch.uint8)
    tokens={'tokenized_prompt':torch.ones(2,5,dtype=torch.long),'tokenized_prompt_mask':torch.ones(2,5,dtype=torch.bool)}
    def transform(raw,transpose):
        return {'image':{'base_0_rgb':image},'image_mask':{'base_0_rgb':torch.ones(2,dtype=torch.bool)},'state':raw['states'],**tokens}
    model=SimpleNamespace(config=SimpleNamespace(dsrl_na_enabled=True),obs_processor=lambda raw:{'prompt':raw['task_descriptions'],'states':raw['states']},input_transform=transform)
    packed=model_operation(model,'pack',{'states':torch.zeros(2,14)},tokens=tokens)
    obs=Observation.from_dict(packed)
    assert obs.images['base_0_rgb'].shape==(2,3,4,4)
    torch.testing.assert_close(obs.tokenized_prompt,tokens['tokenized_prompt'])


if __name__=='__main__':
    torch.set_num_threads(2)
    for test in [test_loss_contract,test_projection_alignment,test_replay_resume,test_total_shift_bound,test_openpi_observation_contract]:
        test();print(test.__name__,'PASS',flush=True)
