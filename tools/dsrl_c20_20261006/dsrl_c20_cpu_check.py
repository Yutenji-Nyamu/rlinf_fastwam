import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['JAX_PLATFORMS']='cpu'
import json,sys,time,types
from pathlib import Path
import torch,yaml
torch.set_num_threads(1)
repo=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dsrl-pi05-u-sz1-20261006')
control=Path('/data/chenyiteng/deployment-20261006/dsrl-u-c20-lease-v2')
sys.path[:0]=[str(repo),str(repo/'tools/dsrl_u_20261006/ops')]
from lease_common import checked,read,save
from lease_owner import request_checked
from rlinf.algorithms.dsrl_ugrow import validate_signal_spec,relative_disagreement,build_dsrl_u_weights
from rlinf.data.storage.replay.dsrl_transition import project_dsrl_trajectory,DSRLTransitionReplayBuffer

p,b,op=checked(control)
out={'time':time.time(),'cuda_visible':os.environ['CUDA_VISIBLE_DEVICES'],'roles':{}}
for role in ('clean','u'):
    req=read(control/(role+'-c20-formal-200-mb256-v2-request.json'))
    rt=request_checked(p,req)
    cfg=yaml.safe_load((rt/'resolved.yaml').read_text())
    model=cfg['actor']['model']; alg=cfg['algorithm'];openpi=model['openpi']
    assert cfg['runner']['resume_dir'] is None and cfg['runner']['max_steps']==200
    assert model['num_action_chunks']==openpi['action_chunk']==20
    assert model['num_steps']==openpi['num_steps']==10
    assert openpi['action_horizon']==50 and openpi['dsrl_action_noise_dim']==32
    assert alg['replay_buffer']['warmup_size']==500 and alg['utd_ratio']==20
    assert cfg['actor']['global_batch_size']==cfg['actor']['micro_batch_size']==256
    C,H,L,T,B,D=20,50,32,10,4,14
    spec=validate_signal_spec(alg['dsrl_u']['spec']) if role=='u' else None
    assert spec==openpi['dsrl_u_spec']
    obs={'main_images':torch.zeros(T,B,224,224,3,dtype=torch.uint8),'states':torch.zeros(T,B,D)}
    tr=types.SimpleNamespace(rewards=torch.zeros(T,B,C),actions=torch.zeros(T,B,H,L),
        curr_obs=obs,next_obs=obs,terminations=torch.zeros(T,B,C,dtype=torch.bool),
        truncations=torch.zeros(T,B,C,dtype=torch.bool),forward_inputs={})
    tr.truncations[-1,:,-1]=True
    if spec:
        torch.manual_seed(1234)
        scores,valid=relative_disagreement(torch.randn(T*B,C,D),torch.randn(T*B,C,D))
        tr.forward_inputs={'dsrl_u':scores.reshape(T,B,C),'dsrl_u_valid':valid.reshape(T,B,C),
                           'dsrl_u_policy_step':torch.zeros(T,B,1,dtype=torch.int64),
                           'dsrl_u_phase':torch.zeros(T,B,1,dtype=torch.int64)}
    batch=project_dsrl_trajectory(tr,action_horizon=H,latent_dim=L,state_dim=D,num_action_chunks=C,gamma=alg['gamma'],u_spec=spec)
    assert batch['actions'].shape==(40,32)
    assert (batch['rewards']==-1).all() and batch['truncations'].sum()==4
    assert torch.allclose(batch['discounts'],torch.full((40,1),.999**20))
    replay=DSRLTransitionReplayBuffer(capacity=64,seed=1234,rank=0,world_size=1,schema_version=2 if spec else 1,u_spec=spec)
    assert len(replay)==0
    assert replay.add_batch(batch)==40 and len(replay)==40
    sample=replay.sample(256)
    metrics={}
    if spec:
        assert sample['dsrl_u'].shape==(256,20)
        weights,metrics=build_dsrl_u_weights(sample['dsrl_u'],sample['dsrl_u_valid'])
        assert weights.shape==(256,1) and torch.isfinite(weights).all()
        assert abs(weights.mean().item()-1)<1e-6
        rejected=False
        try:
            project_dsrl_trajectory(tr,action_horizon=H,latent_dim=L,state_dim=D,num_action_chunks=10,gamma=alg['gamma'],u_spec=spec)
        except ValueError as e:
            rejected='chunk length differs' in str(e)
        assert rejected
    out['roles'][role]={'request_pins_valid':True,'fresh_sft_config':True,'fixture_transitions':len(replay),
                         'latent_shape':list(sample['actions'].shape),'u_shape':list(sample['dsrl_u'].shape) if spec else None,
                         'discount':float(batch['discounts'][0]),'u_metrics':metrics}
assert not torch.cuda.is_initialized()
save(control/'c20-cpu-check.json',out,True)
print(json.dumps(out))
