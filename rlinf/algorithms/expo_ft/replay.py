"""Bounded image+action replay; episode boundaries and actual executed lengths."""
from __future__ import annotations
import copy
import random
from collections import deque
import torch

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    return copy.deepcopy(value)

def stack_env(observations):
    keys=observations[0].keys()
    result={}
    for key in keys:
        values=[o[key] for o in observations]
        if isinstance(values[0],torch.Tensor):result[key]=torch.cat(values,dim=0)
        elif isinstance(values[0],list):result[key]=sum(values,[])
        else:result[key]=values[0]
    return result

class ChunkReplay:
    def __init__(self,capacity=256,seed=42):
        self.capacity=int(capacity)
        self.rows=deque(maxlen=self.capacity)
        self.rng=random.Random(seed)
        self.total_added=0
        self.success_episodes=set()
    def append(self,row):
        assert row['executed_steps']>0
        assert row['terminated'] or row['truncated'] or row['executed_steps']==row['actions'].shape[-2]
        self.rows.append(cpu(row));self.total_added+=1
    def mark_success(self,episode):
        self.success_episodes.add(episode)
    def sample_rows(self,size):
        if not self.rows:raise RuntimeError('No real executed transitions in replay')
        return [self.rows[self.rng.randrange(len(self.rows))] for _ in range(size)]
    def sample(self,size,backend,device):
        rows=self.sample_rows(size)
        current=stack_env([r['env_obs'] for r in rows])
        nxt=stack_env([r['next_env_obs'] for r in rows])
        obs=backend.critic_observation(current)
        next_obs=backend.critic_observation(nxt)
        next_obs['env_obs']=nxt
        return {'obs':obs,'next_obs':next_obs,
                'actions':torch.cat([r['actions'] for r in rows]).to(device),
                'rewards':torch.tensor([r['reward'] for r in rows],device=device),
                'continuations':torch.tensor([r['continuation'] for r in rows],device=device),
                'executed_steps':torch.tensor([r['executed_steps'] for r in rows],device=device),
                'valids':torch.ones(size,device=device)}
    def state_dict(self):
        return {'capacity':self.capacity,'rows':list(self.rows),'rng':self.rng.getstate(),
                'total_added':self.total_added,'success_episodes':sorted(self.success_episodes)}
    def load_state_dict(self,state):
        if state['capacity']!=self.capacity:raise ValueError('Replay capacity mismatch')
        self.rows=deque(state['rows'],maxlen=self.capacity)
        self.rng.setstate(state['rng']);self.total_added=state['total_added']
        self.success_episodes=set(state['success_episodes'])
