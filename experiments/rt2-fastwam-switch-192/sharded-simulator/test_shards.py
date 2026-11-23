"""Exercise the real IPC, ordering and recovery paths with deterministic scene data."""
import json
import os
import tempfile
from pathlib import Path

from rlinf.envs.robotwin.sharded_vector_env import ShardedVectorEnv, SimulatorTransportError

FACTORY = '''
import os
class TestVector:
    def __init__(self,*,n_envs,env_seeds,first_index,**kwargs):
        self.seeds=list(env_seeds); self.offset=first_index
    def reset(self,indices,seeds):
        if seeds is not None:
            for i,s in zip(indices,seeds):self.seeds[i]=s
    def get_obs(self):
        return [(self.offset+i,s,os.getpid()) for i,s in enumerate(self.seeds)]
    def step(self,actions):
        return (self.get_obs(),list(actions),[False]*len(actions),[False]*len(actions),[{} for _ in actions])
    def check_seeds(self,seeds):return list(seeds)
'''
with tempfile.TemporaryDirectory() as t:
    (Path(t)/'shard_test_vector.py').write_text(FACTORY)
    os.environ['PYTHONPATH']=t+':'+os.getenv('PYTHONPATH','')
    env=ShardedVectorEnv(n_envs=8,env_seeds=list(range(8)),shards=4,torch_threads=None)
    for _,_,worker in env.parts:worker._factory='shard_test_vector:TestVector'
    try:
        env.reset(env_seeds=list(range(10,18)))
        env.reset(env_idx=[6,1],env_seeds=[66,11])
        env.reset(env_idx=[1,3],env_seeds=[0]*8)
        observations,rewards,*_=env.step(list(range(8)))
        assert [o[0] for o in observations]==rewards==list(range(8))
        assert [o[1] for o in observations]==[10,11,12,13,14,15,66,17]
        assert env.check_seeds(list(range(8)))==list(range(8))
        workers=[w._process for _,_,w in env.parts]
        workers[1].kill(); workers[1].wait(timeout=5)
        env.reset(env_seeds=list(range(80,88)))
        assert all(p.poll() is not None for p in workers)
        assert [o[1] for o in env.get_obs()]==list(range(80,88))
        failed=env.parts[0][2]._process
        failed.kill(); failed.wait(timeout=5)
        try:env.step(list(range(8)))
        except SimulatorTransportError:pass
        else:raise AssertionError('Lost scene must not replay an action')
        env.close()
        env.reset(env_seeds=list(range(8)))
    finally:
        env.close()
print(json.dumps(dict(ipc=True,indexed_seeds=True,ordered_outputs=True,full_reset_recovered=True,
                      mid_episode_loss_propagated=True,offload_reinitialized=True)))
