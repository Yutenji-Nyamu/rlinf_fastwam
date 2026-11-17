"""Verify stage routing and replay selection independently of GPU simulation."""
import torch
from rlinf.algorithms.rlt.route import CriticalPhaseRLTRoute,RLTRouteContext


def run_case(mode,version):
    student=torch.full((2,10,14),2.);ref=torch.ones_like(student)
    ctx=RLTRouteContext(env_obs={'rlt_critical_phase':torch.tensor([False,True])},
        rlt_obs={'ref_chunk':ref},student_actions=student,result={'forward_inputs':{}},mode=mode,version=version,rlt_switch_flags=torch.tensor([[False],[True]]))
    out=CriticalPhaseRLTRoute(use_schedule=True,warmup_updates=20).route(ctx)
    expected=ref.clone()
    if mode=='eval' or version>=20:expected[1]=student[1]
    torch.testing.assert_close(out.actions,expected)
    f=out.result['forward_inputs']
    assert f['record_transition'].flatten().tolist()==[False,True]
    assert f['actor_switch'].flatten().tolist()==[False,mode=='eval' or version>=20]
    torch.testing.assert_close(f['action'],expected.flatten(1))


if __name__=='__main__':
    for mode,version in [('train',0),('train',20),('eval',0)]:
        run_case(mode,version);print(mode,version,'PASS')


def test_environment_message_path():
    import numpy as np
    from rlinf.envs.robodojo.robodojo_env import RoboDojoEnv
    from rlinf.data.schema.embodied_types import EnvOutput
    env=RoboDojoEnv.__new__(RoboDojoEnv)
    env.reset_state_ids=torch.tensor([0,1]);env.group_size=1
    env._elapsed_steps=torch.zeros(2,dtype=torch.long);env._returns=torch.zeros(2);env._success=torch.zeros(2,dtype=torch.bool)
    flag=np.array([False,True])
    obs=dict(main_images=np.zeros((2,4,4,3),np.uint8),states=np.zeros((2,14),np.float32),task_descriptions=['task','task'],rlt_critical_phase=flag)
    env._rpc=lambda *a,**k:dict(obs=obs,audit={'layout_ids':[0,1]},rewards=np.zeros((2,10),np.float32),terminations=np.zeros((2,10),bool),truncations=np.zeros((2,10),bool),counts=[10,10],success=[False,False])
    reset_obs,reset_info=env.reset()
    next_obs,_,_,_,infos=env.chunk_step(torch.zeros(2,10,14))
    for raw,info in [(reset_obs,reset_info),(next_obs[0],infos[0])]:
        message=EnvOutput(obs=raw,rlt_switch_flags=info['rlt_switch_flags']).to_dict()
        assert 'rlt_critical_phase' not in message['obs']
        assert message['rlt_switch_flags'].flatten().tolist()==[False,True]
    print('environment message path PASS')


if __name__=='__main__':
    test_environment_message_path()
