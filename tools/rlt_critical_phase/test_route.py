"""Verify stage routing and replay selection independently of GPU simulation."""
import torch
from rlinf.algorithms.rlt.route import CriticalPhaseRLTRoute,RLTRouteContext


def run_case(mode,version):
    student=torch.full((2,10,14),2.);ref=torch.ones_like(student)
    ctx=RLTRouteContext(env_obs={'rlt_critical_phase':torch.tensor([False,True])},
        rlt_obs={'ref_chunk':ref},student_actions=student,result={'forward_inputs':{}},mode=mode,version=version)
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
