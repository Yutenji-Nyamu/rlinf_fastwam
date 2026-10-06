"""Start the proven lease owner only after the C10 pair has fully released."""
import json,os,subprocess,sys,time
from pathlib import Path
control=Path('/data/chenyiteng/deployment-20261006/dsrl-u-c20-lease-v2')
code=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dsrl-pi05-u-sz1-20261006/tools/dsrl_u_20261006/ops')
sys.path.insert(0,str(code))
from lease_common import checked,read,save,gpu_released,sha
from lease_owner import request_checked
p,b,op=checked(control)
capture=read(control/'c10-capture.json')
release=read(control/'c10-released.json')
assert release['all_released'] and not b.same(capture['owner'])
assert not (control/'c20-launch-attempt.json').exists()
assert not (control/'owner-identity.json').exists()
for ident in capture['protected'].values():
    assert b.same(ident)
for gpu,row in capture['targets'].items():
    assert not b.same(row['driver'])
    assert gpu_released(p,b,op,gpu,row['namespace'])[0]
    assert not op.active_actors(p,'dsrl-u-20261006-'+row['role']+'-c20-formal-200-mb256-v2')
    assert not (Path(capture['old_control'])/'receipts'/('gpu'+gpu+'-restore-attempt.json')).exists()
    request_checked(p,read(control/(row['role']+'-c20-formal-200-mb256-v2-request.json')))
save(control/'c20-launch-attempt.json',{'time':time.time(),'release_sha256':sha(control/'c10-released.json'),
                                      'plan_sha256':sha(control/'plan.json')},True)
env=os.environ.copy()
for k in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','LD_PRELOAD','RLINF_OPENDW_GPU_SCOPE_MANIFEST'):
    env.pop(k,None)
env['PYTHONDONTWRITEBYTECODE']='1'
argv=[p['dsrl_python'],'-u','-B',str(code/'lease_owner.py'),'--control',str(control),'owner']
with (control/'owner.log').open('x') as log:
    proc=subprocess.Popen(argv,cwd=p['dsrl_repo'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
identity=b.proc(proc.pid)
save(control/'c20-owner-launched.json',{'time':time.time(),'identity':identity,'argv':argv},True)
deadline=time.monotonic()+60
while True:
    assert proc.poll() is None,'New owner exited; inspect owner.log'
    if (control/'status.json').exists():
        state=read(control/'status.json')
        if all(r['state']=='DEV_HOLD' for r in state['slots'].values()):break
    assert time.monotonic()<deadline,'Owner startup exceeded 60 seconds'
    time.sleep(1)
for role in ('clean','u'):
    argv=[p['dsrl_python'],'-B',str(code/'lease_owner.py'),'--control',str(control),'submit',
          '--request',str(control/(role+'-c20-formal-200-mb256-v2-request.json'))]
    result=subprocess.run(argv,cwd=p['dsrl_repo'],env=env,text=True,capture_output=True,timeout=45)
    assert result.returncode==0,result.stderr
    print(result.stdout.strip(),flush=True)
deadline=time.monotonic()+60
while True:
    assert proc.poll() is None
    state=read(control/'status.json')
    assert not any(r['state']=='NEEDS_ATTENTION' for r in state['slots'].values()),state
    if all(r['state']=='FORMAL' and b.same(r['driver']) for r in state['slots'].values()):break
    assert time.monotonic()<deadline,'Formal launch exceeded 60 seconds; inspect receipts'
    time.sleep(2)
for ident in capture['protected'].values():
    assert b.same(ident)
save(control/'c20-formal-started.json',{'time':time.time(),'state':state,'protected_unchanged':True},True)
print(json.dumps({'state':state,'protected_unchanged':True}))
