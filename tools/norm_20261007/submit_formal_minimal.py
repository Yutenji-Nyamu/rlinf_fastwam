import importlib.util,json,shutil,time
from pathlib import Path
S=Path('/data/chenyiteng/projects/norm-bc-dsrl-sz3-20261007');C=S/'control';cycle=S/'rlt-after-norm-g67-v1'
def read(p):return json.loads(Path(p).read_text())
spec=importlib.util.spec_from_file_location('norm_formal',cycle/'rlt_returned_cycle.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.install_helper(cycle);H=m.H
approval=read(C/'minimal-smoke-approval.json');s=read(C/'status.json');plan=read(C/'plan.json')
assert all(r['state']=='FAILED_HOLD' for r in s['roles'].values()),s['roles']
assert not H.gpu_processes([6,7]);actors=H.actors(plan)
for role,identity in approval['stopped_drivers'].items():
    assert not H.same(identity) and not H.active(actors,identity['namespace'])
    rt=Path(plan['roles'][role]['runs']['smoke']['runtime']);finished=read(rt/'finished.json');assert finished['exit_code'] in (15,143) and finished['cleanup_error'] is None,finished
    proof={'time':time.time(),'status':'MINIMAL_SMOKE_PASSED_USER_GATE','intentional_stop':True,'driver_cleanup':finished,'owner_exit_observation':read(C/'receipts'/(role+'-smoke-v1-error.json')),'namespace_released':True,'gpu_released':True,'probe':read(C/'receipts'/('bc-probe-v2-accepted.json' if role=='bc' else 'dsrl-probe-v1-accepted.json')),'gate':approval['gate'],'limitations':['DSRL GPU weighted update awaits formal warmup500','No smoke checkpoint was required under the user revised gate']}
    with (C/'receipts'/(role+'-smoke-v1-minimal-accepted.json')).open('x') as f:json.dump(proof,f,indent=2)
    source=C/'prepared'/(role+'-formal-v1.json');target=C/'requests'/source.name;assert not target.exists();shutil.copyfile(source,target)
print(json.dumps({'formal_submitted':['bc300_gpu6','dsrl200_gpu7'],'fresh_SFT_and_empty_replay':True,'release_verified':True}),flush=True)
