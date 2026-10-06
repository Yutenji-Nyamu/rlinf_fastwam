"""One explicit user-authorized formal start; same training source and recipe."""
import copy,importlib.util,json,math,os,time
from pathlib import Path
import yaml
PREV=Path('/data/chenyiteng/deployment-20261006/ugrow-bc-rlt-g45-v2')
S=PREV.with_name('ugrow-rlt-g5-formal-v3');oldstage=PREV/'rlt';stage=S/'rlt';stage.mkdir(exist_ok=False)
def load(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
rt=load(S/'tools/runtime_v3.py','rt');base=load(S/'tools/rlt_lease.py','base');helper=load(S/'tools/returned_rlt_lease.py','helper')
read=base._read;sha=base._sha
oldplan=read(oldstage/'plan.json');load(PREV/'tools/runtime_v2.py','oldrt').configure(oldstage)[0].checked()
assert oldplan['gpu']==5 and oldplan['lane']=='rlt'
smoke=oldplan['runs']['smoke'];finished=read(Path(smoke['run'])/'runtime/finished.json')
assert finished['exit_code']==0 and read(oldstage/'smoke-released.json')['scope']['released']
metrics=rt.scalars(smoke['run']);train={k:v for k,v in metrics.items() if k.startswith('train/')}
assert train and all(math.isfinite(q['value']) for v in train.values() for q in v)
assert metrics['train/rlt/update_step'][-1]['value']>0 and metrics['train/rlt_ugrow/rollout_u_nonzero_count'][-1]['value']>0
cp_marker=Path(smoke['run'])/Path(smoke['run']).name/'checkpoints/global_step_4/actor/sac_components/rlt_trainer_state/complete.json'
assert read(cp_marker)['complete'] and read(cp_marker)['update_step']==8
authorization={'time':time.time(),'operation_id':'ugrow-rlt-g5-formal-v3','source_head':oldplan['head'],
 'user_requested_formal':True,'user_request':'5卡的rlt也启动正式吧，应该很清晰，没啥问题？简洁精准进行；',
 'engineering_smoke_verified':True,'successful_weighting_coverage':'pending','prior_smoke':smoke,
 'complete_smoke_checkpoint':read(cp_marker),'note':'User directs formal despite zero-success short smoke; no additional smoke or training recipe change.'}
rt.save(stage/'formal-authorization.json',authorization,True)
p=copy.deepcopy(oldplan);p.update(operation_id=authorization['operation_id'],ops=str(S/'tools/runtime_v3.py'),
  lease_dir=str(stage/'rlt-lease'),formal_authorization=str(stage/'formal-authorization.json'))
oldrow=p['runs']['formal'];oldrun=Path(oldrow['run']);run=oldrun.with_name('ugrow-rlt-g5-formal-1006-v3');runtime=run/'runtime';runtime.mkdir(parents=True,exist_ok=False)
def replace(x):
    if isinstance(x,dict):return {k:replace(v) for k,v in x.items()}
    if isinstance(x,list):return [replace(v) for v in x]
    if isinstance(x,str):return x.replace(str(oldrun),str(run)).replace(oldrun.name,run.name)
    return x
before=yaml.safe_load((oldrun/'runtime/resolved.yaml').read_text());cfg=replace(before)
assert cfg['runner']['max_steps']==800 and cfg['runner']['resume_dir'] is None
assert cfg['env']['train']['total_num_envs']==4 and cfg['algorithm']['update_epoch']==5
assert cfg['actor']['global_batch_size']==512 and cfg['actor']['micro_batch_size']==256
assert cfg['algorithm']['rlt_schedule']['warmup_min_size']==10000 and cfg['algorithm']['rlt_schedule']['warmup_post_collect_updates']==15000
diff={k:[v,base._flatten(cfg)[k]] for k,v in base._flatten(before).items() if v!=base._flatten(cfg)[k]}
assert set(diff)<=base.OUTPUT_KEYS,diff
(runtime/'resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
rt.save(runtime/'environment.json',replace(read(oldrun/'runtime/environment.json')),True);(runtime/'environment.json').chmod(0o600)
p['runs']['formal']={**oldrow,'run':str(run),'namespace':run.name}
p['pins'].update({str(f):sha(f) for f in [*(S/'tools').glob('*.py'),stage/'formal-authorization.json',runtime/'resolved.yaml',runtime/'environment.json']})
rt.save(stage/'formal-config-diff.json',diff,True);rt.save(stage/'plan.json',p,True);rt.configure(stage)[0].checked()

# Bind only the live returned GPU5 RLT, with the verified v2 terminal chain.
previous=oldstage/'rlt-lease';ld=Path(p['lease_dir']);ld.mkdir(exist_ok=False)
with base._lock(previous),base._lock(ld):
    old,b,tp,op=base._context(previous)
    assert old['gpu']==5 and not b.same(read(oldstage/'owner-identity.json'))
    terminal=read(oldstage/'terminal.json');assert terminal['released'] and terminal['decision']=='FAILED'
    released=read(previous/'rlt-released.json');assert released['ready'] and released['release_evidence']
    dispatch=read(previous/'return-dispatched.json');prepared=read(previous/'return/prepared.json')
    assert dispatch['checkpoint']==prepared['checkpoint']==released['checkpoint']
    assert all(sha(k)==v for k,v in prepared['pins'].items())
    assert read(previous/'rlt-returned.json')['first_round_verified']
    taskpath=previous/'return/plan.json';task=read(taskpath);row=task['runs'][old['role']]
    identitypath=Path(row['run'])/'runtime/driver-identity.json';identity=read(identitypath)
    assert all(identity[k]==dispatch['identity'][k] for k in ('pid','uid','start')) and b.same(identity)
    assert row['run']==dispatch['run']==old['new_run'] and row['namespace']==dispatch['namespace']==old['new_namespace']
    op.ST=taskpath.parent;op.checked();actors=op.validate_scoped_actors(task,op.active_actors(task,row['namespace']))
    jobs={a['job_id'] for a in actors};assert len(jobs)==1
    watch=read(old['watch'])['runs'][old['watch_key']]
    assert watch==read(previous/'return-watch-updated.json')['target'] and watch['run']==row['run'] and watch['namespace']==row['namespace'] and watch['gpus']==[5]
    checkpoint=helper._select_checkpoint(base,Path(row['run']),released['checkpoint']);base._check_checkpoint(checkpoint)
    next_return='/data/chenyiteng/results/rlinf-rlt/rlt-g5-after-ugrow-1006-v3';assert not Path(next_return).exists()
    original_done=Path(old['returned_from']['original_role_terminal']);assert sha(original_done)==old['pins'][str(original_done)]
    lease=copy.deepcopy(old);lease.update(operation_id=p['operation_id'],lease_kind=helper.KIND,task_plan=str(taskpath),row=row,identity=identity,
       checkpoint=checkpoint,watch_entry=watch,new_run=next_return,new_namespace=Path(next_return).name,prepared_at=time.time(),
       returned_from={'previous_lease':str(previous),'dispatch':str(previous/'return-dispatched.json'),'checkpoint':released['checkpoint'],
         'job_id':next(iter(jobs)),'original_role_terminal':str(original_done)})
    evidence=[previous/'lease.json',previous/'rlt-released.json',previous/'return-dispatched.json',previous/'return/prepared.json',
       previous/'rlt-returned.json',previous/'return-watch-updated.json',oldstage/'terminal.json',oldstage/'owner-identity.json',taskpath,identitypath,
       stage/'formal-authorization.json',*(S/'tools').glob('*.py')]
    lease['pins'].update({str(f):sha(f) for f in evidence});lease['pins'].update(prepared['pins'])
    claim=previous/'returned-gpu5-formal-v3-claim.json';lease['claim']=str(claim)
    b.save(claim,{'operation_id':p['operation_id'],'lease':str(ld/'lease.json')},True);b.save(ld/'lease.json',lease,True)
print(json.dumps({'plan':str(stage/'plan.json'),'formal':p['runs']['formal'],'identity_to_stop':identity,'checkpoint':checkpoint,'config_diff':diff}))
