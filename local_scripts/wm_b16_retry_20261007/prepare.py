"""Prepare the current returned RLT45 contracts, then one B16 formal owner."""
import ast,copy,datetime,hashlib,importlib.util,json,os,socket,sys
from pathlib import Path
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
F=S/'lift-two-gpu-b16-20261007-v1';O=S/'runs/lift-two-gpu-b16-20261007-v1'
PARENT=S/'runs/lift-two-gpu-from0-v2';RETURN=S/'rlt45-return-20261007-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def save(p,v):
 p=Path(p);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 p.write_text(json.dumps(v,indent=2)+'\n');p.chmod(0o600)
def audited_parent(owner,child_path,child_module,helper=None):
 assert Path(owner)==PARENT
 final=read(PARENT/'final.json');ident=read(PARENT/'owner-identity.json')
 assert final['terminal_status']=='failed' and not final['rlt_return_dispatched']
 assert final['recovery_error']['type']=='RuntimeError' and 'precedence audit' in final['recovery_error']['error']
 assert read(PARENT/'cleanup.json')['all_stopped']
 release=read(PARENT/'smoke-release.json');assert release['gpus']==[4,5] and release['all_workers_stopped']
 dispatch=read(RETURN/'dispatched.json');assert dispatch['physical_gpus']==[4,5]
 combo=read(RETURN/'cycle/plan.json')
 group,row=next((g,r) for g,r in combo['children'].items() if Path(r['path'])==Path(child_path))
 assert Path(row['module'])==Path(child_module) and sha(child_module)==row['module_sha256']
 assert sha(Path(child_path)/'plan.json')==row['plan_sha256']
 returned=read(Path(child_path)/'resumed-dispatched.json')
 assert returned['cycle_id']==Path(child_path).name
 assert dispatch['result'][group]['resumed_dispatched']
 if helper:
  assert not helper.same(ident) and all(not helper.same(x) for x in release['managed_processes'])
 paths={'owner_plan':PARENT/'owner-plan.json','final':PARENT/'final.json','identity':PARENT/'owner-identity.json',
 'cleanup':PARENT/'cleanup.json','release':PARENT/'smoke-release.json','owner_return':RETURN/'dispatched.json',
 'return_started':RETURN/'cycle/return-started.json','combined_plan':RETURN/'cycle/plan.json',
 'combined_return':RETURN/'cycle/resumed-dispatched.json','child_return':Path(child_path)/'resumed-dispatched.json'}
 evidence={}
 for k,p in paths.items():evidence[k]=str(p);evidence[k+'_sha256']=sha(p)
 return group,ident,evidence
def prepare():
 assert not O.exists() and not (F/'ready.json').exists()
 old=read(S/'lift-two-gpu-from0-v2/prepared/owner-plan.json')
 parent=read(RETURN/'cycle/plan.json');children={}
 for key,name in [('gpu4','b16-g4'),('gpu567','b16-g5')]:
  row=parent['children'][key];src=Path(row['module']);sub=read(Path(row['path'])/'plan.json')
  m=load('b16_prepare_'+key,src);m.parent_complete=audited_parent
  target=F/'cycles'/name
  m.prepare(target,Path(row['path']),src,PARENT,Path(sub['base_module']),
     scope_activation=sub['scope_activation'],scope_manifest=sub['scope_manifest'],scope_id=sub['scope_id'])
  children[key]={'path':str(target),'module':str(target/'rlt_returned_cycle.py')}
 combo=load('b16_combined',RETURN/'cycle/rlt_returned_multigpu_cycle.py')
 cycle=F/'cycles/cycle';combo.prepare(cycle,children)
 reference=Path(old['trials'][0]['config']);cfg=read(reference)
 cfg=json.loads(json.dumps(cfg).replace(str(PARENT/'formal'),str(O/'formal')))
 cfg['runner']['logger']['experiment_name']='lift-two-gpu-b16-formal-1007'
 for c in ['actor','env','rollout']:cfg[c]['group_name']='LiftB16_'+c
 save(F/'prepared/formal.yaml',cfg)
 service=copy.deepcopy(old['services'][0]);service['argv']=[v.replace(str(PARENT),str(O)) for v in service['argv']]
 idx=service['argv'].index('--wm-batch-size')+1;assert service['argv'][idx]=='32';service['argv'][idx]='16'
 plan=copy.deepcopy(old)
 for k in ['base_owner_module','reused_smoke_owner','smoke_result','smoke_accepted']:plan.pop(k,None)
 plan.update(mode='two_gpu_b16_from0',owner_dir=str(O),lifecycle_path=str(cycle),
   lifecycle_module=str(cycle/'rlt_returned_multigpu_cycle.py'),reference_config=str(reference),
   graphics_fragment=str(RETURN/'scope/environment-fragment.json'),scope_manifest_sha256=sha(RETURN/'scope/scope.json'),
   retry_of=str(PARENT),skip_smoke=True,services=[service])
 plan['trials']=[{'key':'formal','config':str(F/'prepared/formal.yaml'),'namespace':'opendw_lift_b16_200_1007',
                  'episode_steps':384,'num_envs':64,'timeout_seconds':60*86400}]
 # Pin the active implementation rather than retaining inactive startup/smoke
 # wrapper files. Each RLT child still retains its own checkpoint/evidence pins.
 roots=[Path(plan['repo']).resolve(),Path(service['cwd']).resolve(),(S/'opendw').resolve(),(S/'payload_v2/tools').resolve()]
 pins={p:h for p,h in old['source_sha256'].items() if any(Path(p).resolve().is_relative_to(r) for r in roots)}
 active=[F/'code/owner.py',F/'code/prepare.py',reference,F/'prepared/formal.yaml',Path(plan['graphics_fragment']),RETURN/'scope/scope.json']
 active+=list((F/'cycles').rglob('*.py'))+list((F/'cycles').rglob('plan.json'))+list(Path(service['cwd']).glob('*.py'))
 pins.update({str(p):sha(p) for p in active});plan['source_sha256']=pins
 save(F/'prepared/owner-plan.json',plan)
 module=load('b16_owner_validate',(F/'code/owner.py').resolve());module.load_lifecycle(plan);module.validate(plan)
 assert module.normalized(cfg)==module.normalized(read(reference))
 oldservice=copy.deepcopy(old['services'][0]);newservice=copy.deepcopy(service)
 newservice['argv']=[v.replace(str(O),str(PARENT)) for v in newservice['argv']];newservice['argv'][idx]='32'
 assert newservice==oldservice
 save(F/'ready.json',{'time':datetime.datetime.now().astimezone().isoformat(),'plan':str(F/'prepared/owner-plan.json'),
    'entrypoint':str(F/'code/owner.py'),'plan_sha256':sha(F/'prepared/owner-plan.json'),'physical_gpus':[4,5],
    'config_equal_except_output_names':True,'service_only_change':{'wm_batch_size':[32,16]},'skip_smoke':True,
    'source_pins_before':len(old['source_sha256']),'source_pins_after':len(pins),
    'rlt_checkpoints':{k:r['recovery']['checkpoint']['step'] for k,r in read(cycle/'plan.json')['runs'].items()}})
 print(json.dumps(read(F/'ready.json')),flush=True)
if __name__=='__main__':
 assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 prepare()
