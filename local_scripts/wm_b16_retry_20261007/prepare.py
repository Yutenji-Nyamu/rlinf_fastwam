"""Reuse the proven B16 training protocol and current per-card RLT return contracts."""
import copy, datetime, hashlib, importlib.util, json, os, socket, sys
from pathlib import Path
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
F=S/'lift-two-gpu-b16-lean-20261007-v1'; O=S/'runs/lift-two-gpu-b16-lean-20261007-v1'
PREVIOUS=S/'lift-two-gpu-b16-20261007-v1'; PARENT=S/'runs/lift-two-gpu-b16-20261007-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m; spec.loader.exec_module(m); return m
def save(p,value):
 p=Path(p); assert not p.exists(); p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 p.write_text(json.dumps(value,indent=2)+'\n'); p.chmod(0o600)
def prepare():
 assert not O.exists() and not (F/'ready.json').exists()
 old=read(PREVIOUS/'prepared/owner-plan.json'); parent=read(PREVIOUS/'cycles/cycle/plan.json')
 children={}
 for key,name in [('gpu4','lean-g4'),('gpu567','lean-g5')]:
  row=parent['children'][key]; src=Path(row['module']); sub=read(Path(row['path'])/'plan.json')
  assert sha(src)==row['module_sha256']
  module=load('lean_prepare_'+key,src); target=F/'cycles'/name
  module.prepare(target,Path(row['path']),src,PARENT,Path(sub['base_module']),
     scope_activation=sub['scope_activation'],scope_manifest=sub['scope_manifest'],scope_id=sub['scope_id'])
  children[key]={'path':str(target),'module':str(target/'rlt_returned_cycle.py')}
 combo=load('lean_combined',PREVIOUS/'cycles/cycle/rlt_returned_multigpu_cycle.py')
 cycle=F/'cycles/cycle'; combo.prepare(cycle,children)
 reference=Path(old['trials'][0]['config']); cfg=read(reference)
 cfg=json.loads(json.dumps(cfg).replace(str(PARENT/'formal'),str(O/'formal')))
 cfg['runner']['logger']['experiment_name']='lift-two-gpu-b16-lean-1007'
 for key in ['actor','env','rollout']: cfg[key]['group_name']='LiftB16Lean_'+key
 save(F/'prepared/formal.yaml',cfg)
 service=copy.deepcopy(old['services'][0]); service['argv']=[v.replace(str(PARENT),str(O)) for v in service['argv']]
 assert service['argv'][service['argv'].index('--wm-batch-size')+1]=='16'
 plan=copy.deepcopy(old)
 plan.update(owner_dir=str(O),lifecycle_path=str(cycle),lifecycle_module=str(cycle/'rlt_returned_multigpu_cycle.py'),
             reference_config=str(reference),retry_of=str(PARENT),services=[service])
 plan['trials']=[dict(old['trials'][0],config=str(F/'prepared/formal.yaml'),namespace='opendw_lift_b16_lean_1007')]
 roots=[Path(plan['repo']).resolve(),Path(service['cwd']).resolve(),(S/'opendw').resolve(),(S/'payload_v2/tools').resolve()]
 pins={p:h for p,h in old['source_sha256'].items() if any(Path(p).resolve().is_relative_to(r) for r in roots)}
 active=[F/'code/owner.py',reference,F/'prepared/formal.yaml',Path(plan['graphics_fragment'])]
 active+=list((F/'cycles').rglob('*.py'))+list((F/'cycles').rglob('plan.json'))
 pins.update({str(p):sha(p) for p in active}); plan['source_sha256']=pins
 module=load('lean_owner',F/'code/owner.py')
 assert module.normalized(cfg)==module.normalized(read(reference)), 'Training parameters changed'
 assert dict(service,argv=[v.replace(str(O),str(PARENT)) for v in service['argv']])==old['services'][0]
 assert cfg['runner']['max_steps']==200 and cfg['runner']['resume_dir'] is None
 assert cfg['env']['train']['total_num_envs']==64 and cfg['env']['train']['rollout_epoch']==8
 assert cfg['cluster']['component_placement']=={'actor':'4','env':'5','rollout':'4'}
 module.pidfd_probe()
 save(F/'prepared/owner-plan.json',plan)
 save(F/'ready.json',{'time':datetime.datetime.now().astimezone().isoformat(),'plan':str(F/'prepared/owner-plan.json'),
      'entrypoint':str(F/'code/owner.py'),'plan_sha256':sha(F/'prepared/owner-plan.json'),'physical_gpus':[4,5],
      'config_equal_except_output_names':True,'service_parameters_unchanged':True,'skip_smoke':True,
      'rlt_checkpoints':{k:r['recovery']['checkpoint']['step'] for k,r in read(cycle/'plan.json')['runs'].items()}})
 print(json.dumps(read(F/'ready.json')),flush=True)
if __name__=='__main__':
 assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 prepare()
