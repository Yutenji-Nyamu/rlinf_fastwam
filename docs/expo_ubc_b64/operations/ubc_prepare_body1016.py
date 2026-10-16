import os,sys,time,copy,hashlib,math
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
T=Path('/data/chenyiteng/deployment-20261016/expo-ubc-b64-tau123-v1')
sys.path.insert(0,str(S/'ops'))
from common import read,save,same
assert os.getuid()==20001 and not T.exists(), 'Unique preparation; inspect existing transaction instead of replay'
plan=read(S/'plan.json');state=read(S/'status.json')
assert all(state['slots'][g]['phase']=='WAIT_EXTERNAL' and same(plan['slots'][g]['external_identity']) for g in plan['slots'])
prepared=[]
for item in PACK:
 q=item['request'];inputs=item['inputs'];source=Path(q['source']);R=Path(q['run'])
 assert q['uid']==os.getuid() and not R.exists() and inputs['formal']==BASELINE['formal']
 assert inputs['core']==BASELINE['core'] and inputs['core']['target_entropy']==-70+140*math.log(.2)
 assert inputs['signal']['target']=='fm' and inputs['signal']['temperature_local']==q['tau']==inputs['signal']['temperature_chunk']
 assert len(q['gpus'])==2 and set(q['gpus']).issubset({4,5,6,7})
 assert [plan['slots'][str(g)]['gpu_uuid'] for g in q['gpus']]==q['uuids']
 assert all(plan['slots'][str(g)]['external_identity']==item['old_identity'] for g in q['gpus'])
 assert all(plan['slots'][g]==v for g,v in item['protected_slots'].items())
 assert '--resume' not in q['command'] and q['environment']['CUDA_VISIBLE_DEVICES']==','.join(q['uuids'])
 for name,h in inputs['port_source_manifest'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h,name
 for p in [inputs['model']['model_path'], inputs['demo_path'], inputs['env']['assets_path'], inputs['env']['seeds_path'],inputs['evaluation']['seed_path'],q['environment']['LD_PRELOAD'],q['environment']['VK_ICD_FILENAMES']]:assert Path(p).exists(),p
 assert hashlib.sha256(Path(inputs['model']['openpi_data']['norm_stats_path']).read_bytes()).hexdigest()==inputs['norm_sha256']
 assert hashlib.sha256(Path(inputs['evaluation']['seed_path']).read_bytes()).hexdigest()==inputs['evaluation']['config']['_expo_seed_sha256']
 assert hashlib.sha256(Path(inputs['env']['seeds_path']).read_bytes()).hexdigest()=='c4cf9ca97b8a26dc8b28fc82d8cef43421aff594ebbd8164b246c550d1156d2d'
 profiles=[]
 for p in Path('/home/chenyiteng/.nv/nvidia-application-profiles-rc.d').glob('*.json'):
  if Path(q['environment']['LD_PRELOAD']).name in p.read_text():profiles.append({'path':str(p),'data':read(p)})
 assert profiles, 'Existing graphics application profile missing'
 if item.get('scope_file'):
  scope=read(item['scope_file']);assert scope['physical_gpu']==q['gpus'][0] and scope['gpu_uuid']==q['uuids'][0] and scope['mask']==2**q['gpus'][0]
  for k in ['marker','profile']:assert hashlib.sha256(Path(scope[k+'_path']).read_bytes()).hexdigest()==scope[k+'_sha256']
 for g in q['gpus']:
  f=read(plan['slots'][str(g)]['fallback']);rt=Path(f['runtime']);cp=Path(f['checkpoint'])
  assert (rt/'resolved.yaml').is_file() and (rt/'environment.json').is_file() and read(cp/'actor/sac_components/rlt_trainer_state/complete.json')['complete']
  assert not (rt/'started.json').exists()
 sys.path[:0]=[str(source),str(source/'examples/embodiment')]
 from train_expo_formal import validate_inputs
 from rlinf.algorithms.expo_ft.signals import SignalWeighting
 cfg,core=validate_inputs(inputs,60000);w=SignalWeighting(inputs['signal'])
 assert w.active('fm') and not w.active('editor') and not __import__('torch').cuda.is_initialized()
 prepared.append(dict(tau=q['tau'],request=q['request'],run=q['run'],source=q['source'],gpus=q['gpus'],target_entropy=core.resolved_target_entropy,signal_contract=w.contract,command=q['command'],config_diff=item['diff'],old_identity=item['old_identity'],old_release=item['old_release'],old_request=item['old_request'],protected_slots=item['protected_slots'],profiles=profiles))
T.mkdir(parents=True)
save(T/'clean-baseline-inputs.json',BASELINE)
for item,p in zip(PACK,prepared):
 q=item['request'];R=Path(q['run']);(R/'runtime').mkdir(parents=True)
 save(R/'inputs.json',item['inputs']);save(q['request'],q);Path(q['request']).chmod(0o600)
 save(T/('tau'+str(q['tau'])+'-prepared.json'),p)
save(T/'prepared.json',dict(time=time.time(),groups=prepared,plan_before=plan))
print(json.dumps(dict(prepared=True,groups=[{k:p[k] for k in ['tau','gpus','request','run','source','target_entropy','command']} for p in prepared])))
