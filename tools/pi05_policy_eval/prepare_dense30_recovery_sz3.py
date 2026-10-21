"""Prepare the scoped SZ3 repair; requires its reviewed source bundle."""
import os,sys,json,time,copy,shutil,tarfile,subprocess,hashlib
from pathlib import Path
from omegaconf import OmegaConf
import torch
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'));from common import read,save,same,proc
R=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-clean-c20-followups-20261021')
OLD=Path('/data/chenyiteng/deployment-20261021/clean-c20-dense30-sz3-v1');T=Path(str(OLD)+'-recovery1');O=Path('/data/chenyiteng/results/pi05-policy-eval/clean-c20-dense30-20261021-v2')
assert os.getuid()==20001 and not T.exists()and not O.exists()
assert all(not same(read(OLD/f'lane-g{g}-identity.json'))for g in (6,7))
T.mkdir();(T/'source-before').mkdir()
with tarfile.open('/data/chenyiteng/exports/sz3-recovery1021.tar.gz')as tar:
 for m in tar.getmembers():
  assert m.isfile()and m.name.startswith('tools/pi05_policy_eval/')and '..'not in Path(m.name).parts
  p=R/m.name
  if p.exists():shutil.copy2(p,T/'source-before'/p.name)
  p.write_bytes(tar.extractfile(m).read())
py='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
test=subprocess.run([py,'-B','-m','unittest','test_recovery','test_protocol','test_followups'],cwd=R/'tools/pi05_policy_eval',env=dict(os.environ,CUDA_VISIBLE_DEVICES='',ROBOTWIN_PATH='/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support'),capture_output=True,text=True)
(T/'cpu-tests.log').write_text(test.stdout+test.stderr);assert test.returncode==0,(test.stdout+test.stderr)[-3000:]
print('CPU_TESTS_PASS',flush=True)
orig=read(OLD/'prepared.json');oldO=Path(orig['output'])
O.mkdir()
for sub in ('configs','thresholds','summaries'):
 shutil.copytree(oldO/sub,O/sub)
(O/'batches').mkdir()
for name in ('calibration.json','frozen-thresholds.json','manifest.json'):
 if(oldO/name).exists():shutil.copy2(oldO/name,O/name)
for cpath in (O/'configs').glob('*.json'):
 c=read(cpath)
 if not isinstance(c,dict)or 'output'not in c:continue
 def remap(v):
  if isinstance(v,dict):return {k:remap(x)for k,x in v.items()}
  if isinstance(v,list):return [remap(x)for x in v]
  if isinstance(v,str):return v.replace(str(oldO),str(O))
  return v
 c=remap(c);c['environment_step_timeout_seconds']=1200;save(cpath,c)
for g in ('6','7'):
 env=read(oldO/f'environment-g{g}.json');env['TMPDIR']=str(T/f'tmp-g{g}');Path(env['TMPDIR']).mkdir();save(O/f'environment-g{g}.json',env)
 lane=read(OLD/f'lane-g{g}-plan.json');lane=remap(lane);lane['recovery']=dict(attempts=2,stall_seconds=1800,batch_seconds=10800,barrier_seconds=43800,continue_errors=True);save(T/f'lane-g{g}-plan.json',lane)
plan=read(S/'plan.json');state=read(S/'status.json');fallbacks=[]
for g in ('6','7'):
 old=read(plan['slots'][g]['fallback']);rt=Path(old['runtime']);cfg=OmegaConf.to_container(OmegaConf.load(rt/'resolved.yaml'),resolve=True)
 candidates=[Path(old['checkpoint'])]+list((Path(old['run'])/Path(old['run']).name/'checkpoints').glob('global_step_*'))
 required=['actor/dcp_checkpoint/.metadata','actor/dcp_checkpoint/__0_0.distcp','actor/sac_components/rlt_trainer_state/complete.json','actor/sac_components/rlt_trainer_state/checkpoint_rank_0.pt','actor/sac_components/target_model/checkpoint_rank_0.pt','actor/sac_components/replay_buffer/rank_0/metadata.json']
 complete=[p for p in candidates if all((p/f).is_file()and(p/f).stat().st_size>0 for f in required)];cp=max(complete,key=lambda p:int(p.name.split('_')[-1]))
 saved=torch.load(cp/'actor/sac_components/rlt_trainer_state/checkpoint_rank_0.pt',map_location='cpu',weights_only=True);contract=json.loads(saved['rlt_resume_contract']);changes=[]
 if 'rlt_dvac'in contract:
  for k,v in contract['rlt_dvac'].items():
   x=cfg['algorithm']['rlt_dvac'].get(k)
   if x!=v:
    assert k in ('log_eps','minmax_eps')and isinstance(x,str)and isinstance(v,float)and float(x)==v
    changes.append(dict(key=k,before=x,after=v));cfg['algorithm']['rlt_dvac'][k]=v
 computed=dict(cfg['algorithm']['rlt_resume']['contract'])
 if 'rlt_dvac'in contract:computed['rlt_dvac']=cfg['algorithm']['rlt_dvac']
 assert computed==contract
 assert hashlib.sha256(json.dumps(computed,sort_keys=True,separators=(',',':')).encode()).hexdigest()==read(cp/'actor/sac_components/rlt_trainer_state/complete.json')['rlt_resume_contract_sha256']
 run=Path(old['run']+'-recovery1');newrt=run/'runtime';assert not run.exists();newrt.mkdir(parents=True)
 q=dict(old,run=str(run),runtime=str(newrt),namespace=run.name,checkpoint=str(cp),step=int(cp.name.split('_')[-1]))
 def route(v):
  if isinstance(v,dict):return {k:route(x)for k,x in v.items()}
  if isinstance(v,list):return [route(x)for x in v]
  if isinstance(v,str):
   if v==old['namespace']:return q['namespace']
   if v==old['run']or v.startswith(old['run']+'/'):return q['run']+v[len(old['run']):]
  return v
 cfg=route(cfg);cfg['runner']['resume_dir']=str(cp);save(newrt/'resolved.yaml',cfg);save(newrt/'environment.json',route(read(rt/'environment.json')));save(T/f'fallback-g{g}.json',q)
 fallbacks.append(dict(gpu=g,checkpoint=str(cp),step=q['step'],contract_match=True,type_corrections=changes,prior=old))
save(T/'prepared.json',dict(time=time.time(),source=str(R),output=str(O),plan_before=plan,state_before=state,fallbacks=fallbacks,prior_transaction=str(OLD),protocol_unchanged=True,authorization='Restore dense sweep on SZ3 6/7, then original per-card RLT; user 2026-10-21'))
save(O/'manifest.json',dict(orig,output=str(O),transaction=str(T),recovery_of=str(oldO),timeouts=dict(environment_step=1200,stall=1800,batch=10800),max_attempts=2))
print(json.dumps(dict(prepared=str(T),output=str(O),fallbacks=fallbacks)),flush=True)
