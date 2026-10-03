import json,os,pathlib,re,socket,subprocess,sys,time
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
P=pathlib.Path('/data/chenyiteng/projects/wan-goal-sz3');run=P/'evaluations/wm-official-20261002-1738-dc66e5a3'
sys.path.insert(0,str(P/'scripts/resource_switch'))
from common import alive
out={'time':time.time(),'run':str(run),'files':{},'phases':{}}
for name in ('owner.json','status.json','failure.json','complete.json'):
 p=run/name
 if p.is_file():out['files'][name]=json.loads(p.read_text())
owner=out['files'].get('owner.json',{})
out['owner_alive']=alive(owner['identity']) if owner else None
out['protected_training_alive']=all(alive(p) for p in owner.get('protected',[])) if owner else None
for label in ('original','cp40','cp80'):
 phase=run/label
 if not phase.is_dir():continue
 row={}
 for name in ('baseline-zero-gate.json','evaluation-result.json','cleanup.json','policy-rank-0.json','policy-rank-1.json','verified-placement.json'):
  p=phase/name
  if p.is_file():row[name]=json.loads(p.read_text())
 log=phase/'command.log';txt=log.read_text(errors='replace') if log.is_file() else ''
 # The driver log can omit forwarded INFO lines. Read this phase's private
 # Ray worker outputs directly, and de-duplicate against the driver log.
 native_texts=[txt];native_sources=[]
 launch_path=phase/'launch.json'
 if launch_path.is_file():
  launch=json.loads(launch_path.read_text());raytmp=pathlib.Path(launch['ray_temp_dir'])
  assert raytmp.resolve().is_relative_to(pathlib.Path('/data/chenyiteng/we').resolve())
  for p in sorted((raytmp/'session_latest/logs').glob('worker-*.out')):
   assert p.stat().st_uid==20001 and p.stat().st_size<10000000
   raw=p.read_text(errors='replace')
   if '[libero eval]' in raw:
    native_texts.append(raw);native_sources.append({'path':str(p),'bytes':p.stat().st_size})
 pairs={};conflicts=[]
 for task,trial,score in re.findall(r'\[libero eval\]\s+task_id=(\d+),\s*trial_id=(\d+),\s*success=(True|False)\b','\n'.join(native_texts)):
  key=(int(task),int(trial));value=score=='True'
  if key in pairs and pairs[key]!=value:conflicts.append(list(key))
  pairs[key]=value
 ready=[json.loads(p.read_text()) for p in (phase/'graphics-init').glob('*-ready.json')]
 row.update(completed_unique=len(pairs),successes=sum(pairs.values()),conflicts=conflicts,native_sources=native_sources,task_counts={str(t):{'completed':sum(k[0]==t for k in pairs),'successes':sum(v for k,v in pairs.items() if k[0]==t)} for t in range(10)},first_batch20={'completed':sum(k[1]<2 for k in pairs),'successes':sum(v for k,v in pairs.items() if k[1]<2)},graphics_ready=len(ready),software_verified=sum(r['gl_vendor']=='Mesa' and 'llvmpipe' in r['gl_renderer'] and not r['nvidia_egl_loaded'] for r in ready),log_tail=txt[-6500:],log_age_seconds=time.time()-log.stat().st_mtime if log.exists() else None)
 out['phases'][label]=row
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=30)
print(json.dumps(out))
