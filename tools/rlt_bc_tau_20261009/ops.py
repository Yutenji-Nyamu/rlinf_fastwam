import os,sys,json,signal,time,socket,subprocess,zipfile,tarfile,shutil
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');T=S/'rlt-bc-tau-20261009';sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,exact_signal,gpus,actors
mode=os.environ.get('RLT_TAU_MODE','inspect');p=read(S/'plan.json');host=p['host'];uid=os.getuid();assert uid==p['uid']
def alive(q):
 if 'start_ticks'in q:q=dict(pid=q['pid'],uid=q['uid'],start=q['start_ticks'])
 return same(q)
def norm(q):return dict(pid=q['pid'],uid=q['uid'],start=q.get('start',q.get('start_ticks')))
if mode=='inspect':
 import yaml
 st=read(S/'status.json');owner=read(S/'owner-identity.json');assert same(owner);snap=gpus();rows=[]
 for g in ([4,5,6,7]if host=='sz1'else[6,7]):
  slot=p['slots'][str(g)];row=st['slots'][str(g)];q=read(slot['priority']);assert q['kind']=='bc' and row['phase']=='PRIORITY_RUNNING' and same(row['identity']);cfg=yaml.safe_load((Path(q['runtime'])/'resolved.yaml').read_text());root=Path(q['run']);cps=[]
  for cp in (root/root.name/'checkpoints').glob('global_step_*'):
   req=['actor/local_shard_checkpoint/checkpoint_rank_0.pt','actor/online_bc/rank_0/learner.pt','actor/online_bc/rank_0/success_replay.pt','actor/online_bc/rank_0/signal_tau.pt']
   if all((cp/f).is_file()and (cp/f).stat().st_size>0 for f in req):cps.append(cp)
  assert cps
  for i,c in snap.items():
   if i!=g:continue
   for pr in c['processes']:
    a=proc(pr['pid']);assert a and a['uid']==uid
    env=dict(x.split('=',1)for x in (Path('/proc')/str(a['pid'])/'environ').read_bytes().decode(errors='replace').split('\0')if '='in x);assert env.get('CLUSTER_NAMESPACE')==q['namespace']
  rows.append(dict(gpu=g,q=q,identity=row['identity'],checkpoint=str(max(cps,key=lambda x:int(x.name.split('_')[-1]))),config=cfg))
 extra={}
 if host=='sz2':
  meta=read(S/'expo-owner/current.json');assert alive(meta['owner'])and alive(meta['child']);assert all(proc(x['pid'])['uid']==uid and x['pid']==meta['child']['pid'] for i in [4,5]for x in snap[i]['processes']);extra['expo']=meta
  e=Path(meta['train'])/'run';extra['expo_status']=read(e/'status.json');extra['expo_checkpoint']=[dict(name=x.name,bytes=x.stat().st_size)for x in e.glob('checkpoint*.pt')];assert extra['expo_checkpoint']
 if host=='sz3':
  w=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/wm-cycle-20261009-v2/run-retry2');extra['wm_owner']=read(w/'owner.json');assert alive(extra['wm_owner']);extra['wm_status']=read(w/'status.json')
 T.mkdir(exist_ok=True);assert not (T/'prestop.json').exists();pre=dict(time=time.time(),host=host,owner=owner,plan=p,state=st,rows=rows,extra=extra,gpus=snap);save(T/'prestop.json',pre)
 print(json.dumps(pre))
elif mode=='stop':
 pre=read(T/'prestop.json');assert pre['plan']==p;assert not(T/'concluded.json').exists()
 if host in ['sz1','sz2']:
  if(S/'fallback-enabled.json').exists():(S/'fallback-enabled.json').rename(T/'fallback-enabled-before.json')
  assert same(pre['owner']);exact_signal(pre['owner'],signal.SIGTERM)
  for _ in range(40):
   if not same(pre['owner']):break
   time.sleep(1)
  assert not same(pre['owner'])
 if host=='sz3':assert alive(pre['extra']['wm_owner'])
 for r in pre['rows']:assert same(r['identity']);exact_signal(r['identity'],signal.SIGTERM)
 if host=='sz2':exact_signal(norm(pre['extra']['expo']['child']),signal.SIGTERM)
 for _ in range(45):
  if not any(same(r['identity'])for r in pre['rows']):break
  time.sleep(1)
 # Only namespace-matched leftovers of the explicitly stopped BC jobs.
 namespaces={r['q']['namespace']for r in pre['rows']}
 for d in Path('/proc').glob('[0-9]*'):
  try:
   q=proc(d.name)
   if not q or q['uid']!=uid:continue
   ev=dict(x.split('=',1)for x in(d/'environ').read_bytes().decode(errors='replace').split('\0')if '='in x)
   if ev.get('CLUSTER_NAMESPACE')in namespaces:exact_signal(q,signal.SIGTERM)
  except OSError:pass
 assert not any(same(r['identity'])for r in pre['rows'])
 if host=='sz3':assert alive(pre['extra']['wm_owner'])
 save(T/'concluded.json',dict(time=time.time(),reason='USER_CONCLUDED_BC_AND_EXPO',stopped_bc=[r['identity']for r in pre['rows']],expo_driver_stopped=(not alive(pre['extra']['expo']['child']))if host=='sz2'else None,queue_held=host!='sz3',wm_preserved=host=='sz3'))
 print(read(T/'concluded.json'))
elif mode=='archive':
 sys.modules.setdefault('tensorflow',None)
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 pre=read(T/'prestop.json');assert(T/'concluded.json').exists();dest=T/'archive';dest.mkdir(exist_ok=True);summary=[]
 for r in pre['rows']:
  q=r['q'];root=Path(q['run']);pts={}
  for d,dirs,fs in os.walk(root):
   dirs[:]=[n for n in dirs if n not in ['checkpoints','success_data','video','videos','robotwin_data','replay_buffer']]
   for n in fs:
    if not n.startswith('events.out.tfevents'):continue
    a=EventAccumulator(str(Path(d)/n),size_guidance={'scalars':0});a.Reload()
    for tag in a.Tags()['scalars']:
     for x in a.Scalars(tag):pts.setdefault(tag,{})[x.step]=[x.value,x.wall_time]
  z=sorted(pts['eval/success_once'].items());record=dict(host=host,gpu=r['gpu'],request=q,reason='USER_CONCLUDED',completed_rounds=max(pts['env/success_once'])+1,checkpoint=r['checkpoint'],eval_round=z[-1][0]+1,eval_success=z[-1][1][0],last5eval_mean=sum(v[0]for _,v in z[-5:])/len(z[-5:]),curves={k:[[step,v[0],v[1]]for step,v in sorted(z.items())]for k,z in pts.items()});save(dest/f'g{r["gpu"]}.json',record);shutil.copyfile(root/'runtime/resolved.yaml',dest/f'g{r["gpu"]}.yaml');summary.append({k:v for k,v in record.items()if k not in ['curves','request']})
 if host=='sz2':
  e=Path(pre['extra']['expo']['train']);ed=dest/'expo';ed.mkdir(exist_ok=True)
  for n in ['inputs.json','run/status.json','run/events.jsonl','run/complete.json','run/failure.json']:
   f=e/n
   if f.exists():shutil.copyfile(f,ed/f.name)
  for f in(e/'run/evaluations').glob('*/complete.json'):save(ed/(f.parent.name+'.json'),read(f))
 save(dest/'summary.json',summary);shutil.copyfile(T/'concluded.json',dest/'concluded.json')
 with tarfile.open(T/'archive.tar.gz','w:gz')as tf:
  for f in dest.rglob('*'):
   if f.is_file():tf.add(f,arcname=str(f.relative_to(dest)))
 print(json.dumps(summary));print('ARCHIVE',str(T/'archive.tar.gz'),(T/'archive.tar.gz').stat().st_size)
elif mode=='audit':
 import yaml
 out={'host':host,'state':read(S/'status.json'),'gpus':gpus()}
 if host=='sz1':
  root=Path('/data/chenyiteng/deployment-20261006/ugrow-rlt-g5-formal-v3/rlt');old=read(root/'plan.json');out['rlt_plan']=old
  row=old['runs']['formal'];run=Path(row['run']);out['rlt_config']=yaml.safe_load((run/'runtime/resolved.yaml').read_text());out['rlt_environment']=read(run/'runtime/environment.json')
  large=[]
  for d,dirs,files in os.walk('/data/chenyiteng/results'):
   dirs[:]=[n for n in dirs if n not in ['video','videos','robotwin_data','success_data','.git','tensorboard','wandb']]
   if 'checkpoint' not in d:continue
   for name in files:
    f=Path(d)/name
    try:
     z=f.lstat()
     if not f.is_symlink() and z.st_uid==uid and z.st_size>1024**3:large.append(dict(path=str(f),bytes=z.st_size))
    except OSError:pass
  out['large_checkpoints']=large
 if host=='sz2':out['expo_released']=read(S/'expo-owner/expo-released.json')if(S/'expo-owner/expo-released.json').exists()else None
 if host=='sz3':
  root=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/wm-cycle-20261009-v2/run-retry2');out['wm_status']=read(root/'status.json');out['wm_files']=[str(x.relative_to(root))for x in root.glob('*')]
 print(json.dumps(out))
elif mode=='reserve':
 assert host in ['sz2','sz3'];st=read(S/'status.json');pre=read(T/'prestop.json');owner=read(S/'owner-identity.json')
 if same(owner):
  exact_signal(owner,signal.SIGTERM)
  for _ in range(40):
   if not same(owner):break
   time.sleep(1)
  assert not same(owner)
 stopped=[]
 for g in ['6','7']:
  row=st['slots'][g];old=pre['plan']['slots'][g];assert p['slots'][g]==old
  if row.get('identity')and same(row['identity']):
   assert row['phase']=='RLT_RUNNING' and row['request']==old['fallback'];q=read(row['request']);assert q['kind']=='rlt'
   exact_signal(row['identity'],signal.SIGTERM);stopped.append(dict(gpu=g,identity=row['identity'],namespace=q['namespace']))
 for _ in range(45):
  if not any(same(x['identity'])for x in stopped):break
  time.sleep(1)
 assert not any(same(x['identity'])for x in stopped)
 # Retire only these queue slots; no card lock or placeholder process remains.
 for g in ['6','7']:p['slots'].pop(g);st['slots'].pop(g)
 save(S/'plan.json',p);save(S/'status.json',st);save(T/'reserved67.json',dict(time=time.time(),cards=[6,7],stopped=stopped,reason='USER_RESERVED_FOR_OTHER_WINDOWS'))
 if host=='sz3':
  with(T/'owner-reserve.log').open('a')as log:child=subprocess.Popen([p['python'],'-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  time.sleep(2);assert same(proc(child.pid));print('WM queue resumed',child.pid)
 print('RESERVED',host,gpus())
elif mode=='install':
 dest=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-bc-tau-20261009');assert not dest.exists();dest.mkdir()
 with tarfile.open(T/'source.tar.gz')as f:
  assert all((dest/m.name).resolve().is_relative_to(dest.resolve())and not m.issym()and not m.islnk()for m in f.getmembers());f.extractall(dest)
 print('INSTALLED',str(dest))
elif mode=='test':
 repo='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-bc-tau-20261009';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=repo,PYTHONDONTWRITEBYTECODE='1')
 tests=['test_ugrow_signal.py','test_rlt_ugrow_sampler.py','test_rlt_ugrow_worker.py','test_rlt_dvac_two_level_worker.py','test_rlt_dvac_temperature.py','test_rlt_dvac_controls.py']
 rc=subprocess.call([p['python'],'-m','pytest','-q',*[f'tests/unit_tests/{t}'for t in tests]],cwd=repo,env=env);sys.exit(rc)
elif mode=='prepare':
 import copy,yaml,hashlib
 repo=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-bc-tau-20261009');cfg0=read(repo/'tools/rlt_bc_tau_20261009/base-config.json');env0=read(repo/'tools/rlt_bc_tau_20261009/base-environment.json');snap=gpus();assert not same(read(S/'owner-identity.json'));cards=[4,5,6,7]if host=='sz1'else[4,5];new=[]
 for g in cards:
  assert not snap[g]['processes'],(g,snap[g]);tau=(g-4)//2+1 if host=='sz1'else 3;sig='u'if g%2==0 else'norm';slot=p['slots'][str(g)];fallback=read(slot['fallback']);frt=Path(fallback['runtime']);fc=yaml.safe_load((frt/'resolved.yaml').read_text());fe=read(frt/'environment.json')
  run=Path('/data/chenyiteng/results/rlinf-rlt')/f'rlt-bc-{sig}-tau{tau}-{host}-g{g}-800-1009-v1';rt=run/'runtime';assert not run.exists()
  c=copy.deepcopy(cfg0);oldlog=c['runner']['logger']['log_path']
  def rep(x):
   if isinstance(x,dict):return{k:rep(v)for k,v in x.items()}
   if isinstance(x,list):return[rep(v)for v in x]
   if isinstance(x,str):return x.replace(oldlog,str(run))
   return x
  c=rep(c);c['runner']['logger']['experiment_name']=run.name;c['runner'].update(max_steps=800,resume_dir=None,ckpt_path=None)
  c['cluster']=copy.deepcopy(fc['cluster']);feat=c['rollout']['rlt_feature_model']['openpi'];feat.update(rlt_ugrow_enabled=sig=='u',rlt_norm_enabled=sig=='norm',rlt_dvac_mode='off')
  sys.path.insert(0,str(repo));from rlinf.algorithms.ugrow_signal import UGROW_SIGNAL_SPEC
  from rlinf.algorithms.norm_signal import NORM_SIGNAL_SPEC
  dv=c['algorithm']['rlt_dvac'];dv.update(temperature_local=float(tau),temperature_chunk=float(tau),signal_source='ugrow_10_5'if sig=='u'else'norm_residual_t5_l3',signal_spec=dict(UGROW_SIGNAL_SPEC if sig=='u'else NORM_SIGNAL_SPEC))
  feature=Path(c['rollout']['rlt_feature_model']['model_path']);assert feature.exists(),f'Missing original Stage1: {feature}'
  for part in ['train','eval']:
   seed=Path(c['env'][part]['seeds_path']);newseed=repo/'rlinf/envs/robotwin/seeds'/seed.name
   assert newseed.exists();c['env'][part]['seeds_path']=str(newseed)
  # Reuse this card's already proven compute and Vulkan binding, plus original task-specific model paths.
  env=dict(env0);env.update({k:v for k,v in fe.items()if k in ['RAY_ADDRESS','HOME','RLINF_OPENDW_GPU_SCOPE_MANIFEST','LD_PRELOAD','__GL_APPLICATION_PROFILE','ROBOTWIN_PATH','ROBOTWIN_ASSETS_PATH']})
  bootstrap=next(x for x in fe['PYTHONPATH'].split(':')if x.endswith('/bootstrap'));env.update(RLINF_CODE_WORKING_DIR=str(repo),REPO_PATH=str(repo),EMBODIED_PATH=str(repo/'examples/embodiment'),PYTHONPATH=':'.join([bootstrap,str(repo),env['ROBOTWIN_PATH']]),RLT_LOG_ROOT=str(run),CUDA_VISIBLE_DEVICES='')
  for ng in c['cluster']['node_groups']:
   for ec in ng.get('env_configs',[]):
    ev={k:v for x in ec['env_vars']for k,v in x.items()};ev.update({k:env[k]for k in ['HOME','RLINF_OPENDW_GPU_SCOPE_MANIFEST','LD_PRELOAD','__GL_APPLICATION_PROFILE']});ev['PYTHONPATH']=bootstrap;ec['env_vars']=[{k:v}for k,v in ev.items()]
  assert str(g) in str(c['cluster']['component_placement']);assert Path(env['RLINF_OPENDW_GPU_SCOPE_MANIFEST']).exists()and Path(env['LD_PRELOAD']).exists()
  rt.mkdir(parents=True);(rt/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));save(rt/'environment.json',env)
  request=T/'requests'/f'{host}-g{g}.json';q=dict(kind='rlt',signal=sig,tau=tau,gpu=g,run=str(run),runtime=str(rt),repo=str(repo),namespace=run.name);save(request,q)
  new.append(q)
 save(T/'prepared.json',dict(time=time.time(),runs=new,source_sha256=hashlib.sha256((T/'source.tar.gz').read_bytes()).hexdigest()));print(json.dumps(read(T/'prepared.json')))
elif mode=='launch':
 prepared=read(T/'prepared.json');st=read(S/'status.json');assert not same(read(S/'owner-identity.json'));snap=gpus()
 for q in prepared['runs']:
  g=str(q['gpu']);assert not snap[int(g)]['processes'];slot=p['slots'][g];slot.pop('external_release',None);slot.pop('external_identity',None);slot['priority']=str(T/'requests'/f'{host}-g{g}.json');st['slots'][g]={'phase':'PENDING_PRIORITY'}
 save(S/'plan.json',p);save(S/'status.json',st)
 if(T/'fallback-enabled-before.json').exists():(T/'fallback-enabled-before.json').rename(S/'fallback-enabled.json')
 with(T/'owner-launch.log').open('a')as log:child=subprocess.Popen([p['python'],'-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 time.sleep(3);assert same(proc(child.pid));print('OWNER',host,child.pid,read(S/'status.json'))
elif mode=='accept':
 import hashlib,yaml
 sys.modules.setdefault('tensorflow',None)
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 def tail(f,n=5000):
  try:
   with Path(f).open('rb')as h:h.seek(max(0,Path(f).stat().st_size-n));return h.read().decode(errors='replace')
  except OSError:return None
 def scalars(root):
  pts={}
  for f in Path(root).rglob('events.out.tfevents*'):
   a=EventAccumulator(str(f),size_guidance={'scalars':0});a.Reload()
   for tag in a.Tags()['scalars']:
    z=a.Scalars(tag)
    if z:pts[tag]=[[x.step,x.value,x.wall_time]for x in z[-20:]]
  return pts
 st=read(S/'status.json');out=dict(time=time.time(),host=host,state=st,owner_alive=same(read(S/'owner-identity.json')),gpus=gpus(),runs=[],fallback_enabled=(S/'fallback-enabled.json').exists())
 for g,slot in p['slots'].items():
  if 'priority'not in slot:continue
  q=read(slot['priority']);rt=Path(q['runtime']);cfg=yaml.safe_load((rt/'resolved.yaml').read_text());seeds={}
  for part in ['train','eval']:
   f=Path(cfg['env'][part]['seeds_path']);seeds[part]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
  out['runs'].append(dict(gpu=g,request=q,driver_alive=same(st['slots'][g].get('identity',{}))if st['slots'][g].get('identity')else False,log=tail(rt/'driver.log'),finished=read(rt/'finished.json')if(rt/'finished.json').exists()else None,metrics=scalars(Path(q['run'])/'tensorboard'),seed_files=seeds))
 out['fallbacks']={g:read(s['fallback'])for g,s in p['slots'].items()}
 out['gpu_hardware']=subprocess.run(['nvidia-smi','--query-gpu=index,name,power.limit,clocks.max.sm,clocks.max.memory','--format=csv,noheader'],capture_output=True,text=True,timeout=20).stdout
 out['disk']=read(T/'acceptance.json').get('disk') if os.environ.get('RLT_TAU_FAST') else subprocess.run(['df','-B1','/data/chenyiteng','/home/chenyiteng'],capture_output=True,text=True,timeout=60).stdout
 out['cpu']=subprocess.run(['lscpu'],capture_output=True,text=True,timeout=10).stdout
 if host=='sz3':
  root=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/wm-cycle-20261009-v2/run-retry2');w=dict(status=read(root/'status.json'),owner_alive=alive(read(root/'owner.json')),stages=[x.name for x in root.iterdir()],metrics={},log={},evaluations={});out['wm']=w
  for stage in root.glob('rl[0-9][0-9][0-9]'):
   w['metrics'][stage.name]=scalars(stage/'tensorboard');w['log'][stage.name]=tail(stage/'process.log',6000)
  for f in root.rglob('complete.json'):
   if 'checkpoints'not in f.parts:w['evaluations'][str(f.relative_to(root))]=read(f)
 save(T/'acceptance.json',out);print(json.dumps(out))
elif mode=='pack':
 dest=T/'publication';dest.mkdir(exist_ok=True)
 for n in ['concluded.json','reserved67.json','prepared.json','acceptance.json','baseline-comparison.json']:
  if(T/n).exists():shutil.copyfile(T/n,dest/n)
 if(T/'prepared.json').exists():
  for q in read(T/'prepared.json')['runs']:
   shutil.copyfile(Path(q['runtime'])/'resolved.yaml',dest/f'g{q["gpu"]}.yaml')
 shutil.copyfile(S/'plan.json',dest/'plan.json')
 with tarfile.open(T/'publication.tar.gz','w:gz')as tf:
  for f in dest.iterdir():tf.add(f,arcname=f.name)
 print('PACKED',str(T/'publication.tar.gz'))
elif mode=='compare':
 import hashlib,yaml
 assert host=='sz1';pre=read(T/'prestop.json');cfg=pre['rows'][0]['config'];old=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-bc8u5-fixed-clean-20260910');new=Path(pre['rows'][0]['q']['repo']);out={'bc_seeds':{},'bc_code':{},'rlt_seeds':{},'fallbacks':{}}
 for part in ['train','eval']:
  f=Path(cfg['env'][part]['seeds_path']);out['bc_seeds'][part]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
 for label,repo in [('clean',old),('method',new)]:
  f=repo/'rlinf/data/online_bc.py';lines=f.read_text().splitlines();hits=[i for i,x in enumerate(lines)if 'accepted = [ep for ep in episodes' in x];out['bc_code'][label]=dict(path=str(f),filter_lines=[lines[i-2:i+3]for i in hits])
 base=read(Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-bc-tau-20261009/tools/rlt_bc_tau_20261009/base-config.json'))
 for part in ['train','eval']:
  f=Path(base['env'][part]['seeds_path']);g=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-bc-tau-20261009/rlinf/envs/robotwin/seeds')/f.name;out['rlt_seeds'][part]=dict(base=str(f),new=str(g),base_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(g.read_bytes()).hexdigest())
 for g,s in p['slots'].items():
  q=read(s['fallback']);rt=Path(q['runtime']);out['fallbacks'][g]=dict(checkpoint=q.get('checkpoint'),checkpoint_exists=Path(q['checkpoint']).exists(),runtime_files=[x.name for x in rt.iterdir()])
 save(T/'baseline-comparison.json',out);print(json.dumps(out))
