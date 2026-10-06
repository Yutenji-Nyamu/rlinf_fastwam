"""Prepare the explicitly requested fresh BC N8/U5/300, without stopping jobs."""
import copy,hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
import yaml
S=Path('/data/chenyiteng/deployment-20261006/ugrow-bc8u5-300-v1');oldstage=S.with_name('ugrow-bc-rlt-g45-v2')/'bc'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2))
def mod(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def replace(x,pairs):
 if isinstance(x,dict):return {k:replace(v,pairs) for k,v in x.items()}
 if isinstance(x,list):return [replace(v,pairs) for v in x]
 if isinstance(x,str):
  for a,b in pairs:x=x.replace(a,b)
 return x
assert os.getuid()==1003
old=read(oldstage/'plan.json');rt=mod(old['ops'],'oldrt');op,b=rt.configure(oldstage);op.checked()
ext=S.with_name('ugrow-budget-400-2000-v1')/'bc'
assert read(ext/'extension-status.json')['phase']=='WAITING_ORIGINAL_ENDPOINT'
assert b.same(read(oldstage/'owner-identity.json'))
repo=Path(old['repo']).with_name('ugrow-bc8u5-clean-aligned-20261006')
assert not repo.exists();subprocess.run(['git','-C',old['repo'],'worktree','add','-b','codex/ugrow-bc8u5-clean-aligned-20261006',str(repo),old['head']],check=True,capture_output=True)
target=repo/'rlinf/workers/actor/fsdp_online_bc_policy_worker.py';text=target.read_text()
before='if self.demo_weight != 0 or bc.get("max_success_chunks") is not None:\n                raise ValueError("BC U keeps success-only BC and length filtering off.")'
after='if self.demo_weight != 0:\n                raise ValueError("BC U keeps success-only BC.")'
assert text.count(before)==1;target.write_text(text.replace(before,after))
subprocess.run(['git','-C',str(repo),'add',str(target)],check=True);subprocess.run(['git','-C',str(repo),'commit','-m','Allow existing success-length admission filter for BC U'],check=True,capture_output=True)
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
tools=S/'tools';tools.mkdir(parents=True,exist_ok=False)
prior_tools=Path('/data/chenyiteng/deployment-20261006/ugrow-budget-400-2000-v1/tools')
for f in prior_tools.glob('*.py'):(tools/f.name).write_bytes(f.read_bytes())
stage=S/'bc';stage.mkdir();p=copy.deepcopy(old);p.update(repo=str(repo),head=head,ops=str(tools/'runtime_v3.py'),formal_timeout=172800)
p.pop('verified_prior_smoke',None);p.pop('formal_authorization',None)
fixed=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-instruction-reset-fix')
assert subprocess.check_output(['git','-C',str(fixed),'rev-parse','HEAD'],text=True).strip()=='c961671881cca1ae9277bc2dff4918d4513cb778'
assert not subprocess.check_output(['git','-C',str(fixed),'status','--porcelain'],text=True).strip()
p['pins']={}
for k in old['pins']:
 q=Path(k)
 if str(q).startswith(old['repo']+'/'):q=repo/q.relative_to(old['repo'])
 p['pins'][str(q)]=sha(q)
for f in tools.glob('*.py'):p['pins'][str(f)]=sha(f)
for rel in subprocess.check_output(['git','-C',str(fixed),'ls-files'],text=True).splitlines():
 f=fixed/rel
 if f.suffix=='.py':p['pins'][str(f)]=sha(f)
base=mod(prior_tools/'rlt_lease.py','base');diffs={}
for key,steps in [('smoke',2),('formal',300)]:
 oldrun=Path(old['runs']['formal']['run']);run=oldrun.with_name('ugrow-bc8u5-len3-g4-'+key+'-300-1006-v1');runtime=run/'runtime';runtime.mkdir(parents=True,exist_ok=False)
 cfg=replace(yaml.safe_load((oldrun/'runtime/resolved.yaml').read_text()),[(str(oldrun),str(run)),(oldrun.name,run.name),(old['repo'],str(repo))])
 cfg['env']['train']['total_num_envs']=8;cfg['algorithm']['online_bc']['max_success_chunks']=3
 cfg['runner'].update(max_epochs=steps,max_steps=steps,resume_dir=None)
 if key=='smoke':cfg['runner'].update(val_check_interval=0,save_interval=2)
 assert cfg['algorithm']['update_epoch']==5 and cfg['actor']['global_batch_size']==1024 and cfg['actor']['micro_batch_size']==32
 env=replace(read(oldrun/'runtime/environment.json'),[(str(oldrun),str(run)),(oldrun.name,run.name),(old['repo'],str(repo))])
 asset=env['ASSETS_PATH'];env['ROBOTWIN_PATH']=str(fixed);env['PYTHONPATH']=env['PYTHONPATH'].replace(':'+asset,':'+str(fixed))
 for group in cfg['cluster']['node_groups']:
  for ec in group['env_configs']:
   for v in ec['env_vars']:
    for k in tuple(v):
     if k in env:v[k]=env[k]
 # Actor/rollout and EnvWorker import the same fixed instruction environment.
 cfg['cluster']['node_groups'][0]['env_configs'][0]['env_vars'] += [{'ROBOTWIN_PATH':str(fixed)}]
 (runtime/'resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));save(runtime/'environment.json',env);(runtime/'environment.json').chmod(0o600)
 p['runs'][key]={**old['runs'][key],'repo':str(repo),'run':str(run),'namespace':run.name}
 for f in (runtime/'resolved.yaml',runtime/'environment.json'):p['pins'][str(f)]=sha(f)
 hist=yaml.safe_load(Path('/home/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-bc-clean-8u5-len3-fixed-gpu6-formal100-20260910-v1/runtime/resolved.yaml').read_text())
 a=base._flatten(hist);c=base._flatten(cfg);diff={k:[a.get(k),c.get(k)] for k in set(a)|set(c) if a.get(k)!=c.get(k)}
 allowed=lambda k:k.startswith(('cluster.','algorithm.online_bc.dvac.')) or k in base.OUTPUT_KEYS|{'algorithm.online_bc.data_path','runner.max_epochs','runner.max_steps','env.train.seeds_path','env.eval.seeds_path'} or key=='smoke' and k in ('runner.val_check_interval','runner.save_interval')
 assert all(allowed(k) for k in diff),[k for k in diff if not allowed(k)]
 for mode in ('train','eval'):assert sha(hist['env'][mode]['seeds_path'])==sha(cfg['env'][mode]['seeds_path'])
 diffs[key]=diff
save(stage/'plan.json',p);save(stage/'historical-config-diff.json',diffs)
newrt=mod(p['ops'],'newrt');newrt.configure(stage)[0].checked()
save(S/'prepared.json',{'time':time.time(),'repo':str(repo),'head':head,'stage':str(stage),'old_stage':str(oldstage),'old_extension':str(ext),'user_instruction':'Fresh N8/U5 total300, align fixed instruction and max_success_chunks3; preserve old results and other GPUs','runs':p['runs']})
print(json.dumps({'stage':str(stage),'head':head,'runs':p['runs'],'formal_diff_keys':list(diffs['formal'])}))
