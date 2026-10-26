import json, os, sys, subprocess, time
from pathlib import Path
D=Path('/data/chenyiteng/deployment-20261026/fql-smoke-sz1-g23');plan=json.loads((D/'plan.json').read_text());R=Path(plan['source'])
sys.path.insert(0,str(R/'tools/pi05_distillation'));from multitask_eval import cleanup,proc
def read(p):return json.loads(Path(p).read_text())
def save(n,v):(D/n).write_text(json.dumps(v,indent=2))
save('identity.json',proc(os.getpid()));jobs=[];results=[]
try:
 for g,c in zip([2,3],plan['configs']):
  token=f'fql-smoke1026-collect-g{g}';e=dict(os.environ,**read(D/f'environment-g{g}.json'),POLICY_EVAL_TOKEN=token);log=(D/f'collection-g{g}.log').open('x')
  p=subprocess.Popen([sys.executable,'-u','-B',str(R/'tools/pi05_policy_eval/run_batch.py'),c],cwd=R,env=e,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  jobs.append(dict(p=p,log=log,token=token,config=c,gpu=g,started=time.time()))
 save('status.json',{'phase':'collection','jobs':[dict(gpu=j['gpu'],identity=proc(j['p'].pid),config=j['config'])for j in jobs]})
 while jobs:
  for j in list(jobs):
   p=j['p'];expired=time.time()-j['started']>1800
   if p.poll()is not None or expired:
    cleanup(j['token']);p.wait(timeout=30);j['log'].close();results.append(dict(gpu=j['gpu'],config=j['config'],rc=p.returncode,timeout=expired));jobs.remove(j)
  if jobs:time.sleep(3)
 save('collection-results.json',results)
 env=dict(os.environ,**read(D/'environment-g2.json'));env.pop('DOJO_GPU_SCOPE',None);env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(R)+':'+env['ROBOTWIN_PATH'])
 cmd=[sys.executable,str(R/'tools/pi05_distillation/fql_index.py'),str(D/'collection/batches'),'--output',str(D/'collection/transition-manifest.json'),'--config',str(D/'collection/fql.json')]
 r=subprocess.run(cmd,cwd=R,env=env,capture_output=True,text=True,timeout=180);save('index-result.json',dict(rc=r.returncode,out=r.stdout,err=r.stderr));assert r.returncode==0,r.stderr
 config=read(D/'collection/fql.json');config['log_every']=1;assert config['checkpoint_steps']>0;save('fql.json',config)
 env.update(CUDA_VISIBLE_DEVICES='2,3',FQL_GPU_PAIR='2,3',POLICY_EVAL_TOKEN='fql-smoke1026-train')
 for stop in [6,8]:
  command=[sys.executable,'-u','-B','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(R/'tools/pi05_distillation/train_fql.py'),'--config',str(D/'fql.json'),'--output',str(D/'training'),'--stop-step',str(stop)]
  if stop==8:command+=['--resume',str(D/'training/checkpoint-000006.pt'),'--check-export']
  with (D/f'train-{stop}.log').open('x')as log:
   p=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('status.json',dict(phase='train',stop=stop,identity=proc(p.pid),command=command))
   try:rc=p.wait(timeout=900)
   except subprocess.TimeoutExpired:cleanup(env['POLICY_EVAL_TOKEN']);rc=p.wait(timeout=30)
  cleanup(env['POLICY_EVAL_TOKEN']);assert rc==0,f'train stage {stop} rc={rc}'
 save('status.json',dict(phase='complete',time=time.time(),stop=8,collection=results))
except Exception as exc:
 save('status.json',dict(phase='failed',time=time.time(),error=repr(exc),collection=results));raise
finally:
 for j in jobs:cleanup(j['token'])
 cleanup('fql-smoke1026-train')
