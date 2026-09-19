"""Own two-card Stage1/smoke pipeline; preserves shared Ray and unrelated jobs."""
import os,sys,json,time,subprocess,signal,runpy,resource,datetime,urllib.request,math
from pathlib import Path
ST=Path('/data/chenyiteng/deployment-20260919/pi05-rlt-pair')
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2)
def proc(pid):
 try:
  d=Path('/proc')/str(pid);a=(d/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=int(pid),uid=d.stat().st_uid,start=int(a[19]),state=a[0])
 except (FileNotFoundError,ProcessLookupError):return None
def same(i):
 p=proc(i['pid']);return p and p['state']!='Z' and all(p[k]==i[k] for k in ('uid','pid','start'))
def actors():
 with urllib.request.urlopen('http://127.0.0.1:8266/api/v0/actors?limit=10000&detail=1',timeout=20) as f:return json.load(f)['data']['result']['result']
def gpu_pids(gpus):
 ids={s.split(',')[1].strip():int(s.split(',')[0]) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()};out=[]
 for s in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
  u,p=map(str.strip,s.split(','))
  if ids[u] in gpus:out.append(int(p))
 return out
def checked():
 p=read(ST/'plan.json');assert subprocess.check_output(['git','-C',p['repo'],'rev-parse','HEAD'],text=True).strip()==p['head'];assert not subprocess.check_output(['git','-C',p['repo'],'status','--porcelain'],text=True).strip();return p
def cleanup(ns,rt):
 import ray
 if not ray.is_initialized():return
 j=ray.get_runtime_context().get_job_id();j=j.hex() if hasattr(j,'hex') else str(j)
 aa=[a for a in actors() if a.get('ray_namespace')==ns and a.get('state')!='DEAD'];assert all(a['job_id']==j and proc(a['pid'])['uid']==1003 for a in aa)
 save(rt/'cleanup-targets.json',aa)
 for a in aa:
  if a.get('name'):
   try:ray.kill(ray.get_actor(a['name'],namespace=ns),no_restart=True)
   except ValueError:pass
def driver(key):
 p=checked();s=p['runs'][key];rt=Path(s['run'])/'runtime';env=read(rt/'environment.json');assert 'CUDA_VISIBLE_DEVICES' not in env;os.environ.update(env)
 for v in reversed(env['PYTHONPATH'].split(':')):sys.path.insert(0,v)
 soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(max(soft,min(4096,hard)),hard))
 from rlinf.scheduler import Cluster
 Cluster.NAMESPACE=s['namespace'];save(rt/'driver-identity.json',{**proc(os.getpid()),'namespace':s['namespace'],'time':now()})
 def stop(a,b):raise SystemExit(128+a)
 signal.signal(signal.SIGTERM,stop);sys.argv=[str(Path(p['repo'])/s['entry']),'--config-path',str(rt),'--config-name','resolved','hydra.run.dir=.','hydra.output_subdir=null','hydra.job.chdir=false','hydra/job_logging=stdout']
 try:runpy.run_path(sys.argv[0],run_name='__main__')
 finally:
  import ray
  signal.signal(signal.SIGUSR1,signal.SIG_IGN)
  try:cleanup(s['namespace'],rt)
  finally:ray.shutdown()
def launch(key):
 p=checked();s=p['runs'][key];rt=Path(s['run'])/'runtime';assert not gpu_pids(s['gpus']);assert not any(a.get('ray_namespace')==s['namespace'] and a.get('state')!='DEAD' for a in actors())
 with (rt/'driver.log').open('x') as f:q=subprocess.Popen([p['python'],'-u','-B',p['ops'],'driver',key],cwd=p['repo'],env=read(rt/'environment.json'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 save(rt/'launch.json',dict(time=now(),identity=proc(q.pid),key=key));return q
def wait_child(key,q):
 p=read(ST/'plan.json');rt=Path(p['runs'][key]['run'])/'runtime';rc=q.wait();save(rt/'finished.json',dict(time=now(),exit_code=rc));(rt/'exit_code.txt').write_text(str(rc));assert rc==0,(key,rc)
 for _ in range(60):
  if not gpu_pids(p['runs'][key]['gpus']):break
  time.sleep(1)
 else:raise RuntimeError('Own GPU cleanup incomplete: '+key)
def smoke_gate(key):
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 s=read(ST/'plan.json')['runs'][key];e=EventAccumulator(str(Path(s['run'])/'tensorboard'),size_guidance={'scalars':0});e.Reload();tags=e.Tags()['scalars'];updates=e.Scalars('train/rlt/update_step');assert updates and updates[-1].value>0
 bad={t:[v.value for v in e.Scalars(t) if not math.isfinite(v.value)] for t in tags if t.startswith('train/')};assert not any(bad.values()),bad
 save(Path(s['run'])/'runtime/smoke-passed.json',dict(time=now(),update_step=updates[-1].value,tags=tags))
def pipeline():
 p=checked();save(ST/'pipeline-identity.json',{**proc(os.getpid()),'time':now()})
 # Two real Stage1 updates validate the model/data path before the online smoke.
 q=launch('stage1-smoke');wait_child('stage1-smoke',q)
 qs={k:launch(k) for k in ('smoke-clean','smoke-combo')}
 for k,q in qs.items():wait_child(k,q);smoke_gate(k)
 save(ST/'smoke-passed.json',{'time':now(),'groups':list(qs)})
 q=launch('stage1-full');wait_child('stage1-full',q)
 assert Path(p['stage1_full_weights']).is_file()
 save(ST/'stage1-complete.json',{'time':now(),'weights':p['stage1_full_weights']})
 qs={k:launch(k) for k in ('clean','combo')};save(ST/'formal-dispatched.json',{'time':now(),'drivers':{k:proc(q.pid) for k,q in qs.items()}})
 for k,q in qs.items():wait_child(k,q)
if __name__=='__main__':
 assert os.getuid()==1003
 if sys.argv[1]=='driver':driver(sys.argv[2])
 elif sys.argv[1]=='pipeline':pipeline()
