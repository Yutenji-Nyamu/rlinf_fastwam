"""Own four BC dropout-temperature runs; preserve shared Ray and unrelated jobs."""
import os,sys,json,time,subprocess,signal,runpy,resource,datetime,urllib.request,math
from pathlib import Path
ST=Path('/data/chenyiteng/deployment-20260923/bc8-dualtricks')
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
 p=read(ST/'plan.json')
 for info in p['repos'].values():
  assert subprocess.check_output(['git','-C',info['repo'],'rev-parse','HEAD'],text=True).strip()==info['head']
  assert not subprocess.check_output(['git','-C',info['repo'],'status','--porcelain','-uno'],text=True).strip()
 return p

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
 signal.signal(signal.SIGTERM,stop);sys.argv=[str(Path(s['repo'])/s['entry']),'--config-path',str(rt),'--config-name','resolved','hydra.run.dir=.','hydra.output_subdir=null','hydra.job.chdir=false','hydra/job_logging=stdout']
 try:runpy.run_path(sys.argv[0],run_name='__main__')
 finally:
  import ray
  signal.signal(signal.SIGUSR1,signal.SIG_IGN)
  try:cleanup(s['namespace'],rt)
  finally:ray.shutdown()
def launch(key):
 p=checked();s=p['runs'][key];rt=Path(s['run'])/'runtime';assert not gpu_pids(s['gpus']);assert not any(a.get('ray_namespace')==s['namespace'] and a.get('state')!='DEAD' for a in actors())
 with (rt/'driver.log').open('x') as f:q=subprocess.Popen([p['python'],'-u','-B',p['ops'],'driver',key],cwd=s['repo'],env=read(rt/'environment.json'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 save(rt/'launch.json',dict(time=now(),identity=proc(q.pid),key=key));return q
def wait_child(key,q):
 p=read(ST/'plan.json');rt=Path(p['runs'][key]['run'])/'runtime';rc=q.wait();
 if key=='sarm':stop_scorer()
 save(rt/'finished.json',dict(time=now(),exit_code=rc));(rt/'exit_code.txt').write_text(str(rc));assert rc==0,(key,rc)
 for _ in range(60):
  if not gpu_pids(p['runs'][key]['gpus']):break
  time.sleep(1)
 else:raise RuntimeError('Own GPU cleanup incomplete: '+key)

def start_scorer():
 import socket
 p=checked();s=p['runs']['sarm'];root=Path(s['repo']);assert not gpu_pids([4])
 with socket.socket() as sock:sock.bind(('127.0.0.1',18804))
 env=os.environ.copy()
 for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
 cache='/data/chenyiteng/cache/rynnvalue'
 env.update(CUDA_VISIBLE_DEVICES='4',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HOME=cache+'/huggingface',HF_MODULES_CACHE=cache+'/huggingface/modules',HF_HUB_CACHE=cache+'/huggingface/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_HOME=cache+'/torch',XDG_CACHE_HOME=cache+'/xdg',MPLCONFIGDIR=cache+'/matplotlib',CUDA_CACHE_PATH=cache+'/cuda',TRITON_CACHE_DIR=cache+'/triton',TMPDIR=cache+'/tmp',IMAGEIO_USERDIR=cache+'/imageio',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
 model='/data/chenyiteng/models/RynnValue-8B-8738c5e4'
 entry="import runpy;runpy.run_path(%r,run_name='__main__')" % str(root/'rlinf/utils/rynnvalue_scorer.py')
 argv=['/data/chenyiteng/venvs/rynnvalue-8b-py310/bin/python','-u','-B','-c',entry,'serve','--model-path',model,'--manifest',model+'/manifest.json','--device','cuda:0','--port','18804']
 with (ST/'scorer.log').open('x') as f:q=subprocess.Popen(argv,cwd=ST,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 save(ST/'scorer-launch.json',{'identity':proc(q.pid),'time':now()})
 for _ in range(180):
  assert q.poll() is None,'Scorer exited'
  try:
   with urllib.request.urlopen('http://127.0.0.1:18804/health',timeout=2) as f:h=json.load(f)
   if h.get('ok'):save(ST/'scorer-ready.json',h);return
  except Exception:pass
  time.sleep(1)
 raise RuntimeError('Scorer startup timeout')

def stop_scorer():
 f=ST/'scorer-launch.json'
 if f.exists():
  i=read(f)['identity']
  if same(i):os.kill(i['pid'],signal.SIGTERM)

def supervise(key):
 try:
  if key=='sarm':start_scorer()
  q=launch(key);wait_child(key,q)
 finally:
  if key=='sarm':stop_scorer()

if __name__=='__main__':
 assert os.getuid()==1003
 if sys.argv[1]=='driver':driver(sys.argv[2])
 elif sys.argv[1]=='supervise':supervise(sys.argv[2])
