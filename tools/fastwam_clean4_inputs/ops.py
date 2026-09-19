"""Own GPU4 sequential full Stage1 -> FastWAM RLT Clean4 training."""
import os,json,time,datetime,subprocess,sys,signal,runpy,hashlib,urllib.request,resource
from pathlib import Path
ST=Path('/data/chenyiteng/deployment-20260919/fastwam-rlt-turn-switch-full-stage1')
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proc(pid):
 try:
  p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split();return {'pid':int(pid),'uid':p.stat().st_uid,'start':int(v[19]),'state':v[0]}
 except (FileNotFoundError,ProcessLookupError):return None
def same(v):
 q=proc(v['pid']);return bool(q and q['state']!='Z' and all(q[k]==v[k] for k in ('pid','uid','start')))
def actors():
 with urllib.request.urlopen('http://127.0.0.1:8266/api/v0/actors?limit=10000&detail=1',timeout=20) as f:return json.load(f)['data']['result']['result']
def gpu_pids(gpus):
 mapping={r.split(',')[1].strip():int(r.split(',')[0]) for r in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()};out=[]
 for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
  uuid,pid=map(str.strip,row.split(','))
  if mapping[uuid] in gpus:out.append(int(pid))
 return out
def source_check():
 p=read(ST/'plan.json');r=Path(p['repo']);assert subprocess.check_output(['git','-C',str(r),'rev-parse','HEAD'],text=True).strip()==p['head'];assert not subprocess.check_output(['git','-C',str(r),'status','--porcelain'],text=True).strip();return p
def make_formal():
 from hydra import compose,initialize_config_dir
 from omegaconf import OmegaConf
 p=source_check();manifest=ST/'stage1/stage1-manifest.json';m=read(manifest);assert m['step']==2000 and m['smoke'] is False;assert read(ST/'features/features.json')['frames']==read(ST/'features/features.json')['dataset_total_frames']
 env=read(ST/'environment.json');env.update({'FASTWAM_RLT_STAGE1_PATH':str(ST/'stage1/stage1.pt'),'RLT_STAGE1_MODEL_PATH':str(ST/'stage1/stage1.pt'),'RLT_STAGE1_MANIFEST_PATH':str(manifest),'RLT_STAGE1_MANIFEST_SHA256':sha(manifest),'RLT_STAGE1_MANIFEST_ID':'fastwam-clean4-full-'+sha(manifest)[:16],'RLT_NORM_STATS_SHA256':m['identity']['stats_sha256']});env.pop('CUDA_VISIBLE_DEVICES',None);os.environ.update(env)
 with initialize_config_dir(config_dir=str(Path(p['repo'])/'examples/embodiment/config'),version_base=None):cfg=compose(config_name='robotwin_turn_switch_fastwam_rlt_clean4_stage1_20260919');OmegaConf.resolve(cfg)
 assert cfg.runner.max_steps==800 and cfg.env.train.total_num_envs==4 and cfg.algorithm.rlt_dvac.mode=='off';assert cfg.rollout.rlt_feature_model.rlt.minimum_stage1_steps==2000
 rt=Path(p['run'])/'runtime';rt.mkdir(parents=True,exist_ok=False);OmegaConf.save(cfg,rt/'resolved.yaml',resolve=True);save(rt/'environment.json',env);save(rt/'contract.json',{'repo':p['repo'],'run':p['run'],'head':p['head'],'namespace':p['namespace'],'gpus':[4],'stage1_manifest':m,'stage1_manifest_sha256':sha(manifest),'config_sha256':sha(rt/'resolved.yaml')});return rt,env
def run_child(label,args,env,cwd):
 with (ST/(label+'.log')).open('x') as f:child=subprocess.Popen(args,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 save(ST/(label+'-identity.json'),{**proc(child.pid),'time':now(),'argv':args});rc=child.wait();save(ST/(label+'-finished.json'),{'time':now(),'exit_code':rc});assert rc==0,(label,rc)
def pipeline():
 p=source_check();assert not gpu_pids([4]);save(ST/'pipeline-identity.json',{**proc(os.getpid()),'time':now()});env=read(ST/'environment.json');env['CUDA_VISIBLE_DEVICES']='4';py=p['python'];r=Path(p['repo']);entry=str(r/'examples/sft/train_fastwam_rlt_stage1.py');config=str(ST/'stage1-config.yaml')
 run_child('prepare-data',[py,'-u','-B',str(r/'tools/fastwam_clean4_inputs/turn_dataset.py')],env,str(r))
 run_child('extract',[py,'-u','-B',entry,'extract','--config',config,'--out',str(ST/'features')],env,str(r))
 run_child('stage1',[py,'-u','-B',entry,'train','--config',config,'--features',str(ST/'features'),'--out',str(ST/'stage1'),'--steps','2000'],env,str(r))
 rt,env=make_formal();assert not gpu_pids([4]);assert not any(a.get('ray_namespace')==p['namespace'] and a.get('state')!='DEAD' for a in actors())
 args=[py,'-u','-B',str(Path(__file__).resolve()),'driver']
 with (rt/'driver.log').open('x') as f:child=subprocess.Popen(args,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 save(rt/'driver-spawn.json',{'time':now(),'identity':proc(child.pid)});save(ST/'formal-dispatched.json',{'time':now(),'run':p['run'],'identity':proc(child.pid)});rc=child.wait();save(rt/'finished.json',{'time':now(),'exit_code':rc});(rt/'exit_code.txt').write_text(str(rc));return rc
def cleanup(ns,rt):
 import ray
 if not ray.is_initialized():return
 job=ray.get_runtime_context().get_job_id();job=job.hex() if hasattr(job,'hex') else str(job)
 rows=[a for a in actors() if a.get('ray_namespace')==ns and a.get('state')!='DEAD'];assert all(a.get('job_id')==job and proc(a['pid'])['uid']==1003 for a in rows)
 save(rt/'cleanup-targets.json',{'time':now(),'actors':rows})
 for a in rows:
  if a.get('name'):
   try:ray.kill(ray.get_actor(a['name'],namespace=ns),no_restart=True)
   except ValueError:pass
def driver():
 p=source_check();r=Path(p['repo']);rt=Path(p['run'])/'runtime';env=read(rt/'environment.json');assert 'CUDA_VISIBLE_DEVICES' not in env;os.environ.update(env)
 for v in reversed(env['PYTHONPATH'].split(':')):sys.path.insert(0,v)
 soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(max(soft,min(4096,hard)),hard))
 from rlinf.scheduler import Cluster
 Cluster.NAMESPACE=p['namespace'];save(rt/'driver-identity.json',{**proc(os.getpid()),'namespace':p['namespace'],'time':now()})
 def term(s,f):raise SystemExit(128+s)
 signal.signal(signal.SIGTERM,term);sys.argv=[str(r/'examples/embodiment/train_embodied_agent.py'),'--config-path',str(rt),'--config-name','resolved','hydra.run.dir=.','hydra.output_subdir=null','hydra.job.chdir=false','hydra/job_logging=stdout']
 try:runpy.run_path(sys.argv[0],run_name='__main__')
 finally:
  import ray
  signal.signal(signal.SIGUSR1,signal.SIG_IGN)
  try:cleanup(p['namespace'],rt)
  finally:ray.shutdown()
if __name__=='__main__':
 assert os.getuid()==1003
 if sys.argv[1]=='pipeline':raise SystemExit(pipeline())
 if sys.argv[1]=='driver':driver()
