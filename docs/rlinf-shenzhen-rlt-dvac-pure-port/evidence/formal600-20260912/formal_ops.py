"""Run-scoped launch of the already smoked Shenzhen single-GPU RLT pair."""
import copy,datetime,hashlib,json,os,runpy,shlex,shutil,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ST=Path('/data/chenyiteng/results/server-maintenance-20260912/rlt-switch')
BASE=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421')
BASE_HEAD='30349428c37a008b95342121c1455debfeb4805e'
SMOKED_HEAD='b1e01364b01a9f6d6072e2645cd7ba3bdf0df8fd'
PY='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
ROBOTWIN='/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support'
MODEL='/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50'
S1=Path('/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1')
CHECKPOINT=S1/'robotwin_adjust_bottle_rlt_stage1_current_ar_clean50_2k_v1/checkpoints/global_step_2000'
MANIFEST=S1/'artifacts/stage1_artifact_manifest.json'
NORM=Path(MODEL)/'physical-intelligence/robotwin/norm_stats.json'
RAY='172.17.0.1:6389'
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,data):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 with Path(p).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2)
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def select(key):
 assert key in ('control','pure04');gpu=6 if key=='control' else 7
 root=BASE.parent/f'rlt-{key}-single-gpu600-20260912'
 run=Path('/data/chenyiteng/results/rlinf-rlt')/f'current-single-gpu-{key}-phys{gpu}-fresh600-20260912-v1'
 return {'key':key,'gpu':gpu,'root':str(root),'run':str(run),'branch':f'codex/sz-rlt-{key}-single-gpu600-20260912',
  'namespace':f'RLinf_rlt_{key}_single_gpu{gpu}_fresh600_20260912','config':'robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_'+('control' if key=='control' else 'dvac_pure04')}
def proc(pid):
 p=Path('/proc')/str(pid)
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  d={'pid':int(pid),'uid':p.stat().st_uid,'start':int(fields[19]),'state':fields[0]}
  if d['uid']==1003:d['cmd']=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
  return d
 except FileNotFoundError:return None
def same(d):
 p=proc(d['pid']);return bool(p and p['state']!='Z' and all(p[k]==d[k] for k in ('pid','uid','start')))
def actors():
 with urllib.request.urlopen('http://127.0.0.1:8266/api/v0/actors?limit=2000&detail=1&filter_keys=state&filter_predicates=%3D&filter_values=ALIVE',timeout=15) as f:return json.load(f)['data']['result']['result']
def gpu_pids(gpus=None):
 args=['nvidia-smi']+(['-i',gpus] if gpus else [])+['--query-compute-apps=pid','--format=csv,noheader,nounits']
 return [int(x) for x in subprocess.check_output(args,text=True).splitlines() if x.strip().isdigit()]
def environment(s):
 return {'RAY_ADDRESS':RAY,'RLINF_CODE_WORKING_DIR':s['root'],'REPO_PATH':s['root'],'EMBODIED_PATH':s['root']+'/examples/embodiment',
  'PYTHONPATH':s['root']+':'+ROBOTWIN,'ROBOTWIN_PATH':ROBOTWIN,'ROBOTWIN_ASSETS_PATH':ROBOTWIN,'ROBOT_PLATFORM':'ALOHA',
  'ROBOTWIN_PI0_BASE_PATH':MODEL,'RLT_LOG_ROOT':s['run'],'RLT_STAGE1_MODEL_PATH':str(CHECKPOINT),'RLT_STAGE1_MANIFEST_PATH':str(MANIFEST),
  'RLT_STAGE1_MANIFEST_ID':'sz-rlt-stage1-current-ar-clean50-2k-v1','RLT_STAGE1_MANIFEST_SHA256':sha(MANIFEST),
  'RLT_NORM_STATS_SHA256':sha(NORM),'ROBOTWIN_PI0_NORM_STATS_PATH':str(NORM),'OPENPI_DATA_HOME':'/home/chenyiteng/.cache/openpi',
  'JAX_PLATFORMS':'cpu','TOKENIZERS_PARALLELISM':'false','MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl',
  'HYDRA_FULL_ERROR':'1','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1'}
def flat(v,p=''):
 if not isinstance(v,dict):return {p:v}
 return {k:x for n,c in v.items() for k,x in flat(c,p+'.'+n if p else n).items()}
def diff(a,b):
 a,b=flat(a),flat(b);return {k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
def prepare(key):
 from hydra import compose,initialize_config_dir
 from omegaconf import OmegaConf
 assert os.getuid()==1003 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 s=select(key);root=Path(s['root']);run=Path(s['run']);pre=ST/f'prepared-{key}'
 assert git(BASE,'rev-parse','HEAD')==BASE_HEAD and not git(BASE,'status','--porcelain')
 assert not git(BASE,'diff','--name-only',SMOKED_HEAD,BASE_HEAD,'--','rlinf','examples','tests')
 assert not root.exists() and not run.exists() and not pre.exists()
 stage=read(MANIFEST);assert stage['id']=='sz-rlt-stage1-current-ar-clean50-2k-v1'
 weight=CHECKPOINT/'actor/model_state_dict/full_weights.pt';assert weight.is_file() and weight.stat().st_size==stage['full_weights']['size_bytes']
 save(ST/f'create-{key}-attempt.json',s)
 subprocess.run(['git','-C',str(BASE),'worktree','add','-b',s['branch'],str(root),BASE_HEAD],check=True)
 pre.mkdir();env=environment(s);os.environ.update(env)
 with initialize_config_dir(version_base='1.1',config_dir=str(root/'examples/embodiment/config')):
  cfg=compose(config_name=s['config'])
  baseline=OmegaConf.to_container(cfg,resolve=True)
 actual=copy.deepcopy(baseline)
 actual['runner']['max_steps']=600
 if actual['runner']['max_epochs']<600:actual['runner']['max_epochs']=600
 actual['runner']['logger']['log_path']=str(run);actual['runner']['logger']['experiment_name']=run.name
 actual['runner']['resume_dir']=None;actual['runner']['ckpt_path']=None
 actual['cluster']['component_placement']={next(iter(actual['cluster']['component_placement'])):s['gpu']}
 for split in ('train','eval'):
  actual['env'][split]['task_config']['save_path']=str(run/'robotwin_data'/split)
  actual['env'][split]['video_cfg']['video_base_dir']=str(run/'video'/split)
 # Keep the historical video on/off settings and all algorithm/model fields.
 changes=diff(baseline,actual)
 allowed={'runner.max_steps','runner.max_epochs','runner.logger.log_path','runner.logger.experiment_name','runner.resume_dir','runner.ckpt_path','cluster.component_placement.actor,env,rollout'}
 allowed|={f'env.{split}.{field}' for split in ('train','eval') for field in ('task_config.save_path','video_cfg.video_base_dir')}
 assert not(set(changes)-allowed),changes
 assert actual['runner']['max_steps']==600 and actual['runner']['max_epochs']>=600
 assert actual['runner']['val_check_interval']==25 and actual['runner']['save_interval']==25
 assert actual['actor']['global_batch_size']==512 and actual['actor']['micro_batch_size']==256
 assert actual['env']['train']['total_num_envs']==8 and actual['env']['eval']['total_num_envs']==4 and actual['env']['eval']['rollout_epoch']==5
 assert actual['algorithm']['update_epoch']==5 and actual['algorithm']['rlt_schedule']['warmup_min_size']==20000
 assert actual['algorithm']['rlt_schedule']['warmup_post_collect_updates']==30000
 assert actual['algorithm']['replay_buffer']['cache_size']==actual['algorithm']['replay_buffer']['sample_window_size']==80000
 assert actual['algorithm']['rlt_dvac']['mode']==('off' if key=='control' else 'apply')
 if key=='pure04':
  m=actual['algorithm']['rlt_dvac'];assert m['strength']==1.5 and m['selected_l']==3 and m['applied_horizon']==10 and m['application']=='success_episode_bc' and m['success_target']=='reference'
 OmegaConf.save(OmegaConf.create(actual),pre/'resolved.yaml',resolve=True)
 assert OmegaConf.load(pre/'resolved.yaml').algorithm.rlt_dvac.mode==('off' if key=='control' else 'apply')
 save(pre/'baseline-config.json',baseline);save(pre/'config-diff.json',changes);save(pre/'environment.json',env)
 argv=[PY,'-u','-B',str(ST/'formal_ops.py'),'driver',key,'--config-path',str(run/'runtime'),'--config-name','resolved','hydra.run.dir=.','hydra.output_subdir=null','hydra.job.chdir=false','hydra/job_logging=stdout']
 save(pre/'argv.json',argv);(pre/'command.txt').write_text(shlex.join(argv)+'\n')
 (pre/'command.sh').write_text('#!/usr/bin/env bash\nexec '+shlex.join(argv)+'\n')
 (pre/'wrapper.sh').write_text('''#!/usr/bin/env bash
set +e
runtime=$1
date --iso-8601=seconds > "$runtime/started_at.txt"
bash "$runtime/command.sh" > "$runtime/driver.log" 2>&1 &
child=$!
printf '%s\\n' "$child" > "$runtime/child.pid"
wait "$child"
rc=$?
printf '%s\\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
''')
 for fn in ('command.sh','wrapper.sh'):subprocess.run(['bash','-n',str(pre/fn)],check=True)
 seeds={split:sha(actual['env'][split]['seeds_path']) for split in ('train','eval')}
 contract={**s,'time':now(),'base_head':BASE_HEAD,'smoked_source':SMOKED_HEAD,'code_diff_from_smoke':[],
  'stage1_manifest_sha256':sha(MANIFEST),'norm_sha256':sha(NORM),'stage1_weight_stat':{'bytes':weight.stat().st_size,'mtime_ns':weight.stat().st_mtime_ns},
  'baseline_diff':changes,'seed_hashes':seeds,'total_cycles':600,'fresh':True,'global_batch':512,'micro_batch':256,'train_envs':8,'eval_episodes':20,
  'updates_per_data':5,'warmup_min_size':20000,'warmup_post_collect_updates':30000,'replay_capacity':80000,'eval_save_interval':25,
  'method':actual['algorithm']['rlt_dvac'],'max_episode_steps':actual['env']['train']['max_episode_steps'],'source_budget_same_as_aug30_except_total_cycles':True,
  'gpu':s['gpu'],'stop':'600 complete cycles, user request, or unrecoverable runtime error; no arbitrary performance cutoff',
  'files_sha256':{fn:sha(pre/fn) for fn in ('resolved.yaml','environment.json','argv.json','command.sh','wrapper.sh')},'ops_sha256':sha(ST/'formal_ops.py')}
 save(pre/'contract.json',contract)
 print(json.dumps({'key':key,'prepared':True,'contract_sha256':sha(pre/'contract.json'),'baseline_diff':changes,'method':contract['method']}),flush=True)
def pair_check():
 from omegaconf import OmegaConf
 configs={k:OmegaConf.to_container(OmegaConf.load(ST/f'prepared-{k}/resolved.yaml'),resolve=True) for k in ('control','pure04')}
 normalized={}
 for k,cfg in configs.items():
  s=select(k)
  def replace(v):
   if isinstance(v,str):return v.replace(s['root'],'<ROOT>').replace(s['run'],'<RUN>').replace(Path(s['run']).name,'<EXPERIMENT>')
   if isinstance(v,dict):return {a:replace(b) for a,b in v.items()}
   if isinstance(v,list):return [replace(x) for x in v]
   return v
  c=replace(cfg);c['cluster']['component_placement']={next(iter(c['cluster']['component_placement'])):'<GPU>'}
  normalized[k]=c
 changes=diff(*normalized.values())
 allowed=lambda p:p.startswith('algorithm.rlt_dvac.') or p=='rollout.rlt_feature_model.openpi.rlt_dvac_mode'
 assert all(allowed(p) for p in changes),changes
 assert read(ST/'prepared-control/contract.json')['seed_hashes']==read(ST/'prepared-pure04/contract.json')['seed_hashes']
 save(ST/'pair-config-check.json',{'time':now(),'passed':True,'method_only_diff':changes,'budget':{'cycles':600,'train_envs':8,'global_batch':512,'micro_batch':256,'UTD':5,'replay':80000,'warmup':20000,'eval_episodes':20,'eval_save_interval':25}})
 print(json.dumps(read(ST/'pair-config-check.json')),flush=True)
def launch(key):
 s=select(key);root=Path(s['root']);run=Path(s['run']);pre=ST/f'prepared-{key}';c=read(pre/'contract.json')
 assert read(ST/'pair-config-check.json')['passed'] and read(ST/'stop-receipt.json')['completed']
 assert c['ops_sha256']==sha(ST/'formal_ops.py') and git(root,'rev-parse','HEAD')==BASE_HEAD and not git(root,'status','--porcelain')
 for fn,digest in c['files_sha256'].items():assert sha(pre/fn)==digest
 assert not run.exists() and not gpu_pids(str(s['gpu']))
 assert not any(a['ray_namespace']==s['namespace'] for a in actors())
 protected=[proc(pid) for pid in gpu_pids()];assert all(p is not None for p in protected)
 save(ST/f'launch-{key}-attempt.json',{'time':now(),'contract_sha256':sha(pre/'contract.json'),'gpu':s['gpu']})
 run.mkdir();rt=run/'runtime';shutil.copytree(pre,rt);save(rt/'protected-before.json',protected);(rt/'source-head.txt').write_text(BASE_HEAD+'\n')
 env=os.environ.copy()
 for name in ('CUDA_VISIBLE_DEVICES','LD_PRELOAD','RLINF_SCENE_FENCE_LIBRARY','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(name,None)
 env.update(read(rt/'environment.json'))
 with (rt/'wrapper.log').open('xb') as log:
  p=subprocess.Popen(['bash',str(rt/'wrapper.sh'),str(rt)],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 assert all(same(x) for x in protected)
 out={'time':now(),**s,'wrapper':proc(p.pid),'source_head':BASE_HEAD,'contract_sha256':sha(pre/'contract.json'),'protected_unchanged':True}
 save(ST/f'launch-{key}-receipt.json',out);print(json.dumps(out),flush=True)
def cleanup(namespace,receipt):
 import ray
 if not ray.is_initialized():return
 assert namespace in {select(k)['namespace'] for k in ('control','pure04')}
 job=ray.get_runtime_context().get_job_id();job=job.hex() if hasattr(job,'hex') else str(job)
 deadline=time.monotonic()+10
 while True:
  rows=[a for a in actors() if a['ray_namespace']==namespace]
  before={(x['namespace'],x['name']) for x in ray.util.list_named_actors(all_namespaces=True)}
  names={x[1] for x in before if x[0]==namespace}
  identities=[proc(a['pid']) for a in rows]
  if names=={a['name'] for a in rows if a.get('name')} and all(identities):break
  if time.monotonic()>=deadline:raise RuntimeError('Owned actor identity views did not converge; no actor killed')
  time.sleep(.5)
 assert all(a['job_id']==job for a in rows) and all(p['uid']==1003 for p in identities)
 managers={'NodeManager','WorkerManager','CollectiveManager','DeviceLockManager','PortLockManager'}
 for name in sorted(names,key=lambda x:(x in managers,x)):
  try:ray.kill(ray.get_actor(name,namespace=namespace),no_restart=True)
  except ValueError:pass
 after={(x['namespace'],x['name']) for x in ray.util.list_named_actors(all_namespaces=True)}
 save(receipt,{'time':now(),'namespace':namespace,'job_id':job,'killed_names':sorted(names),'processes':identities,'other_names_missing':sorted({x for x in before if x[0]!=namespace}-after)})
def driver(key):
 s=select(key);root=Path(s['root']);rt=Path(s['run'])/'runtime';sys.path.insert(0,str(root))
 from rlinf.scheduler import Cluster
 Cluster.NAMESPACE=s['namespace'];save(rt/'driver-identity.json',{**proc(os.getpid()),'namespace':s['namespace']})
 def terminate(sig,frame):raise SystemExit(128+sig)
 signal.signal(signal.SIGTERM,terminate)
 sys.argv=[str(root/'examples/embodiment/train_embodied_agent.py'),*sys.argv[3:]]
 try:runpy.run_path(sys.argv[0],run_name='__main__')
 finally:
  import ray
  try:signal.signal(signal.SIGUSR1,signal.SIG_IGN);cleanup(s['namespace'],rt/'owned-cleanup.json')
  finally:ray.shutdown()
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='pair_check':pair_check()
 else:{'prepare':prepare,'launch':launch,'driver':driver}[mode](sys.argv[2])
