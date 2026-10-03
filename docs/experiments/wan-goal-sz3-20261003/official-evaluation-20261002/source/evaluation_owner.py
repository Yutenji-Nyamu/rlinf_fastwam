"""Run three matched physical evaluations; isolate and precisely retire each Ray.

The 3-hour per-checkpoint limit is this window's evaluation management bound;
it does not change the WM/RLT training budget or the 500-episode protocol.
"""
import argparse,hashlib,json,os,pathlib,re,signal,socket,subprocess,sys,time,traceback,uuid
P=pathlib.Path('/data/chenyiteng/projects/wan-goal-sz3')
sys.path.insert(0,str(P/'scripts/resource_switch'))
from common import account,alive,atomic,exclusive,identity,ProcessIdentityChanged
from wm_stage import Catalog

def gpu_contexts():
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=30)
 ids={fields[1]:int(fields[0]) for fields in ([part.strip() for part in x.split(',')] for x in gpu.splitlines())}
 query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=30)
 return [{'gpu':ids[fields[0]],'pid':int(fields[1])} for fields in ([part.strip() for part in x.split(',')] for x in query.splitlines() if x.strip())]

def main():
 account();parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
 run=pathlib.Path(args.run);resolved=run.resolve(strict=True)
 assert resolved.is_relative_to((P/'evaluations').resolve()) and run.name.startswith('wm-official-20261002-')
 assert not run.is_symlink() and run.stat().st_uid==20001
 assert not (run/'owner.json').exists()
 source=run/'source';python=P/'envs/pi05-wan/bin/python'
 assert source.is_dir() and not source.is_symlink() and source.stat().st_uid==20001
 assert source.resolve().is_relative_to(resolved)
 vendor=source/'50_mesa.json'
 vendor_data={'file_format_version':'1.0.0','ICD':{'library_path':'libEGL_mesa.so.0'}}
 assert json.loads(pathlib.Path('/usr/share/glvnd/egl_vendor.d/50_mesa.json').read_text())==vendor_data
 assert vendor.is_file() and not vendor.is_symlink() and vendor.stat().st_uid==20001
 assert json.loads(vendor.read_text())==vendor_data
 vendor.chmod(0o600)
 formal_root=P/'runs/wan-goal-sz3-20261001-r6/pi05-formal/tensorboard'
 formal=next((p for p in (formal_root/'all/config.yaml',formal_root/'config.yaml') if p.is_file()),None)
 assert formal is not None,'Missing actual formal TensorBoard config'
 cache_root=pathlib.Path('/dev/shm')/('chenyiteng-wm-eval-'+hashlib.sha256(str(resolved).encode()).hexdigest()[:16])
 assert not cache_root.exists(),'Evaluation cache path already exists; do not reuse another owner'
 cache_root.mkdir(mode=0o700)
 for name in ('triton','inductor'):(cache_root/name).mkdir(mode=0o700)
 base_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
   HF_HOME=str(P/'cache/huggingface'),XDG_CACHE_HOME=str(P/'cache/xdg'),TORCH_HOME=str(P/'cache/torch'),
   OPENPI_DATA_HOME=str(P/'models/openpi-assets'),LIBERO_CONFIG_PATH=str(P/'config/libero-pi05'),
   TMPDIR=str(run/'tmp'),TRITON_CACHE_DIR=str(cache_root/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache_root/'inductor'),
   HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LIBERO_TYPE='standard',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
   __EGL_VENDOR_LIBRARY_FILENAMES=str(vendor),WM_EVAL_RENDER_BACKEND='mesa_llvmpipe',WM_EVAL_MESA_DEVICE_ID='8',
   LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='llvmpipe',LP_NUM_THREADS='2',
   PYTHONPATH=str(source)+os.pathsep+str(P/'RLinf-pi05'),EMBODIED_PATH=str(P/'RLinf-pi05/examples/embodiment'))
 base_env.pop('MUJOCO_EGL_DEVICE_ID',None)
 (run/'tmp').mkdir(parents=True,exist_ok=True,mode=0o700)
 protected=[]
 for row in gpu_contexts():
  assert row['gpu'] not in (2,3),'Evaluation GPUs are already occupied'
  if row['gpu'] in (4,5,6,7):protected.append(identity(row['pid']))
 assert protected,'Expected active WM GPU contexts missing'
 protected={x['pid']:x for x in protected}
 owner=identity(os.getpid());exclusive(run/'owner.json',dict(time=time.time(),identity=owner,physical_gpus=[2,3],protected=list(protected.values()),evaluation_stage_limit_seconds=10800,private_compile_cache=str(cache_root),cache_retained=True,render_backend='mesa_llvmpipe',software_device=8,egl_vendor_file=str(vendor),egl_vendor_sha256=hashlib.sha256(vendor.read_bytes()).hexdigest(),lp_num_threads=2))
 def check_protected():
  assert all(alive(row) for row in protected.values()),'Protected training process changed; stop only evaluator'
 def cleanup(catalog,phase):
  actions=[]
  for sig,seconds in ((signal.SIGTERM,25),(signal.SIGKILL,15)):
   deadline=time.monotonic()+seconds;sent=set()
   while time.monotonic()<deadline:
    catalog.scan();live=[x for x in catalog.rows.values() if alive(x)]
    if not live:break
    for row in live:
     assert not any(all(row[k]==p[k] for k in ('pid','start','boot','uid')) for p in protected.values()),'Protected training identity entered cleanup catalog; refuse to signal'
     key=(row['pid'],row['start'],row['boot'])
     if key not in sent and catalog.signal(row,sig):actions.append({'signal':sig.name,'identity':row});sent.add(key)
    atomic(phase/'cleanup-actions.json',dict(time=time.time(),token=catalog.token,actions=actions))
    time.sleep(1)
  for _ in range(3):
   catalog.scan();assert not any(alive(x) for x in catalog.rows.values()),'Owned evaluator process remains';time.sleep(1)
  contexts=[x for x in gpu_contexts() if x['gpu'] in (2,3)]
  assert not contexts,'Evaluation GPU contexts remain; no foreign process is signaled'
  atomic(phase/'cleanup.json',dict(time=time.time(),token=catalog.token,processes_clear=True,gpus_released=True,actions=actions))
 closing=[False]
 def foreign_gpu_contexts(catalog):
  contexts=[x for x in gpu_contexts() if x['gpu'] in (2,3)]
  pending=[x for x in contexts if not any(x['pid']==p['pid'] and alive(p) for p in catalog.rows.values())]
  if not pending:return []
  # A GPU context can appear between the preceding catalog scan and NVML query.
  # Scan again before deciding a live PID belongs to a different workload.
  catalog.scan();foreign=[]
  for row in pending:
   try:current=identity(row['pid'])
   except ProcessIdentityChanged as exc:
    foreign.append(dict(row,identity_error=str(exc)));continue
   except (FileNotFoundError,ProcessLookupError):continue
   if current['state'] in ('Z','X'):continue
   known=any(all(current[k]==p[k] for k in ('pid','start','boot','uid')) and alive(p) for p in catalog.rows.values())
   if not known:foreign.append(dict(row,identity=current))
  return foreign
 def abort(sig,frame):
  if not closing[0]:raise RuntimeError('Evaluation owner received signal '+str(sig))
 signal.signal(signal.SIGTERM,abort);signal.signal(signal.SIGINT,abort)
 outcomes={};catalog=None;phase=None
 try:
  for kind,label in [('original','original'),('40','cp40'),('80','cp80')]:
   check_protected();assert not any(x['gpu'] in (2,3) for x in gpu_contexts())
   assert os.statvfs('/').f_bavail*os.statvfs('/').f_frsize>1024**3,'Root available below 1GiB'
   phase=run/label;phase.mkdir(mode=0o700)
   with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   command=[str(python),'-B',str(source/'private_eval_driver.py'),'--repo',str(P/'RLinf-pi05'),
    '--formal-config',str(formal),'--config-dir',str(source),'--log-dir',str(phase),'--kind',kind,'--port',str(port)]
   atomic(run/'status.json',dict(time=time.time(),phase='PREFLIGHT',kind=kind,outcomes=outcomes))
   preflight=subprocess.run(command+['--preflight'],env=base_env,cwd=P/'RLinf-pi05',capture_output=True,text=True,timeout=90)
   (phase/'preflight.stdout').write_text(preflight.stdout);(phase/'preflight.stderr').write_text(preflight.stderr)
   assert preflight.returncode==0,preflight.stderr[-5000:]
   token=str(uuid.uuid4());ns='wan_goal_eval_'+uuid.uuid4().hex
   raytmp='/data/chenyiteng/we/'+uuid.uuid4().hex[:10]
   env=dict(base_env,CUDA_VISIBLE_DEVICES='2,3',WM_OWNER_TOKEN=token,CLUSTER_NAMESPACE=ns,
     RAY_ADDRESS='127.0.0.1:'+str(port),WAN_GOAL_RAY_TMPDIR=raytmp)
   catalog=Catalog(phase,token)
   with (phase/'command.log').open('x') as log:
    child=subprocess.Popen(command,env=env,cwd=P/'RLinf-pi05',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    anchor=identity(child.pid);catalog.add(anchor,'evaluation-owner-launch');started=time.monotonic()
    atomic(phase/'driver-identity.json',anchor)
    while child.poll() is None:
     catalog.scan();catalog.persist();check_protected()
     if kind=='original':
      log_text=(phase/'command.log').read_text(errors='replace')
      pairs={}
      for task,trial,score in re.findall(r'\[libero eval\]\s+task_id=(\d+),\s*trial_id=(\d+),\s*success=(True|False)\b',log_text):
       key=(int(task),int(trial));value=score=='True'
       assert key not in pairs or pairs[key]==value,'Conflicting native outcomes'
       pairs[key]=value
      first20=list(pairs.items())[:20]
      if len(first20)==20 and not any(score for _,score in first20):
       atomic(phase/'baseline-zero-gate.json',dict(time=time.time(),completed=len(pairs),gate_completed=20,gate_trials=[list(key) for key,_ in first20],successes=0,decision='Stop evaluation and do not queue CP40/80',diagnostic_only=True))
       raise RuntimeError('Official reference baseline still zero after first20 completed episodes; evaluation stopped for diagnosis')
     foreign=foreign_gpu_contexts(catalog)
     assert not foreign,'Foreign process appeared on evaluator GPUs'
     assert time.monotonic()-started<10800,"Evaluation exceeded this window's 3h per-checkpoint management limit"
     atomic(run/'status.json',dict(time=time.time(),phase='EVALUATING',kind=kind,driver=anchor,elapsed_seconds=time.monotonic()-started,outcomes=outcomes))
     time.sleep(3)
   catalog.scan();catalog.persist();closing[0]=True;cleanup(catalog,phase);catalog=None;closing[0]=False
   assert child.returncode==0,'Evaluation driver failed: '+str(child.returncode)
   result=json.loads((phase/'evaluation-result.json').read_text());assert result['ok'] and result['unique_task_trials']==500
   outcomes[label]=result;check_protected()
   if kind=='original':assert result['successes']>0,'Baseline zero; refuse subsequent checkpoints'
  atomic(run/'complete.json',dict(time=time.time(),ok=True,physical_gpus=[2,3],gpus_released=True,protected_training_alive=True,results=outcomes))
  atomic(run/'status.json',dict(time=time.time(),phase='COMPLETE',outcomes=outcomes))
 except BaseException as exc:
  error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());closing[0]=True
  if catalog is not None:
   try:cleanup(catalog,phase)
   except BaseException as cleanup_exc:error['cleanup_error']=str(cleanup_exc)
  atomic(run/'failure.json',dict(time=time.time(),error=error,outcomes=outcomes))
  atomic(run/'status.json',dict(time=time.time(),phase='FAILED',error=error,outcomes=outcomes))
  raise

if __name__=='__main__':main()
