"""Formal EXPO owns physical4-7; exact cleanup and frozen RLT return on exit."""
import fcntl,hashlib,json,os,signal,socket,subprocess,time,traceback,uuid
from pathlib import Path
from expo_smoke_owner import Roster,identity,atomic,gpu_map,gpu_processes,owned
from expo_process import pidfd_probe
ROOT=Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-20261001')
SOURCE=ROOT/'source';CYCLE=ROOT/'rlt-cycle-expo-formal-20261001-v1'
RLT_PY='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
EXPO_PY='/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin/python'
UUIDS=['GPU-a0a252d6-828d-29e1-1fd2-65187f573f4d','GPU-2cd891ea-180d-da39-6419-2d7033f8b21b',
 'GPU-dc5d6921-fa81-b666-bac7-566c126f1dd4','GPU-3c6321c1-3e58-c071-3867-533391152fe7']

class Terminated(RuntimeError):pass

def main():
 assert os.getuid()==20001 and socket.gethostname()=='h100-gpu02';pidfd_probe()
 with (ROOT/'resource-owner.lock').open('a+') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (ROOT/'owner.json').exists(),'A formal owner was already launched; inspect it'
  scope='expo-formal-'+str(uuid.uuid4());me=identity(os.getpid())
  metadata={'owner':me,'scope':scope,'physical_gpus':[4,5,6,7],'status':'PREPARING','time':time.time()}
  atomic(ROOT/'owner.json',metadata);atomic(ROOT/'current.json',metadata)
  requested=[False];closing=[False];critical=[False]
  def terminate(sig,frame):
   requested[0]=True
   if not closing[0] and not critical[0]:raise Terminated('Formal owner signal '+str(sig))
  signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
  def state(name,**more):
   metadata.update(status=name,time=time.time(),**more);atomic(ROOT/'current.json',metadata)
  def operation(action,release=None):
   intent=ROOT/(action+'-intent.json');assert not intent.exists(),'Refusing repeated '+action
   atomic(intent,{'action':action,'time':time.time(),'owner':me})
   command=[RLT_PY,'-u','-B',str(SOURCE/'tools/expo_formal_resources.py'),action]
   if release:command+=['--release',str(release)]
   critical[0]=True
   try:
    with (ROOT/(action+'.log')).open('xb') as log:
     subprocess.run(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
   finally:critical[0]=False
  roster=None;child=None;error=None;stopped=False;released=False;passed=False;return_status=None
  try:
   mapping=gpu_map();assert [mapping.get(i) for i in (4,5,6,7)]==UUIDS
   inputs=json.loads((ROOT/'inputs.json').read_text())
   assert inputs['formal']['max_physical_actions']==20000
   for name,expected in inputs['port_source_manifest'].items():
    assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==expected,name
   operation('prepare')
   if requested[0]:raise Terminated('Requested during preparation')
   state('STOPPING_RLT');operation('stop')
   proof=json.loads((CYCLE/'rlt-stopped.json').read_text())
   assert proof['all_original_drivers_stopped'] and proof['all_original_namespaces_empty']
   assert set(proof['gpus_released'])=={4,5,6,7};stopped=True
   if requested[0]:raise Terminated('Requested during RLT stop')
   assert not any(row['gpu'] in UUIDS for row in gpu_processes())
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(UUIDS),
    PYTHONPATH=str(SOURCE)+':/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support',
    LD_LIBRARY_PATH='/home/chenyiteng/tools/cuda-12.9/lib64',
    VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
    PYTHONUNBUFFERED='1',MPLBACKEND='Agg',
    EXPO_SMOKE_OWNER_SCOPE=scope,EXPO_SMOKE_OWNER_PHASE='formal',PYTHONDONTWRITEBYTECODE='1')
   (ROOT/'driver-heartbeat').touch()
   command=[EXPO_PY,'-u','-B',str(SOURCE/'examples/embodiment/train_expo_formal.py'),
    '--inputs',str(ROOT/'inputs.json'),'--run',str(ROOT/'run'),'--max-physical-actions','20000','--enable-evaluation']
   critical[0]=True
   try:
    with (ROOT/'driver.log').open('xb') as log:
     child=subprocess.Popen(command,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     anchor=identity(child.pid);roster=Roster(anchor,scope,'formal',ROOT/'process-roster.json');roster.write()
   finally:critical[0]=False
   state('EXPO_RUNNING',child=anchor,command=command)
   if requested[0]:raise Terminated('Requested during child registration')
   last_sample=0
   while child.poll() is None:
    roster.scan()
    if time.time()-(ROOT/'driver-heartbeat').stat().st_mtime>900:raise TimeoutError('EXPO driver heartbeat stale >15min')
    if time.monotonic()-last_sample>30:
     sample=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True,timeout=20)
     with (ROOT/'gpu-samples.jsonl').open('a') as stream:stream.write(json.dumps({'time':time.time(),'gpu':sample})+'\n')
     last_sample=time.monotonic()
    time.sleep(2)
   if child.returncode:raise RuntimeError('EXPO formal exited '+str(child.returncode)+'; inspect driver.log')
   complete=json.loads((ROOT/'run/complete.json').read_text())
   assert complete['ok'] and complete['budget_completed']
   assert complete['cadence']['counters']['physical_actions']==20000
   assert complete['cadence']['counters']['pending_calls']==0;passed=True
  except BaseException as failure:error={'type':type(failure).__name__,'message':str(failure),'traceback':traceback.format_exc()}
  finally:
   closing[0]=True
   try:
    if roster:
     state('CLEANING_EXPO')
     # Let the driver finish its action/checkpoint and close its vector first.
     # Killing simulator children simultaneously would interrupt that commit.
     if child and child.poll() is None and owned(roster.root):
      roster.signal(roster.root,signal.SIGTERM)
      graceful_deadline=time.monotonic()+180
      while child.poll() is None and time.monotonic()<graceful_deadline:
       roster.scan();time.sleep(1)
     cleanup=roster.cleanup();atomic(ROOT/'cleanup.json',cleanup)
     if child:child.wait(timeout=10)
    if not stopped and (CYCLE/'rlt-stopped.json').exists():
     proof=json.loads((CYCLE/'rlt-stopped.json').read_text())
     stopped=bool(proof.get('all_original_drivers_stopped') and proof.get('all_original_namespaces_empty') and set(proof.get('gpus_released',[]))=={4,5,6,7})
    if stopped:
     released=not any(row['gpu'] in UUIDS for row in gpu_processes());assert released,'EXPO GPU remains; refuse RLT resume'
     managed=list(roster.rows.values()) if roster else []
     release={'cycle_id':CYCLE.name,'terminal_status':'completed' if passed else ('failed' if managed else 'not_started'),
      'all_workers_stopped':True,'managed_processes':managed,'physical_gpus':[4,5,6,7]}
     atomic(ROOT/'release.json',release);state('RESTORING_RLT');operation('resume',ROOT/'release.json')
     deadline=time.monotonic()+1800
     while time.monotonic()<deadline:
      with (ROOT/'rlt-status.log').open('w') as log:
       subprocess.run([RLT_PY,'-u','-B',str(SOURCE/'tools/expo_formal_resources.py'),'status'],cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
      status=json.loads((ROOT/'rlt-status.log').read_text().splitlines()[-1]);atomic(ROOT/'rlt-status.json',status)
      if status.get('all_first_rounds_verified'):return_status=status;break
      time.sleep(20)
     assert return_status,'RLT return dispatched; first rounds pending action'
     state('RLT_RESTORED',formal_passed=passed)
    elif (ROOT/'stop-intent.json').exists():
     raise RuntimeError('Partial RLT stop without full receipt; inspect stop.log before recovery')
   except BaseException as failure:error={'type':'CLEANUP_OR_RLT_RETURN_NEEDS_ACTION','message':str(failure),'previous':error,'traceback':traceback.format_exc()};state('ACTION_REQUIRED')
   atomic(ROOT/'final.json',{'time':time.time(),'formal_passed':passed,'error':error,'gpu_released':released,
     'rlt_first_rounds_verified':bool(return_status),'owner':me,'scope':scope})
  return 0 if passed and return_status and error is None else 1

if __name__=='__main__':raise SystemExit(main())
