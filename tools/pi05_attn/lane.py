"""One formal queue per card; validate once, then return to existing RLT."""
import os,sys,json,time,subprocess,signal,traceback
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,gpus,exact_signal
from index import update as update_index
R=Path(__file__).resolve().parents[2];O=Path(os.environ['ATTN_RUN_ROOT']);D=Path(os.environ['ATTN_TRANSACTION_ROOT'])
g=sys.argv[1];assert g in ['6','7'];me=proc(os.getpid());env=dict(os.environ,**read(O/f'environment-g{g}.json'),ATTN_LANE_TOKEN=f'{D.name}-g{g}')
python='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
save(D/f'lane-g{g}-identity.json',me)
child=None

def progress_stalled(out):
 try:return time.time()-(out/'progress.json').stat().st_mtime>600
 except OSError:return False  # Missing or transiently unreadable means unknown.
def native_task_attribute_error(c,out):
 p=out/'error.json'
 return p.exists() and f"AttributeError: '{c['task']}' object has no attribute" in read(p).get('error','')

def skippable_failure(c,out,rc):
 if rc==124 or (rc==1 and native_task_attribute_error(c,out)):return True
 log=D/(out.name+f'-g{g}.log')
 return rc==-6 and log.exists() and 'vk::DeviceLostError' in log.read_text(errors='replace')[-4000:]
def stop(*_):
 if child and child.poll() is None:exact_signal(proc(child.pid),signal.SIGTERM)
 raise SystemExit(143)
signal.signal(signal.SIGTERM,stop)
def cleanup():
 snapshot=gpus();owned=[]
 for gi,card in snapshot.items():
  for pr in card['processes']:
   try:e=dict(x.split('=',1) for x in (Path('/proc')/str(pr['pid'])/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
   except OSError:continue
   if e.get('ATTN_LANE_TOKEN')==env['ATTN_LANE_TOKEN']:
    ident=proc(pr['pid']);assert ident and ident['uid']==os.getuid();owned.append(ident)
 for ident in owned:exact_signal(ident,signal.SIGTERM)
 if owned:time.sleep(2)
 after=gpus();assert not after[int(g)]['processes'],'Assigned card remains occupied after batch exit'
 return dict(time=time.time(),owned_signalled=owned,card_clear=True)
try:
 plan=read(D/f'lane-g{g}-plan.json');failed_jobs=[]
 for row in plan['jobs']:
  path=Path(row['config']);c=read(path);out=Path(c['output'])
  if (out/'validation.json').exists() and read(out/'validation.json')['passed']:continue
  previous=D/(out.name+'-result.json')
  if previous.exists() and skippable_failure(c,out,read(previous).get('returncode')):
   failed_jobs.append(str(path));continue  # Original formal queue continued after native task errors, without replay.
  assert not out.exists(),'An attempted batch cannot be silently replayed'
  assert c['gpu']==int(g) and not c.get('parity',False),'Use the prepared formal config'
  assert not gpus()[int(g)]['processes']
  record=dict(time=time.time(),phase='RUNNING',config=str(path),gpu=g)
  logfile=D/(out.name+f'-g{g}.log')
  with logfile.open('x') as f:
   child=subprocess.Popen([python,'-u','-B',str(R/'tools/pi05_attn/run_batch.py'),str(path)],env=env,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   record['child']=proc(child.pid);save(D/f'lane-g{g}-status.json',record)
   deadline=time.monotonic()+3600
   try:
    while True:
     try:rc=child.wait(timeout=15);break
     except subprocess.TimeoutExpired:
      # Accepted complete batches needed at most 72s per query. A native
      # renderer wait may hold the GIL, so this existing parent checks progress.
      stalled=progress_stalled(out)
      if stalled or time.monotonic()>deadline:raise
   except subprocess.TimeoutExpired:
    exact_signal(record['child'],signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:exact_signal(record['child'],signal.SIGKILL);child.wait(timeout=20)
    save(out/'error.json',dict(time=time.time(),error='No query progress for 600s' if stalled else 'Batch exceeded the accepted 3600-second deadline'))
    rc=124
  clear=cleanup()
  if rc==0:
   with (D/(out.name+f'-g{g}-analysis.log')).open('x') as f:
    child=subprocess.Popen([python,'-u','-B',str(R/'tools/pi05_attn/analyze_batch.py'),str(path)],env=env,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(D/f'lane-g{g}-status.json',dict(record,phase='OFFLINE',child=proc(child.pid)));rc=child.wait()
   clear=cleanup()
  update_index()
  state='VALIDATED' if rc==0 and (out/'validation.json').exists() and read(out/'validation.json')['passed'] else 'FAILED'
  save(D/(out.name+'-result.json'),dict(time=time.time(),state=state,returncode=rc,config=str(path),clearance=clear))
  if state!='VALIDATED':
   if skippable_failure(c,out,rc):failed_jobs.append(str(path));continue
   raise RuntimeError(f'Batch did not validate (exit {rc}); release to the original RLT fallback')
 save(D/f'lane-g{g}-released.json',dict(time=time.time(),identity=me,completed=True,dataset_complete=not plan.get('deferred_jobs') and not failed_jobs,deferred_jobs=plan.get('deferred_jobs',[]),failed_jobs=failed_jobs,clearance=cleanup()))
except BaseException:
 save(D/f'lane-g{g}-error.json',dict(time=time.time(),error=traceback.format_exc()))
 clear=None
 try:clear=cleanup()
 finally:save(D/f'lane-g{g}-released.json',dict(time=time.time(),identity=me,completed=False,dataset_complete=False,reason='Lane stopped; owner must verify exit and GPU clearance before RLT',clearance=clear))
 raise
