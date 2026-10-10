"""Two independent original-config lanes; gated expansion, then exact RLT return."""
import os,sys,json,time,subprocess,signal,hashlib,traceback
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,gpus,exact_signal
from index import update as update_index
R=Path(__file__).resolve().parents[2];O=Path(os.environ['ATTN_RUN_ROOT']);D=Path(os.environ['ATTN_TRANSACTION_ROOT'])
g=sys.argv[1];assert g in ['6','7'];me=proc(os.getpid());env=dict(os.environ,**read(O/f'environment-g{g}.json'),ATTN_LANE_TOKEN=f'{D.name}-g{g}')
python='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
save(D/f'lane-g{g}-identity.json',me)
child=None
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
 plan=read(D/f'lane-g{g}-plan.json')
 for row in plan['jobs']:
  if row['stage']>read(D/'expansion-stage.json')['stage']:
   while row['stage']>read(D/'expansion-stage.json')['stage']:
    # One lightweight stage barrier. Raw tensors are not reopened for admission.
    other='7' if g=='6' else '6'
    if (D/f'lane-g{other}-error.json').exists():raise RuntimeError('Peer lane stopped; expansion withheld')
    manifest=read(O/'manifest.json')
    candidates=[read(p) for p in manifest['configs'] if read(p)['task'] in manifest['stage12'] and read(p)['batch']==0]
    if any((Path(c['output'])/'error.json').exists() for c in candidates):raise RuntimeError('Stage12 batch failed; expansion withheld')
    if g=='6' and all((Path(c['output'])/'validation.json').exists() and read(Path(c['output'])/'validation.json')['passed'] for c in candidates):
     started=time.perf_counter();costs=[v for c in candidates for v in read(Path(c['output'])/'storage_and_timing.json')]
     rawbytes=sum(v['raw_bytes']+v['obs_bytes'] for v in costs);projected=rawbytes/12*100
     save(D/'expansion-stage.json',dict(time=time.time(),stage=100,reason='All 12 original-config batches have complete validated raw/offline receipts; first two passed native exact parity',validated_batches=12,measured_raw_bytes=rawbytes,mean_projection_bytes=projected,free_bytes=__import__('shutil').disk_usage(O).free,receipt_read_seconds=time.perf_counter()-started))
     break
    save(D/f'lane-g{g}-status.json',dict(time=time.time(),phase='WAITING_STAGE',next=row));time.sleep(15)
  path=Path(row['config']);c=read(path);out=Path(c['output'])
  if (out/'validation.json').exists() and read(out/'validation.json')['passed']:continue
  assert not out.exists(),'An attempted batch cannot be silently replayed'
  assert not gpus()[int(g)]['processes'];c.update(gpu=int(g),parity=False);save(path,c)
  record=dict(time=time.time(),phase='RUNNING',config=str(path),gpu=g,stage=row['stage'])
  logfile=D/(out.name+f'-g{g}.log')
  with logfile.open('x') as f:
   child=subprocess.Popen([python,'-u','-B',str(R/'tools/pi05_attn/run_batch.py'),str(path)],env=env,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   record['child']=proc(child.pid);save(D/f'lane-g{g}-status.json',record);rc=child.wait()
  clear=cleanup()
  if rc==0:
   with (D/(out.name+f'-g{g}-analysis.log')).open('x') as f:
    child=subprocess.Popen([python,'-u','-B',str(R/'tools/pi05_attn/analyze_batch.py'),str(path)],env=env,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(D/f'lane-g{g}-status.json',dict(record,phase='OFFLINE',child=proc(child.pid)));rc=child.wait()
   clear=cleanup()
  update_index()
  state='VALIDATED' if rc==0 and (out/'validation.json').exists() and read(out/'validation.json')['passed'] else 'FAILED'
  save(D/(out.name+'-result.json'),dict(time=time.time(),state=state,returncode=rc,config=str(path),clearance=clear))
  if rc in [-11,-6,99,134,139]:raise RuntimeError(f'Native fatal exit {rc}; no automatic continuation')
 save(D/f'lane-g{g}-released.json',dict(time=time.time(),identity=me,completed=True,clearance=cleanup()))
except BaseException:
 save(D/f'lane-g{g}-error.json',dict(time=time.time(),error=traceback.format_exc()))
 clear=cleanup();save(D/f'lane-g{g}-released.json',dict(time=time.time(),identity=me,completed=False,clearance=clear));raise
