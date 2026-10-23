"""Wait for shared accepted data; keep RLT running until this pair can train."""
import fcntl,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
D=Path(sys.argv[1]);S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal
sys.path.insert(0,str(D.parent));from priority_pair_handoff import handoff
me=proc(os.getpid());save(D/'prepare-owner-identity.json',me)
prepared=read(D/'prepared.json');R=Path(prepared['source']);T=Path(read(D/'schedule.json')['data_source']);cards=prepared['physical_gpus'];method=prepared['method'];py='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
def event(phase,**fields):save(D/'prepare-owner-status.json',dict(time=time.time(),phase=phase,**fields));print(phase,flush=True)
try:
 event('WAIT_DATA',cards=cards,data_source=str(T),rlt_continues=True)
 deadline=time.time()+12*3600
 while not(T/'data-ready.json').is_file():
  assert time.time()<deadline,'Shared data preparation exceeded12h; RLT remains running'
  if(T/'data-finish-error.json').is_file():raise RuntimeError('Shared data failed; inspect '+str(T/'data-finish-error.json'))
  time.sleep(30)
 shape=read(T/'data-ready.json');assert shape['tasks']==50 and shape['episodes']==2500
 event('DATA_READY',windows=shape['windows'])
 if method=='onedp':
  import shutil
  smoke=D.parent/'onedp-capacity-sz1-g67'
  while not(smoke/'released.json').exists():
   assert time.time()<deadline,'OneDP smoke exceeded data preparation deadline'
   time.sleep(15)
  assert read(smoke/'released.json')['passed'],'OneDP capacity/resume smoke not accepted'
  assert read(D/'smoke-accepted.json')['passed']
  schedule=read(D/'schedule.json');schedule['run_smoke_before_formal']=False;schedule['smoke_source']=str(smoke);save(D/'schedule.json',schedule)
 # Binding was verified on this host during staging; this transfer is serialized
 # across the two independent pairs so each sees the latest RLT owner and plan.
 assert read(D/'bindings-accepted.json')['passed']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(R))
 with(D/'queue.log').open('x')as log:child=subprocess.Popen([py,'-u','-B',str(R/'tools/pi05_distillation/endpoint_queue.py'),str(D)],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  if(D/'identity.json').exists():break
  assert child.poll()is None;time.sleep(.1)
 identity=read(D/'identity.json');assert same(identity)
 receipt=handoff(D,cards,identity)
 event('FORMAL_QUEUE_OWNS_PAIR',identity=identity,registration=receipt)
except BaseException:
 event('FAILED',error=traceback.format_exc());raise
