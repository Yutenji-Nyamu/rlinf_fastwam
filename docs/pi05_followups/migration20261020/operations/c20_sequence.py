"""Accepted C20 rank/calibration/sweep on SZ2, with original RLT following."""
import json,os,sys,subprocess,time,signal,traceback,importlib.util,shutil
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,gpus,exact_signal
D=Path(sys.argv[1]);p=read(D/'prepared.json');R=Path(p['source'])
PY='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
roots={k:Path(v)for k,v in p['roots'].items()};children=[];me=proc(os.getpid())
save(D/'sequence-identity.json',me)
def stop(*_):
 for c in children:
  if c.poll()is None:exact_signal(proc(c.pid),signal.SIGTERM)
 raise SystemExit(143)
signal.signal(signal.SIGTERM,stop)
def phase(name,key):
 global children
 T=D/name;T.mkdir(exist_ok=True)
 save(D/'status.json',dict(time=time.time(),phase=name,root=str(roots[key])))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',POLICY_EVAL_ROOT=str(roots[key]),POLICY_EVAL_TRANSACTION=str(T),POLICY_EVAL_UID='20001',POLICY_EVAL_GPUS='4,5')
 children=[]
 for g in (4,5):
  with(T/f'lane-g{g}.log').open('x')as log:
   c=subprocess.Popen(['/usr/bin/python3','-u','-B',str(R/'tools/pi05_policy_eval/lane.py'),str(g)],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  children.append(c)
 while any(c.poll()is None for c in children):
  if any(c.poll()not in(None,0)for c in children):raise RuntimeError(name+' lane failed')
  time.sleep(10)
 assert all(c.returncode==0 for c in children)
 assert all(read(T/f'lane-g{g}-released.json')['completed']for g in(4,5))
 assert all(not gpus()[g]['processes']for g in(4,5))
 save(D/f'{name}-complete.json',dict(time=time.time(),completed=True))
try:
 phase('c20-rank','rank')
 with(D/'c20-prepare.log').open('x')as log:
  c=subprocess.Popen([PY,'-u','-B',str(D/'c20_calibrate.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
  children=[c];assert c.wait(timeout=300)==0
 if not(D/'c20-not-stronger.json').exists():phase('c20-sweep','c20')
 save(D/'released.json',dict(time=time.time(),identity=me,completed=True,cards_clear=True))
 save(D/'status.json',dict(time=time.time(),phase='COMPLETE_RLT_HANDOFF'))
except BaseException:
 save(D/'error.json',dict(time=time.time(),error=traceback.format_exc()))
 for c in children:
  if c.poll()is None:exact_signal(proc(c.pid),signal.SIGTERM)
 raise
