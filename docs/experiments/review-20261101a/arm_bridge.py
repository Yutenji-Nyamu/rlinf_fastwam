import fcntl,os,subprocess,time
from pathlib import Path
import sys
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
D=Path('/home/chenyiteng/deployment-20261101/gad-k4-v1');B=D/'g7-collection-wait-rlt'
sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,gpus
assert os.getuid()==1003
lane7=read(D/'signal-recovery/lane-g7-identity.json')
assert not same(lane7) and read(D/'signal-recovery/lane-g7-released.json')['phase']=='complete'
assert same(read(D/'signal-recovery/lane-g6-identity.json'))
assert not gpus()[7]['processes']
with (S/'priority-handoff.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 plan=read(S/'plan.json');state=read(S/'status.json')
 assert state['slots']['7']['phase']=='WAIT_EXTERNAL'
 assert plan['slots']['7']['external_identity']['pid']==read(D/'pair67/identity.json')['pid']
 req=read(plan['slots']['7']['fallback'])
 assert req['gpu']==7 and req['kind']=='rlt' and not (Path(req['runtime'])/'started.json').exists()
 assert not (B/'armed.json').exists()
 save(B/'previous-slot.json',plan['slots']['7'])
 with (B/'bridge.log').open('x') as log:
  child=subprocess.Popen(['/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python','-u','-B',str(B/'bridge.py')],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 plan['slots']['7'].update(external_identity=lane7,external_release=str(D/'signal-recovery/lane-g7-released.json'))
 plan['slots']['7'].pop('external_blockers',None)
 save(S/'plan.json',plan)
 save(B/'armed.json',dict(time=time.time(),identity=proc(child.pid),initial_checkpoint=req['checkpoint'],note='RLT g7 until lane6 exits, then existing exact handoff back to pair67'))
 print(read(B/'armed.json'))
