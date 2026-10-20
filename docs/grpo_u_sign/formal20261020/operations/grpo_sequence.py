"""Authorized EXPO closeout -> signed GRPO -> unchanged per-card RLT."""
import os,sys,time,subprocess,traceback
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,exact_signal,gpus
import signal
D=Path(sys.argv[1]);p=read(D/'handoff.json');cards=p['gpus'];me=proc(os.getpid())
save(D/'sequence-identity.json',me)
def status(phase,**kw):save(D/'status.json',dict(time=time.time(),phase=phase,**kw))
try:
 while not(D/'enqueued.json').exists():time.sleep(1)
 if not(D/'expo-closeout.json').exists():
  status('STOPPING_EXPO')
  assert same(p['old_driver'])
  exact_signal(p['old_driver'],signal.SIGTERM)
  save(D/'stop-requested.json',dict(time=time.time(),driver=p['old_driver']))
  old=p['old_slot']
  while same(old['external_identity'])or not Path(old['external_release']).exists():time.sleep(5)
  release=read(old['external_release']);release=release.get('expo_release',release)
  assert release['owned_contexts_clear']and release['cleanup']['ok']
  assert all(not gpus()[g]['processes']for g in cards)
  save(D/'old-released.json',release)
  with(D/'closeout.log').open('x')as log:
   result=subprocess.run(['/usr/bin/python3','-B',str(D/'closeout.py'),str(D)],stdout=log,stderr=subprocess.STDOUT,timeout=180)
  assert result.returncode==0,'closeout failed'
 # The model algorithm was already tested in the accepted full-round smoke.
 status('LAUNCHING_SIGNED_GRPO')
 with(D/'launch.log').open('x')as log:
  result=subprocess.run(['/usr/bin/python3','-B',str(D/'source/tools/grpo_un/launch.py'),'--plan',str(D/'plan.json'),'--lane','signed'],cwd=D/'source',stdout=log,stderr=subprocess.STDOUT,timeout=120)
 assert result.returncode==0,'formal launch failed'
 q=read(D/'signed-request.json');rt=Path(q['runtime']);identity=read(rt/'started.json')['identity']
 save(D/'grpo-started.json',dict(time=time.time(),identity=identity,request=q))
 status('GRPO_RUNNING',runtime=str(rt))
 while same(identity):time.sleep(15)
 finished=read(rt/'finished.json')
 assert finished.get('cleanup_error') is None
 assert all(not gpus()[g]['processes']for g in cards)
 save(D/'released.json',dict(time=time.time(),identity=me,finished=finished,cards_clear=True))
 status('COMPLETE_RLT_HANDOFF')
except BaseException:
 status('NEEDS_REPAIR',error=traceback.format_exc())
 raise
