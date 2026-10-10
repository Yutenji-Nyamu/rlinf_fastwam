"""Observe one EXPO pair; release its two independent RLT slots after exact cleanup."""
import fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
q=json.loads(Path(sys.argv[1]).read_text());rt=Path(q['runtime']);source=Path(q['source'])
sys.path.insert(0,str(source/'tools'));import expo_smoke_owner as lifecycle
lifecycle.UID=os.getuid()
sys.path.insert(0,'/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops')
from common import read,save,proc,gpus
lock=(rt/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert os.getuid()==q['uid'] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not(rt/'driver.json').exists()
stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
snapshot=gpus();assert all(not snapshot[g]['processes'] and snapshot[g]['uuid']==u for g,u in zip(q['gpus'],q['uuids']))
env=dict(os.environ,**q['environment'],EXPO_SMOKE_OWNER_SCOPE=q['scope'],EXPO_SMOKE_OWNER_PHASE='formal')
env.pop('DISPLAY',None)
with(rt/'driver.log').open('x')as log:
 child=subprocess.Popen(q['command'],cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
root=lifecycle.identity(child.pid);save(rt/'driver.json',proc(child.pid));save(rt/'owner.json',proc(os.getpid()))
roster=lifecycle.Roster(root,q['scope'],'formal',rt/'roster.json');roster.write()
while True:
 if stop[0]:save(rt/'owner-stopped.json',dict(time=time.time(),child_retained=True));break
 try:
  roster.scan();rc=child.poll()
  if rc is None:
   snap=gpus();ids={r['pid']for r in roster.alive()}
   contexts=[dict(gpu=g,**p)for g,c in snap.items()for p in c['processes']if p['pid']in ids]
   if any(p['gpu']not in q['gpus']for p in contexts):
    roster.signal(root,signal.SIGTERM);raise RuntimeError('Own GPU context outside the assigned pair')
   save(rt/'status.json',dict(time=time.time(),phase='EXPO_RUNNING',driver=proc(child.pid),contexts=contexts))
  else:
   cleanup=roster.cleanup();snap=gpus();ids={r['pid']for r in roster.rows.values()}
   assert not roster.alive() and not any(p['pid']in ids for c in snap.values()for p in c['processes'])
   save(rt/'released.json',dict(time=time.time(),exit_code=rc,driver=root,cleanup=cleanup,owned_contexts_clear=True));break
 except Exception:
  save(rt/'status.json',dict(time=time.time(),phase='WAIT_QUERY_OR_RELEASE',error=traceback.format_exc()))
 time.sleep(15)
