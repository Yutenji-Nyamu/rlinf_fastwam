"""CPU-only EXPO observer: query errors never stop training; release cards 4/5."""
import fcntl,importlib.util,os,signal,sys,time,traceback
from pathlib import Path
ROOT=Path('/data/chenyiteng/projects/expo-ft-sz2-20261001')
PREVIOUS=ROOT/'parallel-trial-20261006';OLD=ROOT/'continue-60k-20261005'
TRAIN=ROOT/'formal-turn-switch-repair-20261002'
CURRENT=Path('/data/chenyiteng/deployment-20261006/rlt-q-signals-g67-v1/coexist')
HERE=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/expo-owner')

def main():
 assert os.getuid()==20001 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 sys.path.insert(0,str(PREVIOUS/'source/tools/expo_parallel_20261006'))
 spec=importlib.util.spec_from_file_location('previous_expo_owner',PREVIOUS/'source/tools/expo_parallel_20261006/owner.py')
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 read,save,identity,owned=old.read,old.atomic,old.identity,old.owned
 locks=[]
 for path in [HERE/'owner.lock',CURRENT/'owner.lock',PREVIOUS/'owner.lock',OLD/'owner.lock',TRAIN/'resource-owner.lock',ROOT/'eval10-continuation-20261003/resource-owner.lock',*[q/'owner.lock' for q in old.QUEUES]]:
  f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 meta=read(HERE/'current.json');driver=meta['child'];previous=meta['owner'];me=identity(os.getpid())
 assert not owned(previous), 'Previous observer remains alive'
 original=read(HERE/'expo-roster.json');assert old.same(original['root'],driver)
 active=old.Roster(driver,meta['scope'],'formal',HERE/'expo-roster.json')
 for row in original['registered']:active.rows[(row['pid'],row['start_ticks'])]=row
 active.scan();active.write();requested=[False]
 signal.signal(signal.SIGTERM,lambda *_:requested.__setitem__(0,True))
 signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
 def state(status,**more):
  meta.update(status=status,time=time.time(),owner=me,control=str(HERE),**more)
  for p in [HERE/'current.json',PREVIOUS/'current.json',OLD/'current.json']:save(p,meta)
 save(HERE/'owner.json',dict(owner=me,previous=previous,driver=driver,time=time.time()))
 state('EXPO_RUNNING',observer_revision='20261008-query-isolation-per-card',monitoring_error=None)
 save(HERE/'observer-adopted-20261008.json',dict(time=time.time(),owner=me,driver=driver,driver_unchanged=owned(driver)))
 while True:
  if requested[0]:
   state('MONITOR_STOPPED_CHILD_RETAINED');return 0
  try:
   active.scan();pids={r['pid'] for r in active.rows.values() if owned(r)};gpu=old.gpu_rows()
  except Exception:
   state('EXPO_RUNNING' if owned(driver) else 'WAIT_RELEASE_PROOF',monitoring_error=traceback.format_exc());time.sleep(15);continue
  contexts=[g for g in gpu if g['pid'] in pids]
  if any(g['index'] not in (4,5) for g in contexts):
   state('ACTION_REQUIRED',monitoring_error='Observed owned context outside cards 4/5')
   if owned(driver):active.signal(driver,signal.SIGTERM)
   time.sleep(15);continue
  if owned(driver):
   hb=TRAIN/'driver-heartbeat';age=time.time()-hb.stat().st_mtime if hb.exists() else None
   state('EXPO_RUNNING',gpu_processes=contexts,monitoring_error=None,driver_heartbeat_age=age)
  else:
   try:
    active.cleanup()
    assert not any(owned(r) for r in active.rows.values())
    assert not any(g['pid'] in pids for g in old.gpu_rows())
   except Exception:
    state('WAIT_RELEASE_PROOF',monitoring_error=traceback.format_exc());time.sleep(15);continue
   done=read(TRAIN/'run/complete.json') if (TRAIN/'run/complete.json').exists() else {}
   passed=bool(done.get('ok') and done.get('budget_completed') and done.get('cadence',{}).get('counters',{}).get('physical_actions')==60000)
   save(HERE/'expo-released.json',dict(time=time.time(),driver=driver,expo_completed=passed,owned_contexts_clear=True))
   state('EXPO_RELEASED',expo_completed=passed,expo_released=True);save(HERE/'final.json',meta);return 0
  time.sleep(15)
if __name__=='__main__':raise SystemExit(main())
