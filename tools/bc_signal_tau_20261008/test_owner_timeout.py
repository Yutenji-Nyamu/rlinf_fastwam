import sys,tempfile,types,subprocess,json,os,copy
from pathlib import Path
C=Path(__file__).parent
sys.path.insert(0,str(C))
src=(C/'owner.py').read_text();mod=types.ModuleType('candidate_owner');mod.__file__=str(C/'owner.py');exec(src,mod.__dict__)
with tempfile.TemporaryDirectory(prefix='owner-timeout-test-') as tmp:
 plan=Path(tmp)/'plan.json';plan.write_text(json.dumps(dict(uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),slots={'4':{'gpu_uuid':'test4'},'5':{'gpu_uuid':'test5'}})))
 state={'slots':{g:{'phase':'PRIORITY_RUNNING','identity':{'pid':-int(g)},'request':g} for g in ['4','5']}}
 (Path(tmp)/'status.json').write_text(json.dumps(state));realread=mod.read;realsave=mod.save;records=[];polls=[0];sleeps=[0];prunes=[]
 def read(p):return {'kind':'bc','namespace':'test'+p} if str(p) in ['4','5'] else realread(p)
 def save(p,v):
  if str(p).endswith('status.json'):records.append(copy.deepcopy(v))
  realsave(p,v)
 def gpus():
  polls[0]+=1
  if polls[0]==1:raise subprocess.TimeoutExpired('nvidia-smi',25)
  return {i:{'uuid':'test'+str(i),'processes':[]} for i in [4,5]}
 def prune(q):
  prunes.append(q['namespace'])
  if q['namespace']=='test4':raise OSError('synthetic retention error')
 class Done(Exception):pass
 def sleep(t):
  sleeps[0]+=1
  if sleeps[0]==3:raise Done()
 def forbidden(*a,**kw):raise AssertionError('unexpected launch or signal')
 mod.read=read;mod.save=save;mod.gpus=gpus;mod.same=lambda _:True;mod.retain_latest=prune;mod.launch=forbidden;mod.exact_signal=forbidden;mod.time=types.SimpleNamespace(time=__import__('time').time,sleep=sleep);sys.argv=['owner','--plan',str(plan)]
 try:mod.main()
 except Done:pass
 assert 'gpu_query_error' in records[0] and all(x['phase']=='PRIORITY_RUNNING' for x in records[0]['slots'].values())
 assert 'gpu_query_error' not in records[-1] and 'retention_error' in records[-1]['slots']['4'] and records[-1]['slots']['5']['retention_checked']
 assert prunes==['test4','test5'] and polls[0]==3
 print(json.dumps(dict(passed=['timeout isolation','retry recovery','per-card retention failure isolation','no duplicate launch','retention throttle'],sha=__import__('hashlib').sha256(src.encode()).hexdigest())))
