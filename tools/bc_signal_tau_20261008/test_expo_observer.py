import os,sys,json,types,tempfile,signal,subprocess
from pathlib import Path
from unittest.mock import patch
src=Path(__file__).with_name('expo_owner.py').read_text()
m=types.ModuleType('candidate');exec(src,m.__dict__)
with tempfile.TemporaryDirectory(prefix='expo-observer-check-') as tmp:
 for k in ['ROOT','PREVIOUS','OLD','TRAIN','CURRENT','HERE']:
  p=Path(tmp)/k;p.mkdir();setattr(m,k,p)
 (m.ROOT/'eval10-continuation-20261003').mkdir()
 driver=dict(pid=-2,start_ticks=22);previous=dict(pid=-1,start_ticks=11);states=[];calls=[0];sleeps=[0]
 (m.HERE/'current.json').write_text(json.dumps(dict(child=driver,owner=previous,scope='fake')))
 (m.HERE/'expo-roster.json').write_text(json.dumps(dict(root=driver,registered=[driver])))
 def save(p,v):
  if Path(p)==m.HERE/'current.json':states.append(dict(v))
  Path(p).write_text(json.dumps(v))
 def gpu():
  calls[0]+=1
  if calls[0]==1:raise subprocess.TimeoutExpired('nvidia-smi',25)
  return []
 class Roster:
  def __init__(self,*a):self.rows={};self.root=driver
  def scan(self):pass
  def write(self):pass
  def cleanup(self):raise AssertionError('Live training cleanup forbidden')
  def signal(self,*a):raise AssertionError('Live training signal forbidden')
 fake=types.SimpleNamespace(QUEUES=[],read=lambda p:json.loads(Path(p).read_text()),atomic=save,identity=lambda pid:dict(pid=pid,start_ticks=1),owned=lambda q:q['pid']==-2,same=lambda a,b:a==b,Roster=Roster,gpu_rows=gpu)
 def fill(mod):mod.__dict__.update(vars(fake))
 def sleep(t):
  sleeps[0]+=1
  if sleeps[0]==2:signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
 m.time=types.SimpleNamespace(time=__import__('time').time,sleep=sleep)
 spec=types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=fill))
 with patch.object(m.importlib.util,'spec_from_file_location',return_value=spec),patch.object(m.importlib.util,'module_from_spec',return_value=types.ModuleType('old')),patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''):
  assert m.main()==0
 assert calls[0]==2 and states[-1]['status']=='MONITOR_STOPPED_CHILD_RETAINED'
 assert any(x.get('monitoring_error') for x in states) and states[-2]['monitoring_error'] is None
 assert not (m.HERE/'expo-released.json').exists()
 print(json.dumps(dict(passed=['query timeout does not stop driver','retry clears transient error','monitor SIGTERM retains driver','no false release'],sha=__import__('hashlib').sha256(src.encode()).hexdigest())))

 fake.owned=lambda q:False
 fake.gpu_rows=lambda:[dict(index=4,pid=-9)]
 Roster.cleanup=lambda self:None
 with patch.object(m.importlib.util,'spec_from_file_location',return_value=spec),patch.object(m.importlib.util,'module_from_spec',return_value=types.ModuleType('old')),patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''):
  assert m.main()==0
 release=json.loads((m.HERE/'expo-released.json').read_text())
 assert release['owned_contexts_clear'] and states[-1]['status']=='EXPO_RELEASED'
 print(json.dumps(dict(passed=['foreign occupancy of card 4 does not delay own release; per-card queue still checks local card'],sha=__import__('hashlib').sha256(src.encode()).hexdigest())))
