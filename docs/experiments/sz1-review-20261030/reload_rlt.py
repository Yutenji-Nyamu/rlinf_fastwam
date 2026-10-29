import json,os,signal,subprocess,sys,time
from pathlib import Path
assert os.getuid()==1003
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');T=Path('/home/chenyiteng/deployment-20261030/queue-storage-sz1')
sys.path.insert(0,str(S/'ops'));from common import read,save,same,exact_signal,proc
assert not (T/'rlt-reload.json').exists()
identity=read(S/'owner-identity.json');state=read(S/'status.json');plan=read(S/'plan.json')
assert same(identity) and set(state['slots'])=={'4','5','6','7'}
assert all(x['phase']=='WAIT_EXTERNAL' for x in state['slots'].values())
assert all(same(s['external_identity']) for s in plan['slots'].values())
path=S/'ops/owner.py';code=path.read_text();needle=' while not stop[0]:\n'
assert code.count(needle)==1
(T/'rlt-owner-before.py').write_text(code)
code=code.replace(needle,needle+'  fresh=read(a.plan)\n  assert fresh["uid"]==p["uid"] and fresh["boot_id"]==p["boot_id"] and set(fresh["slots"])==set(p["slots"])\n  p=fresh\n',1)
compile(code,str(path),'exec');path.write_text(code);(T/'rlt-owner-after.py').write_text(code)
exact_signal(identity,signal.SIGTERM);deadline=time.monotonic()+20
while same(identity) and time.monotonic()<deadline:time.sleep(.2)
assert not same(identity)
with (S/'owner.log').open('ab') as log:
 child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(path),'--plan',str(S/'plan.json')],cwd=S,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
deadline=time.monotonic()+8
while time.monotonic()<deadline:
 me=read(S/'owner-identity.json')
 if me['pid']==child.pid and same(me):break
 assert child.poll() is None;time.sleep(.1)
assert me['pid']==child.pid and same(me) and all(same(s['external_identity']) for s in plan['slots'].values())
result=dict(time=time.time(),old_owner=identity,new_owner=me,slots=['4','5','6','7'],phases=read(S/'status.json')['slots'],plan_reload_each_poll=True,training_owners_unchanged=True)
save(T/'rlt-reload.json',result);print(json.dumps(result))
