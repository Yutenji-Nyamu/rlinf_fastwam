import json,os,signal,subprocess,sys,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'))
from common import read,save,same,exact_signal,proc
T=Path('/home/chenyiteng/deployment-20261030/queue-storage-sz1')
plan=read(S/'plan.json');status=read(S/'status.json');old=read(S/'owner-identity.json')
assert same(old) and all(status['slots'][g]['phase']=='WAIT_EXTERNAL' for g in ['4','5','6','7'])
prior=read(Path('/home/chenyiteng/deployment-20261028/physical-v2-formal-sz1-g67/identity.json'))
assert same(prior)
for g in ['6','7']:plan['slots'][g]['external_blockers']=[prior]
save(S/'plan.json',plan)
p=S/'ops/owner.py';s=p.read_text()
before="not same(slot['external_identity']) and not card['processes']"
after="not same(slot['external_identity']) and not any(same(x) for x in slot.get('external_blockers', [])) and not card['processes']"
assert s.count(before)==1
s=s.replace(before,after);compile(s,str(p),'exec');p.write_text(s)
exact_signal(old,signal.SIGTERM)
deadline=time.monotonic()+20
while same(old) and time.monotonic()<deadline:time.sleep(.1)
assert not same(old)
with (S/'owner.log').open('ab') as f:
 child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(p),'--plan',str(S/'plan.json')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=p.parent)
deadline=time.monotonic()+5
while time.monotonic()<deadline:
 me=read(S/'owner-identity.json')
 if me['pid']==child.pid and same(me):break
 time.sleep(.1)
assert me['pid']==child.pid and same(me) and same(prior)
r=dict(time=time.time(),new_owner=me,blocking_current_v2_owner=prior,guards=['queued owner exited','current v2 owner exited','physical GPU clear','root filesystem permit exists'],healthy_training_unchanged=True)
save(T/'rlt-chain-guard.json',r);(T/'rlt-owner-after.py').write_text(s);print(json.dumps(r))
