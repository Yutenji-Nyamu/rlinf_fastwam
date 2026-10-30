"""Replace only idle CPU owner; all priority jobs and shared Ray remain untouched."""
from pathlib import Path
import json,os,signal,subprocess,sys,time,shutil
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
D=Path('/home/chenyiteng/deployment-20261030/rlt-nosave-1030g')
sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,exact_signal
p=read(S/'plan.json');s=read(S/'status.json');old=read(S/'owner-identity.json')
assert os.getuid()==1003 and same(old)
assert all(x['phase']=='WAIT_EXTERNAL' for x in s['slots'].values()),s
cmd=Path(f"/proc/{old['pid']}/cmdline").read_bytes().split(b'\0')
owner_path=next(Path(x.decode()) for x in cmd if x.endswith((b'/owner.py',b'/owner-live-external.py')))
assert owner_path.parent==S/'ops'
external={g:slot['external_identity'] for g,slot in p['slots'].items()}
assert all(same(i) for i in external.values())
root=Path('/tmp/chenyiteng-rlt-nosave-20261030');root.mkdir(mode=0o700,exist_ok=True)
assert root.stat().st_uid==1003 and not root.is_symlink()
backup=root/('before-'+str(time.time_ns()));backup.mkdir(exist_ok=False)
for f in ['plan.json','status.json','owner-identity.json']:
 shutil.copy2(S/f,backup/f)
for f in ['common.py','driver.py',owner_path.name]:shutil.copy2(S/'ops'/f,backup/f)
print(json.dumps({'old_owner':old,'owner_path':str(owner_path),'external_alive':True}),flush=True)
exact_signal(old,signal.SIGTERM)
deadline=time.monotonic()+25
while same(old) and time.monotonic()<deadline:time.sleep(.2)
assert not same(old),'Idle CPU owner did not exit; no live edits applied'
for name in ['common.py','driver.py','nosave.py','priority_pair_handoff.py']:
 target=S/'ops'/name;tmp=target.with_suffix('.py.new1030g');tmp.write_bytes((D/name).read_bytes());os.replace(tmp,target)
tmp=owner_path.with_suffix('.py.new1030g');tmp.write_bytes((D/'owner.py').read_bytes());os.replace(tmp,owner_path)
p['no_checkpoint_root']=str(root/'runs');p['poll_seconds']=2
for slot in p['slots'].values():slot['no_checkpoint_fallback']=True
save(S/'plan.json',p)
for deployment in ['/home/chenyiteng/deployment-20261030/v2-fqlfix400-sz1-g45-recover1','/home/chenyiteng/deployment-20261029/single-extend400-fqlfix-sz1-g67']:
 f=Path(deployment)/'priority_pair_handoff.py'
 if f.exists():
  text=f.read_text()
  original="st=state['slots'][g];old=read(st.get('request',p['slots'][g]['fallback']));rt=Path(old['runtime']);"
  revised="st=state['slots'][g];old=read(st.get('request',p['slots'][g]['fallback']));old=read(old['normal_request']) if old.get('normal_request') else old;rt=Path(old['runtime']);"
  if original in text:
   tmp=f.with_suffix('.py.new1030g');tmp.write_text(text.replace(original,revised));os.replace(tmp,f)
with (root/'owner.log').open('ab') as log:
 child=subprocess.Popen([p['python'],'-u','-B',str(owner_path),'--plan',str(S/'plan.json')],cwd=S,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(3)
assert child.poll() is None,(root/'owner.log').read_text()[-3000:]
new=read(S/'owner-identity.json');now=read(S/'status.json')
assert new['pid']==child.pid and same(new)
assert all(same(i) for i in external.values())
assert all(x['phase']=='WAIT_EXTERNAL' for x in now['slots'].values())
receipt={'time':time.time(),'old_owner':old,'new_owner':new,'phase':{g:r['phase'] for g,r in now['slots'].items()},'no_checkpoint_tail':[4,5,6,7],'external_priority_unchanged':True,'gpu_smoke':False,'reason':'Healthy priority jobs continue; CPU tests validate queue and native save gate','root':str(root),'fallback_steps':{g:read(slot['fallback'])['step'] for g,slot in p['slots'].items()}}
save(D/'deployed.json',receipt);print(json.dumps(receipt))
