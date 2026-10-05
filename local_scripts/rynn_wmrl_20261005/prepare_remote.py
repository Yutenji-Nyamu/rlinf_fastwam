"""Prepare an isolated checkout, a completed CP70 receipt and CPU sample packet."""
import datetime, hashlib, json, os, re, socket, subprocess, sys, zipfile
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
O=S/'runs/formal-b16-v1'; D=S/'rynn-control-v1'; R=S/'rlinf-rynn-v1'
P=json.loads((O/'owner-plan.json').read_text()); old=Path(P['repo'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p)
    with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
D.mkdir(mode=0o700,exist_ok=True)
(D/'prepared').mkdir(mode=0o700,exist_ok=True)
assert not (D/'prepared/base-prepared.json').exists()
assert subprocess.check_output(['git','-C',str(old),'rev-parse','HEAD'],text=True).strip()=='2151a08ee1bd75df1bef0d8190e594bd5c7f7977'
assert not R.exists()
subprocess.run(['git','-C',str(old),'worktree','add','-b','codex/rynnvalue-wmrl-20261005',str(R),'HEAD'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
copied={}
for name in ['rlinf/envs/__init__.py','rlinf/runners/embodied_runner.py','rlinf/envs/world_model/opendw_adapter.py','rlinf/envs/world_model/opendw_robotwin_env.py']:
    p=R/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((old/name).read_bytes());copied[name]=sha(p)
cp=O/'formal/run/opendw-adjust-bottle-formal-b16-v1/checkpoints/global_step_70'
assert 'Global Step:   70/200' in (O/'formal/driver.log').read_text(errors='replace')
sys.path.insert(0,str(old))
import torch
ranks=[]
for rank in range(2):
    p=cp/'actor/local_shard_checkpoint'/f'checkpoint_rank_{rank}.pt'
    with zipfile.ZipFile(p) as z:assert any(i.filename.endswith('/data.pkl') for i in z.infolist())
    ck=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
    keys=list(ck);print('rank',rank,'keys',keys,flush=True)
    assert any('optim' in k for k in keys) and any('scheduler' in k for k in keys),keys
    st=p.stat();ranks.append({'rank':rank,'complete':True,'keys':keys,'files':[{'path':str(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}]})
    del ck
full=cp/'actor/model_state_dict/full_weights.pt';st=full.stat()
with zipfile.ZipFile(full) as z:assert any(i.filename.endswith('/data.pkl') for i in z.infolist())
receipt={'schema':1,'kind':'opendw-formal-resume-checkpoint','time':datetime.datetime.now().astimezone().isoformat(),
 'source_owner':str(O),'source_owner_plan_sha256':sha(O/'owner-plan.json'),'checkpoint_path':str(cp),'completed_step':70,
 'complete':True,'actor_world_size':2,'runner_state':{'global_step':70,'optimizer_state_saved':True,'scheduler_state_saved':True},
 'ranks':ranks,'full_weights':{'path':str(full),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns},
 'save_method':'normal runner CP70 completed before next sampling; both rank archives opened and optimizer/scheduler keys loaded on CPU',
 'owner_identity':json.loads((O/'owner-identity.json').read_text()),'driver_identity':json.loads((O/'formal/driver-identity.json').read_text())}
save(D/'prepared/resume-receipt.json',receipt)
save(D/'prepared/base-prepared.json',{'time':receipt['time'],'repo':str(R),'base_head':P['repo_head'],'copied_source_sha256':copied,'runtime_untouched':True})
print(json.dumps({'prepared':str(D),'repo':str(R),'resume':str(cp)}))
