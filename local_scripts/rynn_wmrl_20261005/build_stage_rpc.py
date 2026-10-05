"""Build an explicit upload packet; local work is serialization, not project testing."""
import base64,hashlib,json
from pathlib import Path
W=Path(__file__).resolve().parents[2];packet=W/'local_patches/rynn_wmrl_20261005';ops=W/'local_scripts/rynn_wmrl_20261005'
files={}
for p in packet.rglob('*.py'):
    relative=p.relative_to(packet).as_posix()
    dest='repo/'+relative if relative.startswith('rlinf/') else 'code/'+p.name
    files[dest]=p.read_bytes()
for name in ['rynn_formal_owner.py','rynn_handoff.py','test_rynn_owner.py','rynn_gate_client.py']:
    files['code/'+name]=(ops/name).read_bytes()
payload={path:{'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()} for path,raw in files.items()}
remote=r'''
import base64,datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'rynn-control-v1';R=S/'rlinf-rynn-v1';C=D/'code'
assert (D/'prepared/base-prepared.json').is_file() and not (D/'handoff').exists()
C.mkdir(mode=0o700,exist_ok=True)
for path,item in json.loads(base64.b64decode(PAYLOAD)).items():
    category,relative=path.split('/',1);dest=(R if category=='repo' else D)/relative if category=='repo' else D/path
    raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
old=json.loads((S/'runs/formal-b16-v1/owner-plan.json').read_text())
for name in ['wm_batch.py','opendw_action_telemetry.py','opendw_reward.py']:
    source=S/'formal-b16-control-v1/code'/name
    assert old['source_sha256'][str(source)]==hashlib.sha256(source.read_bytes()).hexdigest()
    (C/name).write_bytes(source.read_bytes())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R)+':'+str(C),OMP_NUM_THREADS='1')
python='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
actor=R/'rlinf/workers/actor/embodied_fsdp_actor_worker.py'
if 'mask_invalid_rynn_groups' not in actor.read_text():
    subprocess.run([python,'-B',str(C/'patch_actor_rynn_mask.py'),str(actor),'--apply'],env=env,check=True,capture_output=True,text=True)
checks=[]
for command in [[python,'-B',str(C/'test_env_and_mask.py')],
                ['/data/chenyiteng/venvs/rynnvalue-8b-py310/bin/python','-B','-m','unittest','test_rynn_success_service','-v'],
                [python,'-B','-m','unittest','test_rynn_owner','-v']]:
    result=subprocess.run(command,cwd=C,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    checks.append({'command':command,'exit_code':result.returncode,'output':result.stdout})
receipt={'time':datetime.datetime.now().astimezone().isoformat(),'checks':checks,'all_cpu_tests_passed':all(x['exit_code']==0 for x in checks),
    'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in C.glob('*.py')}}
for name in ['rlinf/envs/__init__.py','rlinf/runners/embodied_runner.py','rlinf/workers/actor/embodied_fsdp_actor_worker.py']:
    p=R/name;receipt['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in (R/'rlinf/envs/world_model').glob('*.py'):
    if p.name.startswith(('opendw','rynn')):receipt['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(D/'prepared/cpu-tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));assert receipt['all_cpu_tests_passed'],'CPU contract checks failed'
'''
target=ops/'stage_remote.py';target.write_text('PAYLOAD = '+repr(base64.b64encode(json.dumps(payload).encode()).decode())+'\n'+remote,encoding='utf-8')
print(json.dumps({'files':len(files),'path':str(target),'bytes':target.stat().st_size}))
