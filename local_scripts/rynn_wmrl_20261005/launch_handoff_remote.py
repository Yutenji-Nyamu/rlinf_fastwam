"""Launch the reviewed exact-owner transition once; never replay its receipt."""
import datetime,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'rynn-control-v1';P=D/'prepared'
assert not (D/'launch-receipt.json').exists() and not (D/'handoff').exists() and not (S/'runs/rynn-success-v1').exists()
plan=json.loads((P/'plan-template.json').read_text());ready=json.loads((P/'ready.json').read_text())
for path,digest in ready['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
sys.path.insert(0,str(D/'code'));import rynn_formal_owner
M=rynn_formal_owner.install(plan);M.validate(plan)
previous=json.loads((S/'runs/formal-b16-v1/owner-identity.json').read_text())
assert previous['pid']==394537 and previous['start']==707417273 and M.H.same(previous)
before=M.H.gpu_processes(list(range(8)))
assert not any(row['gpu'] in [0,1,2,3] for row in before),before
argv=[plan['python'],'-u','-B',str(D/'code/rynn_handoff.py'),'--ready-manifest',str(P/'ready.json'),
 '--resume-receipt',str(P/'resume-receipt.json'),'--plan-template',str(P/'plan-template.json')]
env=json.loads(Path(plan['environment_file']).read_text())
with (D/'handoff-launch.log').open('x') as log:
    child=subprocess.Popen(argv,cwd=plan['repo'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
identity=M.H.proc(child.pid);assert identity and identity['uid']==20001
receipt={'time':datetime.datetime.now().astimezone().isoformat(),'identity':identity,'argv':argv,
 'old_owner':previous,'plan_sha256':hashlib.sha256((P/'plan-template.json').read_bytes()).hexdigest(),'gpu_before':before}
with (D/'launch-receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt),flush=True)
