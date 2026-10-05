import importlib.util,json,os,socket,sys
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
D=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/click-bell-v1')
plan=json.loads((D/'prepared/owner-plan.json').read_text())
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
F=load('bell_formal_validate',D/'generated/owner/opendw_formal_owner.py')
M=F.install(plan)
hook=load('bell_prechecks_validate',plan['prechecks_module']);M=hook.install(M,plan)
result=M.validate(plan)
print(json.dumps(dict(valid=True,owner_dir=str(result[0]),budget=plan['budget'],prechecks=[r['key'] for r in plan['prechecks']])),flush=True)
