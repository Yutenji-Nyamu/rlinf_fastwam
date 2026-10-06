import ast,json,os,socket,sys
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');p=json.loads((S/'lift-two-gpu-b32-v1/prepared-v2/owner-plan.json').read_text())
env=json.loads(Path(p['environment_file']).read_text());cfg=json.loads(Path(p['trials'][0]['config']).read_text())
roots=[Path(x) for x in env.get('PYTHONPATH','').split(':') if x]+[Path(x) for x in sys.path if x]
roots += [Path(cfg['env']['eval']['assets_path']).parent]
files=[]
for root in roots:
 for tail in ['robotwin/envs/vector_env.py','envs/vector_env.py']:
  f=root/tail
  if f.is_file() and f not in files:files.append(f)
out={'matches':{}}
for f in files:
 text=f.read_text();lines=text.splitlines();tree=ast.parse(text)
 matches=[]
 for node in ast.walk(tree):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and (node.name in ['close','close_env','worker','_worker']):
   matches.append({'line':node.lineno,'source':'\n'.join(lines[node.lineno-1:node.end_lineno])[-17000:]})
 out['matches'][str(f)]=matches
if not files:out['search_roots']=[str(x) for x in roots]
print(json.dumps(out))
