"""Read the deployed native environment's offload implementation."""
import ast,json,os,socket,subprocess
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
R=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rlinf-opendw-two-gpu-v1')
names=subprocess.check_output(['find','-L',str(R/'rlinf/envs'),'-name','*.py','-type','f'],text=True).splitlines()
out={'robotwin_header':'\n'.join((R/'rlinf/envs/robotwin/robotwin_env.py').read_text().splitlines()[:85])}
for name in names:
 p=Path(name);text=p.read_text();tree=ast.parse(text);lines=text.splitlines();matches=[]
 for node in ast.walk(tree):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ['offload','onload','close','close_env']:
   matches.append({'line':node.lineno,'source':'\n'.join(lines[node.lineno-1:node.end_lineno])})
 if matches and ('vec' in name or 'subproc' in name or 'worker' in name):out[str(p.relative_to(R))]=matches
print(json.dumps(out))
