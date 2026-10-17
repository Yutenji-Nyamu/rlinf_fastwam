import ast,json,zlib,base64
from pathlib import Path
R=Path('E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a');tree=ast.parse(Path('.tmp/prepare_collection1017.py').read_text());body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
body=body.replace('post-iclr-20261017-v1','post-iclr-extra-20261017-v1').replace("('/data/chenyiteng/results/','/home/chenyiteng/results/')","('/data/chenyiteng/projects/',)")
for h in ['sz1','sz2','sz3']:
 d=json.loads((R/h/'extras-inventory1017.out').read_text());fs=[]
 for f in d['files']:
  p=Path(f['path']);a=f['path'].split('/projects/')[1].split('/')
  if any(x in p.parts for x in ['cache','.cache','conda','envs','venvs','packages','pkgs','third_party','lib','build','dist','src','source','assets','Assets']):continue
  if 'robodojo' in a[0] and not (a[1]=='runs' or a[1:3]==['RoboDojo','eval_result']):continue
  if 'opendw' in a[0] and not (a[1] in ['runs','wm-cycle-20261009-v2','rynn-numeric-v1','rynn-success-v1','rlt45-return-20261007-v1','logs','reports']):continue
  if any(x.startswith(('rlinf','publication')) for x in a[1:3]):continue
  fs.append(f)
 payload=base64.b64encode(zlib.compress(json.dumps(fs).encode())).decode();Path(f'.tmp/collectextra1017_{h}.py').write_text('PAYLOAD='+repr(payload)+'\n'+body,encoding='utf-8')
 print(h,len(fs),round(sum(f['bytes'] for f in fs)/2**20),'MiB')
