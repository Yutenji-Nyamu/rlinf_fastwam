import ast,base64,datetime,json,zlib
from pathlib import Path
R=Path('E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a');cut=datetime.datetime(2026,9,26,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
tree=ast.parse(Path('.tmp/prepare_collection1017.py').read_text());body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
body=body.replace('post-iclr-20261017-v1','post-iclr-text-20261017-v1').replace("('/data/chenyiteng/results/','/home/chenyiteng/results/')","('/data/chenyiteng/results/','/home/chenyiteng/results/','/data/chenyiteng/projects/')").replace('20*1024**2','512*1024**2').replace('grew past 20 MiB','grew past 512 MiB')
excluded=[]
for h in ['sz1','sz2','sz3']:
 d=json.loads((R/h/'large-text-inventory1017.out').read_text());fs=[]
 for f in d['large']:
  if f['mtime']<cut or 'server-maintenance-' in f['path']:continue
  if Path(f['path']).name in ['service.log','service-events.jsonl']:
   excluded.append(dict(f,host=h,reason='large WM per-request payload trace; scalar/resource/per-episode results retained'));continue
  if f['bytes']>512*1024**2:excluded.append(dict(f,host=h,reason='individual text file exceeds 512 MiB'));continue
  fs.append(f)
 payload=base64.b64encode(zlib.compress(json.dumps(fs).encode())).decode();Path(f'.tmp/collecttext1017_{h}.py').write_text('PAYLOAD='+repr(payload)+'\n'+body,encoding='utf-8')
 print(h,len(fs),round(sum(f['bytes'] for f in fs)/2**20,1),'MiB')
(R/'archive1017/text-exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2),encoding='utf-8');print('EXCLUDED',len(excluded))
