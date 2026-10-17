import ast,base64,datetime,json,zlib,collections
from pathlib import Path
R=Path('E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a');cut=datetime.datetime(2026,9,26,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
tree=ast.parse(Path('.tmp/prepare_collection1017.py').read_text());body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
body=body.replace('post-iclr-20261017-v1','post-iclr-plots-20261017-v1').replace("('/data/chenyiteng/results/','/home/chenyiteng/results/')","('/data/chenyiteng/results/','/home/chenyiteng/results/','/data/chenyiteng/projects/')")
excluded=[]
for h in ['sz1','sz2','sz3']:
 d=json.loads((R/h/'numeric-inventory1017.out').read_text());fs=[]
 for f in d['files']+d['large']:
  p=Path(f['path']);reason=None
  if f['mtime']<cut:continue
  if 'server-maintenance-' in str(p):continue
  if f['bytes']>20*1024**2:reason='individual numeric/figure artifact exceeds 20 MiB'
  elif p.name.endswith(('_obs.npz','_scores.npz')):reason='large raw observation/full-attention collection; derived JSON/curve arrays retained'
  elif 'opendw-robotwin' in str(p) and any(x in p.parts for x in ['wm_records','service-records','capture','records','samples']):reason='large world-model sample/input collection; JSON result metrics retained'
  elif p.suffix=='.npy':reason='raw depth/point-cloud observation; evaluation receipts and RGB previews retained'
  if reason:excluded.append(dict(f,host=h,reason=reason));continue
  fs.append(f)
 payload=base64.b64encode(zlib.compress(json.dumps(fs).encode())).decode();Path(f'.tmp/collectplots1017_{h}.py').write_text('PAYLOAD='+repr(payload)+'\n'+body,encoding='utf-8')
 print(h,len(fs),round(sum(f['bytes'] for f in fs)/2**20,1),'MiB')
(R/'archive1017/numeric-exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2),encoding='utf-8')
groups=collections.defaultdict(lambda:[0,0])
for f in excluded:groups[f['reason']][0]+=1;groups[f['reason']][1]+=f['bytes']
print('EXCLUDED',json.dumps({k:{'files':v[0],'GiB':round(v[1]/2**30,2)} for k,v in groups.items()}))
