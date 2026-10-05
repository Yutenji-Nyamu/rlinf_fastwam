"""Return bounded scientific results and lifecycle metadata, never image arrays."""
import datetime,hashlib,json,os,socket
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
O=S/'runs/click-bell-v2'
out=dict(time=datetime.datetime.now().astimezone().isoformat(),owner_dir=str(O),receipts={},missing=[])
for rel in ['native_rynn32/dataset/cases.json','rynn32/report.json','rynn32/precheck.json','native_bell32/dataset/samples.json','bell_rm/report.json','bell_rm/summary.json','bell_rm/precheck.json','prechecks.json','startup_smoke/result.json','formal/driver-identity.json','state.json','error.json','final.json']:
 p=O/rel
 if not p.exists():out['missing'].append(rel);continue
 assert p.stat().st_size<2_000_000
 raw=p.read_bytes();r=json.loads(raw)
 if rel=='native_rynn32/dataset/cases.json':r={k:v for k,v in r.items() if k!='cases'}
 if rel=='rynn32/report.json':
  r={k:r[k] for k in ['engineering_passed','summary','batches','started_at_unix','finished_at_unix','elapsed_s','offload','error'] if k in r}|{'cases':[dict(id=x['id'],reference_success=x['reference_success'],last_remaining_value=x.get('last_remaining_value'),language=x.get('language')) for x in r.get('cases',[])]}
 out['receipts'][rel]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),value=r)
print(json.dumps(out),flush=True)
