import json,subprocess,sys
from pathlib import Path
b=Path(__file__).parent;d=Path(sys.argv[1])
r={}
for h in (10,50):
 try:r[h]=subprocess.run([sys.executable,'-uB',str(b/'scan.py'),str(d),str(h)],timeout=3300).returncode
 except subprocess.TimeoutExpired:r[h]='timeout';break
 if r[h]:break
if all(r.get(h)==0 for h in (10,50)):
 a,b=[json.loads((d/f'scan-h{h}.json').read_text()) for h in (10,50)]
 r['paired_layouts']=a['layouts']==b['layouts']
 r['success']=[sum(a['success']),sum(b['success'])]
(d/'trial-status.json').write_text(json.dumps(r));print(r,flush=True)
