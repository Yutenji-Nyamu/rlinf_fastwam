import json
from pathlib import Path
D=Path('/home/chenyiteng/deployment-20261101/gad-k4-v1')
for pair in ['45','67']:
    p=D/f'pair{pair}'/'formal-plan.json'
    c=json.loads(p.read_text())
    c['checkpoint_reserve_gib']['cp_v2']=10
    tmp=p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(c,indent=2)+'\n')
    tmp.replace(p)
print('CP v2 reserve 10 GiB: measured complete smoke pair 8.03 GiB; training budget unchanged.')
