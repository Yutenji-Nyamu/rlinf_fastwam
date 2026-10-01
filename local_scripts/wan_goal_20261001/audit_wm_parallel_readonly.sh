set -eu
/data/chenyiteng/projects/wan-goal-sz3/envs/pi05-wan/bin/python -B - <<'PY'
import json, subprocess, sys, time
from pathlib import Path
R=Path('/data/chenyiteng/projects/wan-goal-sz3')
F=R/'runs/wan-goal-sz3-20261001-r6/pi05-formal'
sys.path.insert(0,str(R/'scripts/resource_switch'))
from common import account, read, identity, alive, sha
account()
catalog=read(F/'managed-identities.json')
rows={int(r['pid']):r for r in catalog['managed_processes']}
gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
cards={}
uuid_index={}
for line in gpu_csv.splitlines():
    idx,uuid,total,used,util=[v.strip() for v in line.split(',')]
    uuid_index[uuid]=int(idx)
    if int(idx) in (4,5,6,7):
        cards[int(idx)]=dict(memory_total_mib=int(total),memory_used_mib=int(used),utilization_pct=int(util),processes=[])
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
for line in apps.splitlines():
    pid,uuid,mem=[v.strip() for v in line.split(',')]
    idx=uuid_index.get(uuid)
    if idx not in cards: continue
    row=rows.get(int(pid)); record=dict(pid=int(pid),memory_used_mib=mem,owned_catalog=False)
    if row and alive(row):
        before=identity(int(pid))
        title=(Path('/proc')/pid/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')[:220]
        after=identity(int(pid))
        if all(before[k]==after[k]==row[k] for k in ('uid','start','boot')):
            record.update(owned_catalog=True,uid=before['uid'],start=before['start'],title=title)
    cards[idx]['processes'].append(record)
repo=R/'RLinf-pi05'
source_paths=[R/'src/diffsynth-studio/diffsynth/models/reward_model.py',
              repo/'rlinf/algorithms/advantages.py',repo/'rlinf/algorithms/utils.py',
              repo/'rlinf/algorithms/losses.py']
sources={}
for path in source_paths:
    if path.is_file(): sources[str(path)]=dict(sha256=sha(path),text=path.read_text()[:22000])
print(json.dumps(dict(time=time.time(),read_only=True,cards=cards,sources=sources,resource_actions=[])))
PY
