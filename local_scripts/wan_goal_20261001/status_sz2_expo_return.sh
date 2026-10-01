set -eu
CUDA_VISIBLE_DEVICES='' /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import importlib.util,json,os,socket,sys,time
from pathlib import Path
sys.modules.setdefault('tensorflow',None)
r=Path('/data/chenyiteng/projects/expo-ft-sz2-20261001');c=r/'rlt-cycle'
read=lambda p:json.loads(p.read_text())
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu02'
guardian=read(r/'rlt-guardian.json')
assert guardian['state']=='RESTORED' and guardian['restoration']['first_round_verified']
assert guardian['owner']['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
spec=importlib.util.spec_from_file_location('returned_sz2_frozen_rlt',c/'rlt_cycle.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
plan=m.load_plan(c);dispatched=read(c/'resumed-dispatched.json');first=read(c/'rlt-first-round.json')
assert first['state']=='verified' and first['status']['all_first_rounds_verified']
status=m.status(c)
watch=read(Path('/data/chenyiteng/deployment-20260927/rlt-six-task-watch/plan.json'))
out={'time':time.time(),'host':'sz2','read_only':True,'cycle':str(c),
     'guardian_state':guardian['state'],'guardian_terminal_status':guardian['terminal_status'],
     'guardian_error':guardian['error'],'restore_first_round_verified':True,
     'resumed_dispatch_present':bool(dispatched),'status':status,'watch_matches':{},
     'source':'Frozen current EXPO return helper status; no bridge CLI writes or resource actions'}
for key,row in plan['runs'].items():
    out['watch_matches'][key]=any(v.get('run')==row['new_run'] for v in watch['runs'].values())
assert all(out['watch_matches'].values())
import torch
out['cuda_initialized']=torch.cuda.is_initialized();assert not out['cuda_initialized']
print(json.dumps(out),flush=True)
PY
