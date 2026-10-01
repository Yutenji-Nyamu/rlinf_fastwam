set -eu
/data/chenyiteng/projects/wan-goal-sz3/envs/pi05-wan/bin/python -B - <<'PY'
import hashlib,json,sys,time
from pathlib import Path
R=Path('/data/chenyiteng/projects/wan-goal-sz3')
D=Path('/data/chenyiteng/projects/robodojo-openwam-sz3/runs/sz3_pi05_official_6300_n4_dual_20260929_r2')
W=R/'runs/wan-goal-sz3-20261001-r5';F=W/'pi05-formal';G=R/'post-wm-direct-rlt-v5'
sys.path.insert(0,str(R/'scripts/resource_switch'))
from common import account,alive,read,sha
account();active=read(D/'active-continuation.json');A=Path(active['attempt_dir']);C=Path(active['cycle_dir'])
assert A==D/'continuation-20261001-wan-goal-v5'
def tail(p,n=5000):
    if not p.is_file():return None
    with p.open('rb') as stream:stream.seek(max(0,p.stat().st_size-n));return stream.read().decode(errors='replace')
report={'time':time.time(),'active_attempt':str(A),'cycle':str(C),'owner_alive':alive(active,command=True),
        'pipeline':read(D/'pipeline-current.json'),'sequence':read(W/'sequence-current.json'),
        'guard_alive':alive(read(G/'guard-identity.json'),command=True),'guard_state':read(G/'current.json'),
        'formal_exit':read(F/'wm-exit.json'),'source_sha256':{},'receipts':{},'logs':{}}
for name in ('wm_sequence.py','resource_switch/common.py','resource_switch/wm_stage.py','resource_switch/cleanup_owned.py'):
    report['source_sha256'][name]=sha(R/'scripts'/name)
for p in (A/'pipeline-final.json',A/'wm-release.json',A/'dojo-release.json',A/'skip-dojo-return-rlt-after-wm-20261001.json',
          C/'resumed-dispatched.json',F/'wm-cleanup.json',W/'pi05-formal-control/wm-release.json'):
    if not p.is_file():continue
    v=read(p)
    # Publication evidence excludes private ownership marker values and large catalogs.
    report['receipts'][str(p)]={'sha256':sha(p),'bytes':p.stat().st_size,
       'fields':{k:v[k] for k in ('time','terminal_status','error','rlt_dispatched','wm_released','outcome','wm_exit_code',
           'all_workers_stopped','processes_clear','gpus_released','physical_gpus','cycle_id','action') if k in v}}
for p in (W/'command.log',W/'sequence.log',A/'wm-command.log',A/'wm-stage.log',A/'pipeline.log',F/'cleanup.log',G/'guard.log'):
    value=tail(p)
    if value is not None:report['logs'][str(p)]=value
report['formal_checkpoint_entries']=[str(p.relative_to(F)) for directory in (F/'checkpoints',F/'checkpoint',F/'tensorboard/checkpoints')
                                      if directory.is_dir() for p in directory.iterdir()]
print(json.dumps(report))
PY
