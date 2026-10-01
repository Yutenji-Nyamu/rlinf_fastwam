set -eu
CUDA_VISIBLE_DEVICES='' /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import importlib.util,json,os,sys,time,traceback
from pathlib import Path
sys.modules.setdefault('tensorflow',None)
p=Path('/data/chenyiteng/projects/robodojo-openwam-sz3')
c=p/'rlt-cycle-sz3-wan-goal-20261001-repair-v1'
d=p/'runs/sz3_pi05_official_6300_n4_dual_20260929_r2'
read=lambda x:json.loads(x.read_text())
active=read(d/'active-continuation.json')
assert os.getuid()==20001 and Path(active['cycle_dir']).resolve()==c.resolve()
spec=importlib.util.spec_from_file_location('own_sz3_return_audit',c/'rlt_cycle_sz3.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
plan=m.load_plan(c);stopped=read(c/'rlt-stopped.json')
out={'time':time.time(),'cycle':str(c),'read_only':True,'runs':{},'scope':'Frozen current SZ3 return checkpoints; no donor repair, launch, signal or config change'}
for key,row in plan['runs'].items():
    cp=stopped['runs'][key]['recovery']['checkpoint']
    cfg=m.config(c/'prepared'/key/'resolved.yaml')
    result={'path':cp['path'],'step':cp['step'],'configured_resume_matches':cfg['runner']['resume_dir']==cp['path']}
    try:
        assert result['configured_resume_matches']
        for relative,sha in cp['small_sha256'].items():
            assert m.sha(Path(cp['path'])/'actor'/relative)==sha,'Frozen small state changed: '+relative
        checked=m.inspect_checkpoint(cp['path'],cfg,plan['repo'])
        assert checked['step']==cp['step'] and checked['contract_sha256']==cp['contract_sha256']
        assert checked['small_sha256']==cp['small_sha256']
        result.update(ok=True,checked=checked)
    except Exception as error:
        result.update(ok=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    out['runs'][key]=result
import torch
out.update(all_valid=len(out['runs'])==4 and all(x['ok'] for x in out['runs'].values()),cuda_initialized=torch.cuda.is_initialized())
assert not out['cuda_initialized']
print(json.dumps(out),flush=True)
PY
