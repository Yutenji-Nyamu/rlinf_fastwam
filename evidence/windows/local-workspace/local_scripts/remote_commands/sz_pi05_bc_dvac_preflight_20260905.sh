set -eu
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 - <<'PY'
import datetime,hashlib,json,os,re,subprocess
from pathlib import Path
base=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc')
old=base.with_name('pi0-online-bc-dvac'); target=base.with_name('pi05-online-bc-dvac')
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1')
files=['rlinf/data/online_bc.py','rlinf/models/embodiment/openpi/openpi_action_model.py','rlinf/workers/actor/fsdp_online_bc_policy_worker.py','rlinf/workers/env/env_worker.py','rlinf/workers/rollout/hf/huggingface_worker.py']
def cmd(*a):return subprocess.check_output(a,text=True,timeout=20).strip()
assert not target.exists()
heads={str(p):{'head':cmd('git','-C',str(p),'rev-parse','HEAD'),'dirty':cmd('git','-C',str(p),'status','--porcelain')} for p in (base,old)}
assert heads[str(base)]=={'head':'6a93605d91dbfdc321c5602108ccca5e2f044001','dirty':''},heads
assert not heads[str(old)]['dirty']
hashes={}
for name in files:
 before=subprocess.check_output(['git','-C',str(old),'show','736b1416^:'+name])
 current=(base/name).read_bytes()
 hashes[name]={'base_sha256':hashlib.sha256(current).hexdigest(),'matches_old_dvac_parent':before==current}
 assert before==current,name
assert not cmd('nvidia-smi','-i','7','--query-compute-apps=pid','--format=csv,noheader,nounits')
log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'driver.log').read_text(errors='replace'))
errors=[s for s in log.splitlines() if re.search(r'Fatal Python|Traceback \(most recent|OutOfMemoryError|CUDA out of memory|ErrorInitializationFailed|cannot create buffer|AssertionError|RuntimeError:',s)]
mem={s.split(':')[0]:s.split(':')[1].strip() for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith(('MemAvailable:','SwapFree:'))}
out={'time':datetime.datetime.now().astimezone().isoformat(),'heads':heads,'shared_files':hashes,'gpu':cmd('nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'),'memory':mem,'disk':cmd('df','-B1','/data','/home'),'bc_base':{'wrapper_alive':Path('/proc/2143105').exists(),'exit':(run/'exit_code.txt').read_text().strip() if (run/'exit_code.txt').exists() else None,'errors':errors[:5],'tail':log.splitlines()[-8:]},'shared_ray_pids_present':{str(p):Path('/proc',str(p)).exists() for p in (321933,322685)},'sidney_wrapper_present':Path('/proc/602620').exists(),'passed':True}
print(json.dumps(out))
PY
