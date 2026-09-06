set -eu
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import datetime,hashlib,json,subprocess
from pathlib import Path
base=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees');bc=base/'pi0-online-bc';pi=base/'sidney-pi05-current-rlinf'
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30);return {'rc':p.returncode,'out':p.stdout,'err':p.stderr}
result={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'files':{}}
audit=Path('/data/chenyiteng/results/rlinf-shenzhen/checkpoint-cleanup-20260905-approved')
result['cleanup_audit_status']={'exists':audit.exists(),'files':[p.name for p in audit.iterdir()] if audit.exists() else []}
names=cmd(['git','-C',str(pi),'grep','-l','class LeRobotAlohaDataConfig'])['out'].splitlines()
names+=['rlinf/envs/robotwin/robotwin_env.py','rlinf/runners/embodied_runner.py','rlinf/utils/metric_utils.py','rlinf/workers/rollout/hf/huggingface_worker.py']
for root in (bc,pi):
 for name in names:
  p=root/name
  if p.is_file():result['files'][str(p)]={'text':p.read_text(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
result['dataconfig_diff']=cmd(['git','diff','--no-index','--',str(bc/'rlinf/models/embodiment/openpi/dataconfig'),str(pi/'rlinf/models/embodiment/openpi/dataconfig')])
result['sidney_implementation']=cmd(['git','-C',str(pi),'show','--stat','--format=fuller','bab221afb8be'])
model=Path('/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab')
result['sidney_model_files']=[{'path':str(p),'bytes':p.stat().st_size} for p in model.rglob('*') if p.is_file()]
result['small_model_metadata']={str(p):p.read_text() for p in model.rglob('*') if p.is_file() and p.suffix=='.json' and p.stat().st_size<2_000_000}
import transformers.models.gemma.modeling_gemma as gemma
p=Path(gemma.__file__);result['transformers_gemma']={'path':str(p),'text':p.read_text(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
root=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1')
result['sidney_runtime_small']={str(p.relative_to(root)):p.read_text() for dirname in ('runtime','runtime-resume100-to200') for p in (root/dirname).iterdir() if p.is_file() and p.suffix in ('.sh','.yaml') and p.stat().st_size<1_000_000}
print(json.dumps(result))
PY
