set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc-dvac
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root:/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support"
export REPO_PATH="$root" EMBODIED_PATH="$root/examples/embodiment"
export ASSETS_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export PI0_MODEL_PATH=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
export ONLINE_BC_RUN_DIR=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-bc-dvac32x1-b1024-u10-gpu7-smoke2-20260905-v1
export RAY_ADDRESS=172.17.0.1:6389 RLINF_CODE_WORKING_DIR="$root"
unset CUDA_VISIBLE_DEVICES
cd "$root"
"$venv/bin/python" - <<'PY'
import hashlib, json, os
from pathlib import Path
import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rlinf.config import validate_cfg
root=Path.cwd()
with initialize_config_dir(version_base='1.1',config_dir=str(root/'examples/embodiment/config')):
    cfg=compose(config_name='robotwin_adjust_bottle_online_bc_openpi',overrides=['+bc_dvac=default'])
try:
    cfg=validate_cfg(cfg)
finally:
    ray.shutdown()
baseline_root=root.parent/'pi0-online-bc'
baseline_run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-bc32x1-b1024-u10-eval16x2-gpu6-formal100-20260905-v1')
base=OmegaConf.load(baseline_run/'runtime/resolved.yaml')
def flat(x,p=''):
    if isinstance(x,dict): return {k2:v2 for k,v in x.items() for k2,v2 in flat(v,p+'.'+k if p else k).items()}
    return {p:x}
a,b=[flat(OmegaConf.to_container(c,resolve=True)) for c in (base,cfg)]
delta={k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
allowed={'cluster.component_placement.actor,env,rollout','runner.logger.experiment_name','runner.max_epochs','runner.val_check_interval','runner.save_interval','actor.optim.total_training_steps'}
for k,(old,new) in delta.items():
    if k.startswith('algorithm.online_bc.dvac.') or k in allowed:continue
    assert isinstance(old,str) and isinstance(new,str), (k,old,new)
    normalized=new.replace(str(root),str(baseline_root)).replace(os.environ['ONLINE_BC_RUN_DIR'],str(baseline_run))
    assert normalized==old,(k,old,new)
for rel in ('rlinf/envs/robotwin/seeds/train_seeds.json','rlinf/envs/robotwin/seeds/eval_seeds.json','examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml'):
    assert (root/rel).read_bytes()==(baseline_root/rel).read_bytes(),rel
packet=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-dvac-20260905')
(packet/'smoke-resolved.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
(packet/'formal-base-to-smoke-diff.json').write_text(json.dumps(delta,indent=2))
print('RESOLVED_VALIDATED',json.dumps(delta))
print('BUDGET: GPU7; 2 collection rounds; 64 train attempts; <=256 new queries; 20 Adam updates; 20480 chunk presentations; 64 fixed eval attempts; 2 checkpoints; <=90min.')
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
df -h /data /home
git diff --check
