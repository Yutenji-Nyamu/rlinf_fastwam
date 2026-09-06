set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root:/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support"
export REPO_PATH="$root" EMBODIED_PATH="$root/examples/embodiment" ASSETS_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export PI0_MODEL_PATH=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
export ONLINE_BC_RUN_DIR=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval8x4-gpu6-20260905-v8
export RAY_ADDRESS=172.17.0.1:6389 RLINF_CODE_WORKING_DIR="$root"
cd "$root"
"$venv/bin/python" -m ruff check --select I tests/unit_tests/test_online_bc.py
"$venv/bin/python" -m ruff format tests/unit_tests/test_online_bc.py
"$venv/bin/python" -m pytest -q tests/unit_tests/test_online_bc.py
"$venv/bin/python" - <<'PY'
import json, os
from pathlib import Path
import ray
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from rlinf.config import validate_cfg
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
root=Path.cwd()
with initialize_config_dir(version_base='1.1',config_dir=str(root/'examples/embodiment/config')):
    cfg=compose(config_name='robotwin_adjust_bottle_online_bc_openpi',overrides=['runner.logger.experiment_name=pi0-bc-u10-eval8x4-smoke-gpu6'])
try:cfg=validate_cfg(cfg)
finally:ray.shutdown()
old_run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval16x2-gpu6-20260905-v7')
old=OmegaConf.load(old_run/'runtime/resolved.yaml')
def flatten(x,p=''):
    if isinstance(x,dict):return {k2:v2 for k,v in x.items() for k2,v2 in flatten(v,p+'.'+k if p else k).items()}
    return {p:x}
a,b=[flatten(OmegaConf.to_container(c,resolve=True)) for c in (old,cfg)]
delta={k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
allowed={'env.eval.total_num_envs','env.eval.rollout_epoch','env.eval.fixed_reset_batch_count','runner.logger.experiment_name'}
for k,(x,y) in delta.items():
    if k in allowed:continue
    assert isinstance(x,str) and isinstance(y,str) and y.replace(os.environ['ONLINE_BC_RUN_DIR'],str(old_run))==x,(k,x,y)
env=RoboTwinEnv.__new__(RoboTwinEnv)
env.cfg=cfg.env.eval;env.task_name='adjust_bottle';env.seed=env.base_seed=cfg.env.eval.seed;env.seed_offset=0;env.total_num_processes=env.group_size=1;env.num_envs=env.num_group=8;env.auto_reset=False;env.use_fixed_reset_state_ids=True
env._init_reset_state_ids()
batches=[]
for _ in range(4):batches.append(env.reset_state_ids.tolist());env.update_reset_state_ids()
expected=json.loads(Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-u10-20260905/fixed-eval-seeds.json').read_text())
assert sum(batches,[])==sum(expected,[]) and len(set(sum(batches,[])))==32
assert env.reset_state_ids.tolist()==batches[0]
packet=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-eval8-20260905')
(packet/'smoke-resolved.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
(packet/'diff.json').write_text(json.dumps(delta,indent=2))
(packet/'fixed-seeds.json').write_text(json.dumps(batches,indent=2))
print('PASSED_RESOLVED_AND_IDENTICAL_32_FIXED_SEEDS',json.dumps(delta))
PY
git diff --check
git diff --stat
git add -- examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml tests/unit_tests/test_online_bc.py
git commit -m 'Reduce online BC eval concurrency to 8x4 while preserving 32 fixed seeds'
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy timeout 90s git -c http.proxy= -c https.proxy= push personal HEAD:refs/heads/codex/sz-pi0-online-bc
git rev-parse HEAD
git status --porcelain
