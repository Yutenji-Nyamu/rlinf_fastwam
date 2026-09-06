#!/usr/bin/env bash
set -eu
id
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root:/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support"
export REPO_PATH="$root"
export EMBODIED_PATH="$root/examples/embodiment"
export ASSETS_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export PI0_MODEL_PATH=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
export ONLINE_BC_RUN_DIR=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-online-bc32x1-m4-gpu6-formal100-20260905-v1
export RAY_ADDRESS=172.17.0.1:6389
export RLINF_CODE_WORKING_DIR="$root"
cd "$root"
test "$(git rev-parse HEAD)" = 72a926041867cbdbf2565ab66a14d742c59a0dad
test -z "$(git status --porcelain)"
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import json,os
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rlinf.config import validate_cfg
import ray
overrides=['runner.max_epochs=100','runner.val_check_interval=5','runner.save_interval=10','actor.optim.total_training_steps=200','runner.logger.experiment_name=pi0-adjust-bottle-online-bc32x1-m4-formal100-gpu6']
with initialize_config_dir(version_base='1.1',config_dir=str(Path.cwd()/'examples/embodiment/config')):
    cfg=compose(config_name='robotwin_adjust_bottle_online_bc_openpi',overrides=overrides)
    try:
        cfg=validate_cfg(cfg)
    finally:
        ray.shutdown()
def leaves(value,prefix=''):
    if isinstance(value,dict):
        return {k:v for name,item in value.items() for k,v in leaves(item,f'{prefix}.{name}' if prefix else name).items()}
    return {prefix:value}
smoke=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v6/runtime/resolved.yaml')
old=leaves(OmegaConf.to_container(OmegaConf.load(smoke),resolve=True))
new=leaves(OmegaConf.to_container(cfg,resolve=True))
changes={k:{'smoke':old.get(k),'formal':new.get(k)} for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
allowed={'runner.max_epochs','runner.val_check_interval','runner.save_interval','actor.optim.total_training_steps','runner.logger.experiment_name','runner.logger.log_path','algorithm.online_bc.data_path','env.train.task_config.save_path','env.eval.task_config.save_path'}
allowed.update({'env.train.video_cfg.video_base_dir','env.eval.video_cfg.video_base_dir'})
assert set(changes)<=allowed,changes
assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None
output=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-20260905')
(output/'formal-resolved-compose.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
(output/'formal-vs-smoke.json').write_text(json.dumps(changes,indent=2))
print(json.dumps(changes,indent=2))
print('FORMAL_COMPOSE_ONLY; no environment, rollout, training or new run launched')
PY
