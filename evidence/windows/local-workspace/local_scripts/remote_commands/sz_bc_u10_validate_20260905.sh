set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root:/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support"
export REPO_PATH="$root" EMBODIED_PATH="$root/examples/embodiment"
export ASSETS_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export PI0_MODEL_PATH=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
export RAY_ADDRESS=172.17.0.1:6389 RLINF_CODE_WORKING_DIR="$root"
cd "$root"
"$venv/bin/python" -m ruff check --select I --fix rlinf/envs/robotwin/robotwin_env.py tests/unit_tests/test_online_bc.py
"$venv/bin/python" -m ruff format rlinf/envs/robotwin/robotwin_env.py tests/unit_tests/test_online_bc.py
git diff --check
"$venv/bin/python" -m pytest -q tests/unit_tests/test_online_bc.py
"$venv/bin/python" - <<'PY'
import json, os
from pathlib import Path
import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rlinf.config import validate_cfg
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
root = Path.cwd()
base = Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc')
packet = base/'implementation-u10-20260905'
packet.mkdir(exist_ok=True)
configs = {}
for mode, run, rounds, interval, save, total in [
    ('smoke', 'pi0-adjust-bottle-smoke32x1-b1024-u10-eval16x2-gpu6-20260905-v7',2,1,1,20),
    ('formal','pi0-adjust-bottle-bc32x1-b1024-u10-eval16x2-gpu6-formal100-20260905-v1',100,5,10,1000),
]:
    os.environ['ONLINE_BC_RUN_DIR'] = str(base/run)
    with initialize_config_dir(version_base='1.1',config_dir=str(root/'examples/embodiment/config')):
        cfg=compose(config_name='robotwin_adjust_bottle_online_bc_openpi',overrides=[
            f'runner.max_epochs={rounds}',f'runner.val_check_interval={interval}',
            f'runner.save_interval={save}',f'actor.optim.total_training_steps={total}',
            f'runner.logger.experiment_name=pi0-bc-u10-{"smoke" if mode=="smoke" else "formal100"}-gpu6',
        ])
    try:
        cfg=validate_cfg(cfg)
    finally:
        ray.shutdown()
    assert cfg.actor.micro_batch_size == 32 and cfg.actor.global_batch_size == 1024
    assert cfg.algorithm.update_epoch == 10 and cfg.env.train.total_num_envs == 32
    assert cfg.env.eval.total_num_envs*cfg.env.eval.rollout_epoch == 32
    assert cfg.actor.model.num_steps == 4 and cfg.runner.resume_dir is None
    configs[mode]=OmegaConf.to_container(cfg,resolve=True)
    (packet/f'{mode}-resolved.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    # Exercise real seed files and native partitioning without creating a scene.
    env=RoboTwinEnv.__new__(RoboTwinEnv)
    env.cfg=cfg.env.eval
    env.task_name='adjust_bottle'
    env.seed=env.base_seed=cfg.env.eval.seed
    env.seed_offset=0
    env.total_num_processes=env.group_size=1
    env.num_envs=env.num_group=16
    env.auto_reset=False
    env.use_fixed_reset_state_ids=True
    env._init_reset_state_ids()
    first=env.reset_state_ids.tolist()
    env.update_reset_state_ids()
    second=env.reset_state_ids.tolist()
    env.update_reset_state_ids()
    assert env.reset_state_ids.tolist()==first and len(set(first+second))==32
    print(mode, 'FIXED_SEED_BATCHES', first, second)
    (packet/'fixed-eval-seeds.json').write_text(json.dumps([first,second]))
def flatten(x,p=''):
    if isinstance(x,dict):
        return {k2:v2 for k,v in x.items() for k2,v2 in flatten(v,p+'.'+k if p else k).items()}
    return {p:x}
a,b=map(flatten,[configs['smoke'],configs['formal']])
delta={k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
print('SMOKE_FORMAL_DIFF',json.dumps(delta))
(packet/'validation.json').write_text(json.dumps({'passed':True,'u':10,'diff':delta},indent=2))
print('Validated GPU6 / train32x1 / eval16x2 / m32 B1024 U10 / original SFT / empty pool')
PY
git diff --stat
