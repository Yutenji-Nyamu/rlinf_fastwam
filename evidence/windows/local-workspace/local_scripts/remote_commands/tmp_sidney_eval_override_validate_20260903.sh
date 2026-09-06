set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
cd "$WT"
export PYTHONPATH="$WT:${PYTHONPATH:-}"
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
"$PY" - <<'PY'
import os
from hydra import compose, initialize_config_dir
from rlinf.config import validate_cfg

config_dir = os.path.join(os.environ['EMBODIED_PATH'], 'config')
overrides = [
    'runner.task_type=embodied_eval',
    'runner.only_eval=true',
    '~cluster.component_placement',
    '+cluster.component_placement={env:4,rollout:4}',
    'env.eval.total_num_envs=1',
    'env.eval.rollout_epoch=1',
]
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(
        config_name='robotwin_move_stapler_pad_grpo_openpi_pi05_sidney',
        overrides=overrides,
    )
assert cfg.runner.task_type == 'embodied_eval'
assert cfg.runner.only_eval is True
assert set(cfg.cluster.component_placement) == {'env', 'rollout'}
assert cfg.cluster.component_placement.env == 4
assert cfg.cluster.component_placement.rollout == 4
assert cfg.rollout.model.model_type == 'openpi'
assert cfg.rollout.model.openpi.config_name == 'pi05_sidney_robotwin'
assert cfg.rollout.model.num_steps == 10
validate_cfg(cfg)
print('EVAL_OVERRIDE_VALIDATE_PASS=1')
print('placement=', dict(cfg.cluster.component_placement))
print('task_type=', cfg.runner.task_type, 'only_eval=', cfg.runner.only_eval)
PY
