#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
cd "$WT"
export PYTHONPATH="$WT:${PYTHONPATH:-}"
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
"$PY" -m pytest -q tests/unit_tests/test_lerobot_pi05_importer.py
"$PY" -m ruff check \
  rlinf/utils/ckpt_convertor/openpi/lerobot_pi05_to_openpi_rlinf.py \
  rlinf/utils/ckpt_convertor/openpi/convert.py \
  rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py \
  rlinf/models/embodiment/openpi/dataconfig/__init__.py \
  tests/unit_tests/test_lerobot_pi05_importer.py
"$PY" - <<'PY'
import os
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rlinf.config import validate_cfg
config_dir = os.path.join(os.environ['EMBODIED_PATH'], 'config')
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name='robotwin_move_stapler_pad_grpo_openpi_pi05_sidney')
assert cfg.actor.model.num_action_chunks == 50
assert cfg.actor.model.num_steps == 10
assert cfg.actor.model.openpi.config_name == 'pi05_sidney_robotwin'
assert cfg.env.train.total_num_envs == 64
assert cfg.env.train.rollout_epoch == 4
assert cfg.env.train.task_config.task_name == 'move_stapler_pad'
assert cfg.env.train.max_episode_steps == 400
assert cfg.env.train.task_config.step_lim == 400
assert cfg.env.eval.task_config.step_lim == 400
assert cfg.algorithm.update_epoch == 2
assert cfg.actor.global_batch_size == 1024
assert cfg.env.train.task_config.embodiment == ['aloha-agilex']
assert cfg.env.train.task_config.camera.collect_wrist_camera is True
assert cfg.env.eval.task_config.camera.collect_wrist_camera is True
assert cfg.env.train.center_crop is False
assert cfg.actor.model.openpi.noise_level == 0.3
assert OmegaConf.to_container(cfg.rollout.model, resolve=True) == OmegaConf.to_container(
    cfg.actor.model, resolve=True
)
assert cfg.rollout.model.model_type == 'openpi'
assert cfg.rollout.model.num_action_chunks == 50
assert cfg.rollout.model.action_dim == 14
assert cfg.rollout.model.num_steps == 10
assert cfg.rollout.model.openpi.config_name == 'pi05_sidney_robotwin'
assert cfg.rollout.model.openpi.num_images_in_input == 3
assert cfg.rollout.model.openpi.action_chunk == 50
assert cfg.rollout.model.openpi.action_env_dim == 14
validate_cfg(cfg)
with initialize_config_dir(version_base=None, config_dir=config_dir):
    eval_cfg = compose(
        config_name='robotwin_move_stapler_pad_grpo_openpi_pi05_sidney',
        overrides=['runner.only_eval=true'],
    )
validate_cfg(eval_cfg)
print('compose_validate_ok 2gpu env64x4 GB1024 update2 H50 M10 aloha3cam move400')
PY
