#!/usr/bin/env bash
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root:/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support"
export REPO_PATH="$root"
export EMBODIED_PATH="$root/examples/embodiment"
export ASSETS_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export PI0_MODEL_PATH=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
export ONLINE_BC_RUN_DIR=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v6
export RAY_ADDRESS=172.17.0.1:6389
export RLINF_CODE_WORKING_DIR="$root"
cd "$root"
date -Is
files=(rlinf/data/online_bc.py rlinf/workers/actor/fsdp_online_bc_policy_worker.py rlinf/models/embodiment/openpi/openpi_action_model.py examples/embodiment/train_embodied_agent.py rlinf/workers/rollout/hf/huggingface_worker.py rlinf/workers/env/env_worker.py tests/unit_tests/test_online_bc.py)
"$venv/bin/python" -m ruff check --select I --fix "${files[@]}"
"$venv/bin/python" -m ruff format "${files[@]}"
git diff --check
"$venv/bin/python" -m pytest -q tests/unit_tests/test_online_bc.py
"$venv/bin/python" - <<'PY'
import ast
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rlinf.workers.actor.fsdp_online_bc_policy_worker import EmbodiedOnlineBCFSDPPolicy
from rlinf.config import validate_cfg
import os, ray
root=Path.cwd()
assert (Path(os.environ['ASSETS_PATH'])/'assets/objects/objaverse/list.json').is_file()
from robotwin.envs.vector_env import VectorEnv
print('Real RoboTwin VectorEnv import and asset-root contract passed; no scenes created')
for rel in ['rlinf/data/online_bc.py','rlinf/workers/actor/fsdp_online_bc_policy_worker.py',
            'rlinf/workers/env/env_worker.py','rlinf/models/embodiment/openpi/openpi_action_model.py',
            'rlinf/workers/rollout/hf/huggingface_worker.py','examples/embodiment/train_embodied_agent.py']:
    ast.parse((root/rel).read_text())
with initialize_config_dir(version_base='1.1', config_dir=str(root/'examples/embodiment/config')):
    cfg=compose(config_name='robotwin_adjust_bottle_online_bc_openpi')
    try:
        assert os.environ['RAY_ADDRESS'] == '172.17.0.1:6389'
        # RLinf owns Ray initialization/namespace selection; RAY_ADDRESS pins
        # its connection to the existing head. Never call ray.start here.
        cfg=validate_cfg(cfg)
    finally:
        ray.shutdown()  # Disconnect this diagnostic driver; never stop the shared head.
    resolved=OmegaConf.to_yaml(cfg,resolve=True)
    assert cfg.actor.model.num_steps == cfg.actor.model.openpi.num_steps == 4
    assert cfg.env.train.total_num_envs == cfg.env.eval.total_num_envs == 32
    assert cfg.actor.micro_batch_size == 32 and cfg.actor.global_batch_size == 1024
    print('COMPOSE: online_bc / expert-only / GPU6 / 32x1 / M4 / MB32 GB1024 / demo_weight=0')
    output=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-20260905')
    output.mkdir(parents=True,exist_ok=True)
    (output/'smoke-resolved-compose.yaml').write_text(resolved)
print('AST / worker import / Hydra compose / validate_cfg passed; no GPU run launched')
PY
git diff --stat
git status --short
