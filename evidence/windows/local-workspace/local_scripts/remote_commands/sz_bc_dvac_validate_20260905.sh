set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc-dvac
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="$root:/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support"
export REPO_PATH="$root" EMBODIED_PATH="$root/examples/embodiment"
export ASSETS_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export PI0_MODEL_PATH=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
export ONLINE_BC_RUN_DIR=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-bc-dvac32x1-b1024-u10-gpu7-smoke2-20260905-v1
cd "$root"
files=(rlinf/algorithms/online_bc_dvac.py rlinf/data/online_bc.py rlinf/models/embodiment/openpi/openpi_action_model.py rlinf/workers/actor/fsdp_online_bc_policy_worker.py rlinf/workers/env/env_worker.py rlinf/workers/rollout/hf/huggingface_worker.py tests/unit_tests/test_online_bc_dvac.py)
"$venv/bin/python" -m ruff check --select I --fix "${files[@]}"
"$venv/bin/python" -m ruff format "${files[@]}"
git diff --check
"$venv/bin/python" -m pytest -q tests/unit_tests/test_online_bc.py tests/unit_tests/test_online_bc_dvac.py
"$venv/bin/python" - <<'PY'
from pathlib import Path
import os, json
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
root=Path.cwd()
with initialize_config_dir(version_base='1.1',config_dir=str(root/'examples/embodiment/config')):
    a=compose(config_name='robotwin_adjust_bottle_online_bc_openpi')
    b=compose(config_name='robotwin_adjust_bottle_online_bc_openpi',overrides=['+bc_dvac=default'])
def flat(x,p=''):
    if isinstance(x,dict):
        return {k2:v2 for k,v in x.items() for k2,v2 in flat(v,p+'.'+k if p else k).items()}
    return {p:x}
af,bf=[flat(OmegaConf.to_container(c,resolve=True)) for c in (a,b)]
diff={k:[af.get(k),bf.get(k)] for k in sorted(af.keys()|bf.keys()) if af.get(k)!=bf.get(k)}
assert all(k.startswith('algorithm.online_bc.dvac.') or k in ('cluster.component_placement.actor,env,rollout','runner.logger.experiment_name') for k in diff),diff
assert b.actor.micro_batch_size==32 and b.actor.global_batch_size==1024 and b.algorithm.update_epoch==10
assert b.env.train.total_num_envs==32 and b.env.train.rollout_epoch==1 and b.actor.model.num_steps==4
packet=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-dvac-20260905')
(packet/'inherited-resolved.yaml').write_text(OmegaConf.to_yaml(b,resolve=True))
(packet/'method-config-diff.json').write_text(json.dumps(diff,indent=2))
print('METHOD_CONFIG_DIFF',json.dumps(diff))
print('TEST_AND_CONFIG_VALIDATION_PASSED')
PY
git diff --stat
