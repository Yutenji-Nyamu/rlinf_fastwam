set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$FW:$ROBOTWIN"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA

git -C "$WT" status --short
git -C "$WT" diff --check
"$VENV/bin/python" -m py_compile \
  "$WT/rlinf/models/embodiment/fastwam/builder.py" \
  "$WT/rlinf/models/embodiment/fastwam/robotwin_adapter.py" \
  "$WT/rlinf/models/embodiment/fastwam/fastwam_rl.py" \
  "$WT/rlinf/models/embodiment/fastwam/fastwam_policy.py"
"$VENV/bin/python" - <<'PY'
import importlib
import yaml

mods = [
    "rlinf.models.embodiment.fastwam.builder",
    "rlinf.models.embodiment.fastwam.fastwam_rl",
    "rlinf.models.embodiment.fastwam.fastwam_policy",
]
for name in mods:
    importlib.import_module(name)
for path in [
    "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo/examples/embodiment/config/model/fastwam_robotwin.yaml",
    "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo/examples/embodiment/config/env/robotwin_move_stapler_pad.yaml",
    "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo/examples/embodiment/config/robotwin_move_stapler_pad_grpo_fastwam.yaml",
]:
    with open(path, encoding="utf-8") as handle:
        yaml.safe_load(handle)
print("STATIC_IMPORT_YAML_OK")
PY

cd "$WT"
"$VENV/bin/python" examples/embodiment/train_embodied_agent.py \
  --config-path "$WT/examples/embodiment/config" \
  --config-name robotwin_move_stapler_pad_grpo_fastwam \
  --cfg job --resolve \
  runner.logger.log_path=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/probes/compose \
  runner.max_steps=1 \
  runner.val_check_interval=1 \
  runner.save_interval=1 \
  env.train.rollout_epoch=1 \
  actor.global_batch_size=256 \
  algorithm.update_epoch=1
