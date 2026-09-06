set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
cd "$WT"
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
"$VENV/bin/python" evaluations/eval_embodied_agent.py \
  --config-path "$WT/examples/embodiment/config" \
  --config-name robotwin_move_stapler_pad_grpo_openpi_pi05_sidney \
  '~cluster.component_placement' \
  '+cluster.component_placement={env\, rollout:"4"}' \
  runner.task_type=embodied_eval runner.only_eval=true \
  runner.val_check_interval=-1 runner.save_interval=-1 \
  --cfg job --resolve | sed -n '/^cluster:/,/^runner:/p'
