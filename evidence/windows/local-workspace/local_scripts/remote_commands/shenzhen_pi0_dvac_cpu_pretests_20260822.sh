#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
model=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
placement='cluster.component_placement={env\,\ rollout:2-3}'

cd "$root"
test "$(git rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test -z "$(git diff --name-only --diff-filter=U)"
test "$(git diff --cached --name-only | wc -l)" -eq 6
git diff --cached --check
test -x "$venv/bin/python"
test -d "$robotwin"
test -d "$model"

export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=''
export NVIDIA_VISIBLE_DEVICES=none
export EMBODIED_PATH="$root/examples/embodiment"
export REPO_PATH="$root"
export ROBOTWIN_PATH="$robotwin"
export ROBOT_PLATFORM=ALOHA
export HYDRA_FULL_ERROR=1

python_files=(
  rlinf/models/embodiment/openpi/openpi_action_model.py
  rlinf/utils/dvac_telemetry.py
  rlinf/workers/env/env_worker.py
  rlinf/workers/rollout/hf/huggingface_worker.py
  tests/unit_tests/test_dvac_telemetry.py
)

"$venv/bin/python" -m ruff check "${python_files[@]}"
"$venv/bin/python" -m ruff format --check "${python_files[@]}"
"$venv/bin/python" -m py_compile "${python_files[@]}"
"$venv/bin/python" -m pytest -q tests/unit_tests/test_dvac_telemetry.py

common_args=(
  --config-path "$root/evaluations/robotwin"
  --config-name robotwin_adjust_bottle_openpi_dvac_eval
  "$placement"
  "runner.logger.log_path=/data/chenyiteng/results/dvac-observation/pretest-not-launched"
  "rollout.model.model_path=$model"
  "env.eval.assets_path=$robotwin"
  env.eval.total_num_envs=2
  env.eval.rollout_epoch=1
  env.eval.max_episode_steps=200
  env.eval.max_steps_per_rollout_epoch=200
)

off_resolved=$("$venv/bin/python" "$root/evaluations/eval_embodied_agent.py" \
  "${common_args[@]}" --cfg job --resolve)
printf '%s\n' "$off_resolved" | grep -Fq 'enabled: false'
printf '%s\n' "$off_resolved" | grep -Fq \
  'common_base_commit: 7d07a4212ee6858cc333e1d4fab7a37256d1f839'

on_resolved=$("$venv/bin/python" "$root/evaluations/eval_embodied_agent.py" \
  "${common_args[@]}" \
  rollout.dvac_telemetry.enabled=true \
  rollout.dvac_telemetry.run_id=pi0-dvac-compose-only \
  rollout.dvac_telemetry.source_commit=working-tree-precommit \
  rollout.dvac_telemetry.seed_file_sha256=compose-only \
  'rollout.dvac_telemetry.launch_command=compose-only' \
  --cfg job --resolve)
printf '%s\n' "$on_resolved" | grep -Fq 'enabled: true'
printf '%s\n' "$on_resolved" | grep -Fq \
  'robotwin_commit: 0008ae6800df9f75fc8de7098bacb01735fd8fd2'
printf '%s\n' "$on_resolved" | grep -Fq 'total_num_envs: 2'

echo 'COMPOSE_OFF_ON_OK=1'
echo 'CUDA_VISIBLE_DEVICES_EMPTY=1'
echo 'STATUS_BEGIN'
git status --short
echo 'STATUS_END'
