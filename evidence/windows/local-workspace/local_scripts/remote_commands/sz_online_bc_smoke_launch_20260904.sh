#!/usr/bin/env bash
# Prepared launch contract. Do not run until the user approves this smoke budget.
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
run_dir=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke-v1
test "$(git -C "$root" branch --show-current)" = codex/sz-pi0-online-bc
test -d "$robotwin/assets"
test ! -e "$run_dir"
# Check GPU3 ownership again immediately before invoking this prepared script.
mkdir "$run_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root:$robotwin"
export REPO_PATH="$root"
export EMBODIED_PATH="$root/examples/embodiment"
export ASSETS_PATH="$robotwin/assets"
export PI0_MODEL_PATH=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
export ONLINE_BC_RUN_DIR="$run_dir"
export RAY_ADDRESS=172.17.0.1:6389
export RLINF_CODE_WORKING_DIR="$root"
export TORCHINDUCTOR_COMPILE_THREADS=1
cd "$root"
nohup "$venv/bin/python" -u "$root/examples/embodiment/train_embodied_agent.py" \
  --config-name robotwin_adjust_bottle_online_bc_openpi \
  > "$run_dir/driver.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run_dir/driver.pid"
printf 'Started only this BC driver: %s\n' "$(cat "$run_dir/driver.pid")"
