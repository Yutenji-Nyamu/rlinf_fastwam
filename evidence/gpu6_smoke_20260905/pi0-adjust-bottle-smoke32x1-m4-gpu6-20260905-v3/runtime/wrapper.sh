#!/usr/bin/env bash
set -u
run_dir=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v3
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root:$robotwin"
export REPO_PATH="$root"
export EMBODIED_PATH="$root/examples/embodiment"
export ASSETS_PATH="$robotwin"
export PI0_MODEL_PATH=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
export ONLINE_BC_RUN_DIR="$run_dir"
export RAY_ADDRESS=172.17.0.1:6389
export RLINF_CODE_WORKING_DIR="$root"
export TORCHINDUCTOR_COMPILE_THREADS=1
unset CUDA_VISIBLE_DEVICES LD_PRELOAD RLINF_SCENE_FENCE_LIBRARY
cd "$root" || exit 91
date -Is > "$run_dir/started_at.txt"
timeout --signal=TERM --kill-after=180s 5400s "$venv/bin/python" -u \
  "$root/examples/embodiment/train_embodied_agent.py" \
  --config-name robotwin_adjust_bottle_online_bc_openpi \
  > "$run_dir/driver.log" 2>&1 &
child=$!
printf '%s\n' "$child" > "$run_dir/timeout.pid"
wait "$child"
rc=$?
printf '%s\n' "$rc" > "$run_dir/exit_code.txt"
date -Is > "$run_dir/finished_at.txt"
exit "$rc"
