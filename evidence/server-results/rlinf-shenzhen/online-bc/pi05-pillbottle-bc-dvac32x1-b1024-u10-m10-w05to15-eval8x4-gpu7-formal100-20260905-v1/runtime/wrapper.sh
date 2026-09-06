#!/usr/bin/env bash
set -u
run_dir=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-w05to15-eval8x4-gpu7-formal100-20260905-v1
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc-dvac
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root:$robotwin"
export REPO_PATH="$root" EMBODIED_PATH="$root/examples/embodiment" ASSETS_PATH="$robotwin"
export PI05_MODEL_PATH=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
export ONLINE_BC_RUN_DIR="$run_dir" RAY_ADDRESS=172.17.0.1:6389 RLINF_CODE_WORKING_DIR="$root"
export TORCHINDUCTOR_COMPILE_THREADS=1
unset CUDA_VISIBLE_DEVICES LD_PRELOAD RLINF_SCENE_FENCE_LIBRARY
cd "$root" || exit 91
date -Is > "$run_dir/started_at.txt"
git rev-parse HEAD > "$run_dir/runtime/source-head.txt"
timeout --signal=TERM --kill-after=180s 172800 "$venv/bin/python" -u \
  "$root/examples/embodiment/train_embodied_agent.py" \
  --config-name robotwin_adjust_bottle_online_bc_openpi \
  +online_bc_model=pi05_sidney +bc_dvac=bounded_half \
  runner.max_epochs=100 runner.val_check_interval=5 runner.save_interval=10 \
  actor.optim.total_training_steps=1000 \
  runner.logger.experiment_name=pi05-pillbottle-bc-dvac-u10-w05to15-formal100-gpu7 \
  > "$run_dir/driver.log" 2>&1 &
child=$!
printf '%s\n' "$child" > "$run_dir/timeout.pid"
wait "$child"
rc=$?
printf '%s\n' "$rc" > "$run_dir/exit_code.txt"
date -Is > "$run_dir/finished_at.txt"
exit "$rc"
