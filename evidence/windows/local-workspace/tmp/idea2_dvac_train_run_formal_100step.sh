#!/usr/bin/env bash
set -u

run_name=idea2_dvac_apply_formal_100step_2gpu16env_20260821
run_dir="/root/autodl-tmp/idea2_dvac_train_runs/$run_name"
runtime_dir="/root/autodl-tmp/idea2_dvac_train_runtime/$run_name"
source_root=/root/autodl-tmp/RLinf_idea2_dvac_train
robotwin_root=/root/autodl-tmp/idea2_dvac_train_wamppo/RoboTwin_RLinf
python_bin=/root/autodl-tmp/RLinf/.venv/bin/python
observer_script="$runtime_dir/observe_resources.sh"

test ! -e "$run_dir" || {
  printf 'REFUSING_NONEMPTY_OR_EXISTING_RUN_DIR=%s\n' "$run_dir" >&2
  exit 20
}
mkdir -p "$runtime_dir"

export CUDA_VISIBLE_DEVICES=0,1
export EMBODIED_PATH="$source_root/examples/embodiment"
export REPO_PATH="$source_root"
export ROBOTWIN_PATH="$robotwin_root"
export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$source_root:$robotwin_root${PYTHONPATH:+:$PYTHONPATH}"

printf '%s\n' \
  "$python_bin $source_root/examples/embodiment/train_embodied_agent.py --config-path $source_root/examples/embodiment/config/ --config-name robotwin_adjust_bottle_grpo_openpi_dvac_train_100step_formal" \
  > "$runtime_dir/launch_command.txt"
date --iso-8601=seconds > "$runtime_dir/launch_started_at.txt"

"$python_bin" "$source_root/examples/embodiment/train_embodied_agent.py" \
  --config-path "$source_root/examples/embodiment/config/" \
  --config-name robotwin_adjust_bottle_grpo_openpi_dvac_train_100step_formal \
  > "$runtime_dir/driver.log" 2>&1 &
driver_pid=$!
printf '%s\n' "$driver_pid" > "$runtime_dir/driver.pid"

bash "$observer_script" "$driver_pid" "$runtime_dir/resource_monitor" \
  > "$runtime_dir/observer.log" 2>&1 &
observer_pid=$!
printf '%s\n' "$observer_pid" > "$runtime_dir/observer.pid"

wait "$driver_pid"
driver_rc=$?
wait "$observer_pid"
observer_rc=$?

printf '%s\n' "$driver_rc" > "$runtime_dir/driver.exitcode"
printf '%s\n' "$observer_rc" > "$runtime_dir/observer.exitcode"
date --iso-8601=seconds > "$runtime_dir/launch_finished_at.txt"
exit "$driver_rc"
