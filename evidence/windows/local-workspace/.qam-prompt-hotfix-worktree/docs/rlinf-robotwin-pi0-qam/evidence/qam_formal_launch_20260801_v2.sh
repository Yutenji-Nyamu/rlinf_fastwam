#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
run_root=/root/autodl-tmp/experiments/qam_formal_20260801_v2
experiment=robotwin_adjust_bottle_qam_formal_20260801_v2
runtime_root=/root/autodl-tmp/experiment_exports/qam_formal_20260801_v2/runtime
monitor=/root/autodl-tmp/qam_resource_monitor_20260731_v1.sh
approved_base=/root/autodl-tmp/qam_formal_resolved_20260731_v1.yaml
model_dir=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
deadline='2026-08-01 09:00:00 CST'

expected_head=d5f6d7d1da0fc355a71ca653be027282cad040d2
expected_tree=d63c2a038ee6d6d0413e42dffd3a523b91dbba76
expected_config_sha=0aca13bfd8b24c4f08dc867599c9cedc55f0be7c379f822a661ab71a626b112d
expected_monitor_sha=01f0e4087f58a6a6c72e1865cb683639df377bb9bf3a85a54f4ae634bd0282d7

source_config="$repo/examples/embodiment/config/robotwin_adjust_bottle_qam_openpi.yaml"
test "$(git -C "$repo" branch --show-current)" = codex/qam-pi0-robotwin
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$repo" rev-parse 'HEAD^{tree}')" = "$expected_tree"
test -z "$(git -C "$repo" status --short)"
test "$(sha256sum "$source_config" | awk '{print $1}')" = "$expected_config_sha"
test "$(sha256sum "$monitor" | awk '{print $1}')" = "$expected_monitor_sha"
test -f "$approved_base"
test -f "$model_dir/model.safetensors.index.json"
test -f "$model_dir/physical-intelligence/robotwin/norm_stats.json"
test ! -e "$run_root"
test ! -e "$runtime_root"
if pgrep -x raylet >/dev/null || pgrep -x gcs_server >/dev/null; then
  printf 'PRECHECK_FAIL=existing_ray\n' >&2
  exit 1
fi
if pgrep -af '[t]rain_embodied_agent.py|[t]orch.distributed.run' >/dev/null; then
  printf 'PRECHECK_FAIL=existing_training\n' >&2
  exit 1
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F, '
    {gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>100 || $2>0) exit 1}'
test "$(df -B1 --output=avail /root/autodl-tmp | tail -n 1)" -ge 536870912000

deadline_epoch=$(date -d "$deadline" +%s)
wall_limit=$((deadline_epoch - $(date +%s)))
test "$wall_limit" -ge 3600
mkdir -p "$runtime_root"
cp "$approved_base" "$runtime_root/approved_base_resolved.yaml"

export PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin_RLinf"
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

command=(
  "$venv/bin/python"
  -B
  examples/embodiment/train_embodied_agent.py
  --config-path "$repo/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_qam_openpi
  "runner.logger.log_path=$run_root"
  "runner.logger.experiment_name=$experiment"
  runner.max_steps=500
  runner.save_interval=25
  runner.resume_dir=null
  runner.ckpt_path=null
  algorithm.qam.phase=am_on
  algorithm.qam.inv_temp=1.0
  algorithm.qam.warmup_global_inserts=512
  algorithm.qam.q_only_updates_before_am=512
  algorithm.qam.min_replay_per_rank=32
  algorithm.qam.max_updates_per_step=32
  actor.global_batch_size=64
  actor.micro_batch_size=32
  +actor.fsdp_config.save_full_model_weights=false
)

{
  printf 'cd %q\n' "$repo"
  printf 'export PYTHONPATH=%q\n' "$PYTHONPATH"
  printf 'export EMBODIED_PATH=%q\n' "$EMBODIED_PATH"
  printf 'export REPO_PATH=%q\n' "$REPO_PATH"
  printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$CUDA_VISIBLE_DEVICES"
  printf 'timeout --signal=TERM --kill-after=180s %qs' "$wall_limit"
  printf ' %q' "${command[@]}"
  printf '\n'
} >"$runtime_root/exact_command.txt"

cat >"$runtime_root/budget.json" <<EOF
{
  "runner_outer_cycle_ceiling": 500,
  "train_envs": 2,
  "global_macro_warmup": 512,
  "critic_only_updates": 512,
  "first_joint_logical_update": 513,
  "utd_ratio": 1.0,
  "global_batch_size": 64,
  "local_batch_size": 32,
  "inv_temp": 1.0,
  "deadline": "$deadline",
  "wall_clock_hard_limit_seconds": $wall_limit,
  "checkpoint_interval_outer_cycles": 25
}
EOF

{
  printf 'time\t%s\n' "$(date --iso-8601=seconds)"
  printf 'deadline\t%s\n' "$deadline"
  printf 'wall_limit_seconds\t%s\n' "$wall_limit"
  printf 'host\t%s\n' "$(hostname)"
  printf 'head\t%s\n' "$expected_head"
  printf 'tree\t%s\n' "$expected_tree"
  printf 'config_sha256\t%s\n' "$expected_config_sha"
  printf 'disk_available_bytes\t%s\n' \
    "$(df -B1 --output=avail /root/autodl-tmp | tail -n 1)"
} >"$runtime_root/run_provenance.tsv"

cd "$repo"
set +e
timeout --signal=TERM --kill-after=180s "${wall_limit}s" \
  "${command[@]}" >"$runtime_root/driver.log" 2>&1 &
driver_pid=$!
set -e
printf '%s\n' "$driver_pid" >"$runtime_root/driver.pid"
bash "$monitor" "$driver_pid" "$runtime_root/resources.csv" 10 \
  >"$runtime_root/monitor.log" 2>&1 &
monitor_pid=$!
printf '%s\n' "$monitor_pid" >"$runtime_root/monitor.pid"

set +e
wait "$driver_pid"
status=$?
wait "$monitor_pid"
monitor_status=$?
set -e
printf '%s\n' "$status" >"$runtime_root/exit_code.txt"
printf '%s\n' "$monitor_status" >"$runtime_root/monitor_exit_code.txt"
printf 'QAM_FORMAL_EXIT=%s MONITOR_EXIT=%s\n' "$status" "$monitor_status"
exit "$status"
