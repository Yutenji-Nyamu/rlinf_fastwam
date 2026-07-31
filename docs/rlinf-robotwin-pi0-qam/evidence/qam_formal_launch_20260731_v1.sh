#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
run_root=/root/autodl-tmp/experiments/qam_formal_20260731_v1
experiment=robotwin_adjust_bottle_qam_formal_20260731_v1
runtime_root=/root/autodl-tmp/experiment_exports/qam_formal_20260731_v1/runtime
monitor=/root/autodl-tmp/qam_resource_monitor_20260731_v1.sh
source_resolved=/root/autodl-tmp/qam_source_resolved_20260731_formal_v1.yaml
formal_resolved=/root/autodl-tmp/qam_formal_resolved_20260731_v1.yaml
resolved_diff=/root/autodl-tmp/qam_source_to_formal_20260731_v1.diff
model_dir=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
norm_file="$model_dir/physical-intelligence/robotwin/norm_stats.json"
model_index="$model_dir/model.safetensors.index.json"
deadline='2026-08-01 09:00:00 CST'

expected_branch=codex/qam-pi0-robotwin
expected_head=4a15699e10971e306ed756dcbbf8aa65632553d5
expected_tree=d86082209c07866c13fef7e9051355cf54e6511c
expected_source_yaml_sha=0aca13bfd8b24c4f08dc867599c9cedc55f0be7c379f822a661ab71a626b112d
expected_source_resolved_sha=45bea3edcd28d9b7d8475ce66fe7ef1cf9533dcbc795a78aa94fd210ad4310b4
expected_formal_resolved_sha=c26133cd7462d7c30d5779b9a6bba224209ec0781ea003fb99cc1d74e7644915
expected_resolved_diff_sha=851dd01876ce4cfbc4893981a360eba9c11fd02bfed504791da91fcf3fb0a07c
expected_norm_sha=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a
expected_model_index_sha=79b9eae15b87f8757471b1040bd27fba4b7731feb302c347bbdc55e4765f0311
expected_monitor_sha=01f0e4087f58a6a6c72e1865cb683639df377bb9bf3a85a54f4ae634bd0282d7

source_yaml="$repo/examples/embodiment/config/robotwin_adjust_bottle_qam_openpi.yaml"

require_sha() {
  local expected=$1
  local path=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if test "$actual" != "$expected"; then
    printf 'SHA_MISMATCH expected=%s actual=%s path=%s\n' \
      "$expected" "$actual" "$path" >&2
    exit 1
  fi
}

test "$(git -C "$repo" branch --show-current)" = "$expected_branch"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$repo" rev-parse 'HEAD^{tree}')" = "$expected_tree"
test -z "$(git -C "$repo" status --porcelain=v1)"
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

nvidia-smi \
  --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  | awk -F, '
      {
        gsub(/ /, "", $2);
        gsub(/ /, "", $3);
        if ($2 > 100 || $3 > 0) {
          printf "PRECHECK_FAIL=gpu_busy row=%s\n", $0 > "/dev/stderr";
          exit 1;
        }
      }
    '

disk_available=$(df -B1 --output=avail /root/autodl-tmp | tail -n 1)
test "$disk_available" -ge 536870912000

require_sha "$expected_source_yaml_sha" "$source_yaml"
require_sha "$expected_source_resolved_sha" "$source_resolved"
require_sha "$expected_formal_resolved_sha" "$formal_resolved"
require_sha "$expected_resolved_diff_sha" "$resolved_diff"
require_sha "$expected_norm_sha" "$norm_file"
require_sha "$expected_model_index_sha" "$model_index"
require_sha "$expected_monitor_sha" "$monitor"
bash -n "$monitor"

deadline_epoch=$(date -d "$deadline" +%s)
wall_limit=$((deadline_epoch - $(date +%s)))
test "$wall_limit" -ge 3600

mkdir -p "$runtime_root"
cp "$source_resolved" "$runtime_root/source_resolved.yaml"
cp "$formal_resolved" "$runtime_root/resolved.yaml"
cp "$resolved_diff" "$runtime_root/source_to_formal.diff"

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
  "checkpoint_interval_outer_cycles": 25,
  "save_full_model_weights": false
}
EOF

cat >"$runtime_root/stop_conditions.txt" <<EOF
The hard deadline is $deadline; timeout terminates the driver at that boundary.
NaN/Inf, CUDA OOM, NCCL/Ray fatal, rank death, or cgroup OOM are fatal.
Outcome mix and action-sensitivity diagnostics are recorded but do not block AM.
The source config is launch-closed; this run is resolved as phase=am_on.
Updates 1-512 are critic-only; logical update 513 starts joint critic+AM.
Checkpoints are periodic every 25 outer cycles; full-weight export is deferred.
EOF

{
  printf 'time\t%s\n' "$(date --iso-8601=seconds)"
  printf 'deadline\t%s\n' "$deadline"
  printf 'wall_limit_seconds\t%s\n' "$wall_limit"
  printf 'host\t%s\n' "$(hostname)"
  printf 'uid\t%s\n' "$(id -u)"
  printf 'repo\t%s\n' "$repo"
  printf 'branch\t%s\n' "$expected_branch"
  printf 'head\t%s\n' "$expected_head"
  printf 'tree\t%s\n' "$expected_tree"
  printf 'source_yaml_sha256\t%s\n' "$expected_source_yaml_sha"
  printf 'source_resolved_sha256\t%s\n' "$expected_source_resolved_sha"
  printf 'formal_resolved_sha256\t%s\n' "$expected_formal_resolved_sha"
  printf 'resolved_diff_sha256\t%s\n' "$expected_resolved_diff_sha"
  printf 'norm_stats_sha256\t%s\n' "$expected_norm_sha"
  printf 'model_index_sha256\t%s\n' "$expected_model_index_sha"
  printf 'monitor_sha256\t%s\n' "$expected_monitor_sha"
  printf 'disk_available_bytes\t%s\n' "$disk_available"
  printf 'cgroup_memory_current\t%s\n' \
    "$(cat /sys/fs/cgroup/memory.current 2>/dev/null || printf unknown)"
  printf 'cgroup_memory_max\t%s\n' \
    "$(cat /sys/fs/cgroup/memory.max 2>/dev/null || printf unknown)"
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
