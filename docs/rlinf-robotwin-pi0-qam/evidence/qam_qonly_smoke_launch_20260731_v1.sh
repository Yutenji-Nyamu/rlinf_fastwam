#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
run_root=/root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1
experiment=robotwin_adjust_bottle_qam_qonly_smoke_20260731_v1
runtime_root=/root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1/runtime
monitor=/root/autodl-tmp/qam_resource_monitor_20260731_v1.sh
source_resolved=/root/autodl-tmp/qam_source_resolved_20260731_v1.yaml
smoke_resolved=/root/autodl-tmp/qam_qonly_smoke_resolved_20260731_v1.yaml
resolved_diff=/root/autodl-tmp/qam_source_to_qonly_smoke_20260731_v1.diff
model_dir=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
norm_file="$model_dir/physical-intelligence/robotwin/norm_stats.json"
model_index="$model_dir/model.safetensors.index.json"

expected_branch=codex/qam-pi0-robotwin
expected_head=7bc5f87086035087adf6d44ddda76eb5a9e54ee8
expected_tree=6dc9124ba63b5712918ba2dbdcffde203cfb5eed
expected_source_yaml_sha=d3da1b66d24233300e2a5cebebdf9cb9bcb9e17db959c72e9efcf85dcff1cc6f
expected_source_resolved_sha=ae1faf2e177f6ca5abce17a27056c191c11aaaf0c0d30a6063cb14f17f0dfdfd
expected_smoke_resolved_sha=ce8661de889992357f473068e481ac5b6c56f44fb9eddeaee3e20858db9cefee
expected_resolved_diff_sha=ea19fd759f653f2ee924ca45a3a67524a6a9d5e9dc6a34779f3396c56a37c998
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
test "$disk_available" -ge 107374182400

require_sha "$expected_source_yaml_sha" "$source_yaml"
require_sha "$expected_source_resolved_sha" "$source_resolved"
require_sha "$expected_smoke_resolved_sha" "$smoke_resolved"
require_sha "$expected_resolved_diff_sha" "$resolved_diff"
require_sha "$expected_norm_sha" "$norm_file"
require_sha "$expected_model_index_sha" "$model_index"
require_sha "$expected_monitor_sha" "$monitor"
bash -n "$monitor"

mkdir -p "$runtime_root"
cp "$source_resolved" "$runtime_root/source_resolved.yaml"
cp "$smoke_resolved" "$runtime_root/resolved.yaml"
cp "$resolved_diff" "$runtime_root/source_to_smoke.diff"

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
  runner.max_steps=1
  runner.save_interval=1
  runner.resume_dir=null
  runner.ckpt_path=null
  algorithm.qam.phase=q_only
  algorithm.qam.warmup_global_inserts=2
  algorithm.qam.min_replay_per_rank=1
  algorithm.qam.max_updates_per_step=2
  actor.global_batch_size=2
  actor.micro_batch_size=1
  +actor.fsdp_config.save_full_model_weights=false
)

{
  printf 'cd %q\n' "$repo"
  printf 'export PYTHONPATH=%q\n' "$PYTHONPATH"
  printf 'export EMBODIED_PATH=%q\n' "$EMBODIED_PATH"
  printf 'export REPO_PATH=%q\n' "$REPO_PATH"
  printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$CUDA_VISIBLE_DEVICES"
  printf 'timeout --signal=TERM --kill-after=180s 7200s'
  printf ' %q' "${command[@]}"
  printf '\n'
} >"$runtime_root/exact_command.txt"

printf '%s\n' \
  '{' \
  '  "outer_cycles": 1,' \
  '  "train_episodes": 2,' \
  '  "requested_action_slots_max": 400,' \
  '  "global_macro_inserts_expected_min": 2,' \
  '  "global_macro_inserts_expected_max": 20,' \
  '  "critic_updates_exact": 2,' \
  '  "fine_updates_exact": 0,' \
  '  "eval_episodes": 0,' \
  '  "checkpoints": 1,' \
  '  "wall_clock_hard_limit_seconds": 7200,' \
  '  "gpu_hours_hard_limit": 4,' \
  '  "gpu_memory_mib_per_card_stop": 61440,' \
  '  "host_anon_gib_stop": 180,' \
  '  "new_disk_gib_stop": 25' \
  '}' >"$runtime_root/budget.json"

printf '%s\n' \
  'Stop on any NaN/Inf in loss, Q, TD target, gradient, or action.' \
  'Stop on CUDA OOM, NCCL/Ray fatal, rank death, or cgroup OOM/oom_kill increment.' \
  'Stop if either GPU memory exceeds 60 GiB, host anon exceeds 180 GiB, or new disk exceeds 25 GiB.' \
  'Stop if no new driver progress is logged for 30 minutes after initialization.' \
  'Stop if QAM payload/contract validation fails, frozen behavior changes, or DCP/sidecars are incomplete.' \
  'The hard timeout is 7200 seconds; this smoke never enables AM.' \
  >"$runtime_root/stop_conditions.txt"

{
  printf 'time\t%s\n' "$(date --iso-8601=seconds)"
  printf 'host\t%s\n' "$(hostname)"
  printf 'uid\t%s\n' "$(id -u)"
  printf 'repo\t%s\n' "$repo"
  printf 'branch\t%s\n' "$expected_branch"
  printf 'head\t%s\n' "$expected_head"
  printf 'tree\t%s\n' "$expected_tree"
  printf 'source_yaml_sha256\t%s\n' "$expected_source_yaml_sha"
  printf 'source_resolved_sha256\t%s\n' "$expected_source_resolved_sha"
  printf 'smoke_resolved_sha256\t%s\n' "$expected_smoke_resolved_sha"
  printf 'norm_stats_sha256\t%s\n' "$expected_norm_sha"
  printf 'model_index_sha256\t%s\n' "$expected_model_index_sha"
  printf 'monitor_sha256\t%s\n' "$expected_monitor_sha"
  printf 'disk_available_bytes\t%s\n' "$disk_available"
} >"$runtime_root/run_provenance.tsv"

cd "$repo"
set +e
timeout --signal=TERM --kill-after=180s 7200s \
  "${command[@]}" >"$runtime_root/driver.log" 2>&1 &
driver_pid=$!
set -e
printf '%s\n' "$driver_pid" >"$runtime_root/driver.pid"

"$monitor" "$driver_pid" "$runtime_root/resources.csv" 2 \
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
printf 'QAM_QONLY_SMOKE_EXIT=%s MONITOR_EXIT=%s\n' "$status" "$monitor_status"
exit "$status"
