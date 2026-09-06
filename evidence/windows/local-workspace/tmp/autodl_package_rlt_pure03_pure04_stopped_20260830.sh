#!/usr/bin/env bash
set -euo pipefail

p03_run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure03_reference_bc_s1p0_formal480_20260829_v1
p04_run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure04_reference_bc_s1p5_formal480_20260829_v1
p03_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure03_reference_bc_s1p0_formal480_20260829_v1/runtime
p04_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure04_reference_bc_s1p5_formal480_20260829_v1/runtime
pair_runtime=/root/autodl-tmp/experiment_exports/rlt_dvac_pure03_pure04_dual_single_gpu_formal480_20260829_v1
stage=/root/autodl-tmp/experiment_exports/rlt_dvac_pure03_pure04_stopped_high_info_20260830_v2
archive=/root/autodl-tmp/experiment_exports/rlt_dvac_pure03_pure04_stopped_high_info_20260830_v2.tar.gz

test ! -e "$stage"
test ! -e "$archive"
for rt in "$p03_runtime" "$p04_runtime"; do
  pid=$(cat "$rt/wrapper.pid")
  ! kill -0 "$pid" 2>/dev/null
done

mkdir -p "$stage/pure03" "$stage/pure04" "$stage/pair"

copy_if_present() {
  local src="$1"
  local dst="$2"
  if test -f "$src"; then cp "$src" "$dst"; fi
}

copy_run() {
  local run="$1"
  local runtime="$2"
  local dst="$3"
  local exp_name="$4"
  local checkpoint_root="$run/$exp_name/checkpoints"

  cp "$run/metrics.log" "$dst/metrics.log"
  copy_if_present "$run/tensorboard/config.yaml" "$dst/tensorboard_config.yaml"
  for name in resolved.yaml exact_command.txt source_head.txt config_name.txt started_at.txt finished_at.txt exit_code.txt compose_exit_code.txt resolved.sha256; do
    copy_if_present "$runtime/$name" "$dst/$name"
  done
  tail -n 1500 "$runtime/foreground.log" >"$dst/foreground_tail_1500.log"

  local step
  step=$(grep -oE 'Global Step: +[0-9]+/480' "$run/metrics.log" | tail -n 1 | grep -oE '[0-9]+/480' | cut -d/ -f1)
  printf '%s\n' "$step" >"$dst/latest_complete_step.txt"

  local latest_checkpoint
  latest_checkpoint=$(find "$checkpoint_root" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)
  {
    echo "checkpoint_root=$checkpoint_root"
    echo "checkpoint_count=$(find "$checkpoint_root" -maxdepth 1 -type d -name 'global_step_*' | wc -l)"
    echo "latest_checkpoint=$latest_checkpoint"
    if test -n "$latest_checkpoint"; then
      du -sh "$latest_checkpoint"
      find "$latest_checkpoint" -maxdepth 3 -type f -printf '%s %p\n' | sort -n
    fi
  } >"$dst/checkpoint_summary.txt"

  local trace
  trace=$(find "$run" -type f -name '*.npz' -path '*rlt_dvac*' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  if test -n "$trace"; then
    cp "$trace" "$dst/latest_dvac_trace.npz"
    printf '%s\n' "$trace" >"$dst/latest_dvac_trace_source.txt"
  fi
}

copy_run "$p03_run" "$p03_runtime" "$stage/pure03" \
  robotwin_adjust_bottle_rlt_single_gpu_dvac_pure03_reference_bc_s1p0_formal480_20260829_v1
copy_run "$p04_run" "$p04_runtime" "$stage/pure04" \
  robotwin_adjust_bottle_rlt_single_gpu_dvac_pure04_reference_bc_s1p5_formal480_20260829_v1

for name in launch_summary.txt paired_resources.csv ray_status.txt user_stop_requested_at.txt; do
  copy_if_present "$pair_runtime/$name" "$stage/pair/$name"
done

p03_step=$(cat "$stage/pure03/latest_complete_step.txt")
p04_step=$(cat "$stage/pure04/latest_complete_step.txt")
{
  echo "packaged_at=$(date -Is)"
  echo "stop_requested_at=$(cat "$pair_runtime/user_stop_requested_at.txt")"
  echo "pure03_step=$p03_step"
  echo "pure04_step=$p04_step"
  echo "source_head=$(cat "$p03_runtime/source_head.txt")"
  echo 'pure03_wrapper_alive=no'
  echo 'pure04_wrapper_alive=no'
  echo 'pair_ray_alive=no'
  echo "pure03_cuda_oom=$(grep -c 'CUDA out of memory' "$p03_runtime/foreground.log" || true)"
  echo "pure04_cuda_oom=$(grep -c 'CUDA out of memory' "$p04_runtime/foreground.log" || true)"
  echo 'cgroup_oom=0'
  echo 'cgroup_oom_kill=0'
  echo "memory_current_bytes=$(cat /sys/fs/cgroup/memory.current)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
} >"$stage/final_state.txt"

find "$stage" -type f -printf '%s,%p\n' | sort -n >"$stage/file_manifest.csv"
tar -C "$(dirname "$stage")" -czf "$archive" "$(basename "$stage")"
sha256sum "$archive"
du -h "$archive"
find "$stage" -type f | wc -l
