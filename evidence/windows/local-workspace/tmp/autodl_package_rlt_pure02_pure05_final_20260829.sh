#!/usr/bin/env bash
set -euo pipefail

s05_run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1
s20_run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1
s05_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1/runtime
s20_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1/runtime
pair_runtime=/root/autodl-tmp/experiment_exports/rlt_dvac_pure_dual_single_gpu_formal480_20260828_v1
stage=/root/autodl-tmp/experiment_exports/rlt_dvac_pure02_pure05_final480_high_info_20260829_v1
archive=/root/autodl-tmp/experiment_exports/rlt_dvac_pure02_pure05_final480_high_info_20260829_v1.tar.gz

test ! -e "$stage"
test ! -e "$archive"
test "$(cat "$s05_runtime/exit_code.txt")" = 0
test "$(cat "$s20_runtime/exit_code.txt")" = 0
grep -q 'Global Step:  480/480' "$s05_run/metrics.log"
grep -q 'Global Step:  480/480' "$s20_run/metrics.log"

mkdir -p "$stage/pure02" "$stage/pure05" "$stage/pair"

copy_run() {
  local run="$1"
  local runtime="$2"
  local dst="$3"
  local exp_name="$4"
  local checkpoint_root="$run/$exp_name/checkpoints"

  cp "$run/metrics.log" "$dst/metrics.log"
  cp "$run/tensorboard/config.yaml" "$dst/tensorboard_config.yaml"
  for name in resolved.yaml exact_command.txt source_head.txt config_name.txt started_at.txt finished_at.txt exit_code.txt compose_exit_code.txt resolved.sha256; do
    cp "$runtime/$name" "$dst/$name"
  done
  tail -n 1500 "$runtime/foreground.log" >"$dst/foreground_tail_1500.log"
  printf '480\n' >"$dst/latest_checkpoint_step.txt"
  {
    echo "checkpoint_root=$checkpoint_root"
    echo "checkpoint_count=$(find "$checkpoint_root" -maxdepth 1 -type d -name 'global_step_*' | wc -l)"
    echo "latest_checkpoint=$checkpoint_root/global_step_480"
    du -sh "$checkpoint_root/global_step_480"
    find "$checkpoint_root/global_step_480" -maxdepth 3 -type f -printf '%s %p\n' | sort -n
  } >"$dst/checkpoint_summary.txt"
}

copy_run \
  "$s05_run" "$s05_runtime" "$stage/pure02" \
  robotwin_adjust_bottle_rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1
copy_run \
  "$s20_run" "$s20_runtime" "$stage/pure05" \
  robotwin_adjust_bottle_rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1

for name in launch_summary.txt paired_resources.csv cleanup_finished_at.txt ray_status.txt; do
  cp "$pair_runtime/$name" "$stage/pair/$name"
done

{
  echo "packaged_at=$(date -Is)"
  echo 'pure02_step=480'
  echo 'pure02_exit_code=0'
  echo 'pure05_step=480'
  echo 'pure05_exit_code=0'
  echo "source_head=$(cat "$s05_runtime/source_head.txt")"
  echo 'oom=0'
  echo 'oom_kill=0'
} >"$stage/final_state.txt"

tar -C "$(dirname "$stage")" -czf "$archive" "$(basename "$stage")"
sha256sum "$archive"
du -h "$archive"
find "$stage" -type f | wc -l
