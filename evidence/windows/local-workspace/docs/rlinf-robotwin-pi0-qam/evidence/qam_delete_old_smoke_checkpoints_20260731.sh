#!/usr/bin/env bash
set -euo pipefail

audit=/root/autodl-tmp/experiment_exports/qam_old_smoke_checkpoint_cleanup_20260731.txt
mkdir -p "$(dirname "$audit")"
exec > >(tee "$audit") 2>&1

roots=(
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260718_013332-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260719_120513-robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke
  /root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1
)
targets=(
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke/checkpoints
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260718_013332-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke/robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke/checkpoints
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260719_120513-robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke/robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke/checkpoints
  /root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1/s1a/robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1/checkpoints
)

echo "START_TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
echo "HOST=$(hostname) UID=$(id -u)"
echo "DISK_BEFORE"
df -B1 /root/autodl-tmp

total_bytes=0
for index in "${!targets[@]}"; do
  root=${roots[$index]}
  target=${targets[$index]}
  resolved_root=$(readlink -f -- "$root")
  resolved_target=$(readlink -f -- "$target")
  test "$resolved_root" = "$root"
  test "$resolved_target" = "$target"
  test -d "$root"
  test -d "$target"
  test ! -L "$root"
  test ! -L "$target"
  case "$target" in
    "$root"/*/checkpoints) ;;
    *)
      echo "INVALID_TARGET root=$root target=$target" >&2
      exit 1
      ;;
  esac
  case "$root" in
    *smoke*) ;;
    *)
      echo "NON_SMOKE_ROOT=$root" >&2
      exit 1
      ;;
  esac
  bytes=$(du -sx --block-size=1 "$target" | awk '{print $1}')
  total_bytes=$((total_bytes + bytes))
  echo "VERIFIED index=$index bytes=$bytes root=$root target=$target"
  find "$target" -maxdepth 2 -type d -name 'global_step_*' -print | sort
done

echo "TOTAL_CHECKPOINT_BYTES_BEFORE=$total_bytes"

for index in "${!targets[@]}"; do
  target=${targets[$index]}
  echo "DELETE_BEGIN index=$index target=$target"
  rm -rf -- "$target"
  test ! -e "$target"
  echo "DELETE_DONE index=$index target=$target"
done

echo "RETAINED_ROOTS"
for index in "${!roots[@]}"; do
  root=${roots[$index]}
  target=${targets[$index]}
  test -d "$root"
  test ! -e "$target"
  du -sx --block-size=1 "$root"
  find "$root" -maxdepth 2 -type f -printf '%s\t%p\n' |
    sort -nr |
    head -20
done

echo "FORMAL_GUARDS"
for path in \
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1 \
  /root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1; do
  test -d "$path"
  echo "PRESENT $path"
done

echo "DISK_AFTER"
df -B1 /root/autodl-tmp
echo "END_TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
