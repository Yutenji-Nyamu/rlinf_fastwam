set -u

targets=(
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260718_013332-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260719_120513-robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke
  /root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1
  /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40
)

echo "AUDIT_TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
df -h /root/autodl-tmp

for path in "${targets[@]}"; do
  echo
  echo "TARGET=$path"
  if ! test -d "$path"; then
    echo "STATUS=MISSING"
    continue
  fi
  stat --format='MTIME=%y' "$path"
  du -sx --block-size=1 "$path" | awk '{print "TOTAL_BYTES="$1}'
  echo "TOP_CHILDREN_BYTES"
  du -x --block-size=1 --max-depth=3 "$path" 2>/dev/null |
    sort -nr |
    head -35
  echo "CHECKPOINT_DIRS"
  find "$path" -xdev -type d -name 'global_step_*' -print | sort
  echo "LARGEST_FILES"
  find "$path" -xdev -type f \
    -printf '%s\t%TY-%Tm-%Td %TH:%TM:%TS\t%p\n' 2>/dev/null |
    sort -nr |
    head -25
  echo "CAPSULE_COUNTS"
  printf 'mp4='
  find "$path" -xdev -type f -name '*.mp4' -printf x 2>/dev/null | wc -c
  printf 'distcp='
  find "$path" -xdev -type f -name '*.distcp' -printf x 2>/dev/null | wc -c
  printf 'full_weights='
  find "$path" -xdev -type f -name 'full_weights.pt' -printf x 2>/dev/null | wc -c
  printf 'resolved_yaml='
  find "$path" -xdev -type f -iname '*resolved*.yaml' -printf x 2>/dev/null | wc -c
  printf 'logs='
  find "$path" -xdev -type f -name '*.log' -printf x 2>/dev/null | wc -c
done

backup=/root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40
echo
echo "BACKUP_TOP_LEVEL"
find "$backup" -mindepth 1 -maxdepth 1 \
  -printf '%y\t%TY-%Tm-%Td %TH:%TM:%TS\t%p\n' | sort
echo "BACKUP_VENVS_BYTES"
find "$backup" -mindepth 1 -maxdepth 1 -type d -name '.venv*' \
  -exec du -sx --block-size=1 {} + | sort -nr
echo "BACKUP_LOG_RUNS_BYTES"
du -x --block-size=1 --max-depth=1 "$backup/logs" 2>/dev/null |
  sort -nr |
  head -30
echo "BACKUP_GIT_STATUS"
git -C "$backup" rev-parse --is-inside-work-tree 2>&1 || true

