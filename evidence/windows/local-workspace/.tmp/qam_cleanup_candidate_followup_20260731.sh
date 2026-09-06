set -u

backup=/root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40
grpo=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260718_013332-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke
ppo=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260719_120513-robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke

echo "AUDIT_TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"

echo "BACKUP_GIT_RESOLUTION"
git -C "$backup" rev-parse --show-toplevel 2>&1 || true
git -C "$backup" rev-parse --absolute-git-dir 2>&1 || true
test -e "$backup/.git" && ls -ld "$backup/.git" || echo "NO_OWN_DOT_GIT"

echo "BACKUP_HISTORY_RUNS_BYTES"
du -x --block-size=1 --max-depth=1 "$backup/logs/logs_his" 2>/dev/null |
  sort -nr |
  head -35

echo "BACKUP_PRIMARY_PPO_CHECKPOINTS_BYTES"
ppo_root="$backup/logs/20260713_124420-robotwin_adjust_bottle_ppo_openpi_a800_2gpu_baseline-step0-to-60/robotwin_ppo_openpi_a800_2gpu_baseline/checkpoints"
du -x --block-size=1 --max-depth=1 "$ppo_root" 2>/dev/null | sort -nr
find "$ppo_root" -xdev -type f \
  \( -name '*.distcp' -o -name 'full_weights.pt' -o -name '.metadata' \) \
  -printf '%s\t%p\n' 2>/dev/null |
  sort -nr

for run in "$grpo" "$ppo"; do
  echo
  echo "RUN=$run"
  echo "COMMAND"
  sed -n '1,10p' "$run/command.txt"
  echo "METRICS"
  sed -n '1,160p' "$run/metrics.log"
  echo "RUN_LOG_KEY_LINES"
  grep -Ei \
    'global_step|checkpoint|success|reward|loss|oom|nan|fatal|traceback|error|peak|finished|complete' \
    "$run/run_embodiment.log" |
    tail -80 || true
  echo "RESOURCE_PEAK"
  cat "$run/resource_monitor/peak.txt" 2>/dev/null || true
done

