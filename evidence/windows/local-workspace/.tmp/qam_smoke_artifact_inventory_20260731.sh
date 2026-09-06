set -u

runtime=/root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1/runtime
run_root=/root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1
checkpoint="$run_root/robotwin_adjust_bottle_qam_qonly_smoke_20260731_v1/checkpoints/global_step_1"

echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
echo "PROCESSES"
pgrep -af '[q]am_qonly_smoke_launch_20260731_v1.sh|[t]rain_embodied_agent.py|[r]aylet|[g]cs_server' || true
echo "GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
echo "CGROUP"
cat /sys/fs/cgroup/memory.stat |
  awk '$1=="anon" || $1=="file" || $1=="shmem" {print}'
cat /sys/fs/cgroup/memory.events
echo "DISK"
df -h /root/autodl-tmp
du -sh "$run_root" "$runtime" 2>/dev/null || true
echo "RUNTIME_FILES"
find "$runtime" -maxdepth 2 -type f \
  -printf '%s\t%TY-%Tm-%Td %TH:%TM:%TS\t%p\n' |
  sort -nr
echo "CHECKPOINT_TREE"
if test -d "$checkpoint"; then
  find "$checkpoint" -maxdepth 6 -type f \
    -printf '%s\t%TY-%Tm-%Td %TH:%TM:%TS\t%p\n' |
    sort -nr
else
  echo "MISSING_CHECKPOINT=$checkpoint"
fi
echo "LOG_FAILURE_SCAN"
grep -R -n -Ei \
  'nan|inf|cuda out of memory|oom|nccl.*(error|fail)|ray.*fatal|traceback|exception' \
  "$runtime/driver.log" \
  "$run_root"/*.log \
  2>/dev/null |
  tail -100 || true
echo "LAUNCH_EXIT"
cat "$runtime/exit_code.txt" "$runtime/monitor_exit_code.txt"
cat /root/autodl-tmp/qam_qonly_smoke_launcher_20260731_v1.log

