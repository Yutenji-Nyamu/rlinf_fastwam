set -u

runtime=/root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1/runtime
run_root=/root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1
launcher_log=/root/autodl-tmp/qam_qonly_smoke_launcher_20260731_v1.log

echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
echo "PROCESSES"
pgrep -af '[q]am_qonly_smoke_launch_20260731_v1.sh|[t]rain_embodied_agent.py|[r]aylet|[g]cs_server' || true
echo "GPU"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
echo "CGROUP"
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.stat |
  awk '$1=="anon" || $1=="file" || $1=="shmem" {print}'
cat /sys/fs/cgroup/memory.events
echo "DISK"
df -h /root/autodl-tmp
du -sh "$run_root" "$runtime" 2>/dev/null || true
echo "LAUNCHER_LOG"
tail -60 "$launcher_log" 2>/dev/null || true
echo "DRIVER_LOG"
tail -100 "$runtime/driver.log" 2>/dev/null || true
echo "FILES"
find "$runtime" -maxdepth 1 -type f \
  -printf '%s\t%TY-%Tm-%Td %TH:%TM:%TS\t%f\n' 2>/dev/null |
  sort

