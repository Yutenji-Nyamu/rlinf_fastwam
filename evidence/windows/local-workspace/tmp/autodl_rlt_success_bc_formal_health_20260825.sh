set -u
pair=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_formal480_20260825_v1
control=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_formal480_20260825_v1/runtime
method=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v1/runtime
ray_cli=/root/autodl-tmp/RLinf/.venv/bin/ray

echo '== time and launch =='
date -Is
cat "$pair/launch_summary.txt" 2>/dev/null || true

echo '== process groups =='
for path in "$pair/ray_head.pid" "$control/wrapper.pid" "$method/wrapper.pid" "$pair/monitor.pid" "$pair/cleanup.pid"; do
  printf '%s=' "$path"
  cat "$path" 2>/dev/null || true
done
ps -eo pid,pgid,etimes,cmd --sort=pid | grep -E '50011|rlt_single_gpu_(control|success_episode_bc_dvac)_formal480|train_embodied_agent|raylet|gcs_server' | grep -v grep || true

echo '== ray =='
address=$(cat "$pair/ray_address.txt")
"$ray_cli" status --address="$address" 2>&1 || true

echo '== gpu =='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

echo '== memory =='
awk '/MemAvailable/ {print}' /proc/meminfo
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events

echo '== control markers =='
grep -E 'Ray namespace conflict|Using flexible placement|OpenPI model|Actor|EnvWorker|Rollout|Global Step|Start running|Traceback|CUDA out of memory|KeyError|RuntimeError' "$control/foreground.log" 2>/dev/null | tail -80 || true
echo '-- control tail --'
tail -50 "$control/foreground.log" 2>/dev/null || true

echo '== method markers =='
grep -E 'Ray namespace conflict|Using flexible placement|OpenPI model|Actor|EnvWorker|Rollout|Global Step|Start running|Traceback|CUDA out of memory|KeyError|RuntimeError' "$method/foreground.log" 2>/dev/null | tail -80 || true
echo '-- method tail --'
tail -50 "$method/foreground.log" 2>/dev/null || true

echo '== outputs =='
for path in "$control/resolved.yaml" "$method/resolved.yaml" "$control/started_at.txt" "$method/started_at.txt" "$control/exit_code.txt" "$method/exit_code.txt"; do
  if test -e "$path"; then printf 'present '; else printf 'absent '; fi
  echo "$path"
done
wc -l "$pair/paired_resources.csv" 2>/dev/null || true
