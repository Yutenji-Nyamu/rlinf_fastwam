set -u
echo '=== verify ==='
date '+%Y-%m-%d %H:%M:%S %Z'
ps -o pid,ppid,pgid,stat,etime,rss,args -p 70610,70614,70615 2>&1 || true
ps -eo pid,pgid,stat,args | awk '$2==70608 {count++; if (count<=20) print} END {print "PGID70608_COUNT=" (count+0)}'
pgrep -af 'idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true

run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
echo '=== final_artifact_count ==='
find "$run_dir" -type f -name '*.npz' 2>/dev/null | wc -l
find "$run_dir" -type f -name '*global_step_49*' -o -name '*step_49*' 2>/dev/null | head -n 20
tail -n 30 "$runtime_dir/driver.log" 2>/dev/null || true
echo '=== launch_exit ==='
tail -n 30 "$runtime_dir/launch.out" 2>/dev/null || true
