set -u
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
echo '=== time_pids ==='
date '+%Y-%m-%d %H:%M:%S %Z'
for f in wrapper.pid driver.pid observer.pid; do printf '%s=' "$f"; cat "$runtime/$f" 2>/dev/null || true; done
wp=$(cat "$runtime/wrapper.pid" 2>/dev/null || printf 0)
dp=$(cat "$runtime/driver.pid" 2>/dev/null || printf 0)
op=$(cat "$runtime/observer.pid" 2>/dev/null || printf 0)
ps -o pid,ppid,pgid,stat,etime,%cpu,rss,args -p "$wp,$dp,$op" 2>&1 || true

echo '=== core_workers ==='
ps -eo pid,ppid,pgid,stat,rss,args | grep -E 'ray::(EmbodiedFSDPActor|MultiStepRolloutWorker|EnvWorker)' | grep -v grep || true

echo '=== progress_tail ==='
grep -aE 'Generating Rollout Epochs|Global Step|ALIVE|Traceback|CUDA out of memory|OutOfMemory|ActorDiedError|RayActorError|NCCL|FATAL|fatal' "$runtime/driver.log" 2>/dev/null | tail -n 100 || true
tail -n 50 "$runtime/driver.log" 2>/dev/null || true

echo '=== resources_artifacts ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
find "$run" -maxdepth 2 -type f -printf '%P %s\n' 2>/dev/null | head -n 40
wc -l "$runtime/resource_monitor/resources.csv" 2>/dev/null || true
echo '=== exit_files ==='
for f in driver.exitcode observer.exitcode launch_finished_at.txt; do test -e "$runtime/$f" && { printf '%s=' "$f"; cat "$runtime/$f"; }; done
