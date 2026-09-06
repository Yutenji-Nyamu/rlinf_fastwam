#!/usr/bin/env bash
set -u

runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822

date --iso-8601=seconds
for name in wrapper driver observer; do
  pid=$(cat "$runtime_dir/${name}.pid")
  if [[ -d "/proc/$pid" ]]; then printf '%s=%s:alive\n' "$name" "$pid"; else printf '%s=%s:dead\n' "$name" "$pid"; fi
done
ps -eo pid,stat,comm,args | grep -E 'ray::EmbodiedFSDPActor|ray::MultiStepRolloutWorker|ray::EnvWorker' | grep -v grep || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
grep -aoE 'Generating Rollout Epochs:[^\r\n]*' "$runtime_dir/driver.log" | tail -n 3 || true
grep -E 'CUDA out of memory|NCCL|WorkerCrashed|RayTaskError|No space left|SIGSEGV|SIGBUS|Killed|all_gather' "$runtime_dir/driver.log" | tail -n 20 || true
for manifest in "$run_dir"/dvac_train/actor_rank*/run_manifest.json; do
  printf 'MANIFEST=%s\n' "$manifest"
  cat "$manifest"
done
wc -l "$runtime_dir/resource_monitor/resources.csv"
tail -n 2 "$runtime_dir/resource_monitor/resources.csv"
du -sh "$run_dir"
