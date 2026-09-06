#!/usr/bin/env bash
set +e

runtime_root=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime
out="${runtime_root}/resources.csv"
printf '%s\n' 'unix_time,cgroup_current_bytes,cgroup_high_bytes,cgroup_max_bytes,event_high,event_max,event_oom,event_oom_kill,anon_bytes,file_bytes,shmem_bytes,inactive_file_bytes,active_file_bytes,slab_reclaimable_bytes,pgscan,pgsteal,refault_file,psi_some_avg10,psi_full_avg10,gpu0_used_mib,gpu0_util_pct,gpu1_used_mib,gpu1_util_pct,env_rss_kib,actor_rss_kib,rollout_rss_kib,driver_rss_kib,ray_system_rss_kib,matched_total_rss_kib,compute_process_count' >"$out"

while true; do
  [ -f "${runtime_root}/finished_at.txt" ] && break
  now=$(date +%s)
  current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
  high=$(cat /sys/fs/cgroup/memory.high 2>/dev/null || echo 0)
  max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0)
  event_high=$(awk '$1=="high" {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null); event_high=${event_high:-0}
  event_max=$(awk '$1=="max" {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null); event_max=${event_max:-0}
  event_oom=$(awk '$1=="oom" {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null); event_oom=${event_oom:-0}
  event_oom_kill=$(awk '$1=="oom_kill" {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null); event_oom_kill=${event_oom_kill:-0}
  stat_value() { awk -v key="$1" '$1==key {print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null; }
  anon=$(stat_value anon); anon=${anon:-0}
  file=$(stat_value file); file=${file:-0}
  shmem=$(stat_value shmem); shmem=${shmem:-0}
  inactive_file=$(stat_value inactive_file); inactive_file=${inactive_file:-0}
  active_file=$(stat_value active_file); active_file=${active_file:-0}
  slab_reclaimable=$(stat_value slab_reclaimable); slab_reclaimable=${slab_reclaimable:-0}
  pgscan=$(stat_value pgscan); pgscan=${pgscan:-0}
  pgsteal=$(stat_value pgsteal); pgsteal=${pgsteal:-0}
  refault=$(stat_value workingset_refault_file); refault=${refault:-0}
  psi_some=$(awk '$1=="some" {for(i=1;i<=NF;i++) if($i ~ /^avg10=/){split($i,a,"="); print a[2]}}' /sys/fs/cgroup/memory.pressure 2>/dev/null); psi_some=${psi_some:-0}
  psi_full=$(awk '$1=="full" {for(i=1;i<=NF;i++) if($i ~ /^avg10=/){split($i,a,"="); print a[2]}}' /sys/fs/cgroup/memory.pressure 2>/dev/null); psi_full=${psi_full:-0}
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  gpu0_used=$(printf '%s\n' "$gpu" | sed -n '1s/,.*//p'); gpu0_used=${gpu0_used:-0}
  gpu0_util=$(printf '%s\n' "$gpu" | sed -n '1s/^[^,]*, //p'); gpu0_util=${gpu0_util:-0}
  gpu1_used=$(printf '%s\n' "$gpu" | sed -n '2s/,.*//p'); gpu1_used=${gpu1_used:-0}
  gpu1_util=$(printf '%s\n' "$gpu" | sed -n '2s/^[^,]*, //p'); gpu1_util=${gpu1_util:-0}
  sum_rss() { ps -eo rss=,args= | awk -v pattern="$1" '$0 ~ pattern {sum += $1} END {print sum+0}'; }
  env_rss=$(sum_rss 'ray::EnvWorker')
  actor_rss=$(sum_rss 'ray::Actor')
  rollout_rss=$(sum_rss 'ray::Rollout')
  driver_rss=$(sum_rss 'train_embodied_agent.py')
  ray_system_rss=$(sum_rss 'raylet|gcs_server|dashboard|log_monitor')
  matched_total=$((env_rss + actor_rss + rollout_rss + driver_rss + ray_system_rss))
  compute_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {n++} END {print n+0}')
  printf '%s\n' "$now,$current,$high,$max,$event_high,$event_max,$event_oom,$event_oom_kill,$anon,$file,$shmem,$inactive_file,$active_file,$slab_reclaimable,$pgscan,$pgsteal,$refault,$psi_some,$psi_full,$gpu0_used,$gpu0_util,$gpu1_used,$gpu1_util,$env_rss,$actor_rss,$rollout_rss,$driver_rss,$ray_system_rss,$matched_total,$compute_count" >>"$out"
  sleep 2
done
