#!/usr/bin/env bash
set +e

control_runtime="$1"
method_runtime="$2"
control_driver_pgid="$3"
control_head_pgid="$4"
method_driver_pgid="$5"
method_head_pgid="$6"
out="$7"

printf '%s\n' 'unix_time,cgroup_current_bytes,cgroup_high_bytes,cgroup_max_bytes,event_high,event_max,event_oom,event_oom_kill,anon_bytes,file_bytes,psi_some_avg10,psi_full_avg10,gpu0_used_mib,gpu0_util_pct,gpu1_used_mib,gpu1_util_pct,control_driver_rss_kib,control_head_rss_kib,method_driver_rss_kib,method_head_rss_kib,compute_process_count' >"$out"

stat_value() { awk -v key="$1" '$1==key {print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null; }
event_value() { awk -v key="$1" '$1==key {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null; }
pgid_rss() { ps -eo pgid=,rss= | awk -v target="$1" '$1==target {sum += $2} END {print sum+0}'; }

while true; do
  now=$(date +%s)
  current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
  high=$(cat /sys/fs/cgroup/memory.high 2>/dev/null || echo 0)
  max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0)
  event_high=$(event_value high); event_high=${event_high:-0}
  event_max=$(event_value max); event_max=${event_max:-0}
  event_oom=$(event_value oom); event_oom=${event_oom:-0}
  event_oom_kill=$(event_value oom_kill); event_oom_kill=${event_oom_kill:-0}
  anon=$(stat_value anon); anon=${anon:-0}
  file=$(stat_value file); file=${file:-0}
  psi_some=$(awk '$1=="some" {for(i=1;i<=NF;i++) if($i ~ /^avg10=/){split($i,a,"="); print a[2]}}' /sys/fs/cgroup/memory.pressure 2>/dev/null); psi_some=${psi_some:-0}
  psi_full=$(awk '$1=="full" {for(i=1;i<=NF;i++) if($i ~ /^avg10=/){split($i,a,"="); print a[2]}}' /sys/fs/cgroup/memory.pressure 2>/dev/null); psi_full=${psi_full:-0}
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  gpu0_used=$(printf '%s\n' "$gpu" | sed -n '1s/,.*//p'); gpu0_used=${gpu0_used:-0}
  gpu0_util=$(printf '%s\n' "$gpu" | sed -n '1s/^[^,]*, //p'); gpu0_util=${gpu0_util:-0}
  gpu1_used=$(printf '%s\n' "$gpu" | sed -n '2s/,.*//p'); gpu1_used=${gpu1_used:-0}
  gpu1_util=$(printf '%s\n' "$gpu" | sed -n '2s/^[^,]*, //p'); gpu1_util=${gpu1_util:-0}
  printf '%s\n' "$now,$current,$high,$max,$event_high,$event_max,$event_oom,$event_oom_kill,$anon,$file,$psi_some,$psi_full,$gpu0_used,$gpu0_util,$gpu1_used,$gpu1_util,$(pgid_rss "$control_driver_pgid"),$(pgid_rss "$control_head_pgid"),$(pgid_rss "$method_driver_pgid"),$(pgid_rss "$method_head_pgid"),$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {n++} END {print n+0}')" >>"$out"
  if [ -f "${control_runtime}/finished_at.txt" ] && [ -f "${method_runtime}/finished_at.txt" ]; then
    break
  fi
  sleep 2
done
