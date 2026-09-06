#!/usr/bin/env bash
set -u

printf '%s\n' 'IDENTITY'
date -Is
hostname
uptime

printf '%s\n' 'MEMORY'
free -h
printf 'cgroup_current_gib='
awk '{printf "%.2f\n",$1/1073741824}' /sys/fs/cgroup/memory.current
printf 'cgroup_high_gib='
awk '{if ($1=="max") print "max"; else printf "%.2f\n",$1/1073741824}' /sys/fs/cgroup/memory.high
printf 'cgroup_max_gib='
awk '{if ($1=="max") print "max"; else printf "%.2f\n",$1/1073741824}' /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
printf '%s\n' 'PSI_MEMORY'
cat /proc/pressure/memory

printf '%s\n' 'GPU'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits

printf '%s\n' 'DISK'
df -h / /root/autodl-tmp

printf '%s\n' 'TOP_RSS_MIB'
ps -eo pid,ppid,pgid,stat,rss,etimes,comm,args --sort=-rss | head -n 12

printf '%s\n' 'TRAIN_RAY_COUNTS'
printf 'train_drivers='
pgrep -fc 'train_embodied_agent.py' || true
printf 'raylet='
pgrep -fc 'raylet' || true
printf 'gcs_server='
pgrep -fc 'gcs_server' || true
printf 'core_rlt_actors='
ps -eo cmd | grep -E 'ray::(RLTACFSDPPolicy|MultiStepRolloutWorker|EnvWorker)' | grep -v grep | wc -l
