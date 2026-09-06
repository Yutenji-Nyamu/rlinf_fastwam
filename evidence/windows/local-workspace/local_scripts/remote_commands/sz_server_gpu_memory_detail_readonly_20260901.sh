#!/usr/bin/env bash
set -u

echo '=== GPU_UUID_INDEX ==='
nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits
echo '=== GPU_PID_MEMORY ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true
echo '=== RL_DRIVERS ==='
ps -eo user=,pid=,ppid=,etimes=,stat=,%cpu=,rss=,args= | grep -E 'train_embodied_agent.py|run_.*formal|launch_.*formal' | grep -v grep || true
echo '=== MEMORY_COMPACT ==='
free -h
printf 'vm_oom_kill_total='; awk '$1=="oom_kill"{print $2}' /proc/vmstat
printf 'raylet_rss_gib='; ps -C raylet -o rss= | awk '{s+=$1} END {printf "%.3f\n",s/1048576}'
printf 'gcs_rss_gib='; ps -C gcs_server -o rss= | awk '{s+=$1} END {printf "%.3f\n",s/1048576}'
