#!/usr/bin/env bash
set -u

runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight

printf 'IDENTITY\n'; hostname; pwd; id -u; date --iso-8601=seconds
printf 'OWNED_PIDS\n'
for name in wrapper driver observer; do
  pid=$(cat "$runtime/$name.pid" 2>/dev/null || true)
  printf '%s_PID=%s ALIVE=%s\n' "${name^^}" "$pid" "$(test -n "$pid" && test -d "/proc/$pid" && echo 1 || echo 0)"
  test -n "$pid" && ps -p "$pid" -o pid=,ppid=,stat=,etime=,rss=,pcpu=,comm=,args= 2>/dev/null || true
done
printf 'EXIT_MARKERS\n'; find "$runtime" -maxdepth 1 -type f \( -name '*.exitcode' -o -name 'launch_finished_at.txt' \) -printf '%f ' -exec cat {} \; 2>/dev/null || true

printf 'RAY_CORE_ACTORS\n'
/root/autodl-tmp/RLinf/.venv/bin/ray list actors --detail 2>/dev/null | awk '/class_name: (EmbodiedFSDPActor|MultiStepRolloutWorker|EnvWorker)/ {c=$2} /state:/ && c {print c,$2; c=""}' | sort

printf 'DRIVER_PROGRESS\n'
grep -aE 'Generating Rollout Epochs|Global Step|global_step' "$runtime/driver.log" 2>/dev/null | tail -n 35 || true
printf 'DRIVER_ERRORS\n'
grep -aE 'Traceback|ActorDiedError|RayActorError|CUDA out of memory|OutOfMemory|NCCL|Killed|fatal|FATAL|ERROR' "$runtime/driver.log" 2>/dev/null | tail -n 30 || true
printf 'DRIVER_LOG\n'; stat -c 'size=%s mtime=%y' "$runtime/driver.log" 2>/dev/null || true

printf 'RUN_SIZE\n'; du -sh "$run" 2>/dev/null || true
printf 'TOP_LEVEL\n'; find "$run" -mindepth 1 -maxdepth 2 -printf '%y %p %s\n' 2>/dev/null | sort | tail -n 100
printf 'CHECKPOINTS\n'; find "$run" -maxdepth 2 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V
printf 'TELEMETRY_COUNTS\n'
find "$run" -type f \( -name '*.npz' -o -name '*.csv' -o -name '*.json' -o -name '*.mp4' -o -name '*.png' \) 2>/dev/null | awk -F. '{ext=tolower($NF); n[ext]++} END {for (e in n) print e,n[e]}' | sort
printf 'RECENT_FILES\n'; find "$run" -type f -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -n 80

printf 'GPU_CURRENT\n'; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
printf 'GPU_PROCESSES\n'; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null || true
printf 'CGROUP_CURRENT_BYTES=%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'CGROUP_MAX_BYTES=%s\n' "$(cat /sys/fs/cgroup/memory.max)"
printf 'MEMORY_EVENTS\n'; cat /sys/fs/cgroup/memory.events
printf 'HOST_MEM_AVAILABLE_KIB=%s\n' "$(awk '$1=="MemAvailable:" {print $2}' /proc/meminfo)"
printf 'DISK\n'; df -h /root/autodl-tmp /dev/shm

printf 'RESOURCE_SUMMARY\n'
resources="$runtime/resource_monitor/resources.csv"
if test -f "$resources"; then
  awk -F, 'NR>1 {
    n++;
    g=$3+0; gm=$4+0; gu=$6+0; temp=$8+0; p=$9+0; cg=$10+0;
    if (gm>maxgm[g]) maxgm[g]=gm; if (gu>maxgu[g]) maxgu[g]=gu;
    if (temp>maxt[g]) maxt[g]=temp; if (p>maxp[g]) maxp[g]=p;
    if (cg>maxcg) maxcg=cg;
    lastts=$1; lastelapsed=$2; lasteventmax=$15; lastoom=$16; lastoomkill=$17;
  } END {
    printf "rows=%d last_ts=%s elapsed_s=%s max_cgroup_bytes=%.0f event_max=%s oom=%s oom_kill=%s\n", n,lastts,lastelapsed,maxcg,lasteventmax,lastoom,lastoomkill;
    for (g in maxgm) printf "gpu=%s max_mem_mib=%.0f max_util_pct=%.0f max_temp_c=%.0f max_power_w=%.2f\n",g,maxgm[g],maxgu[g],maxt[g],maxp[g];
  }' "$resources"
  printf 'RESOURCE_LAST\n'; tail -n 4 "$resources"
fi

printf 'SOURCE\n'; git -C "$source_root" rev-parse HEAD; git -C "$source_root" status --short
