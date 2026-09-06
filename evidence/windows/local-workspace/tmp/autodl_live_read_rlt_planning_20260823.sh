set -u

printf 'TIME_IDENTITY\n'
date '+%Y-%m-%d %H:%M:%S %Z'
hostname
pwd
id -u

printf 'GPU_SUMMARY\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits

printf 'GPU_COMPUTE_PROCESSES\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 || true

printf 'RELEVANT_PROCESSES\n'
ps -eo pid,ppid,pgid,etimes,%cpu,%mem,rss,stat,cmd --sort=-rss | grep -E 'train_embodied_agent|idea2_dvac|ray::|raylet|gcs_server|python.*RLinf' | grep -v grep | head -n 40 || true

printf 'CGROUP_MEMORY\n'
for f in memory.current memory.max memory.high memory.events; do
  if [ -r "/sys/fs/cgroup/$f" ]; then
    printf '%s=' "$f"
    tr '\n' ' ' < "/sys/fs/cgroup/$f"
    printf '\n'
  fi
done

printf 'HOST_MEMORY\n'
free -h

printf 'GLOBAL_Z_RUN\n'
run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
if [ -d "$run" ]; then
  du -sh "$run"
  find "$run" -maxdepth 2 -type d -name 'global_step_*' -printf '%f\n' | sort -V | tail -n 12
  find "$run" -maxdepth 3 -type f \( -name '*.out' -o -name '*.log' -o -name '*.csv' -o -name 'events.out.tfevents.*' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort | tail -n 20
fi
if [ -d "$runtime" ]; then
  du -sh "$runtime"
  find "$runtime" -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort | tail -n 20
fi

printf 'LATEST_TRAIN_MARKERS\n'
if [ -d "$runtime" ]; then
  grep -R -h -E 'Global Step|Generating Rollout Epochs|success|Traceback|CUDA out of memory|OutOfMemory|worker.*died|driver.*exit|DONE|finished' "$runtime" 2>/dev/null | tail -n 80 || true
fi

printf 'DISK\n'
df -h /root/autodl-tmp
