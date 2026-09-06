#!/usr/bin/env bash
set -u

source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
robotwin_root=/root/autodl-tmp/idea2_dvac_train_wamppo
python_bin=/root/autodl-tmp/RLinf/.venv/bin/python
v2_config="$source_root/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_100step_formal.yaml"
v3_config="$source_root/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_v3_w0to2_100step_formal.yaml"
v3_run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
v3_runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822

printf 'IDENTITY_TIME\n'
hostname
pwd
id -u
date --iso-8601=seconds

printf 'GPU_AND_COMPUTE\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,name --format=csv,noheader || true

printf 'TARGET_PROCESSES\n'
ps -eo pid=,ppid=,comm=,args= | awk '
  $3 ~ /^(python|python3|raylet|gcs_server)$/ &&
  ($0 ~ /train_embodied_agent.py/ || $3 == "raylet" || $3 == "gcs_server") {print}
' || true

printf 'MEMORY_DISK\n'
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
printf 'memory.max='; cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
awk '$1 == "MemAvailable:" {print}' /proc/meminfo
df -h /root/autodl-tmp /dev/shm

printf 'SOURCE_GIT\n'
git -C "$source_root" status --short
git -C "$source_root" branch --show-current
git -C "$source_root" rev-parse HEAD
git -C "$source_root" rev-parse --abbrev-ref '@{upstream}' 2>/dev/null || true
git -C "$source_root" rev-parse '@{upstream}' 2>/dev/null || true
git -C "$robotwin_root" status --short
git -C "$robotwin_root" rev-parse HEAD

printf 'CONFIG_AND_RUNTIME\n'
test -x "$python_bin" && printf 'python_ok=%s\n' "$python_bin"
test -f "$v2_config" && sha256sum "$v2_config"
if test -e "$v3_config"; then printf 'V3_CONFIG_EXISTS=%s\n' "$v3_config"; else printf 'V3_CONFIG_ABSENT=%s\n' "$v3_config"; fi
if test -e "$v3_run"; then printf 'V3_RUN_EXISTS=%s\n' "$v3_run"; else printf 'V3_RUN_ABSENT=%s\n' "$v3_run"; fi
if test -e "$v3_runtime"; then printf 'V3_RUNTIME_EXISTS=%s\n' "$v3_runtime"; else printf 'V3_RUNTIME_ABSENT=%s\n' "$v3_runtime"; fi

printf 'V2_KEY_VALUES\n'
grep -nE 'log_path:|experiment_name:|max_steps:|save_interval:|group_size:|update_epoch:|rollout_epoch:|total_num_envs:|micro_batch_size:|global_batch_size:|weight_min:|weight_max:|residual_clip:|lr:|clip_grad:' "$v2_config"

printf 'V2_RETAINED_OUTPUT\n'
v2_run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
du -sh "$v2_run" 2>/dev/null || true
find "$v2_run" -maxdepth 7 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V || true
