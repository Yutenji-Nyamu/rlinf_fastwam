#!/usr/bin/env bash
set -u

PI0=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-p1-16ep-800baf80-v1
FW=/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2
FWWT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
FWOFF="$FWWT/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2"
TASK_CONFIG="$FWWT/third_party/RoboTwin/task_config/_eval_step_limit.yml"
TASKS=(move_stapler_pad turn_switch pick_diverse_bottles)

printf 'time=%s\n' "$(date --iso-8601=seconds)"
id

echo '=== GPU_AND_MEMORY ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
free -b
df -B1 / /home/chenyiteng /data/chenyiteng
printf 'raylet='; pgrep -u "$(id -u)" -x raylet | tr '\n' ',' || true; echo
printf 'gcs_server='; pgrep -u "$(id -u)" -x gcs_server | tr '\n' ',' || true; echo
printf 'grpo_driver='; pgrep -u "$(id -u)" -af 'train_embodied_agent.py.*robotwin_adjust_bottle_grpo_openpi' || true

echo '=== EXISTING_DATA ==='
for root in "$PI0" "$FW" "$FWOFF"; do
  if test -e "$root"; then
    du -sb "$root"
    for ext in csv npz png mp4; do
      find "$root" -type f -iname "*.$ext" -printf '%s\n' 2>/dev/null \
        | awk -v e="$ext" '{n++; b+=$1} END {printf "%s_count=%d %s_bytes=%d\n",e,n,e,b}'
    done
  else
    printf 'absent=%s\n' "$root"
  fi
done

echo '=== FASTWAM_TASKS ==='
for task in "${TASKS[@]}"; do
  env_file="$FWWT/third_party/RoboTwin/envs/$task.py"
  printf 'task=%s env=%s step_limit=' "$task" "$(test -f "$env_file" && echo present || echo absent)"
  if test -f "$TASK_CONFIG"; then
    awk -v t="$task" '$1==t":" {print $2; found=1} END {if(!found) print "absent"}' "$TASK_CONFIG"
  else
    echo 'task_config_absent'
  fi
done

echo 'SZ_DVAC_EXPAND_LIVE_PREFLIGHT_DONE'
