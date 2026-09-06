#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1
exp_root="$run_root/robotwin_adjust_bottle_rlt_stage2_formal_100c_v1"
runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime

echo "SECTION=IDENTITY"
date --iso-8601=seconds
hostname
id

echo "SECTION=GIT"
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD' 2>/dev/null || true

echo "SECTION=PROCESS"
for name in driver monitor; do
  pid_file="$runtime/${name}_pid.txt"
  if [[ -f "$pid_file" ]]; then
    pid=$(tr -d '[:space:]' < "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      echo "${name}_pid=$pid alive=1"
      ps -p "$pid" -o pid=,ppid=,etimes=,stat=,%cpu=,%mem=,rss=,cmd=
    else
      echo "${name}_pid=$pid alive=0"
    fi
  else
    echo "${name}_pid_file=MISSING"
  fi
done
mapfile -t matched_pids < <(
  pgrep -f 'train_embodied_agent.py|raylet|gcs_server|rlt_stage2_formal_100c_20260730_v1' || true
)
printf 'matched_pid_count=%s\n' "${#matched_pids[@]}"
if ((${#matched_pids[@]})); then
  ps -o pid=,ppid=,etimes=,stat=,%cpu=,%mem=,rss=,cmd= -p "$(IFS=,; echo "${matched_pids[*]}")" || true
fi

echo "SECTION=RUNTIME_FILES"
for name in started_at.txt finished_at.txt exit_code.txt driver_pid.txt monitor_pid.txt \
  driver.log resources.csv exact_command.txt resolved.yaml run_provenance.tsv \
  stop_command.txt; do
  path="$runtime/$name"
  if [[ -e "$path" ]]; then
    stat -c '%n size=%s mtime=%y' "$path"
    case "$name" in
      started_at.txt|finished_at.txt|exit_code.txt|driver_pid.txt|monitor_pid.txt)
        printf '%s=' "$name"
        tr '\n' ' ' < "$path"
        echo
        ;;
    esac
  else
    echo "$path MISSING"
  fi
done

echo "SECTION=LOG_SUMMARY"
if [[ -f "$runtime/driver.log" ]]; then
  wc -l -c "$runtime/driver.log"
  printf 'epoch_start_count='
  grep -c 'Generating Rollout Epochs' "$runtime/driver.log" || true
  printf 'checkpoint_save_count='
  grep -cE 'Saving checkpoint|global_step_[0-9]+' "$runtime/driver.log" || true
  printf 'fatal_cuda_oom_count='
  grep -Eic 'CUDA out of memory|CUDA error: out of memory' "$runtime/driver.log" || true
  printf 'fatal_nccl_count='
  grep -Eic 'NCCL.*(error|fatal|abort)' "$runtime/driver.log" || true
  printf 'ray_actor_death_count='
  grep -Eic 'RayActorError|actor died|worker died' "$runtime/driver.log" || true
  printf 'nan_metric_line_count='
  grep -Ei '(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "$runtime/driver.log" |
    grep -Eiv 'curobo|vulkan|warning|traceback' |
    wc -l || true
  printf 'traceback_count='
  grep -c 'Traceback (most recent call last)' "$runtime/driver.log" || true
  echo "LOG_TAIL_BEGIN"
  tail -n 180 "$runtime/driver.log"
  echo "LOG_TAIL_END"
fi

echo "SECTION=RESOURCE_SUMMARY"
if [[ -f "$runtime/resources.csv" ]]; then
  wc -l -c "$runtime/resources.csv"
  head -n 2 "$runtime/resources.csv"
  tail -n 5 "$runtime/resources.csv"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo
df -B1 /root/autodl-tmp
for path in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.stat \
  /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory.pressure; do
  if [[ -f "$path" ]]; then
    echo "FILE=$path"
    case "$path" in
      */memory.stat) grep -E '^(anon|file|inactive_file|active_file|slab) ' "$path" ;;
      *) cat "$path" ;;
    esac
  fi
done

echo "SECTION=ARTIFACTS"
if [[ -d "$run_root" ]]; then
  du -sb "$run_root"
  find "$run_root" -maxdepth 3 -type f -printf '%p size=%s mtime=%TY-%Tm-%TdT%TH:%TM:%TS\n' |
    sort
else
  echo "run_root=MISSING"
fi
if [[ -d "$exp_root/checkpoints" ]]; then
  echo "CHECKPOINT_DIRS_BEGIN"
  find "$exp_root/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V
  echo "CHECKPOINT_DIRS_END"
  for checkpoint in "$exp_root"/checkpoints/global_step_*; do
    [[ -d "$checkpoint" ]] || continue
    du -sb "$checkpoint"
    find "$checkpoint" -maxdepth 3 -type f -printf '%P size=%s\n' | sort
  done
else
  echo "checkpoints=MISSING"
fi

echo "SECTION=TENSORBOARD"
if [[ -d "$run_root/tensorboard" ]]; then
  find "$run_root/tensorboard" -maxdepth 2 -type f \
    -printf '%p size=%s mtime=%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort
else
  echo "tensorboard=MISSING"
fi

echo "RLT_FORMAL100_STATUS_AUDIT_DONE"
