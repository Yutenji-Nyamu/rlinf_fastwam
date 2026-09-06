#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1
experiment=robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1
checkpoint_root="$run_root/$experiment/checkpoints"
runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime

echo "SECTION=IDENTITY"
date --iso-8601=seconds
hostname
pwd
id -u

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
      ps -p "$pid" -o pid=,ppid=,lstart=,etimes=,stat=,%cpu=,%mem=,rss=,cmd=
    else
      echo "${name}_pid=$pid alive=0"
    fi
  else
    echo "${name}_pid_file=MISSING"
  fi
done
ps -eo pid=,comm=,args= |
  awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}'
pgrep -ax raylet || true
pgrep -ax gcs_server || true

echo "SECTION=RUNTIME"
for name in started_at.txt finished_at.txt exit_code.txt driver.log resources.csv; do
  path="$runtime/$name"
  if [[ -e "$path" ]]; then
    stat -c '%n size=%s mtime=%y' "$path"
    case "$name" in
      started_at.txt|finished_at.txt|exit_code.txt)
        printf '%s=' "$name"
        tr '\n' ' ' < "$path"
        echo
        ;;
    esac
  else
    echo "$path MISSING"
  fi
done

if [[ -f "$runtime/driver.log" ]]; then
  echo "SECTION=LOG"
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
  grep -a 'Global Step:' "$runtime/driver.log" | tail -n 3 || true
fi

echo "SECTION=LIVE_RESOURCE"
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits
grep -E 'MemAvailable|SwapTotal|SwapFree' /proc/meminfo
df -B1 /root/autodl-tmp
cat /sys/fs/cgroup/memory.current
grep -E '^(anon|file|inactive_file|active_file|slab) ' /sys/fs/cgroup/memory.stat
cat /sys/fs/cgroup/memory.events
cat /sys/fs/cgroup/memory.pressure

echo "SECTION=ARTIFACTS"
du -sb "$run_root"
if [[ -d "$checkpoint_root" ]]; then
  for checkpoint in "$checkpoint_root"/global_step_*; do
    [[ -d "$checkpoint" ]] || continue
    step=${checkpoint##*_}
    bytes=$(du -sb "$checkpoint" | awk '{print $1}')
    files=$(find "$checkpoint" -type f | wc -l)
    printf 'checkpoint=%s bytes=%s files=%s\n' "$step" "$bytes" "$files"
    complete="$checkpoint/actor/sac_components/rlt_trainer_state/rlt_trainer_state_complete.json"
    if [[ -f "$complete" ]]; then
      printf 'completion_%s=' "$step"
      tr -d '\n' < "$complete"
      echo
    else
      echo "completion_${step}=MISSING"
    fi
    for rank in 0 1; do
      metadata="$checkpoint/actor/sac_components/replay_buffer/rank_${rank}/metadata.json"
      if [[ -f "$metadata" ]]; then
        printf 'replay_%s_rank%s=' "$step" "$rank"
        tr -d '\n' < "$metadata"
        echo
      else
        echo "replay_${step}_rank${rank}=MISSING"
      fi
    done
  done
fi

echo "RLT_FORMAL250_COMPACT_STATUS_DONE"
