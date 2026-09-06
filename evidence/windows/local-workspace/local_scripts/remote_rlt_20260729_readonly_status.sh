#!/usr/bin/env bash
set -uo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
run_root=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1
experiment=robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1
run_dir="$run_root/$experiment"
export_root=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1
runtime="$export_root/runtime"
driver_pid_file="$runtime/driver_pid.txt"
monitor_pid_file="$runtime/monitor_pid.txt"

section() {
  printf '\n=== %s ===\n' "$1"
}

section SNAPSHOT
date --iso-8601=seconds
hostname
uptime

section GIT
if test -d "$repo"; then
  GIT_OPTIONAL_LOCKS=0 git -C "$repo" rev-parse HEAD
  GIT_OPTIONAL_LOCKS=0 git -C "$repo" branch --show-current
  GIT_OPTIONAL_LOCKS=0 git -C "$repo" status --porcelain=v1
  GIT_OPTIONAL_LOCKS=0 git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD' 2>&1 || true
else
  printf 'WORKTREE_MISSING\n'
fi

section PIDS
for file in "$driver_pid_file" "$monitor_pid_file"; do
  if test -f "$file"; then
    pid=$(tr -d '[:space:]' < "$file")
    printf '%s=%s\n' "$(basename "$file")" "$pid"
    ps -p "$pid" -o pid=,ppid=,stat=,etimes=,lstart=,cmd= || true
  else
    printf '%s=MISSING\n' "$(basename "$file")"
  fi
done
ps -eo pid=,ppid=,stat=,etimes=,lstart=,cmd= |
  grep -E 'train_vla_sft|rlt_stage1_formal|robotwin_adjust_bottle_rlt_stage1|resource.*monitor' |
  grep -v grep || true

section GPU
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true

section HOST_MEMORY
free -h
printf 'MemAvailable_kB='
awk '/^MemAvailable:/ {print $2}' /proc/meminfo

section DISK
df -h /root/autodl-tmp
du -sh "$run_root" "$export_root" 2>/dev/null || true

section RUNTIME_CONTROL
for file in \
  "$runtime/started_at.txt" \
  "$runtime/finished_at.txt" \
  "$runtime/exit_code.txt" \
  "$runtime/driver_pid.txt" \
  "$runtime/monitor_pid.txt"; do
  if test -f "$file"; then
    printf '%s=' "$(basename "$file")"
    tr '\n' ' ' < "$file"
    printf '\n'
  else
    printf '%s=MISSING\n' "$(basename "$file")"
  fi
done

section RUN_FILES
if test -d "$run_dir"; then
  find "$run_dir" -maxdepth 4 -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' |
    sort
else
  printf 'RUN_DIR_MISSING=%s\n' "$run_dir"
fi

section EXPORT_FILES
if test -d "$export_root"; then
  find "$export_root" -maxdepth 3 -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' |
    sort
else
  printf 'EXPORT_DIR_MISSING=%s\n' "$export_root"
fi

section CHECKPOINTS
checkpoint_root="$run_dir/checkpoints"
if test -d "$checkpoint_root"; then
  find "$checkpoint_root" -mindepth 1 -maxdepth 2 -type d -printf '%p\n' | sort
  du -sh "$checkpoint_root" "$checkpoint_root"/* 2>/dev/null || true
  find "$checkpoint_root" -type f -printf '%s\t%p\n' | sort -n
  printf 'TMP_RESIDUE_COUNT='
  find "$checkpoint_root" -name '*.tmp' -o -name '*tmp*' | wc -l
else
  printf 'CHECKPOINT_ROOT_MISSING=%s\n' "$checkpoint_root"
fi

section DRIVER_LOG_SUMMARY
driver_log="$runtime/driver.log"
if test -f "$driver_log"; then
  stat -c 'bytes=%s mtime=%y' "$driver_log"
  printf 'lines='
  wc -l < "$driver_log"
  printf 'error_oom='
  grep -Eic 'out of memory|CUDA.*OOM|CUBLAS_STATUS_ALLOC_FAILED' "$driver_log" || true
  printf 'error_nan_inf='
  grep -Eic '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$driver_log" || true
  printf 'error_traceback='
  grep -Eic 'Traceback|ChildFailed|NCCL.*error|rank.*(died|failed)|SIG(KILL|TERM)' "$driver_log" || true
  printf '%s\n' '-- metric-bearing tail --'
  grep -aE 'train/loss|train/rlt_loss|global_step|checkpoint|Saving|finished|exit|Training' "$driver_log" |
    tail -n 40 || true
  printf '%s\n' '-- raw tail --'
  tail -n 80 "$driver_log"
else
  printf 'DRIVER_LOG_MISSING\n'
fi

section RESOURCE_CSV_SUMMARY
resource_csv="$runtime/resources.csv"
if test -f "$resource_csv"; then
  stat -c 'bytes=%s mtime=%y' "$resource_csv"
  printf 'lines='
  wc -l < "$resource_csv"
  head -n 3 "$resource_csv"
  printf '%s\n' '-- tail --'
  tail -n 12 "$resource_csv"
else
  printf 'RESOURCE_CSV_MISSING\n'
fi

section SMALL_HASHES
for file in \
  "$export_root/formal_resolved.yaml" \
  "$export_root/source_config.yaml" \
  "$export_root/dataset_manifest.json" \
  "$export_root/run_provenance.tsv" \
  "$export_root/early_health.json"; do
  if test -f "$file"; then
    sha256sum "$file"
  fi
done
