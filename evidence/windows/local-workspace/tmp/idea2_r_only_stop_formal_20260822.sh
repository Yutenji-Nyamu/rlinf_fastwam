#!/usr/bin/env bash
set -u

run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
expected_source=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight/examples/embodiment/train_embodied_agent.py
expected_config='--config-name robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_100step_formal'

printf 'IDENTITY_AND_TIME\n'
hostname
pwd
id -u
date --iso-8601=seconds

printf 'PRE_STOP_CONTROLLERS\n'
for name in wrapper driver observer; do
  pid=$(cat "$runtime/${name}.pid" 2>/dev/null || true)
  if [[ -n "$pid" && -d "/proc/$pid" ]]; then
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    printf '%s\tpid=%s\talive\tcmd=%s\n' "$name" "$pid" "$cmd"
  else
    printf '%s\tpid=%s\tdead_or_missing\n' "$name" "${pid:-missing}"
  fi
done

driver=$(cat "$runtime/driver.pid" 2>/dev/null || true)
if [[ -z "$driver" || ! -d "/proc/$driver" ]]; then
  printf 'STOP_ABORTED=driver_not_alive\n'
  exit 43
fi
driver_cmd=$(tr '\0' ' ' <"/proc/$driver/cmdline")
if [[ "$driver_cmd" != *"$expected_source"* || "$driver_cmd" != *"$expected_config"* ]]; then
  printf 'STOP_ABORTED=driver_cmdline_mismatch\n'
  printf 'driver_cmd=%s\n' "$driver_cmd"
  exit 44
fi

printf 'LATEST_COMPLETE_BEFORE_STOP\n'
grep -o 'Global Step:[[:space:]]*[0-9]\+/100' "$run/metrics.log" 2>/dev/null | tail -n 8 || true

printf 'CHECKPOINTS_BEFORE_STOP\n'
find "$run" -maxdepth 7 -type d -name 'global_step_*' -printf '%p\n' 2>/dev/null | sort -V || true
for d in $(find "$run" -maxdepth 7 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -n 2); do
  printf 'CHECKPOINT=%s\n' "$d"
  du -sb "$d" 2>/dev/null || true
  find "$d" -type f -printf '%f\t%s\n' 2>/dev/null | sort || true
done

printf 'PRE_STOP_RESOURCE\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events

printf 'SIGNAL_INT\tpid=%s\n' "$driver"
kill -INT "$driver"
for _ in $(seq 1 10); do
  [[ ! -d "/proc/$driver" ]] && break
  sleep 1
done

if [[ -d "/proc/$driver" ]]; then
  current_cmd=$(tr '\0' ' ' <"/proc/$driver/cmdline")
  if [[ "$current_cmd" != *"$expected_source"* || "$current_cmd" != *"$expected_config"* ]]; then
    printf 'TERM_ABORTED=driver_identity_changed\n'
    exit 45
  fi
  printf 'SIGNAL_TERM\tpid=%s\n' "$driver"
  kill -TERM "$driver"
else
  printf 'TERM_NOT_NEEDED=driver_exited_after_INT\n'
fi

for _ in $(seq 1 30); do
  alive=0
  for name in wrapper driver observer; do
    pid=$(cat "$runtime/${name}.pid" 2>/dev/null || true)
    [[ -n "$pid" && -d "/proc/$pid" ]] && alive=1
  done
  [[ "$alive" -eq 0 ]] && break
  sleep 1
done

printf 'POST_SIGNAL_CONTROLLERS\n'
for name in wrapper driver observer; do
  pid=$(cat "$runtime/${name}.pid" 2>/dev/null || true)
  if [[ -n "$pid" && -d "/proc/$pid" ]]; then
    printf '%s\tpid=%s\talive\tstate=%s\n' "$name" "$pid" "$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
  else
    printf '%s\tpid=%s\texited\n' "$name" "${pid:-missing}"
  fi
done

printf 'POST_SIGNAL_MARKERS\n'
for f in "$runtime"/*exitcode "$runtime"/*finished* "$runtime"/*rc; do
  [[ -e "$f" ]] && { printf '%s=' "$f"; cat "$f"; }
done

printf 'LATEST_COMPLETE_AFTER_SIGNAL\n'
grep -o 'Global Step:[[:space:]]*[0-9]\+/100' "$run/metrics.log" 2>/dev/null | tail -n 8 || true

printf 'POST_SIGNAL_RESOURCE\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
