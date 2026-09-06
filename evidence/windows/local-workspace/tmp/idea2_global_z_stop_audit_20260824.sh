#!/usr/bin/env bash
set -u

run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823

printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
printf 'HOST_START\n'
uptime -s 2>/dev/null || true
ps -p 1 -o pid,lstart,etimes,cmd 2>/dev/null || true

printf 'CONTROL_FILES\n'
find "$runtime" -maxdepth 2 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort

printf 'CONTROL_STATUS\n'
for name in wrapper driver observer; do
  pid=$(cat "$runtime/$name.pid" 2>/dev/null || true)
  alive=0
  [ -n "$pid" ] && [ -d "/proc/$pid" ] && alive=1
  printf '%s pid=%s alive=%s\n' "$name" "$pid" "$alive"
done

printf 'EXIT_OR_SIGNAL_FILES\n'
find "$runtime" -maxdepth 2 -type f \( -iname '*exit*' -o -iname '*rc*' -o -iname '*status*' -o -iname '*signal*' \) -print -exec sed -n '1,80p' {} \; 2>/dev/null || true

printf 'EXIT_OR_ERROR_MATCHES\n'
grep -aEn 'FORMAL_DRIVER_RC=|DRIVER_RC=|exit code|return code|Killed|Terminated|SIG(INT|TERM|KILL)|KeyboardInterrupt|ActorDiedError|RayActorError|CUDA out of memory|OutOfMemory|NCCL.*(error|Error)|fatal|FATAL|Traceback' \
  "$runtime"/*.log 2>/dev/null | tail -n 160 || true

printf 'WRAPPER_TAIL\n'
tail -n 120 "$runtime/wrapper.log" 2>/dev/null || true
printf 'OBSERVER_TAIL\n'
tail -n 160 "$runtime/observer.log" 2>/dev/null || true
printf 'DRIVER_TAIL\n'
tail -n 240 "$runtime/driver.log" 2>/dev/null || true

printf 'LAST_COMPLETE_METRIC\n'
grep -a 'Global Step:' "$run/metrics.log" 2>/dev/null | tail -n 3 || true
printf 'LAST_ROLLOUT_PROGRESS\n'
grep -a 'Generating Rollout Epochs:' "$runtime/driver.log" 2>/dev/null | tail -n 8 || true

printf 'LOG_END_TIMES\n'
stat -c '%y %s %n' "$runtime/wrapper.log" "$runtime/observer.log" "$runtime/driver.log" \
  "$runtime/resource_monitor/resources.csv" "$run/metrics.log" 2>/dev/null || true

printf 'KERNEL_OOM_RECENT\n'
dmesg --ctime 2>/dev/null | grep -Ei 'out of memory|oom-kill|killed process' | tail -n 30 || true

printf 'RESOURCE_LAST\n'
tail -n 5 "$runtime/resource_monitor/resources.csv" 2>/dev/null || true

