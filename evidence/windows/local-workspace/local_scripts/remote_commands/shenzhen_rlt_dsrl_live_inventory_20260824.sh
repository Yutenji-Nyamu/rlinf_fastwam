#!/usr/bin/env bash
set -u

rlt_stage1=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1
rlt_root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3
rlt_run="$rlt_root/runtime"
dsrl_root=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2
dsrl_run="$dsrl_root/run"

date --iso-8601=seconds
id

printf '%s\n' '--- wrappers ---'
for pair in "rlt:$rlt_run/wrapper.pid" "dsrl:$dsrl_run/wrapper.pid"; do
  name=${pair%%:*}
  pidfile=${pair#*:}
  pid=$(cat "$pidfile" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    ps -o pid=,ppid=,etimes=,stat=,rss=,cmd= -p "$pid" | sed "s/^/$name /"
  else
    printf '%s pid=%s state=dead_or_missing\n' "$name" "${pid:-missing}"
  fi
done

printf '%s\n' '--- compact progress ---'
for pair in "rlt:$rlt_run/driver.log" "dsrl:$dsrl_run/driver.log"; do
  name=${pair%%:*}
  logfile=${pair#*:}
  printf '[%s]\n' "$name"
  tr '\r' '\n' < "$logfile" 2>/dev/null \
    | grep -E 'Global Step:|success_once=|success_at_end=|reward=|rlt/update_step=|rlt/ready_for_online=|rlt/replay_buffer_size=|sac/global_resident_transitions=|sac/planned_optimizer_updates=|sac/update_step=|actor_loss=|critic_loss=|q_loss=|Saving checkpoint|Eval' \
    | tail -n 100 || true
done

printf '%s\n' '--- fatal pattern counts ---'
for pair in "rlt:$rlt_run/driver.log" "dsrl:$dsrl_run/driver.log"; do
  name=${pair%%:*}
  logfile=${pair#*:}
  printf '%s ' "$name"
  for pattern in 'Traceback' 'CUDA out of memory' 'OutOfMemory' 'WorkerCrashed' 'nonfinite' 'nan' 'ERROR'; do
    count=$(grep -ai -c "$pattern" "$logfile" 2>/dev/null || true)
    printf '%s=%s ' "${pattern// /_}" "$count"
  done
  printf '\n'
done

printf '%s\n' '--- lightweight source files ---'
for file in \
  "$rlt_stage1/driver.log" "$rlt_stage1/resource.csv" "$rlt_stage1/resolved.yaml" \
  "$rlt_run/driver.log" "$rlt_run/resource.csv" "$rlt_run/resolved.yaml" \
  "$dsrl_run/driver.log" "$dsrl_run/resource.csv" "$dsrl_run/resolved.yaml"; do
  if [ -f "$file" ]; then
    stat -c '%s|%y|%n' "$file"
  fi
done

printf '%s\n' '--- resource headers and ranges ---'
for pair in "rlt_stage1:$rlt_stage1/resource.csv" "rlt_stage2:$rlt_run/resource.csv" "dsrl:$dsrl_run/resource.csv"; do
  name=${pair%%:*}
  file=${pair#*:}
  printf '[%s]\n' "$name"
  sed -n '1,2p' "$file" 2>/dev/null || true
  tail -n 2 "$file" 2>/dev/null || true
  wc -l "$file" 2>/dev/null || true
done

printf '%s\n' '--- checkpoints ---'
for pair in "rlt_stage1:$rlt_stage1" "rlt_stage2:$rlt_root" "dsrl:$dsrl_root"; do
  name=${pair%%:*}
  root=${pair#*:}
  printf '[%s]\n' "$name"
  find "$root" -type d -name 'global_step_*' -print 2>/dev/null | sort -V | while IFS= read -r checkpoint; do
    bytes=$(du -sb "$checkpoint" 2>/dev/null | awk '{print $1}')
    printf '%s|%s\n' "${bytes:-unknown}" "$checkpoint"
  done
done

printf '%s\n' '--- artifact inventory ---'
for pair in "rlt_stage1:$rlt_stage1" "rlt_stage2:$rlt_root" "dsrl:$dsrl_root"; do
  name=${pair%%:*}
  root=${pair#*:}
  printf '[%s]\n' "$name"
  du -sh "$root" 2>/dev/null || true
  for sub in video robotwin_data logs checkpoints; do
    [ -e "$root/$sub" ] && du -sh "$root/$sub" 2>/dev/null || true
  done
  printf 'files=%s videos=%s events=%s\n' \
    "$(find "$root" -type f 2>/dev/null | wc -l)" \
    "$(find "$root" -type f \( -name '*.mp4' -o -name '*.avi' \) 2>/dev/null | wc -l)" \
    "$(find "$root" -type f -name 'events.out.tfevents.*' 2>/dev/null | wc -l)"
  find "$root" -type f -name 'events.out.tfevents.*' -printf '%s|%TY-%Tm-%TdT%TH:%TM:%TS|%p\n' 2>/dev/null | sort || true
done

printf '%s\n' '--- live resources ---'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits
free -b | sed -n '1,2p'
df -B1 / /home /data | sed -n '1p;2,$p'

printf '%s\n' '--- exit markers and current log tails ---'
for pair in "rlt:$rlt_run" "dsrl:$dsrl_run"; do
  name=${pair%%:*}
  root=${pair#*:}
  printf '[%s]\n' "$name"
  [ -s "$root/exit_code.txt" ] && { printf 'exit_code='; tr '\n' ' ' < "$root/exit_code.txt"; printf '\n'; } || printf 'exit_code=none\n'
  tail -n 12 "$root/driver.log" 2>/dev/null || true
done
