#!/usr/bin/env bash
set -u

stage2=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3
runtime=$stage2/runtime
raylogs=/data/chenyiteng/ray/rlt-dsrl-v3/session_latest/logs

date --iso-8601=seconds
printf '%s\n' '--- driver and checkpoint filesystem ---'
stat -c '%y %s %n' "$runtime/driver.log" "$runtime/resource.csv" 2>/dev/null || true
find "$stage2" -maxdepth 6 -printf '%y %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null \
  | sort -k2,2 | tail -n 180 || true

printf '%s\n' '--- relevant process state ---'
pids='390911 391536 391538 391539 391541 391542 391559'
ps -o pid,ppid,stat,etimes,time,pcpu,pmem,rss,wchan:36,cmd -p $pids 2>/dev/null || true
for pid in $pids; do
  if [ -r "/proc/$pid/status" ]; then
    printf 'pid=%s ' "$pid"
    grep -E '^(State|Threads|VmRSS|VmSwap|voluntary_ctxt_switches|nonvoluntary_ctxt_switches):' "/proc/$pid/status" \
      | tr '\n' ' '; printf '\n'
    grep -E '^(read_bytes|write_bytes|cancelled_write_bytes):' "/proc/$pid/io" 2>/dev/null \
      | tr '\n' ' '; printf '\n'
  fi
done

printf '%s\n' '--- ray log files for RLT actor pids ---'
for pid in $pids; do
  find "$raylogs" -maxdepth 1 -type f -name "*-$pid.*" -printf '%y %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null || true
done

printf '%s\n' '--- targeted tails and errors from RLT actor logs ---'
for pid in 391536 391538 391539 391541 391542 391559; do
  for file in "$raylogs"/*-"$pid".*; do
    [ -f "$file" ] || continue
    printf '### %s\n' "$file"
    grep -a -Ei 'traceback|error|exception|failed|timeout|NCCL|GLOO|checkpoint|saving|saved|barrier|oom|killed' "$file" 2>/dev/null | tail -n 100 || true
    printf '%s\n' '--- tail ---'
    tail -n 40 "$file" 2>/dev/null || true
  done
done

printf '%s\n' '--- relevant Ray system errors after checkpoint start ---'
grep -a -Ei '391536|391538|RLTACFSDPPolicy|checkpoint|NCCL|GLOO|deadlock|timeout|exception|error' \
  "$raylogs"/raylet.err "$raylogs"/raylet.out "$raylogs"/gcs_server.err "$raylogs"/gcs_server.out 2>/dev/null \
  | tail -n 200 || true

