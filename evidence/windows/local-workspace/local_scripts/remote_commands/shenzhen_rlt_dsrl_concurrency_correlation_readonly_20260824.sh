#!/usr/bin/env bash
set -u

rlt=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3
dsrl=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run
rlt_ckpt="$rlt/robotwin_adjust_bottle_rlt_stage2_current_ar_8env250_output_isolated_v2/checkpoints/global_step_25"
dsrl_ckpt="$dsrl/dsrl-current-formal-200c-v2/checkpoints"

date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds

printf '%s\n' '--- per-job distributed environment ---'
for item in rlt-r0:391536 rlt-r1:391538 dsrl-r0:371579 dsrl-r1:371581; do
  label=${item%%:*}; pid=${item#*:}
  printf '### %s pid=%s\n' "$label" "$pid"
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
    | grep -E '^(RAY_JOB_ID|RAY_NAMESPACE|MASTER_ADDR|MASTER_PORT|RANK|LOCAL_RANK|WORLD_SIZE|GROUP_RANK|TORCHELASTIC|NCCL_|GLOO_)=' \
    | sort || true
done

printf '%s\n' '--- independent progress and mtime anchors ---'
stat -c '%y|%s|%n' "$rlt/runtime/driver.log" "$rlt/runtime/resource.csv" "$dsrl/driver.log" "$dsrl/resource.csv" 2>/dev/null || true
printf 'rlt_global_steps='; tr '\r' '\n' < "$rlt/runtime/driver.log" | grep -c 'Global Step:' || true
printf 'dsrl_global_steps='; tr '\r' '\n' < "$dsrl/driver.log" | grep -c 'Global Step:' || true
find "$dsrl_ckpt" -maxdepth 1 -type d -name 'global_step_*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS|%p\n' 2>/dev/null | sort || true

printf '%s\n' '--- RLT partial checkpoint exact ---'
find "$rlt_ckpt" -printf '%y|%TY-%Tm-%TdT%TH:%TM:%TS|%s|%p\n' 2>/dev/null | sort || true

printf '%s\n' '--- current actor state and IO ---'
ps -o pid,stat,etimes,time,pcpu,rss,wchan:28,cmd -p 391536,391538,371579,371581 2>/dev/null || true
for pid in 391536 391538 371579 371581; do
  printf 'pid=%s|' "$pid"
  grep -E '^(read_bytes|write_bytes|cancelled_write_bytes):' "/proc/$pid/io" 2>/dev/null | tr '\n' ' '
  printf '\n'
done

printf '%s\n' '--- current resources ---'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -b | sed -n '1,2p'
df -B1 /data | tail -n 1
tail -n 1 "$rlt/runtime/resource.csv" 2>/dev/null || true
tail -n 1 "$dsrl/resource.csv" 2>/dev/null || true

printf '%s\n' '--- recent shared ray health errors only ---'
raylogs=/data/chenyiteng/ray/rlt-dsrl-v3/session_latest/logs
for f in "$raylogs/raylet.err" "$raylogs/gcs_server.err"; do
  stat -c '%y|%s|%n' "$f" 2>/dev/null || true
  tail -n 100 "$f" 2>/dev/null | grep -a -Ei 'error|fatal|oom|dead|timeout|heartbeat|rpc' | tail -n 40 || true
done
