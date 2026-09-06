set -u
runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1/runtime
printf 'NCCL_FIRST_CONTEXT\n'
line=$(grep -an -m1 'Watchdog caught collective operation timeout\|work timed out' "$runtime/driver.log" | cut -d: -f1 || true)
if [ -n "$line" ]; then start=$((line-30)); [ "$start" -lt 1 ] && start=1; end=$((line+100)); sed -n "${start},${end}p" "$runtime/driver.log"; fi
printf 'DRIVER_LAST300\n'
tail -300 "$runtime/driver.log"
printf 'RAY_ACTOR_ERR_MATCHES\n'
for f in /tmp/ray/session_latest/logs/*7876*.err /tmp/ray/session_latest/logs/*7894*.err; do echo FILE=$f; grep -an -E 'Watchdog|timeout|Collective|SeqNum|OpType|NCCL' "$f" | head -80; done
