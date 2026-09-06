set -u
run=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1
printf 'NEWEST_FILES\n'
find "$run" -type f -printf '%T@ %s %p\n' 2>/dev/null | sort -nr | head -40
printf 'OPEN_FDS_ROLLOUT\n'
for p in 7898 7900 7876 7894; do printf 'PID=%s ' "$p"; ls /proc/$p/fd 2>/dev/null | wc -l; done
printf 'TCP_COUNT\n'
ss -tnp 2>/dev/null | grep -E '7898|7900|7876|7894' | wc -l || true
