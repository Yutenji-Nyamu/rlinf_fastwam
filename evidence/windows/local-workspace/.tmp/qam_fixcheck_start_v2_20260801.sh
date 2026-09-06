set -euo pipefail
nohup bash /root/autodl-tmp/qam_prompt_fixcheck_launch_20260801_v2.sh \
  >/root/autodl-tmp/qam_prompt_fixcheck_supervisor_20260801_v2.log \
  2>&1 </dev/null &
printf 'SUPERVISOR_PID=%s\n' "$!"
