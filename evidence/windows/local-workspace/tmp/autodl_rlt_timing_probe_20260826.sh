set -euo pipefail
for log in \
  /root/autodl-tmp/experiment_exports/rlt_single_gpu_control_formal480_20260825_v2/runtime/foreground.log \
  /root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v2/runtime/foreground.log
do
  echo "LOG=$log"
  grep -E 'Global Step:|Elapsed:' "$log" | tail -n 2
done
