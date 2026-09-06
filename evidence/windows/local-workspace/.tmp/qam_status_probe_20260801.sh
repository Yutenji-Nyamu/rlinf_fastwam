set -u
hostname
pwd
id -u
date '+%F %T %Z'
if kill -0 380841 2>/dev/null; then
  echo DRIVER_ALIVE=yes
else
  echo DRIVER_ALIVE=no
fi
printf 'EXIT='
if test -f /root/autodl-tmp/experiment_exports/qam_formal_resume100_to380_20260801_v4/runtime/driver.exit; then
  cat /root/autodl-tmp/experiment_exports/qam_formal_resume100_to380_20260801_v4/runtime/driver.exit
else
  echo none
fi
grep -a 'Global Step:' /root/autodl-tmp/experiment_exports/qam_formal_resume100_to380_20260801_v4/runtime/driver.log | tail -n 1
grep -aE 'success_once=|qam/global_total_inserts=|qam/critic_loss=|qam/critic_updates=|qam/fine_updates=|qam/fine_policy_version=|qam/pending_update_credit=|qam/am_loss=|qam/fine_grad_norm=|qam/q_mean=|qam/q_std_heads=|qam/td_target_mean=|qam/terminal_adjoint_norm=' /root/autodl-tmp/experiment_exports/qam_formal_resume100_to380_20260801_v4/runtime/driver.log | tail -n 16
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
find /root/autodl-tmp/experiments/qam_formal_resume100_to380_20260801_v4/robotwin_adjust_bottle_qam_formal_resume100_to380_20260801_v4 -maxdepth 2 -type d -name 'global_step_*' -printf '%f\n' | sort -V | tail -n 3
