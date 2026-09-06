#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime
run=/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1/robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1

printf 'observed_at\t%s\n' "$(date -Iseconds)"
for name in started_at finished_at exit_code; do
  printf '%s\t' "${name}"
  cat "${runtime}/${name}.txt"
done
printf 'active_processes\n'
pgrep -af 'rlt_stage2_formal_8env_250c|train_embodied_agent.py|raylet|gcs_server' || true
printf 'gpu\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
for step in 225 250; do
  checkpoint="${run}/checkpoints/global_step_${step}"
  completion="${checkpoint}/actor/sac_components/rlt_trainer_state/rlt_trainer_state_complete.json"
  printf 'checkpoint_%s_size\t' "${step}"
  du -sh "${checkpoint}"
  printf 'checkpoint_%s_files\t%s\n' "${step}" "$(find "${checkpoint}" -type f | wc -l)"
  printf 'checkpoint_%s_completion\t' "${step}"
  cat "${completion}"
done
printf 'checkpoint_total\t'
du -sh "${run}/checkpoints"
printf 'run_total\t'
du -sh "${run}"
printf 'runtime_total\t'
du -sh "${runtime}"
printf 'tensorboard\n'
find "$(dirname "${run}")/tensorboard" -maxdepth 1 -type f -printf '%f\t%s\n'
printf 'disk\n'
df -h /root/autodl-tmp
