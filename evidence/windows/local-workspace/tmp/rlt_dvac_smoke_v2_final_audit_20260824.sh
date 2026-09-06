set -euo pipefail
runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v2/runtime
run_root=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v2
experiment=${run_root}/robotwin_adjust_bottle_rlt_teacher_dvac_w0to2_smoke_8env1c_v2

printf 'LIFECYCLE\n'
for f in source_head.txt started_at.txt finished_at.txt exit_code.txt resources_after.txt; do
  printf '%s\n' "---$f"; cat "$runtime/$f"
done
printf 'METRICS_LOG\n'
cat "$run_root/metrics.log"
printf 'CHECKPOINT_FILES\n'
find "$experiment/checkpoints/global_step_1" -type f -printf '%s %p\n' | sort -n
printf 'CHECKPOINT_TOTAL\n'
du -sh "$experiment/checkpoints/global_step_1"
printf 'TRACE_FILES\n'
find "$experiment/rlt_dvac" -type f -name '*.npz' -printf '%s %p\n' | sort
printf 'GIT\n'
cd /root/autodl-tmp/RLinf_rlt_teacher_dvac
git rev-parse HEAD
git status --short
git rev-list --left-right --count '@{upstream}...HEAD'
printf 'LIVE\n'
pgrep -af 'rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v2|ray::|raylet|gcs_server' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
