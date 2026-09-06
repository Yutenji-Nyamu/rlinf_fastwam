set -u
runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1/runtime
run_root=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1
printf 'STATUS\n'
if [ -f "$runtime/driver_pid.txt" ]; then
  driver_pid=$(cat "$runtime/driver_pid.txt")
  ps -p "$driver_pid" -o pid=,etimes=,%cpu=,%mem=,cmd= || true
fi
for f in started_at.txt finished_at.txt exit_code.txt; do
  if [ -f "$runtime/$f" ]; then printf '%s=' "$f"; cat "$runtime/$f"; fi
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'PROGRESS\n'
grep -aE 'Generating Rollout|Evaluating Rollout|Global Step:|Saving checkpoint|rlt_dvac/|baseline_frozen|actor_loss|critic_loss|update_step' "$runtime/driver.log" | tail -120 || true
printf 'RECENT_ERRORS\n'
grep -aEi 'CUDA out of memory|OOM|worker died|RayActorError|NCCL.*error|nan|inf|Traceback|RuntimeError|ValueError|AssertionError' "$runtime/driver.log" | tail -40 || true
printf 'OUTPUTS\n'
find "$run_root" -maxdepth 6 -type f \( -name 'runner_step_metrics.csv' -o -name '*.npz' -o -name 'rlt_trainer_state.pt' -o -name 'metrics.log' \) -printf '%s %p\n' 2>/dev/null | sort -n | tail -60
printf 'RESOURCE_LAST\n'
tail -3 "$runtime/resources.csv" 2>/dev/null || true
