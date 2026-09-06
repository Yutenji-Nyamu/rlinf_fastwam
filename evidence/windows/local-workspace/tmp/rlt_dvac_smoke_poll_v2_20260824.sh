set -u
runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v2/runtime
run_root=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v2
printf 'STATUS\n'
if [ -f "$runtime/driver_pid.txt" ]; then
  driver_pid=$(cat "$runtime/driver_pid.txt")
  ps -p "$driver_pid" -o pid=,etimes=,%cpu=,%mem=,cmd= || true
fi
for f in source_head.txt started_at.txt finished_at.txt exit_code.txt; do
  if [ -f "$runtime/$f" ]; then printf '%s=' "$f"; cat "$runtime/$f"; fi
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'PROGRESS\n'
grep -aE 'Generating Rollout|Evaluating Rollout|Global Step:|Saving checkpoint|rlt_dvac/|actor_loss|critic_loss|update_step' "$runtime/driver.log" | tail -100 || true
printf 'ERRORS\n'
grep -aEi 'Watchdog caught|CUDA out of memory|OOM|worker died|RayActorError|NCCL.*error|RuntimeError|ValueError|AssertionError' "$runtime/driver.log" | tail -30 || true
printf 'OUTPUTS\n'
find "$run_root" -maxdepth 7 -type f \( -name '*.npz' -o -name 'rlt_trainer_state.pt' -o -name 'metrics.log' \) -printf '%s %p\n' 2>/dev/null | sort -n | tail -40
printf 'RESOURCE_LAST\n'
tail -2 "$runtime/resources.csv" 2>/dev/null || true
