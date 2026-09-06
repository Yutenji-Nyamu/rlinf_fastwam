set +e
control=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v9/runtime
method=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v9/runtime
pair=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_pair_v9
echo EXITS
for d in "$control" "$method"; do
  printf '%s ' "$d"
  if test -f "$d/exit_code.txt"; then cat "$d/exit_code.txt"; else echo running; fi
done
echo CONTROL_EVENTS
grep -E 'Global Step|global_step|replay/|Traceback|KeyError|RuntimeError|Saving|Evaluation|Training' "$control/foreground.log" | tail -40
echo METHOD_EVENTS
grep -E 'namespace conflict|Global Step|global_step|replay/|rlt_dvac|Traceback|KeyError|RuntimeError|Saving|Evaluation|Training' "$method/foreground.log" | tail -60
echo GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo RAM
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
echo RESOURCE_TAIL
tail -5 "$pair/paired_resources.csv"
