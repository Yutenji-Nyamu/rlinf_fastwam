set +e
control_run=/root/autodl-tmp/experiments/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v9
control_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v9/runtime
method_run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v9
method_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v9/runtime
pair=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_pair_v9

echo EXITS_AND_TIMES
for d in "$control_runtime" "$method_runtime"; do
  echo "$d"
  cat "$d/started_at.txt" "$d/finished_at.txt" "$d/exit_code.txt"
done
echo PROCESSES
pgrep -af '[r]aylet|[g]cs_server|[t]rain_embodied_agent.py|[r]un_one_shared_smoke.sh' || true
echo GPU
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo MEMORY
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
echo RUN_SIZES
du -sh "$control_run" "$method_run" "$control_runtime" "$method_runtime" "$pair"
echo CHECKPOINTS
find "$control_run" "$method_run" -path '*checkpoints/global_step_1*' -maxdepth 8 -type f -printf '%p %s\n' | sort
echo KEY_ARTIFACT_COUNTS
for root in "$control_run" "$method_run"; do
  echo "$root"
  find "$root" -type f | wc -l
  find "$root" -type f \( -name '*.npz' -o -name 'events.out.tfevents*' -o -name '*.mp4' -o -name '*.jsonl' \) -printf '%p %s\n' | sort
done
echo FATAL_SCAN
for f in "$control_runtime/foreground.log" "$method_runtime/foreground.log"; do
  echo "$f"
  grep -En 'KeyError|CUDA out of memory|OutOfMemory|OOM|worker died|RayActorError|RuntimeError|ValueError|AssertionError' "$f" | tail -20
done
echo RESOURCE_HEADER_AND_TAIL
head -2 "$pair/paired_resources.csv"
tail -10 "$pair/paired_resources.csv"
echo GIT
git -C /root/autodl-tmp/RLinf_rlt_dvac_success_bc status --short
git -C /root/autodl-tmp/RLinf_rlt_dvac_success_bc rev-parse HEAD
