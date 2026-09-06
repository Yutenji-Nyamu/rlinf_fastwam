set -euo pipefail
runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime
run_root=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
printf 'LIFECYCLE\n'
for f in started_at.txt exit_code.txt finished_at.txt; do
  if [ -f "$runtime/$f" ]; then printf '%s=' "$f"; cat "$runtime/$f"; else printf '%s=MISSING\n' "$f"; fi
done
printf 'WORKER_COUNTS\n'
for pattern in 'ray::RLTACFSDPPolicy' 'ray::MultiStepRolloutWorker' 'ray::EnvWorker'; do
  printf '%s=' "$pattern"
  pgrep -af "$pattern" | grep -v 'pgrep -af' | wc -l
done
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'MEMORY\n'
printf 'current='; cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
printf 'RESOURCE_LAST\n'
tail -n 1 "$runtime/resources.csv"
printf 'PROGRESS\n'
tail -n 1200 "$runtime/foreground.log" | grep -E 'Global Step|Generate trajectories|generate trajectories|rollout|Rollout|episode|success|replay|Loaded|initialized|ready' | tail -n 80 || true
printf 'FATAL\n'
tail -n 1600 "$runtime/foreground.log" | grep -Ei 'cuda out of memory|nccl.*(error|timeout)|work(er|ercrashederror).*(died|failed)|raytaskerror|floatingpointerror|killed due to memory pressure' | tail -n 60 || true
printf 'OUTPUT_DIR\n'
find "$run_root" -maxdepth 2 -type f -printf '%s %p\n' | sort | tail -n 60
