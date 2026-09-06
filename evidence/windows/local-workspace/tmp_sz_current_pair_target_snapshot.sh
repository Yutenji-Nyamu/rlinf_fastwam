set -u
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
PI_RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2
FW_RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2

echo '===TIME==='; TZ=Asia/Shanghai date --iso-8601=seconds
for item in "PI|$PI_RUN|RLinf|4,5" "FW|$FW_RUN|RLinf_1|6,7"; do
  IFS='|' read -r label run ns devs <<EOF
$item
EOF
  echo "===${label}_IDENTITY==="
  for f in wrapper.pid owned.pgid observer.pid exit_code.txt started_at.txt finished_at.txt; do
    if [ -e "$run/runtime/$f" ]; then printf '%s=' "$f"; cat "$run/runtime/$f"; fi
  done
  pid=$(cat "$run/runtime/wrapper.pid")
  printf 'wrapper_alive='; kill -0 "$pid" 2>/dev/null && echo yes || echo no
  ps -o user=,pid=,ppid=,pgid=,sid=,lstart=,etime=,stat=,args= -p "$pid" || true
  printf 'last_complete_step='; grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$run/runtime/driver.log" | tail -n1 || true
  printf 'last_rollout_epoch='; grep -aoE 'Generating Rollout Epochs:[^\r\n]*' "$run/runtime/driver.log" | tail -n1 || true
  printf 'fatal_count='; grep -aEic 'Traceback|CUDA out of memory|OutOfMemoryError|Fatal Python error|SIGSEGV|nonfinite|NaN|NCCL.*error|worker.*died|invalid handle' "$run/runtime/driver.log" || true
  echo 'latest_metrics:'; tail -n 12 "$run/metrics.log" 2>/dev/null || true
  echo 'checkpoints:'; find "$run" -maxdepth 5 -type d -name 'global_step_*' -printf '%T@\t%p\n' 2>/dev/null | sort -n | tail -12
  echo 'gpu_jobs:'
  while read -r gp; do
    [ -n "$gp" ] || continue
    job=''; [ -r "/proc/$gp/environ" ] && job=$(tr '\0' '\n' < "/proc/$gp/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    printf 'pid=%s user=%s job=%s args=' "$gp" "$(ps -o user= -p "$gp" | xargs)" "$job"
    ps -o comm= -p "$gp"
  done < <(nvidia-smi -i "$devs" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  echo "namespace=$ns"
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE="$ns" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ['RAY_ADDRESS'], namespace='codex_readonly_pair_snapshot', logging_level='ERROR')
rows=sorted(r['name'] for r in ray.util.list_named_actors(all_namespaces=True) if r.get('namespace') == os.environ['TARGET_NAMESPACE'])
print('actor_count=', len(rows), 'names=', ','.join(rows), sep='')
ray.shutdown()
PY
done

echo '===ALL_GPU_COMPUTE_POSTTARGET===' 
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
