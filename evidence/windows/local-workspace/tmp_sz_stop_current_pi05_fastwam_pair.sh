set -euo pipefail

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
PI_RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2
FW_RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2

namespace_names() {
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE="$1" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ['RAY_ADDRESS'], namespace='codex_exact_pair_stop_probe', logging_level='ERROR')
print('\n'.join(sorted(r['name'] for r in ray.util.list_named_actors(all_namespaces=True) if r.get('namespace') == os.environ['TARGET_NAMESPACE'])))
ray.shutdown()
PY
}

verify_wrapper() {
  local run=$1 expected=$2 pid pgid
  pid=$(cat "$run/runtime/wrapper.pid")
  pgid=$(cat "$run/runtime/owned.pgid")
  test "$pid" = "$expected"
  test "$pgid" = "$expected"
  kill -0 "$pid"
  test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
  ps -o args= -p "$pid" | grep -F "$run/runtime/wrapper.sh" >/dev/null
}

verify_gpu_job() {
  local devices=$1 expected_job=$2 count=0 pid job
  while read -r pid; do
    [ -n "$pid" ] || continue
    test -r "/proc/$pid/environ"
    test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
    job=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    test "$job" = "$expected_job"
    count=$((count + 1))
  done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  test "$count" -eq 6
}

stop_tree() {
  local root=$1
  mapfile -t pids < <("$VENV/bin/python" - "$root" <<'PY'
import os, sys
root=int(sys.argv[1]); children={}
for name in os.listdir('/proc'):
    if not name.isdigit(): continue
    try:
        fields=open(f'/proc/{name}/stat').read().split()
        children.setdefault(int(fields[3]), []).append(int(name))
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError): pass
out=[]
def visit(pid):
    for child in children.get(pid, []): visit(child)
    out.append(pid)
visit(root)
print('\n'.join(map(str,out)))
PY
  )
  for pid in "${pids[@]}"; do
    [ ! -d "/proc/$pid" ] || test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
  done
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 90); do
    local alive=0
    for pid in "${pids[@]}"; do [ ! -d "/proc/$pid" ] || alive=$((alive + 1)); done
    [ "$alive" -eq 0 ] && return 0
    sleep 1
  done
  for pid in "${pids[@]}"; do [ ! -d "/proc/$pid" ] || kill -KILL "$pid"; done
}

cleanup_namespace() {
  RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE="$1" "$VENV/bin/python" - <<'PY'
import os,time,ray
target=os.environ['TARGET_NAMESPACE']
ray.init(address=os.environ['RAY_ADDRESS'], namespace='codex_exact_pair_stop_cleanup', logging_level='ERROR')
def names(): return sorted(r['name'] for r in ray.util.list_named_actors(all_namespaces=True) if r.get('namespace') == target)
rows=names()
assert len(rows) in (0,15), (target,rows)
managers={'CollectiveManager','DeviceLockManager','NodeManager','PortLockManager','WorkerManager'}
for name in sorted(rows,key=lambda x:(x in managers,x)):
    try: ray.kill(ray.get_actor(name,namespace=target),no_restart=True)
    except ValueError: pass
for _ in range(120):
    if not names(): break
    time.sleep(1)
else: raise RuntimeError((target,names()))
print(f'cleaned_namespace={target};actors={len(rows)}')
ray.shutdown()
PY
}

wait_job_gone() {
  local devices=$1 expected_job=$2 pid job remain
  for _ in $(seq 1 120); do
    remain=0
    while read -r pid; do
      [ -n "$pid" ] || continue
      [ -r "/proc/$pid/environ" ] || continue
      job=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
      [ "$job" != "$expected_job" ] || remain=$((remain + 1))
    done < <(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
    [ "$remain" -eq 0 ] && return 0
    sleep 1
  done
  return 1
}

stop_one() {
  local label=$1 run=$2 wrapper=$3 namespace=$4 devices=$5 job=$6 step observer before_count
  verify_wrapper "$run" "$wrapper"
  verify_gpu_job "$devices" "$job"
  mapfile -t before_names < <(namespace_names "$namespace")
  before_count=${#before_names[@]}
  test "$before_count" -eq 15
  step=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$run/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)
  observer=$(cat "$run/runtime/observer.pid" 2>/dev/null || true)
  stop_tree "$wrapper"
  if [ -n "$observer" ] && [ -d "/proc/$observer" ]; then stop_tree "$observer"; fi
  cleanup_namespace "$namespace"
  wait_job_gone "$devices" "$job"
  cat > "$run/runtime/stopped_by_user_for_model_expansion_20260902.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=user authorized stopping current pi0.5 and Fast-WAM runs for Sidney pi0.5 inference and Fast-WAM offload smoke
last_complete_step=$step
ray_job_id=$job
namespace=$namespace
cleanup=exact owned process tree and namespace only; shared Ray and other users unchanged
EOF
  printf '%s stopped step=%s wrapper=%s job=%s namespace=%s devices=%s\n' "$label" "$step" "$wrapper" "$job" "$namespace" "$devices"
}

echo "stop_started=$(TZ=Asia/Shanghai date --iso-8601=seconds)"
stop_one FASTWAM "$FW_RUN" 826980 RLinf_1 6,7 6e010000
stop_one PI05 "$PI_RUN" 1477572 RLinf 4,5 3f010000

test -z "$(namespace_names RLinf_1)"
test -z "$(namespace_names RLinf)"
! kill -0 826980 2>/dev/null
! kill -0 1477572 2>/dev/null

echo '===FINAL_GPU===' 
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo '===FINAL_GPU_COMPUTE===' 
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
echo '===SHARED_RAY===' 
ps -o user=,pid=,lstart=,etime=,stat=,args= -p 321933
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status | sed -n '1,24p'
echo '===RAM===' 
free -h
cat /proc/pressure/memory
echo "stop_finished=$(TZ=Asia/Shanghai date --iso-8601=seconds)"
