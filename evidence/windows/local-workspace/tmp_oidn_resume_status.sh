set -u
RAY_ADDRESS=172.17.0.1:6389
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2

date '+TIME %Y-%m-%d %H:%M:%S %Z'
pid=$(cat "$RUN/runtime/wrapper.pid" 2>/dev/null || true)
printf 'wrapper_pid=%s alive=' "$pid"
test -n "$pid" && kill -0 "$pid" 2>/dev/null && echo yes || echo no
test -f "$RUN/runtime/exit_code.txt" && printf 'exit=' && cat "$RUN/runtime/exit_code.txt"
printf 'GPU6_7\n'
nvidia-smi -i 6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>/dev/null | grep -E 'GPU-671b216c|GPU-4887e1b9' || true
printf 'NAMESPACES\n'
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os, ray, collections
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_fastwam_renderlife_status", logging_level="ERROR")
rows=ray.util.list_named_actors(all_namespaces=True)
print(dict(collections.Counter(x.get("namespace") for x in rows)))
for x in rows:
    if x.get("namespace") == "RLinf_1":
        print(x.get("name"), x.get("pid"))
ray.shutdown()
PY
printf 'BOUNDARY\n'
grep -aE 'Loading checkpoint|Loaded checkpoint|resume|Global Step:|Rollout Epoch:|OIDN Error|pthread_key_create|Fatal Python error|Traceback|CUDA out of memory|RayActorError' "$RUN/runtime/driver.log" 2>/dev/null | tail -60 || true
printf 'TAIL\n'
tail -n 35 "$RUN/runtime/driver.log" 2>/dev/null || true
printf 'RESOURCE\n'
tail -n 5 "$RUN/runtime/resource.csv" 2>/dev/null || true
