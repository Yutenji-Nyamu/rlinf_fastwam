set -u
RAY_ADDRESS=172.17.0.1:6389
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2
date '+TIME %Y-%m-%d %H:%M:%S %Z'
printf 'PIDS\n'
for name in wrapper observer; do
  pid=$(cat "$RUN/runtime/$name.pid" 2>/dev/null || true)
  test -z "$pid" || ps -o user,pid,ppid,pgid,etimes,rss,stat,args -p "$pid"
done
printf 'DRIVER_MARKERS\n'
grep -anE 'Global Step:|Generating Rollout Epochs|Fatal|Traceback|Error|Exception|OOM|out of memory' "$RUN/runtime/driver.log" 2>/dev/null | tail -80 || true
printf 'DRIVER_TAIL\n'
tail -n 100 "$RUN/runtime/driver.log" 2>/dev/null || true
printf 'GPU\n'
nvidia-smi -i 6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'NAMESPACES\n'
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import ray
ray.init(address='172.17.0.1:6389', namespace='codex_fastwam_256_status', logging_level='ERROR')
rows = ray.util.list_named_actors(all_namespaces=True)
for ns in ('RLinf', 'RLinf_1', 'RLinf_2'):
    names = sorted(row['name'] for row in rows if row.get('namespace') == ns)
    print(ns, len(names), ','.join(names))
ray.shutdown()
PY
