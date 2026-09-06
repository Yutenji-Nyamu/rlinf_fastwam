set -euo pipefail
pkg=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_runtime_package
repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc

for script in "$pkg"/*.sh; do
  bash -n "$script"
done
test "$(git -C "$repo" rev-parse HEAD)" = 5ace6d9792a74b8f8436f40382461e34d47d2aa3
test -z "$(git -C "$repo" status --short)"
/root/autodl-tmp/RLinf/.venv/bin/python - <<'PY'
import socket

for port in (48001, 49001):
    probe = socket.socket()
    try:
        probe.bind(("0.0.0.0", port))
    finally:
        probe.close()
print("PORTS_FREE")
PY
echo '[GPU]'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo '[MEMORY]'
printf 'current='; cat /sys/fs/cgroup/memory.current
printf 'high='; cat /sys/fs/cgroup/memory.high
printf 'max='; cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
echo '[RELEVANT_PROCESSES]'
ps -eo pid,pgid,cmd | grep -E 'train_embodied_agent|raylet|gcs_server' | grep -v grep || true
echo 'DUAL_SMOKE_PRELAUNCH_OK'
