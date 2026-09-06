set -euo pipefail
pkg=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_runtime_package
repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
for script in "$pkg"/start_shared_ray_head.sh "$pkg"/run_one_shared_smoke.sh "$pkg"/cleanup_shared_after_both.sh "$pkg"/launch_shared_dual_smoke.sh; do
  bash -n "$script"
done
test "$(git -C "$repo" rev-parse HEAD)" = f01bbb95d65f3dd8df395c5f922a5e55bf84114a
test -z "$(git -C "$repo" status --short)"
/root/autodl-tmp/RLinf/.venv/bin/python - <<'PY'
import socket
s = socket.socket()
try:
    s.bind(("0.0.0.0", 50001))
finally:
    s.close()
print("SHARED_PORT_FREE")
PY
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'memory_current='; cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
if ps -eo cmd | grep -E 'train_embodied_agent|raylet|gcs_server' | grep -v grep; then
  exit 2
fi
echo 'SHARED_PRELAUNCH_OK'
