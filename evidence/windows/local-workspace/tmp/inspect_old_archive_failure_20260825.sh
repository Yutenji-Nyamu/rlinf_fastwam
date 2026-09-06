set -euo pipefail
queue_root="/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_queue"
runtime_package="/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_runtime_package"
echo "=== archive failure ==="
cat "$queue_root/archive_old.log" 2>/dev/null || true
echo "=== archive script ==="
sed -n '1,260p' "$runtime_package/archive_old.sh"
echo "=== old runtime files ==="
find /root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime -maxdepth 2 -type f -printf '%s %p\n' | sort | tail -n 120
echo "=== old run top files ==="
find /root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1 -maxdepth 3 -type f -printf '%s %p\n' | sort | tail -n 160
