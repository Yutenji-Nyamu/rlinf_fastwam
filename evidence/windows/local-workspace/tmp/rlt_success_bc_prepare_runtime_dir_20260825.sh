set -euo pipefail
dst=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_runtime_package
test ! -e "$dst"
mkdir -p "$dst"
