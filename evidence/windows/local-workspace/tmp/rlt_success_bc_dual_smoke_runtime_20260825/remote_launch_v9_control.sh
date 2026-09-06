set -euo pipefail
if pgrep -af '[r]aylet|[g]cs_server|[t]rain_embodied_agent.py|[r]un_one_shared_smoke.sh'; then
  exit 17
fi
test ! -e /root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_pair_v9
STAGGER_METHOD=1 bash /root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_runtime_package/launch_shared_dual_smoke.sh
