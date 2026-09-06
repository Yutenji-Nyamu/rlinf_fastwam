set -euo pipefail
control=/root/autodl-tmp/experiments/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v9/robotwin_adjust_bottle_rlt_single_gpu_control_success_bc_pair_smoke_v9
method=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v9/robotwin_adjust_bottle_rlt_single_gpu_success_episode_bc_dvac_smoke_v9
for root in "$control" "$method"; do
  echo "ROOT=$root"
  find "$root/checkpoints/global_step_1" -type f \( -name '*complete*.json' -o -name 'metadata.json' \) -print | sort | while read -r file; do
    echo "FILE=$file"
    cat "$file"
  done
done
