set -euo pipefail
python=/root/autodl-tmp/RLinf/.venv/bin/python
script=/tmp/compare_resolved_yaml_20260825.py
pair=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_pair_v9
historical=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_resume250_to480_20260730_v1/runtime/resolved.yaml
control=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v9/runtime/resolved.yaml
method=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v9/runtime/resolved.yaml
"$python" "$script" "$historical" "$control" >"$pair/historical_two_gpu_vs_single_control_leaf_diff.json"
"$python" "$script" "$control" "$method" >"$pair/control_vs_success_bc_method_leaf_diff.json"
sha256sum "$pair/historical_two_gpu_vs_single_control_leaf_diff.json" "$pair/control_vs_success_bc_method_leaf_diff.json"
