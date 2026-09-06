#!/usr/bin/env bash
set -euo pipefail
package_root=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_runtime_package
for f in "$package_root"/*.sh; do
  bash -n "$f"
done
grep -n 'CUDA_VISIBLE_DEVICES\|RAY_TMPDIR\|TMPDIR\|RAY_ADDRESS' "$package_root/run_one.sh"
sha256sum "$package_root"/*.sh
git -C /root/autodl-tmp/RLinf_rlt_teacher_dvac rev-parse HEAD
git -C /root/autodl-tmp/RLinf_rlt_teacher_dvac status --short
