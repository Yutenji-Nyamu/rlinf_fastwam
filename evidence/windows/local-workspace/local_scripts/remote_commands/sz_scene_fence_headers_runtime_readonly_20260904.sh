#!/usr/bin/env bash
set -euo pipefail
V=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
R=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
ls "$V/lib/python3.11/site-packages/sapien/include"
sed -n '1,250p' "$V/lib/python3.11/site-packages/sapien/include/svulkan2/renderer/rt_renderer.h"
find "$V/lib/python3.11/site-packages/sapien/include" -maxdepth 3 -type f \( -name '*denoiser*' -o -name 'logger.h' -o -name 'spdlog.h' \)
grep -R -n -E 'LD_LIBRARY_PATH|LD_PRELOAD|PYTHONPATH|env_vars|runtime_env' "$R/rlinf/scheduler" --include='*.py' | head -85
sed -n '1,200p' "$R/examples/embodiment/run_embodiment.sh"
cat /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1/runtime/environment.sh
date -Is
