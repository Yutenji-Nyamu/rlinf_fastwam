#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
rl=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
rt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
find "$rl/tests" "$rl/scripts" "$rl/examples" -iname '*fastwam*' -type f
find /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2 -maxdepth 5 -type f \( -name '*.distcp' -o -name '.metadata' -o -name '*resolved*' \) -printf '%p %s\n'
sed -n '1,260p' "$rl/rlinf/models/embodiment/fastwam/builder.py"
sed -n '1,250p' "$rl/rlinf/models/embodiment/fastwam/fastwam_policy.py"
sed -n '1,180p' "$rl/examples/embodiment/config/model/fastwam.yaml"
sed -n '1,180p' "$rl/examples/embodiment/config/env/robotwin_move_stapler_pad_fastwam.yaml"
sed -n '1,150p' "$rt/envs/_base_task.py"
sed -n '1,180p' "$rt/robotwin/envs/vector_env.py"
