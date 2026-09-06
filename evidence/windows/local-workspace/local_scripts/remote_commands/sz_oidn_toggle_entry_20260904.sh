#!/usr/bin/env bash
set -eu
rl=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
rt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
find "$rl" -maxdepth 2 -type d -not -path '*/.git*' -not -path '*/docs*'
find "$rl/tools" -maxdepth 3 -type f -iname '*fastwam*' 2>/dev/null || true
find /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2 -maxdepth 8 -type f \( -name '*.distcp' -o -name '.metadata' \) -printf '%p %s\n'
cat /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2/runtime/resolved.yaml
sed -n '255,380p' "$rl/rlinf/models/embodiment/fastwam/builder.py"
sed -n '1,240p' "$rl/rlinf/models/embodiment/fastwam/robotwin_adapter.py"
sed -n '175,470p' "$rt/robotwin/envs/vector_env.py"
sed -n '1,150p' "$rt/envs/camera.py"
ls -ld "$rt/assets" "$rt/task_config" "$rt/description" "$rt/robotwin"
find "$rt/robotwin" -maxdepth 1 -type f
git -C "$rt" worktree list --porcelain
