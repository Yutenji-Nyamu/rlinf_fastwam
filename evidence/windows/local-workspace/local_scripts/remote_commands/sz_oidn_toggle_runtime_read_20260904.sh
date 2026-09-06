#!/usr/bin/env bash
set -eu
rt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-oidn-toggle-20260904
rl=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
sed -n '330,465p' "$rt/robotwin/envs/vector_env.py"
grep -n -E 'def |worker_info|success' "$rl/rlinf/envs/robotwin/robotwin_env.py"
sed -n '185,330p' "$rl/rlinf/envs/robotwin/robotwin_env.py"
sed -n '330,435p' "$rl/rlinf/envs/robotwin/robotwin_env.py"
sed -n '1,150p' "$rt/envs/camera/camera.py"
find /data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6/assets -maxdepth 2 -printf '%y %p %l\n'
find "$rt/assets" -maxdepth 2 -printf '%y %p %l\n'
find /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2 -maxdepth 1 -type f
grep -R -n -E 'FASTWAM|PYTHONPATH|LD_LIBRARY_PATH|TORCH|HF_' /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2 --include='*.sh'
grep -n 'denoiser' /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/sapien/pysapien/render.pyi
head -40 "$rl/rlinf/envs/robotwin/seeds/eval_seeds.json"
