#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
date -Is
root=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
find "$root" -maxdepth 4 -type f -iname '*base*task*' -print
grep -Rnl --include='*.py' 'def gen_sparse_reward_data' "$root/envs" | while read src; do
  grep -n 'def gen_sparse_reward_data' "$src" | cut -d: -f1 | while read line; do sed -n "$((line+100)),$((line+265))p" "$src"; done
done
run=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1
find "$run" -maxdepth 3 -type f -name '*.sh' -print
head -n 65 "$run/runtime/wrapper.sh"
grep -nE 'batch_size|num_workers|distributed' /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/openpi/training/data_loader.py | head -n 45
find /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages -maxdepth 1 -iname '*openpi*' -o -iname '*robotwin*'
find /data/chenyiteng/projects/rlinf-shenzhen -maxdepth 2 -type d -iname '*openpi*' -o -iname '*asset*'
find /data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50 -maxdepth 2 -type f -printf '%P %s\n' | head -n 25
git -C /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc status --short
