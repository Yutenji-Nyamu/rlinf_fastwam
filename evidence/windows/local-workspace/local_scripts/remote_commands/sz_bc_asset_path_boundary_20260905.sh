#!/usr/bin/env bash
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
sed -n '35,150p' "$root/rlinf/envs/robotwin/robotwin_env.py"
sed -n '1,85p' "$robotwin/envs/_GLOBAL_CONFIGS.py"
sed -n '1,42p' "$robotwin/envs/utils/rand_create_cluttered_actor.py"
ls -ld "$robotwin/assets" "$robotwin/assets/objects/objaverse/list.json"
git -C "$root" worktree list --porcelain
grep -n 'assets_path' /data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2/runtime/resolved.yaml
git -C "$robotwin" status --short
