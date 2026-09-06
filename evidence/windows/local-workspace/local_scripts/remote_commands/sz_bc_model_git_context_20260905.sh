#!/usr/bin/env bash
set -eu
export PYTHONDONTWRITEBYTECODE=1
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
date -Is
git -C "$root" ls-tree -r --name-only HEAD examples/embodiment/config | grep -E 'robotwin.*(grpo|dagger).*yaml'
find /data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50 -maxdepth 2 -type f -not -name '*.safetensors' -not -path '*/.cache/*' -printf '%p %s bytes\n'
for f in config.json config.yaml; do
  target=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50/$f
  if test -f "$target"; then cat "$target"; fi
done
grep -n -A 30 -B 10 'pi0_aloha_robotwin' "$root/rlinf/models/embodiment/openpi/dataconfig.py"
git -C "$root" worktree list --porcelain
ps -o user:16,pid,lstart,etime,pcpu,rss,args -p 201345,200301,1568973,3176215
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
git -C "$root" config --get remote.personal.url
