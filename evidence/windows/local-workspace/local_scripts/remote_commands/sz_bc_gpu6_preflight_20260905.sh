#!/usr/bin/env bash
set -eu
export PYTHONDONTWRITEBYTECODE=1
date -Is
id
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits
free -h
df -h / /data /home
cat /proc/pressure/memory /proc/pressure/io
ps -eo user,pid,pgid,etime,rss,args | grep -E '[t]rain_embodied_agent|[g]cs_server|[r]aylet|[E]nvWorker|[F]SDPPolicy|[M]ultiStepRollout' | cut -c 1-600
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
git --no-optional-locks -C "$root" status --short
git -C "$root" branch --show-current
git -C "$root" rev-parse HEAD
git -C "$root" show HEAD:examples/embodiment/config/model/pi0.yaml
git -C "$root" show HEAD:examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml
git -C "$root" ls-tree -r --name-only HEAD examples/embodiment/config | grep -E 'robotwin.*(grpo|dagger).*yaml'
find /data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50 -maxdepth 2 -type f -not -name '*.safetensors' -not -path '*/.cache/*' -printf '%p %s bytes\n'
for f in config.json config.yaml; do
  target=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50/$f
  if test -f "$target"; then cat "$target"; fi
done
grep -n -A 30 -B 10 'pi0_aloha_robotwin' "$root/rlinf/models/embodiment/openpi/dataconfig.py"
git -C "$root" worktree list --porcelain
