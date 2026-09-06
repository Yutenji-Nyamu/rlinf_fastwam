#!/usr/bin/env bash
set -u

echo '=== identity ==='
date --iso-8601=seconds
hostname
id
pwd

echo '=== accelerator ==='
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

echo '=== memory-and-disk ==='
free -h
df -h /root/autodl-tmp

echo '=== relevant-processes ==='
ps -eo pid,ppid,stat,etimes,cmd --sort=pid \
  | grep -E 'train_embodied_agent|raylet|gcs_server|python.*(ogpo|qam|rlt|dsrl|fastwam)' \
  | grep -v -E 'grep -E|remote_ogpo_20260807_readonly_refresh' \
  || true

echo '=== rlinf-base ==='
git -C /root/autodl-tmp/RLinf rev-parse HEAD
git -C /root/autodl-tmp/RLinf branch --show-current
git -C /root/autodl-tmp/RLinf status --short --branch
git -C /root/autodl-tmp/RLinf worktree list --porcelain

echo '=== ogpo-target-worktree ==='
if [ -e /root/autodl-tmp/RLinf_ogpo_pi0_robotwin ]; then
  echo PRESENT
  git -C /root/autodl-tmp/RLinf_ogpo_pi0_robotwin status --short --branch || true
else
  echo ABSENT
fi

echo '=== official-ogpo-clone ==='
find /root/autodl-tmp -maxdepth 2 -type d \( -iname 'OGPO_public' -o -iname '*ogpo*' \) -print 2>/dev/null | sort

echo '=== pi0-sft-and-norm ==='
model=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
if [ -d "$model" ]; then
  echo MODEL_PRESENT
  du -sh "$model"
  find "$model" -type f -name norm_stats.json -print -exec sha256sum {} \;
else
  echo MODEL_ABSENT
fi

echo '=== shared-runtime ==='
/root/autodl-tmp/RLinf/.venv/bin/python - <<'PY'
import importlib.util
import platform

print('python=' + platform.python_version())
for name in ('torch', 'ray', 'jax', 'flax'):
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f'{name}=ABSENT')
        continue
    module = __import__(name)
    print(f'{name}=' + str(getattr(module, '__version__', 'PRESENT')))
PY
