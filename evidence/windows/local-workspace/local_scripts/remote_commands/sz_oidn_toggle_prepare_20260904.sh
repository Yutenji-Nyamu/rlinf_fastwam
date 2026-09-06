#!/usr/bin/env bash
set -eu
trial=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-oidn-toggle-20260904
rl=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
fw=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
assets=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support/assets
out=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/diagnostics/oidn-toggle-20260904
for name in embodiments objects background_texture; do
  test -d "$assets/$name"
  if test -L "$trial/assets/$name"; then
    test "$(readlink "$trial/assets/$name")" = "$assets/$name"
  else
    test ! -e "$trial/assets/$name"
    ln -s "$assets/$name" "$trial/assets/$name"
  fi
done
mkdir -p "$out"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="$trial:$rl:$fw/src"
export FASTWAM_CONFIG_DIR="$fw/configs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$fw"
git rev-parse HEAD
git status --porcelain
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python "$trial/oidn_toggle_trial.py" --source-config /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2/runtime/resolved.yaml --output "$out" --denoiser oidn --prepare
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python "$trial/oidn_toggle_trial.py" --source-config /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2/runtime/resolved.yaml --output "$out" --denoiser none --prepare
