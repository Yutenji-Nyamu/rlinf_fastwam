#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
model=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
seed_file="$root/rlinf/envs/robotwin/seeds/eval_seeds.json"

cd "$root"
echo "RLINF_HEAD=$(git rev-parse HEAD)"
echo "RLINF_BRANCH=$(git branch --show-current)"
echo "RLINF_STATUS_BEGIN"
git status --short
echo "RLINF_STATUS_END"
echo "ROBOTWIN_HEAD=$(git -C "$robotwin" rev-parse HEAD)"
test -d "$model"
sha256sum "$seed_file"

CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
  "$venv/bin/python" - "$seed_file" <<'PY'
import json
import sys

import torch

with open(sys.argv[1], encoding="utf-8") as handle:
    seeds = torch.as_tensor(
        json.load(handle)["adjust_bottle"]["success_seeds"], dtype=torch.long
    )
generator = torch.Generator()
generator.manual_seed(0)
shuffled = seeds[torch.randperm(seeds.numel(), generator=generator)]
print(f"SUCCESS_SEED_COUNT={seeds.numel()}")
print("P0_ONE_ENV_WORKER_FIXED_IDS=" + ",".join(map(str, shuffled[:2].tolist())))
PY

echo 'GPU_OR_SIM_USED=0'
