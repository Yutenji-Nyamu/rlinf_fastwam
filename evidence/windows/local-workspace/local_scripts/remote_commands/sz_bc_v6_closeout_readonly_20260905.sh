#!/usr/bin/env bash
set -eu
id
date -Is
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
run=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v6
git -C "$root" rev-parse HEAD
git -C "$root" status --short
tail -n 75 "$run/driver.log"
nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
free -h
df -h /data
vmstat 1 2
export PYTHONDONTWRITEBYTECODE=1
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import json,torch
from pathlib import Path
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v6')
episodes=torch.load(run/'success_data/rank_0/batch_000000.pt',map_location='cpu',weights_only=True)
print(json.dumps({'success_episodes':len(episodes),'query_records':sum(map(len,episodes)),'sft_calls':(run/'driver.log').read_text().count('Disabled gradient checkpointing for PI0Pytorch model')}))
PY
