#!/usr/bin/env bash
set -eu
id
date -Is
printf 'SSH_NOFILE_SOFT='; ulimit -Sn
printf 'SSH_NOFILE_HARD='; ulimit -Hn
grep 'Max open files' /proc/321933/limits /proc/322685/limits
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root"
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import json,re,torch
from pathlib import Path
run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v5')
episodes=torch.load(run/'success_data/rank_0/batch_000000.pt',map_location='cpu',weights_only=True)
log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','', (run/'driver.log').read_text())
print(json.dumps({'success_episodes':len(episodes),'query_records':sum(map(len,episodes)),
'sft_disable_lines':log.count('Disabled gradient checkpointing for PI0Pytorch model'),
'evaluation_entered':'Evaluating Rollout Epochs' in log,
'first_error':'vk::Device::getSemaphoreFdKHR: ErrorInitializationFailed',
'exit':(run/'exit_code.txt').read_text().strip()},indent=2))
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
import inspect
print(inspect.getsource(RoboTwinEnv.reset))
print(inspect.getsource(RoboTwinEnv.update_reset_state_ids))
PY
