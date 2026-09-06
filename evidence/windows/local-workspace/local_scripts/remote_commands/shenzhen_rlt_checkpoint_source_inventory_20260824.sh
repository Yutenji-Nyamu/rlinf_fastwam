#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

printf '%s\n' '--- source identity ---'
git -C "$worktree" status --short
git -C "$worktree" rev-parse HEAD

printf '%s\n' '--- checkpoint call sites ---'
rg -n 'class Checkpoint|def state_dict|def save_checkpoint|get_state_dict\(|dcp\.save|FSDP\.optim_state_dict|_init_optim_state' \
  "$worktree/rlinf/workers/actor/fsdp_sac_policy_worker.py" \
  "$worktree/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py" \
  "$worktree/rlinf/utils" \
  "$worktree/rlinf/algorithms" \
  "$worktree/rlinf" \
  | head -n 240

printf '%s\n' '--- installed torch source identities ---'
"$venv/bin/python" - <<'PY'
import inspect
import torch
import torch.distributed.checkpoint.state_dict as sd
import torch.distributed.checkpoint.state_dict_saver as saver
import torch.distributed.fsdp._optim_utils as optim_utils

print("torch", torch.__version__)
print("state_dict", inspect.getsourcefile(sd))
print("state_dict_saver", inspect.getsourcefile(saver))
print("optim_utils", inspect.getsourcefile(optim_utils))
PY
