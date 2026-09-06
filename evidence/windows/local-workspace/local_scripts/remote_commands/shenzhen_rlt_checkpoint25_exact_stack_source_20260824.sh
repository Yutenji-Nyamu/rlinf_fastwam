#!/usr/bin/env bash
set -u

venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
torchroot=$venv/lib/python3.11/site-packages/torch
worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421

"$venv/bin/python" - <<'PY'
import torch
print("torch_version=" + torch.__version__)
print("torch_git=" + str(torch.version.git_version))
print("torch_cuda=" + str(torch.version.cuda))
PY

printf '%s\n' '--- RLinf Checkpoint wrapper ---'
grep -R -n --include='*.py' 'class Checkpoint' "$worktree/rlinf" | head -n 30 || true
file=$(grep -R -l --include='*.py' 'class Checkpoint' "$worktree/rlinf/hybrid_engines/fsdp" | head -n 1)
printf 'file=%s\n' "$file"
if [ -n "$file" ]; then
  line=$(grep -n 'class Checkpoint' "$file" | head -n 1 | cut -d: -f1)
  start=$((line - 20)); end=$((line + 180))
  nl -ba "$file" | sed -n "${start},${end}p"
fi

printf '%s\n' '--- RLinf warmup optimizer state ---'
grep -R -n --include='*.py' 'def warmup_optimizer_state' "$worktree/rlinf" | head -n 20 || true
file=$(grep -R -l --include='*.py' 'def warmup_optimizer_state' "$worktree/rlinf" | head -n 1)
if [ -n "$file" ]; then
  line=$(grep -n 'def warmup_optimizer_state' "$file" | head -n 1 | cut -d: -f1)
  start=$((line - 20)); end=$((line + 100))
  nl -ba "$file" | sed -n "${start},${end}p"
fi

printf '%s\n' '--- torch state_dict optimizer initialization ---'
grep -n -E 'def _init_optim_state|def get_optimizer_state_dict|def _get_optim_state_dict|def get_state_dict' "$torchroot/distributed/checkpoint/state_dict.py" || true
nl -ba "$torchroot/distributed/checkpoint/state_dict.py" | sed -n '585,725p'
nl -ba "$torchroot/distributed/checkpoint/state_dict.py" | sed -n '1110,1275p'

printf '%s\n' '--- torch DCP save call order ---'
grep -n -E 'def _stateful_to_state_dict|def _save_state_dict|def save\(' "$torchroot/distributed/checkpoint/state_dict_saver.py" || true
nl -ba "$torchroot/distributed/checkpoint/state_dict_saver.py" | sed -n '45,180p'
nl -ba "$torchroot/distributed/checkpoint/state_dict_saver.py" | sed -n '220,360p'

printf '%s\n' '--- exact last warning source ---'
nl -ba "$torchroot/distributed/fsdp/_optim_utils.py" | sed -n '1135,1205p'

printf '%s\n' '--- process group env subset ---'
for pid in 391536 391538 371579 371581; do
  printf 'pid=%s\n' "$pid"
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
    | grep -E '^(MASTER_ADDR|MASTER_PORT|RANK|LOCAL_RANK|WORLD_SIZE|GROUP_RANK|TORCHELASTIC|NCCL_|GLOO_)=' \
    | sort || true
done

