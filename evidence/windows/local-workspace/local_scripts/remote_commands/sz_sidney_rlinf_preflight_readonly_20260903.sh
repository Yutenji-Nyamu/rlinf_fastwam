#!/usr/bin/env bash
set -u

RAY_ADDRESS=172.17.0.1:6389
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
PI05_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl
FASTWAM_RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu16x16-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-pi0style-v3

date --iso-8601=seconds
hostname
whoami
id

echo GPU_STATE
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
echo GPU_PROCESSES
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits

echo HOST_MEMORY
free -h
awk '/^some / || /^full / {print}' /proc/pressure/memory
echo STORAGE
df -h / /home /data

echo TRAINING_PROCESSES
ps -eo user,pid,pgid,etimes,rss,args --sort=pid | grep -E '[t]rain_embodied_agent.py|[f]astwam-grpo-control-formal100|[s]tarvla' || true

echo RAY_STATUS
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status 2>&1 | sed -n '1,36p'

echo FASTWAM_STATUS
if test -d "$FASTWAM_RUN"; then
  test -f "$FASTWAM_RUN/runtime/driver.pid" && { printf 'driver_pid='; cat "$FASTWAM_RUN/runtime/driver.pid"; }
  test -f "$FASTWAM_RUN/runtime/exit_code.txt" && { printf 'exit_code='; cat "$FASTWAM_RUN/runtime/exit_code.txt"; }
  printf 'latest_step='; grep -h -oE 'Global Step: [0-9]+' "$FASTWAM_RUN/runtime/driver.log" 2>/dev/null | tail -1 || true
  printf 'latest_rollout='; grep -h -oE 'Generating Rollout Epochs: *[0-9]+/[0-9]+' "$FASTWAM_RUN/runtime/driver.log" 2>/dev/null | tail -1 || true
  printf 'fatal_count='; grep -Eic 'Traceback|CUDA out of memory|OOM|Fatal Python error|RayTaskError|NCCL.*error|non.?finite' "$FASTWAM_RUN/runtime/driver.log" 2>/dev/null || true
  tail -n 35 "$FASTWAM_RUN/runtime/driver.log" 2>/dev/null
else
  echo MISSING_FASTWAM_RUN
fi

echo PI05_WORKTREE
git -C "$PI05_WT" rev-parse HEAD 2>/dev/null || true
git -C "$PI05_WT" status --short 2>/dev/null || true
find "$PI05_WT/examples/embodiment/config" -maxdepth 1 -type f -iname '*pi05*robotwin*' -printf '%f\n' 2>/dev/null | sort

echo TASK_SUPPORT
python3 - <<'PY'
import json
from pathlib import Path
root=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl')
for rel in ['rlinf/envs/robotwin/seeds/train_seeds.json','rlinf/envs/robotwin/seeds/eval_seeds.json']:
    p=root/rel
    print(rel, p.exists())
    if p.exists():
        d=json.load(open(p))
        for k in ['adjust_bottle','move_stapler_pad','place_a2b_left','move_can_pot','place_mouse_pad']:
            v=d.get(k)
            print(k, None if v is None else len(v))
for needle in ['place_a2b_left','move_stapler_pad']:
    hits=[]
    for p in (root/'rlinf/envs/robotwin').rglob('*.py'):
        try:
            if needle in p.read_text(errors='ignore'):
                hits.append(str(p.relative_to(root)))
        except OSError:
            pass
    print('reward_hits',needle,sorted(hits)[:12])
PY
