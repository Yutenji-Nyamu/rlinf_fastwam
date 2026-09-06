#!/usr/bin/env bash
set -u
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

python3 - "$WT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
for rel in ['rlinf/envs/robotwin/seeds/train_seeds.json','rlinf/envs/robotwin/seeds/eval_seeds.json']:
    p=root/rel
    d=json.load(open(p))
    print('FILE',rel,'type',type(d).__name__)
    for k in ['adjust_bottle','move_stapler_pad','place_a2b_left','move_can_pot','place_mouse_pad']:
        v=d.get(k)
        print(k,repr(v)[:500])
PY

echo ROBOTWIN_REWARD_TASK_HITS
grep -RIl --exclude-dir=.git -E 'place_a2b_left|move_stapler_pad' "$ROBOTWIN" 2>/dev/null | head -60

echo EXISTING_PI05_SMOKE_ASSETS
find /data/chenyiteng/results/rlinf-shenzhen/pi05 -maxdepth 3 -type f \
  \( -name resolved.yaml -o -name command.txt -o -name exit_code.txt \) -printf '%p\n' 2>/dev/null | sort | tail -80

echo CURRENT_CONFIG_MODEL_CONTRACT
grep -nE 'task_name|group_size|global_batch_size|micro_batch_size|rollout_epoch|total_num_envs|max_episode_steps|num_steps|model_path|update_epoch|checkpoint_format|enable_offload|adapt_to_pi|extra_delta_transform|action_chunk_size' \
  "$WT/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05.yaml" 2>/dev/null

echo AVAILABLE_TASK_CONFIGS
find "$ROBOTWIN" -path '*/task_config/*' -type f \( -iname '*place_a2b_left*' -o -iname '*move_stapler_pad*' \) -printf '%p\n' 2>/dev/null | head -50
