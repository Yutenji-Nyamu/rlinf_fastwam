#!/usr/bin/env bash
set -u
SRC=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
printf '%s\n' '=== SEED UTILS ==='
sed -n '1,240p' "$SRC/rlinf/envs/robotwin/seed_utils.py"
printf '%s\n' '=== OFFICIAL EVAL SEED CONFIG ==='
grep -R -nE '^([[:space:]]+)?seed:|use_fixed_reset_state_ids|total_num_envs|rollout_epoch|group_size' \
  "$SRC/evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml" \
  "$SRC/examples/embodiment/config/env/robotwin_adjust_bottle.yaml" 2>/dev/null || true
printf '%s\n' '=== COMPUTED TOTAL64 AND TWO SHARDS ==='
cd "$SRC" || exit 1
PYTHONPATH="$SRC" /root/autodl-tmp/RLinf/.venv/bin/python - <<'PY'
import json
from pathlib import Path
import torch
from rlinf.envs.robotwin.seed_utils import partition_success_seeds

p=Path('rlinf/envs/robotwin/seeds/eval_seeds.json')
seeds=torch.tensor(json.loads(p.read_text())['adjust_bottle']['success_seeds'], dtype=torch.long)
for base_seed in [0,1234]:
    by_rank=[]
    for rank in range(2):
        got=partition_success_seeds(seeds, base_seed=base_seed, seed_offset=rank, total_num_processes=2, num_group=32)
        by_rank.append([int(x) for x in got[:32]])
    selected=by_rank[0]+by_rank[1]
    print('base_seed',base_seed,'count',len(selected),'unique',len(set(selected)))
    print('rank0',by_rank[0])
    print('rank1',by_rank[1])
    print('shardA',by_rank[0][:16]+by_rank[1][:16])
    print('shardB',by_rank[0][16:]+by_rank[1][16:])
PY
