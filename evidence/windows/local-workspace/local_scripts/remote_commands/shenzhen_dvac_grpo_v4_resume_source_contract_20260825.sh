#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4
CKPT="$RUN/robotwin_grpo_openpi_dvac_global_z_matched/checkpoints/global_step_30"

echo '=== embodied_runner_resume_and_loop ==='
sed -n '165,205p' "$WT/rlinf/runners/embodied_runner.py"
grep -n -A8 -B6 -E 'range\(|while .*global_step|max_steps|global_step \+=' "$WT/rlinf/runners/embodied_runner.py" | tail -n 100

echo '=== dvac_sidecar_save_load ==='
sed -n '875,955p' "$WT/rlinf/workers/actor/embodied_fsdp_actor_worker.py"

echo '=== sidecar_contract ==='
python3 - "$CKPT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]) / 'actor'
for p in sorted(root.glob('dvac_state_rank*.json')):
    d = json.loads(p.read_text())
    stats = d['recent_stats']
    print(p.name, 'rank=', d['actor_rank'], 'world=', d['actor_world_size'],
          'runner_step=', d['runner_step'], 'mode=', d['mode'],
          'L=', d['selected_l'], 'recent_keys=', sorted(stats),
          'history_len=', len(stats.get('recent_stats', stats.get('history', []))))
PY

echo SZ_DVAC_GRPO_V4_RESUME_SOURCE_CONTRACT_OK
