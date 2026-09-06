#!/usr/bin/env bash
set -euo pipefail

REPO=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
WORKER="$REPO/rlinf/workers/actor/embodied_fsdp_actor_worker.py"
NESTED="$REPO/rlinf/utils/nested_dict_process.py"

echo '=== FILTER AND ADVANTAGE 220-430 ==='
nl -ba "$WORKER" | sed -n '220,430p'

echo '=== TRAIN LOOP 500-680 ==='
nl -ba "$WORKER" | sed -n '500,680p'

echo '=== PROCESS TRAIN HELPERS ==='
grep -n -E 'def process_nested_dict_for_train|def split_dict_to_chunk|def process_nested_dict_for_adv' "$NESTED"
nl -ba "$NESTED" | sed -n '180,360p'

echo 'SZ_GRPO_WORKER_NARROW_READONLY_OK'
