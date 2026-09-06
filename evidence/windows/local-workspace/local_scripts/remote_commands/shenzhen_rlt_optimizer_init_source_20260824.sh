#!/usr/bin/env bash
set -u

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
grep -R -n --include='*.py' -E 'fake_optimizer_step|init_optimizer_state|optimizer.state' "$worktree/rlinf/hybrid_engines/fsdp" "$worktree/rlinf/workers/actor" \
  | head -n 260 || true
nl -ba "$worktree/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py" | sed -n '535,605p'
nl -ba "$worktree/rlinf/hybrid_engines/fsdp/strategy/fsdp.py" | sed -n '180,225p'
grep -n -A90 -B20 'qf_optimizer' "$worktree/rlinf/workers/actor/fsdp_sac_policy_worker.py" | head -n 260 || true
