#!/usr/bin/env bash
set -u

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
ckpt=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3/robotwin_adjust_bottle_rlt_stage2_current_ar_8env250_output_isolated_v2/checkpoints/global_step_25

printf '%s\n' '--- checkpoint files exact ---'
find "$ckpt" -printf '%y %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort
printf '%s\n' '--- base checkpoint implementation ---'
nl -ba "$worktree/rlinf/hybrid_engines/fsdp/strategy/base.py" | sed -n '170,285p'
printf '%s\n' '--- SAC worker checkpoint implementation ---'
nl -ba "$worktree/rlinf/workers/actor/fsdp_sac_policy_worker.py" | sed -n '730,810p'
printf '%s\n' '--- RLT state save implementation ---'
nl -ba "$worktree/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py" | sed -n '780,970p'

