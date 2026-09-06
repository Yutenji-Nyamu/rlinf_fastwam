#!/usr/bin/env bash
set -eu
id
date -Is
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
cd "$root"
test "$(git branch --show-current)" = codex/sz-pi0-online-bc
test "$(git rev-parse HEAD)" = 9876c28de25bba429d10e2c8b4c85fb19b24c387
test -z "$(git diff --cached --name-only)"
git diff --check
git diff --stat
git diff -- examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml rlinf/workers/actor/fsdp_online_bc_policy_worker.py
git add -- docs/online_bc.md examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml rlinf/workers/actor/fsdp_online_bc_policy_worker.py tests/unit_tests/test_online_bc.py
git commit -m 'Align expert-only joint SFT FSDP wrapping and explicit BC checkpoint format'
git push personal codex/sz-pi0-online-bc
local_head=$(git rev-parse HEAD)
remote_head=$(git ls-remote personal refs/heads/codex/sz-pi0-online-bc | cut -f1)
test "$local_head" = "$remote_head"
printf 'VERIFIED_HEAD=%s\n' "$local_head"
git status --short
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
free -h
df -h /data
vmstat 1 2
ps -p 3176215,321933,322685 -o user,pid,etime,stat,comm
