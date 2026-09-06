#!/usr/bin/env bash
set -eu
id
date -Is
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
cd "$root"
test "$(git branch --show-current)" = codex/sz-pi0-online-bc
test "$(git rev-parse HEAD)" = 5ae809d5a8c8c293540b57d14e423050c2c1e8d2
test -z "$(git diff --cached --name-only)"
git diff --check
git diff --stat
git diff -- rlinf/workers/env/env_worker.py examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml
git add -- docs/online_bc.md rlinf/workers/env/env_worker.py examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml tests/unit_tests/test_online_bc.py
git commit -m 'Set explicit per-EnvWorker file descriptor capacity for BC camera concurrency'
git push personal codex/sz-pi0-online-bc
local_head=$(git rev-parse HEAD)
remote_head=$(git ls-remote personal refs/heads/codex/sz-pi0-online-bc | cut -f1)
test "$local_head" = "$remote_head"
printf 'VERIFIED_HEAD=%s\n' "$local_head"
git status --short
nvidia-smi -i 6 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
free -h
df -h /data
