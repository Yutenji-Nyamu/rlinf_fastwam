#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
cd "$WT"
git apply --check /tmp/rlt_worker_unresolved_contract.patch
git apply /tmp/rlt_worker_unresolved_contract.patch

echo '== narrow dataset inventory =='
for root in /data/chenyiteng/datasets /data/chenyiteng/models; do
  if [[ -d "$root" ]]; then
    find "$root" -maxdepth 3 -mindepth 1 \
      \( -iname '*robotwin*' -o -iname '*clean50*' -o -iname '*processed*' \) \
      -print | sort | head -200
  fi
done

echo '== status =='
git status --short
echo 'RLT_GUARD_AND_UPLOADS_READY'
