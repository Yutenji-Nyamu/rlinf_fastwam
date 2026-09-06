#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
printf 'HEAD='
git -C "$repo" rev-parse HEAD
printf 'UPSTREAM='
git -C "$repo" rev-parse '@{upstream}'
printf 'STATUS\n'
git -C "$repo" status --short
printf 'TRAIN\n'
ps -o pid,stat,etime,cmd -p 103802,103857,103859
printf 'DOC_TARGETS\n'
for file in \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
do
  if test -f "$repo/$file"; then
    sha256sum "$repo/$file"
  else
    printf 'MISSING %s\n' "$file"
  fi
done
