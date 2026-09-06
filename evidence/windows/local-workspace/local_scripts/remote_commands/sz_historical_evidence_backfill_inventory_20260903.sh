#!/usr/bin/env bash
set -euo pipefail

printf 'TIME='; date '+%F %T %Z'
printf '\n=== ACTIVE_RLINF ===\n'
ps -eo user:16,pid=,pgid=,etimes=,args= --sort=pid | grep -E 'chenyiteng.*(ray job submit|runner\.py|resource_observer)' | grep -v grep || true

printf '\n=== RESULT_ROOTS ===\n'
for root in \
  /data/chenyiteng/results/rlinf-shenzhen \
  /data/chenyiteng/results/rlinf-rlt \
  /data/chenyiteng/results/rlinf-dsrl \
  /data/chenyiteng/results/rlinf-rlt-dvac-pure; do
  test -d "$root" || continue
  printf 'ROOT=%s\n' "$root"
  find "$root" -mindepth 2 -maxdepth 4 -type d \( -name runs -o -name packets -o -name exports -o -name bundles \) -prune -o -type f -name exit_code.txt -printf '%h\n' 2>/dev/null | sort -u | sed 's/^/EXIT_RUN=/'
done

printf '\n=== HIGH_INFO_ARCHIVES ===\n'
find /data/chenyiteng/results -xdev -type f \
  \( -iname '*.zip' -o -iname '*.tar.gz' -o -iname '*bundle*' -o -iname '*summary*.json' -o -iname '*manifest*.json' \) \
  -size -64M -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' 2>/dev/null | sort -k3,3 | sed -n '1,600p'

printf '\n=== WORKTREES ===\n'
root=/data/chenyiteng/projects/rlinf-shenzhen
for wt in "$root"/RLinf "$root"/worktrees/*; do
  test -d "$wt" || continue
  git -C "$wt" rev-parse --is-inside-work-tree >/dev/null 2>&1 || continue
  printf 'WT=%s\tBRANCH=%s\tHEAD=%s\tDIRTY=%s\n' \
    "$wt" "$(git -C "$wt" branch --show-current)" "$(git -C "$wt" rev-parse HEAD)" \
    "$(git -C "$wt" status --porcelain --untracked-files=all | wc -l)"
done

printf '\n=== BRANCHES ===\n'
git -C "$root/RLinf" for-each-ref --format='%(refname:short)\t%(objectname)' refs/heads refs/remotes/personal/codex | sort
