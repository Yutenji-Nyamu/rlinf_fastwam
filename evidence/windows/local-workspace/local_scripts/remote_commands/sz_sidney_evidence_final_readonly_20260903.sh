set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
OUT="$WT/evidence/smoke_20260903"
BRANCH=codex/sz-sidney-pi05-current-rlinf

printf 'branch=%s\n' "$(git -C "$WT" branch --show-current)"
printf 'head=%s\n' "$(git -C "$WT" rev-parse HEAD)"
printf 'remote=%s\n' "$(git -C "$WT" rev-parse "personal/$BRANCH")"
printf 'status_count=%s\n' "$(git -C "$WT" status --porcelain --untracked-files=all | wc -l)"
printf 'tracked_files=%s\n' "$(git -C "$WT" ls-files evidence/smoke_20260903 | wc -l)"
printf 'disk_files=%s\n' "$(find "$OUT" -type f | wc -l)"
printf 'file_bytes=%s\n' "$(find "$OUT" -type f -printf '%s\n' | awk '{sum += $1} END {print sum}')"
printf 'du_bytes=%s\n' "$(du -sb "$OUT" | awk '{print $1}')"
printf 'largest_file=%s\n' "$(find "$OUT" -type f -printf '%s %P\n' | sort -nr | head -n 1)"
git -C "$WT" log -3 --format='%H %P %s'
git -C "$WT" show --stat --oneline --summary HEAD
