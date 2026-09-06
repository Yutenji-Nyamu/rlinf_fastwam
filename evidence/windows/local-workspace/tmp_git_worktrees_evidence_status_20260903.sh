set -u
ROOT=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
echo 'RLINF_WORKTREES'
git -C "$ROOT" worktree list --porcelain 2>/dev/null || true
echo 'ROBOTWIN_WORKTREES'
git -C /data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support worktree list --porcelain 2>/dev/null || true
echo 'EVIDENCE_DIRS'
for wt in /data/chenyiteng/projects/rlinf-shenzhen/RLinf /data/chenyiteng/projects/rlinf-shenzhen/worktrees/*; do
  test -d "$wt/.git" -o -f "$wt/.git" || continue
  branch=$(git -C "$wt" branch --show-current 2>/dev/null || true)
  case "$branch" in codex/*)
    head=$(git -C "$wt" rev-parse --short=12 HEAD 2>/dev/null || true)
    ev=$(du -sb "$wt/evidence" 2>/dev/null | awk '{print $1}' || true)
    tracked=$(git -C "$wt" ls-files evidence 2>/dev/null | wc -l)
    dirty=$(git -C "$wt" status --porcelain --untracked-files=all 2>/dev/null | wc -l)
    printf '%s\t%s\t%s\ttracked=%s\tevidence_bytes=%s\tdirty=%s\n' "$branch" "$head" "$wt" "$tracked" "${ev:-0}" "$dirty"
  esac
done
