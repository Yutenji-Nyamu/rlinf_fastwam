set -u
echo 'RESULT_DIRS'
find /data/chenyiteng/results -mindepth 2 -maxdepth 5 -type d \( -iname '*rlt*' -o -iname '*dsrl*' -o -iname '*ppo*formal*' -o -iname '*fastwam*' \) 2>/dev/null | sort | sed -n '1,500p'

echo 'TARGET_GIT'
for wt in \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/st-dvac-local-shard \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-action-dvac-adv \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421 \
 /data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421; do
  test -e "$wt/.git" || continue
  branch=$(git -C "$wt" branch --show-current)
  head=$(git -C "$wt" rev-parse HEAD)
  remote=$(git -C "$wt" ls-remote personal "refs/heads/$branch" 2>/dev/null | awk '{print $1}')
  dirty=$(git -C "$wt" status --porcelain --untracked-files=all | wc -l)
  printf '%s\t%s\tremote=%s\tdirty=%s\t%s\n' "$branch" "$head" "${remote:-NONE}" "$dirty" "$wt"
done
