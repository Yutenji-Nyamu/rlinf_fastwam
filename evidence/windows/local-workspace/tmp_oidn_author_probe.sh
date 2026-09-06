set -eu
for repo in \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo \
  /data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support; do
  printf 'repo=%s\n' "$repo"
  git -C "$repo" log -1 --format='%an <%ae>'
  git -C "$repo" config --get user.name || true
  git -C "$repo" config --get user.email || true
done
