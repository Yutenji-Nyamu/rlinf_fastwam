set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
cd "$WT"
find toolkits -maxdepth 2 -type f | head -50 || true
find tests -maxdepth 2 -type d | sort | head -60
git status --short
git log -3 --oneline
