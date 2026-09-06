set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
cd "$WT"
git diff --check 256eeeb4459b4bd5db85bfc6a0eb315771e8c38c..HEAD
git diff --stat 256eeeb4459b4bd5db85bfc6a0eb315771e8c38c..HEAD
sha256sum toolkits/lerobot/sidney_pi05_parity.py
git status --short
git rev-parse HEAD
