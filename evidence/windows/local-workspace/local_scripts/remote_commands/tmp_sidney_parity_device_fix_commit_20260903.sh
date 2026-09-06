set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
cd "$WT"
git add toolkits/lerobot/sidney_pi05_parity.py
git diff --cached --check
git commit -m "fix(openpi): move native parity model to selected device"
git push personal HEAD:codex/sz-sidney-pi05-current-rlinf
git status --short
git rev-parse HEAD
git show --stat --oneline --summary HEAD
