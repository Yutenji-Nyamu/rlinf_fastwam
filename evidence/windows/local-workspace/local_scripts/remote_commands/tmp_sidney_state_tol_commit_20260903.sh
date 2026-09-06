set -euo pipefail
cd /data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
git add toolkits/lerobot/sidney_pi05_parity.py
git commit -m "test: allow float32 state parity tolerance"
git push personal codex/sz-sidney-pi05-current-rlinf
git rev-parse HEAD
git status --short
