set -eu
F=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf/toolkits/lerobot/sidney_pi05_parity.py
nl -ba "$F" | sed -n '260,345p'
