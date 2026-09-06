set -eu
F=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf/rlinf/config.py
nl -ba "$F" | sed -n '880,940p'
