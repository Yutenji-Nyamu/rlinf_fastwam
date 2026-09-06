set -eu
F=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf/rlinf/runners/embodied_runner.py
nl -ba "$F" | sed -n '35,245p'
nl -ba "$F" | sed -n '330,470p'
