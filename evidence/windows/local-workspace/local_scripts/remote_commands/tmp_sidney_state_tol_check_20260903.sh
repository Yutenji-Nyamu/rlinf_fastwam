set -euo pipefail
cd /data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -m py_compile toolkits/lerobot/sidney_pi05_parity.py
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -m ruff check toolkits/lerobot/sidney_pi05_parity.py
git diff --check
GIT_PAGER=cat git diff -- toolkits/lerobot/sidney_pi05_parity.py
git status --short
