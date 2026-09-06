set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
RLPY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
cd "$WT"
$RLPY -m ruff check --fix toolkits/lerobot/sidney_pi05_parity.py
$RLPY -m ruff check toolkits/lerobot/sidney_pi05_parity.py
sed -n '1,28p' toolkits/lerobot/sidney_pi05_parity.py
