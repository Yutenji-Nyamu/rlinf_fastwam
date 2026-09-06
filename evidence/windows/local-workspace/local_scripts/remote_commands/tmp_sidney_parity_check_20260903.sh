set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
RLPY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
LRPY=/home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/python
ART=/data/chenyiteng/results/rlinf-shenzhen/sidney-pi05-parity-preflight
cd "$WT"
git status --short
$RLPY -m py_compile toolkits/lerobot/sidney_pi05_parity.py
$LRPY -m py_compile toolkits/lerobot/sidney_pi05_parity.py
$RLPY toolkits/lerobot/sidney_pi05_parity.py --help | head -8
mkdir -p "$ART"
$RLPY toolkits/lerobot/sidney_pi05_parity.py prepare --output "$ART/input.npz"
$RLPY - <<'PY'
import numpy as np
p='/data/chenyiteng/results/rlinf-shenzhen/sidney-pi05-parity-preflight/input.npz'
with np.load(p, allow_pickle=False) as d:
    print({k:(d[k].shape,str(d[k].dtype)) for k in d.files})
PY
$RLPY -m ruff check toolkits/lerobot/sidney_pi05_parity.py
