set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
RLPY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
LRPY=/home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/python
cd "$WT"
$RLPY -m ruff format toolkits/lerobot/sidney_pi05_parity.py
$RLPY -m ruff check toolkits/lerobot/sidney_pi05_parity.py
$RLPY -m py_compile toolkits/lerobot/sidney_pi05_parity.py
$LRPY -m py_compile toolkits/lerobot/sidney_pi05_parity.py
$LRPY toolkits/lerobot/sidney_pi05_parity.py export-native --help | head -10
$RLPY toolkits/lerobot/sidney_pi05_parity.py export-rlinf --help | head -10
$RLPY - <<'PY'
import pathlib
import torch

root = pathlib.Path('/tmp/sidney-parity-cli-check')
root.mkdir(exist_ok=True)
base = {
    'camera_order': ('cam_high', 'cam_left_wrist', 'cam_right_wrist'),
    'images': torch.zeros(3, 1, 2, 2, 3),
    'image_masks': torch.ones(3, 1, dtype=torch.bool),
    'normalized_state14': torch.zeros(1, 14),
    'padded_state32': torch.zeros(1, 32),
    'tokens': torch.ones(1, 4, dtype=torch.long),
    'token_mask': torch.ones(1, 4, dtype=torch.bool),
    'model_actions32': torch.zeros(1, 50, 32),
    'final_actions14': torch.zeros(1, 50, 14),
}
native = dict(base, provenance={'backend':'native-lerobot-v0.6','model_path':'/native','source_revision':'rev'})
rlinf = dict(base, provenance={'backend':'current-rlinf-openpi','model_path':'/converted','source_revision':'rev','state_contract_digest':'sha256:test'})
torch.save(native, root/'native.pt')
torch.save(rlinf, root/'rlinf.pt')
PY
$RLPY toolkits/lerobot/sidney_pi05_parity.py compare \
  --native /tmp/sidney-parity-cli-check/native.pt \
  --rlinf /tmp/sidney-parity-cli-check/rlinf.pt | head -16
git diff --check
git diff --stat
