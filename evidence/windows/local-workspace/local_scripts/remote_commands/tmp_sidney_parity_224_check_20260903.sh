set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
LR=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
MODEL=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
INPUT=/tmp/sidney-pi05-parity-input224.npz
RLPY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
LRPY=/home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/python
cd "$WT"
$RLPY -m ruff format toolkits/lerobot/sidney_pi05_parity.py
$RLPY -m ruff check toolkits/lerobot/sidney_pi05_parity.py
$RLPY -m py_compile toolkits/lerobot/sidney_pi05_parity.py
$RLPY toolkits/lerobot/sidney_pi05_parity.py prepare --output "$INPUT"
cd "$LR"
HF_HOME=/data/chenyiteng/cache/huggingface-sidney HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$LR/src" \
"$LRPY" - "$MODEL" "$INPUT" <<'PY'
import sys
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import OBS_STATE

class ImageAdapter(torch.nn.Module):
    _preprocess_images = PI05Policy._preprocess_images
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.anchor = torch.nn.Parameter(torch.empty(()), requires_grad=False)

model, artifact = sys.argv[1:]
config = PreTrainedConfig.from_pretrained(model, local_files_only=True)
config.device = 'cpu'
preprocessor, _ = make_pre_post_processors(
    config, pretrained_path=model,
    preprocessor_overrides={'device_processor': {'device': 'cpu'}},
)
with np.load(artifact, allow_pickle=False) as raw:
    assert raw['main_image'].shape == (224,224,3)
    batch = preprocessor({
        OBS_STATE: torch.from_numpy(raw['state']),
        'observation.images.cam_high': torch.from_numpy(raw['main_image']).permute(2,0,1).contiguous().float()/255.0,
        'observation.images.cam_left_wrist': torch.from_numpy(raw['wrist_images'][0]).permute(2,0,1).contiguous().float()/255.0,
        'observation.images.cam_right_wrist': torch.from_numpy(raw['wrist_images'][1]).permute(2,0,1).contiguous().float()/255.0,
        'task': str(raw['prompt'].item()),
    })
images, _ = ImageAdapter(config)._preprocess_images(batch)
assert [tuple(x.shape) for x in images] == [(1,3,224,224)] * 3
print('native_224_core_input_ok', [tuple(x.shape) for x in images])
PY
cd "$WT"
git diff --check
git diff --stat
