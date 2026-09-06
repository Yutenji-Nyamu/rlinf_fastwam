set -euo pipefail

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
CUROBO="$ROBOTWIN/envs/curobo"

printf '%s\n' '=== use_cuda_kernel source ==='
grep -Rns --exclude-dir=.git 'use_cuda_kernel' "$CUROBO/src/curobo/opt/newton" | head -n 40 || true

cd "$ROBOTWIN"
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 TORCH_DISABLE_ADDR2LINE=1 \
timeout --signal=TERM --kill-after=30s 300s python3 - <<'PY'
import os

import torch
import warp as wp
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

print("torch", torch.__version__, torch.version.cuda, torch.__file__)
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("warp", wp.__version__, wp.__file__)

yml_path = os.path.abspath("assets/embodiments/aloha-agilex/curobo_left.yml")
world_config = {
    "cuboid": {
        "table": {
            "dims": [0.7, 2, 0.04],
            "pose": [-0.65, 0.0, 0.74, 1, 0, 0, 0.0],
        }
    }
}
cfg = MotionGenConfig.load_from_robot_config(
    yml_path,
    world_config,
    interpolation_dt=1 / 250,
    num_trajopt_seeds=1,
    use_cuda_graph=False,
)
mg = MotionGen(cfg)
print("MotionGen constructed")
mg.warmup()
torch.cuda.synchronize()
print("cuRobo MotionGen warmup OK")
PY
