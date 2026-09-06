set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate act

robotwin_tree=/data/chenyiteng/projects/robotwin-native/RoboTwin
act_dir="$robotwin_tree/XPolicyLab/policy/ACT"

export CUDA_VISIBLE_DEVICES=0

printf '%s\n' '=== PRECHECK ==='
cd "$robotwin_tree"
test "$(git rev-parse HEAD)" = 30954692d06ba7e89f7a6b76064f4062c488fa81
test "$(git -C XPolicyLab rev-parse HEAD)" = c07a09614dd44cc4a67483bcb9a82e7439d99926
test ! -d "$CONDA_PREFIX/lib/python3.10/site-packages/torch"
python --version
python -m pip --version
df -h "$CONDA_PREFIX" /tmp

printf '%s\n' '=== RUN LOCKED OFFICIAL ACT INSTALL SCRIPT ==='
cd "$act_dir"
bash install.sh

printf '%s\n' '=== VERIFY ACT ENV ==='
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import torchvision
import mujoco
import dm_control
import cv2
import h5py
import XPolicyLab
import detr
print("torch", torch.__version__, "torch_cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("torch_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torchvision", torchvision.__version__)
print("mujoco", mujoco.__version__)
print("dm_control", dm_control.__file__)
print("opencv", cv2.__version__, "h5py", h5py.__version__)
print("xpolicylab", XPolicyLab.__file__)
print("detr", detr.__file__)
PY
python -m pip check
du -sh "$CONDA_PREFIX"
