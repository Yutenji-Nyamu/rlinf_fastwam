set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

robotwin_tree=/data/chenyiteng/projects/robotwin-native/RoboTwin
expected_robotwin=30954692d06ba7e89f7a6b76064f4062c488fa81
expected_xpolicylab_before=c37109c500be67d0dea6b36bf7337bbd26e763cd
expected_xpolicylab_main=c07a09614dd44cc4a67483bcb9a82e7439d99926

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=16
export CUDA_VISIBLE_DEVICES=0

printf '%s\n' '=== PRECHECK ==='
cd "$robotwin_tree"
test "$(git rev-parse HEAD)" = "$expected_robotwin"
test "$(git -C XPolicyLab rev-parse HEAD)" = "$expected_xpolicylab_before"
test -z "$(git status --short)"
test ! -e envs/curobo
remote_xpolicylab_main=$(git ls-remote https://github.com/XPolicyLab/XPolicyLab.git refs/heads/main | awk '{print $1}')
printf 'remote_xpolicylab_main=%s\n' "$remote_xpolicylab_main"
test "$remote_xpolicylab_main" = "$expected_xpolicylab_main"
python --version
python -m pip --version
df -h "$CONDA_PREFIX" "$robotwin_tree" /tmp

printf '%s\n' '=== INSTALL MINIMAL CUDA 12.1 BUILD TOOLCHAIN ==='
conda install --yes --prefix "$CONDA_PREFIX" \
  --channel nvidia/label/cuda-12.1.1 --channel conda-forge \
  cuda-nvcc=12.1.105 cuda-cudart-dev=12.1.105 cuda-cccl=12.1.109
nvcc --version

printf '%s\n' '=== RUN LOCKED OFFICIAL ROBOTWIN INSTALL SCRIPT ==='
bash scripts/_install.sh

printf '%s\n' '=== VERIFY INSTALL ==='
printf 'robotwin_head='; git rev-parse HEAD
printf 'xpolicylab_head='; git -C XPolicyLab rev-parse HEAD
git status --short --branch
python - <<'PY'
import torch
import torchvision
import pytorch3d
import sapien
import mplib
import XPolicyLab
print("torch", torch.__version__, "torch_cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("torch_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torchvision", torchvision.__version__)
print("pytorch3d", getattr(pytorch3d, "__version__", "unknown"))
print("sapien", getattr(sapien, "__version__", "unknown"))
print("mplib", getattr(mplib, "__version__", "unknown"))
print("xpolicylab", XPolicyLab.__file__)
PY
printf 'curobo_head='; git -C envs/curobo rev-parse HEAD
du -sh "$CONDA_PREFIX" envs/curobo
df -h /home/chenyiteng /data/chenyiteng /tmp
