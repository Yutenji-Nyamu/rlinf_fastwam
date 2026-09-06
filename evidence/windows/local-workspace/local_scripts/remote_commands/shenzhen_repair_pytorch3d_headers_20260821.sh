set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

robotwin_tree=/data/chenyiteng/projects/robotwin-native/RoboTwin
nvidia_wheel_root="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=16
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=9.0

printf '%s\n' '=== NVIDIA WHEEL HEADER/LIB INVENTORY ==='
for required_header in cusparse.h cublas_v2.h cublasLt.h cusolverDn.h; do
  found_header=$(find "$nvidia_wheel_root" -name "$required_header" -print -quit)
  printf '%s=%s\n' "$required_header" "$found_header"
  test -n "$found_header"
done

include_path=$(find "$nvidia_wheel_root" -mindepth 2 -maxdepth 2 -type d -name include -print | sort | paste -sd: -)
library_path=$(find "$nvidia_wheel_root" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)
test -n "$include_path"
test -n "$library_path"
export CPATH="$include_path:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$include_path:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$library_path:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$library_path:${LD_LIBRARY_PATH:-}"
printf 'include_path=%s\n' "$include_path"
printf 'library_path=%s\n' "$library_path"

printf '%s\n' '=== INSTALL NINJA BUILD BACKEND ==='
python -m pip install ninja
ninja --version

printf '%s\n' '=== REBUILD LOCKED PYTORCH3D STABLE ==='
python -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47' --no-build-isolation

printf '%s\n' '=== VERIFY PYTORCH3D AND CORE RUNTIME ==='
cd "$robotwin_tree"
python - <<'PY'
import torch
import torchvision
import pytorch3d
import sapien
import mplib
import XPolicyLab
from curobo.types.base import TensorDeviceType
print("torch", torch.__version__, "torch_cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("torch_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torchvision", torchvision.__version__)
print("pytorch3d", getattr(pytorch3d, "__version__", "unknown"), pytorch3d.__file__)
print("sapien", getattr(sapien, "__version__", "unknown"))
print("mplib", getattr(mplib, "__version__", "unknown"))
print("xpolicylab", XPolicyLab.__file__)
print("curobo_tensor_args", TensorDeviceType())
PY
printf 'robotwin_head='; git rev-parse HEAD
printf 'xpolicylab_head='; git -C XPolicyLab rev-parse HEAD
printf 'curobo_head='; git -C envs/curobo rev-parse HEAD
git status --short --branch
du -sh "$CONDA_PREFIX" envs/curobo
