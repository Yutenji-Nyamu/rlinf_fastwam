# Fast-WAM 官方 RoboTwin standalone 跑通手册

更新时间：2026-07-17

本手册只负责迁移计划的阶段 0：在 AutoDL 双 A800 服务器上，用官方发布 checkpoint 跑通一个 RoboTwin 单任务、单 episode。它不是第二份 RLinf 集成计划；Fast-WAM 接入 RLinf 仍只按 [05_IMPLEMENTATION_PLAN.md](05_IMPLEMENTATION_PLAN.md) 执行。

阶段状态：**已完成（用户于 2026-07-17 确认）**。本文件保留为可复核命令与故障依据，不要求为了迁移重复安装或重跑；2026-07-17 20:05 已只读确认成功run的resolved config、日志位置、视频与`H/N/S/shift`。迁移前只补小型黄金fixture；资源峰值改在首个真实RLinf runner记录。

旧的 `fastwam-official-robotwin-install-guide-20260716.md` 保留为详细调查底稿；本文件是当前项目内的精简执行版本。每次只执行一个阶段，返回输出并验收后再继续。

## 1. 来源锁与最终选择

| 项目 | 固定值 | 性质与原因 |
|---|---|---|
| Fast-WAM | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` | **官方**：2026-07-17 再次核对，仍是官方 `main` HEAD；避免安装期间上游变化 |
| Python | 3.10 | **官方**：Fast-WAM README 的环境命令；也与 RoboTwin 官方推荐一致 |
| PyTorch | `2.7.1+cu128` | **官方**：Fast-WAM README/`pyproject.toml` 精确固定 |
| Torchvision | `0.22.1+cu128` | **官方**：同上 |
| RoboTwin 代码 | Fast-WAM 自带 `third_party/RoboTwin` | **官方**：Fast-WAM 已适配的 evaluator；不能用旧 RoboTwin 整仓覆盖 |
| RoboTwin vendor | `bf44be51cf5717a5595ce59447f2cf5263d2aa95` | **官方代码记录**：`README.vendor.md` 锁定的上游版本 |
| CuRobo | `v0.7.8` / `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` | **兼容性 pin**：不是 Fast-WAM 明文锁定；用于匹配 vendored RoboTwin 使用的 CuRobo v1 API，避免当前 main API 漂移 |
| Warp | `warp-lang==1.11.1` | **兼容性 pin**：CuRobo 0.7.8 仍调用 `wp.torch`，但只声明 `warp-lang>=0.9.0`；Warp 1.13 起删除该旧入口。1.11.1 与 CuRobo tag 同期，且 RLinf 官方 RoboTwin 安装脚本也精确固定此版本 |
| release checkpoint | `yuanty/fastwam` 的 `robotwin_uncond_3cam_384.pt` + 对应 stats | **官方**：官方 Hugging Face 发布物 |
| Wan VAE/T5/tokenizer | ModelScope：`DiffSynth-Studio/Wan-Series-Converted-Safetensors` + `Wan-AI/Wan2.1-T2V-1.3B` | **官方代码默认**：固定 commit 的 `io.py` 默认 `modelscope`，且转换仓实际托管在 ModelScope；不能强制转成 Hugging Face repo |
| 首次任务 | `adjust_bottle`、`demo_clean`、unseen、1 episode | **本机验收选择**：已有 π0/Motus 历史，便于之后交叉验证；不是官方唯一任务 |
| action | 14D qpos，`H=32`，首次 `N=24` | **官方当前完整入口**：`sim_robotwin.yaml` 的 24 会覆盖 deploy policy YAML 的孤立默认值 8 |

直接来源：

- [Fast-WAM 官方仓库与 README](https://github.com/yuantianyuan01/FastWAM)
- [Fast-WAM 官方发布 checkpoint](https://huggingface.co/yuanty/fastwam)
- [RoboTwin 官方安装说明](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
- [Fast-WAM vendor 对应的 RoboTwin requirements](https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/bf44be51cf5717a5595ce59447f2cf5263d2aa95/script/requirements.txt)
- [CuRobo v0.7.8](https://github.com/NVlabs/curobo/tree/v0.7.8)
- [NVIDIA Warp 兼容性变更](https://github.com/NVIDIA/warp/blob/main/CHANGELOG.md)
- [RLinf 官方 RoboTwin 安装脚本](https://github.com/RLinf/RLinf/blob/main/requirements/install.sh)

本机网络规则来自既有 Motus、LaWAM、OpenPI/RLinf 的成功安装记录：

- 默认清除 `http_proxy`/`https_proxy`/`all_proxy`；AutoDL 学术加速不常驻。
- GitHub clone 直连慢时，才在单个子 shell 中临时 `source /etc/network_turbo`；clone 完成后代理不会泄漏到父 shell。
- 普通 pip 包和官方 `pip install -e .` 使用本机实测较快的清华 PyPI 镜像；官方 PyTorch cu128 index 只加在显式安装 Torch/Torchvision 的那条命令上，不作为全局 extra index。
- `yuanty/fastwam` release checkpoint 使用 Hugging Face/`hf-mirror.com`；Fast-WAM loader 的 Wan VAE/T5/tokenizer 使用代码默认的 ModelScope。两者保持学术加速代理关闭，分别放到数据盘 cache；不能因为 Motus 的 HF 仓曾在镜像下载成功，就把 ModelScope 转换仓也强制交给 Hugging Face。
- 这些只改变下载路由，不改变 Fast-WAM 官方 `pyproject.toml` 的版本约束。网络变量只作用于当前 shell，不写入全局 pip/conda 配置。

环境必须分开：

- **现在的官方 standalone**：干净 Python 3.10 conda 环境，只负责官方推理。
- **未来的 RLinf 集成**：另建环境/worktree，再处理 Python 3.11、RLinf 与 Fast-WAM 的联合依赖。
- 当前 π0/RLinf venv 和旧 RoboTwin conda 环境都不复制为 Fast-WAM 环境；旧 CuRobo `.so` 也不复用，因为 Torch ABI 不同。

执行规则：阶段 A 验收后才进入 B；之后也一次只执行一个阶段。任一命令报错就停在当前阶段，不继续用半安装状态运行后续命令。

## 2. 阶段 A：只读现场预检（现在先执行这一段）

**性质：验收门 + 本机适配。** 官方 README 不负责判断这台服务器是否仍有 RLinf/Ray 训练、数据盘是否够用、CUDA toolkit 是否能编译 CuRobo。先确认这些事实，避免安装与正在运行的实验争抢 240 GiB cgroup RAM。

```bash
# [本机验收] 记录时间，并确认训练/Ray、GPU 和主机内存已经释放。
date

pgrep -af 'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' || true
nvidia-smi
free -h

# [本机验收] 安装、模型和 cache 都放数据盘；先确认容量与 inode。
df -h /root/autodl-tmp
df -i /root/autodl-tmp

# [本机适配] 历史 Conda base 是 /root/miniconda3，但现场动态确认，不硬编码旧路径。
command -v conda || true
conda info --base 2>/dev/null || true
conda env list 2>/dev/null || true

# [兼容性验收] CuRobo 要在最终 Torch 环境编译，所以先确认 CUDA toolkit。
command -v nvcc || true
nvcc --version || true
ls -ld /usr/local/cuda* 2>/dev/null || true

# [RoboTwin 官方前置的本机验收] 检查 SAPIEN 所需 Vulkan 能力。
command -v vulkaninfo || true
vulkaninfo --summary 2>/dev/null | head -n 40 || true

# [安全验收] 发现目标路径时先停下审计，绝不覆盖已有安装或模型。
for p in \
  /root/autodl-tmp/FastWAM \
  /root/autodl-tmp/conda/envs/FastWAM-official \
  /root/autodl-tmp/models/fastwam
do
  if [ -e "$p" ] || [ -L "$p" ]; then
    ls -ld "$p"
  else
    echo "ABSENT $p"
  fi
done

# [本机复用检查] 只看现有 RoboTwin assets/task_config 是否可作为受控输入。
git -C /root/autodl-tmp/RoboTwin rev-parse HEAD 2>/dev/null || true
git -C /root/autodl-tmp/RoboTwin status --short -- task_config 2>/dev/null || true
du -sh /root/autodl-tmp/RoboTwin/assets /root/autodl-tmp/RoboTwin/task_config 2>/dev/null || true
```

继续条件：

- 没有仍在运行的 RLinf/Ray 训练 worker；GPU 和主机 RAM 已释放。
- `/root/autodl-tmp` 有足够空间和 inode。除 12 GB release checkpoint 外，VAE/T5/tokenizer/cache 还需要明显额外空间。
- `conda info --base` 有有效结果。历史上是 `/root/miniconda3`，但不硬编码。
- `nvcc`/`CUDA_HOME` 能与 Torch cu128 的 CuRobo 编译需求匹配；不匹配就停在此处。
- Vulkan 可用；若 `vulkaninfo` 不存在或报错，先处理图形能力。
- 三个目标路径不存在；若已存在，先只读审计，不能覆盖。

## 3. 阶段 B：克隆并固定官方 Fast-WAM

**性质：官方 + 可复现性。** clone 官方仓库；固定 commit 是本项目的可复现性要求。这里不创建新的“自有仓库”，也不碰 RLinf。

其中 `/etc/network_turbo` 是 AutoDL 的可选下载加速，不是 Fast-WAM 官方要求；只让它在 clone 的子 shell 中生效，避免它设置的 Aliyun pip 索引泄漏到阶段 C。

```bash
ROOT=/root/autodl-tmp
FW="$ROOT/FastWAM"
ENV="$ROOT/conda/envs/FastWAM-official"
PIN=45d8e1458921d83f8ad6cf9ce993d371208dabd0

if [ -e "$FW" ] || [ -L "$FW" ]; then
  echo "STOP: $FW already exists"
  exit 1
fi

if [ -f /etc/network_turbo ]; then
  (
    source /etc/network_turbo
    git clone https://github.com/yuantianyuan01/FastWAM.git "$FW"
  )
else
  git clone https://github.com/yuantianyuan01/FastWAM.git "$FW"
fi

git -C "$FW" checkout --detach "$PIN"

git -C "$FW" rev-parse HEAD
git -C "$FW" remote -v
git -C "$FW" status --short --branch
```

验收：HEAD 精确等于 `PIN`，origin 是官方仓库，没有意外 tracked 修改。

## 4. 阶段 C：建立官方 Python 3.10 联合环境

**性质：官方。** Python、Torch、Torchvision 和 `pip install -e .` 直接来自 Fast-WAM README。

**本机适配：** 环境、pip/HF/Torch extension cache 和临时目录放在 `/root/autodl-tmp`，避免小系统盘；Conda hook 由 `conda info --base` 动态确定。先清除学术加速代理，普通包使用本机历史实测更快的清华 PyPI；CUDA wheel 仍来自官方 PyTorch cu128 index。

`wheel`、`ninja`、`setuptools_scm` 是 editable install 与后续 CuRobo 编译工具。运行环境中的 `setuptools` 精确固定为 `80.9.0`：RoboTwin 固定的 `sapien==3.0.0b1` 仍会导入 `pkg_resources`，而 `setuptools>=82` 已删除该模块；`80.9.0` 是本机 Motus/RoboTwin 已验证组合，同时满足 Fast-WAM `pyproject.toml` 的 `setuptools>=68`。这是一项兼容性 pin，不是 Fast-WAM 模型版本选择。

下面的完整建环境块只在目标环境不存在时执行一次。当前现场已经完成 Python、Torch 和编译工具安装，不要再次执行 `conda create` 或 Torch 安装；直接使用后面的“原地恢复”块。

```bash
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

ROOT=/root/autodl-tmp
FW="$ROOT/FastWAM"
ENV="$ROOT/conda/envs/FastWAM-official"

conda create -p "$ENV" python=3.10 -y
conda activate "$ENV"

export PIP_CACHE_DIR="$ROOT/cache/pip"
export HF_HOME="$ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export MODELSCOPE_CACHE="$ROOT/cache/modelscope"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions/fastwam-torch271-cu128"
export TMPDIR="$ROOT/tmp/fastwam"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
export PIP_DEFAULT_TIMEOUT=120

# [本机环境清理] 非法的继承值会触发 libgomp 警告；只清理当前 shell，不修改 rc 文件。
printf 'OMP_NUM_THREADS=<%q>\n' "${OMP_NUM_THREADS-}"
unset OMP_NUM_THREADS

mkdir -p "$PIP_CACHE_DIR" "$HUGGINGFACE_HUB_CACHE" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

python -m pip install -U pip wheel
python -m pip install "setuptools==80.9.0"
python -m pip install ninja setuptools_scm
python -m pip install \
  torch==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "$FW"
```

如果环境和 Torch 已经成功，但 Aliyun 索引缺包，或切换官方 PyPI 后疑似受残留学术代理影响而下载极慢，不删除环境、不重装 Torch。保持 `FastWAM-official` 激活，只执行下面的原地恢复：

```bash
# [本机适配] 恢复数据盘 cache；关闭学术加速代理，普通包切到历史实测较快的清华源。
ROOT=/root/autodl-tmp
FW="$ROOT/FastWAM"

export PIP_CACHE_DIR="$ROOT/cache/pip"
export HF_HOME="$ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions/fastwam-torch271-cu128"
export TMPDIR="$ROOT/tmp/fastwam"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
export PIP_DEFAULT_TIMEOUT=120

# [本机环境清理] 记录并清除格式非法的 OpenMP 变量；它不是官方 Fast-WAM 配置。
printf 'OMP_NUM_THREADS=<%q>\n' "${OMP_NUM_THREADS-}"
unset OMP_NUM_THREADS

printf 'PIP_INDEX_URL=%s\n' "$PIP_INDEX_URL"
printf 'PIP_EXTRA_INDEX_URL=%s\n' "${PIP_EXTRA_INDEX_URL-<unset>}"
env | grep -iE '^(http|https|all)_proxy=' || true

# [官方安装语义 + 本机索引修正] 仍是官方的 pip install -e .，保留 PEP 517 隔离构建。
python -m pip install -e "$FW"
```

这里不加 `--no-build-isolation`：pip 会按官方 `pyproject.toml` 新建隔离构建环境并安装其中声明的 `setuptools>=68`、`wheel`；隔离构建环境不改变主运行环境的 `setuptools==80.9.0`。索引/速度问题属于传输路径，不是 Fast-WAM 版本冲突。中断下载后直接重跑即可复用 pip cache，不删除 cache 或临时目录。

验收：

```bash
python - <<'PY'
import sys
import torch
import torchvision
import transformers
import fastwam

print("python", sys.version.split()[0])
print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("transformers", transformers.__version__)
print("fastwam", fastwam.__file__)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
PY

python -m pip show fastwam
python -m pip check
```

必须看到 Python 3.10、Torch 2.7.1+cu128、Torchvision 0.22.1+cu128、Transformers 4.49.0、`fastwam` 指向当前源码目录和 CUDA 可用。若 `fastwam` 尚未注册，`pip check` 即使显示无冲突也不算阶段 C 通过。

### C1. 每个后续阶段的 shell 恢复块

**性质：本机适配。** 官方 README 默认在同一个 shell 连续执行；这里会逐阶段验收，SSH 也可能重连。因此阶段 D–H 每次开始前先运行这一小段，避免空的 `$FW`/`$ENV` 或 cache 回落到系统盘。

```bash
ROOT=/root/autodl-tmp
FW="$ROOT/FastWAM"
ENV="$ROOT/conda/envs/FastWAM-official"
VRT="$FW/third_party/RoboTwin"
MODEL_ROOT="$ROOT/models/fastwam"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV"

export PIP_CACHE_DIR="$ROOT/cache/pip"
export HF_HOME="$ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions/fastwam-torch271-cu128"
export TMPDIR="$ROOT/tmp/fastwam"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
export PIP_DEFAULT_TIMEOUT=120
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_ROOT/diffsynth"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export PYTHONUNBUFFERED=1
unset OMP_NUM_THREADS

test "$(python -c 'import sys; print(sys.executable)')" = "$ENV/bin/python" || {
  echo "STOP: FastWAM-official environment is not active"
  exit 1
}
```

## 5. 阶段 D：补齐 RoboTwin 与 CuRobo

### D1. 仿真 Python 包

**性质：兼容性修正。** RoboTwin 官方要求运行安装脚本，但 Fast-WAM vendored 目录缺 `script/requirements.txt`；对应上游 requirements 还会把 Torch 降为 2.4.1、HF Hub 降为 0.25.0。不能直接执行 `_install.sh`，只安装首轮评测需要且不冲突的仿真依赖。Fast-WAM 自身已经提供 `av`、`imageio`、`termcolor`、`wandb` 等共同依赖；描述生成、数据转换和云端 API 包不为一次推理额外安装。

上游 requirements 固定了 `sapien==3.0.0b1`，但没有声明它运行时仍需要的 `pkg_resources`，也没有约束 `setuptools`。因此这里再次声明已验证的精确 pin，避免新 Conda 环境解析到已经删除 `pkg_resources` 的 `setuptools>=82`。

```bash
conda activate "$ENV"

python -m pip install "setuptools==80.9.0"
python -m pip install \
  transforms3d==0.4.2 \
  sapien==3.0.0b1 \
  scipy==1.10.1 \
  mplib==0.2.1 \
  gymnasium==0.29.1 \
  trimesh==4.4.3 \
  open3d==0.18.0 \
  h5py \
  'pyglet<2' \
  toppra \
  opencv-python-headless \
  matplotlib \
  PyYAML

command -v ffmpeg || conda install -c conda-forge ffmpeg -y
```

PyTorch3D 不作为 RGB+qpos 首轮 smoke 门；RoboTwin 官方也说明不使用 3D 数据时它失败不影响基本功能。

### D2. CuRobo

**性质：RoboTwin 必需 + 兼容性 pin。** vendored robot 会构造 `CuroboPlanner`，evaluator 默认还会先跑专家 seed check。Motus/RLinf 中“走 MPLib 可忽略 CuRobo”的旧结论不适用于这里。

先检查最终编译环境。`nvidia-smi` 显示的 CUDA 13.0 是驱动最高兼容能力，不是本地 toolkit；本机应显式使用与 Torch cu128 匹配的 `/usr/local/cuda-12.8`：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
test -x "$CUDA_HOME/bin/nvcc" || {
  echo "STOP: missing $CUDA_HOME/bin/nvcc"
  exit 1
}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

readlink -f /usr/local/cuda
"$CUDA_HOME/bin/nvcc" --version

python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("CUDA_HOME", CUDA_HOME)
PY
```

只有 CUDA toolkit 检查通过后才执行：

```bash
VRT="$FW/third_party/RoboTwin"
CUROBO_SRC="$VRT/envs/curobo"

if [ -e "$CUROBO_SRC" ] || [ -L "$CUROBO_SRC" ]; then
  echo "STOP: $CUROBO_SRC already exists"
  exit 1
fi

if [ -f /etc/network_turbo ]; then
  (
    source /etc/network_turbo
    git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_SRC"
  )
else
  git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_SRC"
fi
test "$(git -C "$CUROBO_SRC" rev-parse HEAD)" = d64c4b005459db10c5dd867d8b30a87d5bda9bdb

python -m pip install "warp-lang==1.11.1"
MAX_JOBS=8 python -m pip install -e "$CUROBO_SRC" --no-build-isolation
```

`MAX_JOBS=8` 是本机编译并发上限，用于避免 C++/CUDA 编译瞬时占满主机内存；不是 CuRobo 官方数学配置。

验证：

```bash
python - <<'PY'
from importlib.metadata import version

import curobo
import torch
import warp as wp
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

wp.init()
print("curobo", curobo.__file__)
print("warp-lang", version("warp-lang"))
print("warp cuda", wp.is_cuda_available())
print("warp legacy device", wp.torch.device_from_torch(torch.device("cuda:0")))
print("CuRobo v1 API import OK")
PY
```

### D3. SAPIEN/MPLib 官方语义补丁

**性质：RoboTwin 官方安装脚本要求。** 补丁只作用于新 Fast-WAM 环境的 `pip show` 路径，不触碰旧环境。执行前后都查看目标行。

```bash
SAPIEN_LOCATION="$(python -m pip show sapien | awk '/^Location:/{print $2}')/sapien"
MPLIB_LOCATION="$(python -m pip show mplib | awk '/^Location:/{print $2}')/mplib"
URDF_LOADER="$SAPIEN_LOCATION/wrapper/urdf_loader.py"
PLANNER="$MPLIB_LOCATION/planner.py"

test -f "$URDF_LOADER" || { echo "STOP: missing $URDF_LOADER"; exit 1; }
test -f "$PLANNER" || { echo "STOP: missing $PLANNER"; exit 1; }
grep -n 'with open(urdf_file' "$URDF_LOADER" || true
grep -n 'delta_twist' "$PLANNER" | tail -n 5 || true

sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"

grep -n 'with open(urdf_file.*encoding="utf-8"' "$URDF_LOADER"
grep -nF 'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:' "$PLANNER"
if grep -Fq 'or collide or not within_joint_limit:' "$PLANNER"; then
  echo "STOP: MPLib planner patch did not apply"
  exit 1
fi
python -m pip check
```

## 6. 阶段 E：准备 assets、task_config 与 policy

### E1. assets

**性质：官方要求 assets；软链接是本机适配。** 复用已经支撑 ACT/Motus 的约 16 GB 官方 assets，避免重复下载；不覆盖 Fast-WAM vendored RoboTwin 代码。

```bash
RT="$ROOT/RoboTwin"
VRT="$FW/third_party/RoboTwin"

test -d "$RT/assets/objects"
test -d "$RT/assets/embodiments"
test -d "$RT/assets/background_texture"

if [ -e "$VRT/assets" ] || [ -L "$VRT/assets" ]; then
  echo "STOP: $VRT/assets already exists"
  exit 1
fi

ln -s "$RT/assets" "$VRT/assets"
readlink -f "$VRT/assets"
```

### E2. task_config

**性质：RoboTwin 官方运行输入 + 本机版本一致性适配。** 这不是 Fast-WAM README 逐字给出的复制命令。Fast-WAM vendored 代码没有提交 `task_config`，但 manager 和 evaluator 会直接读取 `demo_clean.yml`、`demo_randomized.yml`、`_camera_config.yml`、`_embodiment_config.yml` 与 `_eval_step_limit.yml`；缺失会在模型推理前触发 `FileNotFoundError`。

阶段 A 记录的现有 RoboTwin HEAD 为 `c3ddfa8...`，Fast-WAM `README.vendor.md` 记录的上游版本为 `bf44be5...`。两个官方 commit 下整个 `task_config` Git tree SHA 已核验完全相同，都是 `fdb995fb05a65f4ee6bdfaf633a89631af93db33`。因此主路径不再下载第二份仓库，而是现场复核 tree 与工作区干净后复制现有目录。使用复制而不是软链接，是因为阶段 F 会在 Fast-WAM 目录新建专用 smoke 配置，不能污染旧 RoboTwin。

```bash
RT="$ROOT/RoboTwin"
VRT="$FW/third_party/RoboTwin"
EXPECTED_TASK_CONFIG_TREE=fdb995fb05a65f4ee6bdfaf633a89631af93db33

test -d "$RT/task_config" || {
  echo "STOP: missing $RT/task_config"
  exit 1
}

SOURCE_HEAD="$(git -C "$RT" rev-parse HEAD)"
ACTUAL_TASK_CONFIG_TREE="$(git -C "$RT" rev-parse HEAD:task_config)"
printf 'source_head=%s\n' "$SOURCE_HEAD"
printf 'task_config_tree=%s\n' "$ACTUAL_TASK_CONFIG_TREE"

test "$ACTUAL_TASK_CONFIG_TREE" = "$EXPECTED_TASK_CONFIG_TREE" || {
  echo "STOP: existing task_config does not match the Fast-WAM vendor version"
  exit 1
}
test -z "$(git -C "$RT" status --porcelain -- task_config)" || {
  echo "STOP: existing task_config has working-tree changes"
  exit 1
}

if [ -e "$VRT/task_config" ] || [ -L "$VRT/task_config" ]; then
  echo "STOP: $VRT/task_config already exists"
  exit 1
fi

cp -a "$RT/task_config" "$VRT/task_config"

for f in \
  demo_clean.yml \
  demo_randomized.yml \
  _camera_config.yml \
  _config_template.yml \
  _embodiment_config.yml \
  _eval_step_limit.yml \
  create_task_config.sh
do
  test -s "$VRT/task_config/$f" || {
    echo "STOP: missing or empty task_config/$f"
    exit 1
  }
done

echo "task_config copied OK"
```

### E3. policy 软链接

**性质：Fast-WAM README 官方命令。** policy 名必须是 `fastwam_policy`。

```bash
ln -sfn "$FW/experiments/robotwin/fastwam_policy" "$VRT/policy/fastwam_policy"
readlink -f "$VRT/policy/fastwam_policy"
```

不要运行 `update_embodiment_config_path.py`，除非后续明确出现 embodiment 绝对路径失效；共享 assets 的原路径仍存在。

## 7. 阶段 F：先验证 RoboTwin，不加载 12 GB checkpoint

这些是逐层门，不是一组可以忽略前序失败后继续执行的命令：F1 失败就停；F1 通过后才运行 F2；F2 通过后直接运行 F4。F3 是 F4 失败时用于隔离场景初始化问题的诊断门，正常主路径无需重复执行。

### F1. import

**性质：验收门。** 用于把 Python/ABI 问题与模型问题分开。

```bash
cd "$VRT"
python -m pip check

python - <<'PY'
import scipy
import skimage
import torch
import sapien
import mplib
import open3d
import cv2
import toppra
from curobo.wrap.reacher.motion_gen import MotionGen
from envs.adjust_bottle import adjust_bottle
print("RoboTwin imports OK", torch.__version__)
print("scipy", scipy.__version__, "skimage", skimage.__version__)
PY
```

若出现 `ModuleNotFoundError: No module named 'pkg_resources'`，说明主环境仍是 `setuptools>=82`。这是旧 SAPIEN 的隐含依赖与新工具链的兼容断层，不要安装名为 `pkg_resources` 的包，也不要重装 Torch/CuRobo；在当前环境做一次最小修复：

```bash
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST

python -m pip install "setuptools==80.9.0"

python - <<'PY'
import setuptools
import pkg_resources
import sapien
print("setuptools", setuptools.__version__)
print("pkg_resources", pkg_resources.__file__)
print("sapien", sapien.__file__)
PY
python -m pip check
```

这段通过后，重新执行完整 F1；不要直接跳到 F2。

当前现场预期为 SciPy 1.15.3、scikit-image 0.25.2：CuRobo 的未固定 `scikit-image` 依赖使 pip 合法升级了 SciPy。`pip check` 与 CuRobo import 已通过，所以不先做额外回退；F2 render 与 F4 专家 episode 才是行为兼容性的主路径门。

### F2. Vulkan render

**性质：RoboTwin 官方 smoke。** 该脚本捕获异常后退出码仍可能为 0，因此不能只看 `$?`，必须在输出里看到 `Render Well`。

```bash
cd "$VRT"
python script/test_render.py
```

### F3. 最小 `adjust_bottle` 场景（仅在 F4 失败时诊断）

**性质：本机诊断门，非主路径必跑。** 这是官方要求“RoboTwin 环境可用”的最小化拆分检查；不加载 Fast-WAM checkpoint，只验证 task config、assets、robot、三路相机和 observation 能共同初始化。F4 已覆盖场景初始化，只有 F4 报错且需要区分场景与 planner 时才运行本节。

```bash
cd "$VRT"

python - <<'PY'
import os
import yaml

from envs import CONFIGS_PATH
from envs.adjust_bottle import adjust_bottle

with open("./task_config/demo_clean.yml", "r", encoding="utf-8") as f:
    args = yaml.safe_load(f)

with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
    embodiment_table = yaml.safe_load(f)

embodiment = args["embodiment"]
assert len(embodiment) == 1, embodiment
robot_file = embodiment_table[embodiment[0]]["file_path"]

with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
    robot_cfg = yaml.safe_load(f)

args.update({
    "task_name": "adjust_bottle",
    "task_config": "demo_clean",
    "seed": 0,
    "now_ep_num": 0,
    "eval_mode": True,
    "save_data": False,
    "render_freq": 0,
    "left_robot_file": robot_file,
    "right_robot_file": robot_file,
    "left_embodiment_config": robot_cfg,
    "right_embodiment_config": robot_cfg,
    "dual_arm_embodied": True,
})

env = adjust_bottle()
try:
    env.setup_demo(**args)
    obs = env.get_obs()
    print("observation keys:", sorted(obs.keys()))
    print("RoboTwin adjust_bottle scene smoke OK")
finally:
    env.close_env(clear_cache=True)
PY
```

必须出现 `RoboTwin adjust_bottle scene smoke OK`，且没有 asset、camera、SAPIEN、MPLib 或 CuRobo 错误。

### F4. 单个专家 episode

**性质：验收门。** Fast-WAM evaluator 正式运行前本来也会做 expert check；提前执行可以验证 CuRobo/MPLib/SAPIEN/assets，而不用先下载模型。

创建或复用独立的一集配置。该块可重复执行；配置已存在时只核验，不调用 `exit` 关闭交互式 SSH：

```bash
cd "$VRT"
SMOKE_CFG="$VRT/task_config/demo_fastwam_env_smoke.yml"
if [ ! -e "$SMOKE_CFG" ] && [ ! -L "$SMOKE_CFG" ]; then
  cp "$VRT/task_config/demo_clean.yml" "$SMOKE_CFG"
  sed -i -E 's/^episode_num:[[:space:]]*.*/episode_num: 1/' "$SMOKE_CFG"
  sed -i -E 's|^save_path:[[:space:]]*.*|save_path: ./data_fastwam_env_smoke|' "$SMOKE_CFG"
fi

if test -s "$SMOKE_CFG" \
  && grep -Eq '^episode_num:[[:space:]]*1[[:space:]]*$' "$SMOKE_CFG" \
  && grep -Eq '^save_path:[[:space:]]*\./data_fastwam_env_smoke[[:space:]]*$' "$SMOKE_CFG"; then
  grep -n -E '^(episode_num|save_path):' "$SMOKE_CFG"
  CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore::UserWarning \
    python script/collect_data.py adjust_bottle demo_fastwam_env_smoke
else
  echo "STOP: smoke config missing or values changed; do not run F4"
fi
```

必须看到专家规划成功并完成任务。若不断换 seed 或长时间无进展，不要无限等待：依赖/规划错误在官方 evaluator 中也可能表现为 expert-check 循环。

若第一个 seed 报 `module 'warp' has no attribute 'torch'`，而后续 seed 报 `Robot` 缺少 `left_planner`，根因是 pip 按 CuRobo 的无上界约束安装了过新的 Warp；后者只是第一次 planner 构造中断后复用半初始化对象的连锁错误。不要修改 RoboTwin/CuRobo 源码，也不要重编 Torch/CuRobo；在当前环境精确恢复兼容版本，并用全新 Python 进程重跑 F4：

```bash
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST

python -m pip install --no-deps "warp-lang==1.11.1"
python -m pip check

python - <<'PY'
from importlib.metadata import version
import torch
import warp as wp

wp.init()
print("warp-lang", version("warp-lang"))
print("warp", wp.__file__)
print("cuda", wp.is_cuda_available())
print("legacy device", wp.torch.device_from_torch(torch.device("cuda:0")))
print("public device", wp.device_from_torch(torch.device("cuda:0")))
PY
```

`device_from_torch()` 本身不负责初始化 Warp runtime，因此独立 probe 必须先执行 `wp.init()`；否则会出现 `NoneType.cuda_devices` 假阴性。CuRobo 的真实调用链会先执行自己的 `init_warp()`，其中包含 `wp.init()`。出现旧入口的 deprecation warning 可以接受；必须看到版本为 1.11.1、CUDA 可用且两个 device 转换均成功。之前失败产生的空 `seed.txt` 可原样保留，重新启动脚本会从 seed 0 开始。

2026-07-17 现场已经通过 F4：seed 0 成功，失败 `0 / 1`，并保存 141 帧专家视频与 instructions。这是比独立 device probe 更强的端到端证据，后续不要再为该 warning 修改 Warp 或重编 CuRobo。

## 8. 阶段 G：下载官方 release checkpoint

**性质：官方。** 只下载 RoboTwin checkpoint 和对应 stats；首轮推理不需要 RoboTwin 训练数据集或 LIBERO 数据。

```bash
MODEL_ROOT="$ROOT/models/fastwam"
mkdir -p "$MODEL_ROOT/release" "$MODEL_ROOT/diffsynth"

export HF_HOME="$ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600

huggingface-cli download yuanty/fastwam \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir "$MODEL_ROOT/release"

ls -lh "$MODEL_ROOT/release"
sha256sum \
  "$MODEL_ROOT/release/robotwin_uncond_3cam_384.pt" \
  "$MODEL_ROOT/release/robotwin_uncond_3cam_384_dataset_stats.json"
```

这一步只处理 Hugging Face release。首次模型构建所需的 Wan VAE/T5/tokenizer 是下一阶段的 ModelScope 下载，不能继承为 Hugging Face 后端。

**ActionDiT 说明：** README 通用说明称推理也要预生成 ActionDiT；但当前完整 release 路径在 `sim_robotwin.yaml` 明确设置 `skip_dit_load_from_pretrain=true`、path 为 null，随后由完整 checkpoint 覆盖 MoT。因此首轮 release smoke 跳过 ActionDiT preprocessing。这只是当前 release 配置的代码特例，不推广到训练或其他 checkpoint。

## 9. 阶段 H：官方单卡、单任务、单 episode smoke

**性质：官方仓库入口 + 本机缩小规模。** README 示例是默认 8-GPU manager；这里使用同一官方仓库内的 `eval_robotwin_single.py` 和同一 release 配置，缩成单 GPU/单任务/单 episode，只减少评测规模，不改 policy 数学。

当前完整入口的 resolved 值应为 `H=32`、`replan N=24`、10 denoise steps、14D qpos。显式写出 24 是为了避免 deploy YAML 中默认 8 造成歧义。

```bash
ROOT=/root/autodl-tmp
FW="$ROOT/FastWAM"
ENV="$ROOT/conda/envs/FastWAM-official"
MODEL_ROOT="$ROOT/models/fastwam"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
unset OMP_NUM_THREADS

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
unset HF_ENDPOINT
export MODELSCOPE_CACHE="$ROOT/cache/modelscope"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_ROOT/diffsynth"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
unset DIFFSYNTH_SKIP_DOWNLOAD
export PYTHONUNBUFFERED=1

mkdir -p "$MODELSCOPE_CACHE" "$DIFFSYNTH_MODEL_BASE_PATH"

python - <<'PY'
from importlib.metadata import version
from fastwam.models.wan22.helpers.loader import _resolve_configs

cfgs = _resolve_configs("Wan-AI/Wan2.2-TI2V-5B", "Wan-AI/Wan2.1-T2V-1.3B")
print("modelscope", version("modelscope"))
for name, cfg in zip(("dit", "text", "vae", "tokenizer"), cfgs):
    print(name, cfg.model_id, cfg.origin_file_pattern, cfg.parse_download_source())
assert all(cfg.parse_download_source() == "modelscope" for cfg in cfgs)
PY

test -s "$MODEL_ROOT/release/robotwin_uncond_3cam_384.pt"
test -s "$MODEL_ROOT/release/robotwin_uncond_3cam_384_dataset_stats.json"
test "$(readlink -f "$FW/third_party/RoboTwin/policy/fastwam_policy")" = \
  "$FW/experiments/robotwin/fastwam_policy"
pgrep -af '[e]val_robotwin_single.py|[c]ollect_data.py' || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader

cd "$FW"
python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt="$MODEL_ROOT/release/robotwin_uncond_3cam_384.pt" \
  EVALUATION.dataset_stats_path="$MODEL_ROOT/release/robotwin_uncond_3cam_384_dataset_stats.json" \
  EVALUATION.task_name=adjust_bottle \
  EVALUATION.task_config=demo_clean \
  EVALUATION.eval_num_episodes=1 \
  EVALUATION.instruction_type=unseen \
  EVALUATION.replan_steps=24 \
  EVALUATION.skip_get_obs_within_replan=true \
  gpu_id=0
```

首次运行会从 ModelScope 下载约 11.36 GB 的 T5、约 1.41 GB 的 Wan VAE 和约 21 MB tokenizer，最终路径为 `$DIFFSYNTH_MODEL_BASE_PATH/<repo_id>/...`；当前磁盘余量足够。为防 VS Code/SSH 断线，长下载建议在 `tmux` 内执行。`timing_enabled` 不再显式开启，因为当前 official eval 调用链只累加该计时而不读取或保存它，不能作为验收证据。

若日志在 `DiffSynth-Studio/Wan-Series-Converted-Safetensors` 上出现 Hugging Face/`hf-mirror.com` 的 `401 RepositoryNotFound`，说明 shell 被错误设置成 `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`。这不是 Token、checkpoint 或 GPU 问题；恢复本节的 ModelScope 变量即可，不登录 Hugging Face、不删 cache、不重装依赖。2026-07-17 16:25/16:29 的两次现场失败正是该手册旧值造成，均停在模型构造的首次元数据请求，12 GB release checkpoint 尚未进入加载阶段。

不再套外层 `| tee`：single 入口会流式打印、检查子进程退出码，并把 log 与 `eval_config_adjust_bottle.yaml` 保存到 `evaluate_results/robotwin/...`。未开启 `pipefail` 的 `tee` 反而可能掩盖 Python 失败。

验收：

- checkpoint、VAE、T5、tokenizer 加载完成，且官方 loader 没有报告 checkpoint/模型加载错误。官方这里用 `strict=False` 且不打印 missing/unexpected key，因此首轮不能把“没打印”误写成完整 key parity 已验证。
- `adjust_bottle` 场景创建成功，policy 输出 14D qpos。
- expert check 不陷入持续换 seed。
- 一个模型 episode 完整结束；single 入口打印成功日志路径。
- 无 CUDA OOM、segfault、缺 asset、非法 action shape；进程退出后显存释放。
- 保存自动生成的 eval config，其中真实 `replan_steps` 必须是 24。

单集最终显示任务 `Success` 或 `Fail` 都可以证明功能调用链完成；这里是 `demo_clean + 1 episode` 功能 smoke，不是官方 randomized 100-episode 性能复现。

## 10. 成功后留存的最小证据

**性质：为后续 RLinf parity 服务的本机验收。** 不复制大模型或大日志到 Windows，只在服务器保留小型文本证据。

```bash
EVIDENCE="$ROOT/fastwam_runs/official-standalone-evidence-$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$EVIDENCE"

git -C "$FW" rev-parse HEAD > "$EVIDENCE/fastwam-head.txt"
git -C "$FW" status --short --branch > "$EVIDENCE/fastwam-status.txt"
python -m pip freeze > "$EVIDENCE/pip-freeze.txt"
nvidia-smi > "$EVIDENCE/nvidia-smi-after.txt"
sha256sum \
  "$MODEL_ROOT/release/robotwin_uncond_3cam_384.pt" \
  "$MODEL_ROOT/release/robotwin_uncond_3cam_384_dataset_stats.json" \
  > "$EVIDENCE/release-sha256.txt"

echo "$EVIDENCE"
```

另外记录自动生成的 eval log/config 路径。之后再从一次固定 observation/seed 提取黄金 action fixture；这一步属于 RLinf adapter parity 准备，不阻塞“官方单 episode 已跑通”的判定。

## 11. 首轮明确不做

- 不启动默认 8-GPU manager。
- 不下载完整 RoboTwin/LIBERO 训练数据。
- 不运行 Fast-WAM 训练或 T5 cache 预计算。
- 不复制旧 RoboTwin/π0/Motus/LaWAM 环境后覆盖 Torch。
- 不安装 Motus 专属 flash-attn/Qwen 修包。
- 不复用旧 CuRobo 编译产物。
- 不整体覆盖 `third_party/RoboTwin`。
- 不进入 RLinf 代码、配置或 1-step GRPO smoke；官方 standalone 验收后再开始下一阶段。
