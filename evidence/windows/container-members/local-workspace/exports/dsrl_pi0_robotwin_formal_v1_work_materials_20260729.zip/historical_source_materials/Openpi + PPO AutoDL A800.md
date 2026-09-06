# Openpi \+ PPO AutoDL A800



### guide

TODO

motus部分

空间太大就清理下tts的ckpt；

复原代码？tts的代码改为备份；测试跑通

开发阶段后建立git



从上次的机子：[Exp\_snd](https://my.feishu.cn/wiki/JAfbwIKFui0jR6kkbfScpJ0vn7f)

上次的Rlinf部署，可参考：[OpenVLA\-oft \+ GRPO  Compshare a800](https://my.feishu.cn/wiki/AbTJwXLbDiEjazkOA2hcbhIvn1f)

还是上次的RLinf文档：https://rlinf\.readthedocs\.io/zh\-cn/latest/rst\_source/examples/embodied/robotwin\.html

模型选择：pi0，更干净：

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a2cf8a0\-1824\-83e8\-9eac\-16cf63b97f26

大小3B\+300M左右



跑通：

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a2cf6ac\-4ac0\-83e8\-b348\-54b4a16e134b



目录：

```Bash
/root/autodl-tmp/RLinf/              # RLinf 仓库，名字不改
/root/autodl-tmp/RLinf/.venv/        # RLinf 的 uv 虚拟环境
/root/autodl-tmp/RoboTwin_RLinf/     # 专门给 RLinf_support 分支用
/root/autodl-tmp/models/rlinf/       # OpenPI / π0 / π0.5 SFT 模型
/root/autodl-tmp/cache/uv/           # uv 包缓存
/root/autodl-tmp/cache/uv_python/    # uv 管理的 Python
/root/autodl-tmp/cache/rlinf_assets/ # RLinf 自己下载的 OpenPI tokenizer 等
```



复现流程

```Bash
**只检查，不改动**：记录当前目录、磁盘、GPU、Vulkan、网络、已有缓存。 
**写一个 RLinf 专用环境变量脚本**：以后所有 RLinf 操作都先 source /root/autodl-tmp/rlinf_env.sh，把 uv/HF/pip/tmp/download cache 固定到 /root/autodl-tmp。 
**clone RLinf 到 /root/autodl-tmp/RLinf**。 
**安装 OpenPI+RoboTwin 依赖**：bash requirements/install.sh embodied --model openpi --env robotwin。官方支持的模型名列表包含 openpi，环境名包含 robotwin。
**clone RoboTwin RLinf_support 到 /root/autodl-tmp/RoboTwin_RLinf**，下载 assets。 
**下载 π0.5 SFT 模型到 /root/autodl-tmp/models/rlinf**。 
**先 eval π0.5 SFT**：用官方 robotwin_adjust_bottle_ppo_OpenPI_pi05_eval.yaml，确认环境、模型、相机输入、视频保存全通。官方也说明 π0.5 eval 配置就是这个文件。
**再 PPO smoke training**：先 20 step，小 batch，小并发，确认 checkpoint、video、tensorboard、success rate 产物都正常。 
**最后再追官方结果**：官方 OpenPI 表中 adjust_bottle 上 Pi0.5 SFT 是 85.94%，Pi0.5 RLinf-PPO 是 96.09%；OpenPI 都在 demo_clean 设置下训练。
```



### env



#### 外层

check

```Bash
# =========================
# CONFIG
# =========================
WORK_ROOT="/root/autodl-tmp"
LOG_ROOT="${WORK_ROOT}/setup_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/rlinf_preflight_${STAMP}.log"

mkdir -p "${LOG_ROOT}"

{
echo "===== TIME ====="
date

echo
echo "===== PWD / ROOTS ====="
pwd
ls -lah /root | sed -n '1,120p'
echo
ls -lah "${WORK_ROOT}" | sed -n '1,160p'

echo
echo "===== TREE: /root/autodl-tmp maxdepth=2 ====="
find "${WORK_ROOT}" \
  -maxdepth 2 \
  -mindepth 1 \
  -printf '%y %p\n' \
  | sort \
  | sed -n '1,240p'

echo
echo "===== DISK ====="
df -h /
df -h "${WORK_ROOT}"
du -sh /root 2>/dev/null || true
du -sh "${WORK_ROOT}"/* 2>/dev/null | sort -h || true

echo
echo "===== CACHE DIRS ====="
du -sh "${WORK_ROOT}/cache" 2>/dev/null || true
du -sh "${WORK_ROOT}/cache/"* 2>/dev/null | sort -h || true

echo
echo "===== MODEL DIRS ====="
du -sh "${WORK_ROOT}/models" 2>/dev/null || true
find "${WORK_ROOT}/models" \
  -maxdepth 2 \
  -mindepth 1 \
  -type d \
  -printf '%p\n' \
  2>/dev/null \
  | sort

echo
echo "===== EXISTING MOTUS / ROBOTWIN ====="
ls -ld "${WORK_ROOT}/RoboTwin" 2>/dev/null || true
ls -ld "${WORK_ROOT}/RoboTwin/policy/Motus" 2>/dev/null || true
ls -ld "${WORK_ROOT}/Motus" 2>/dev/null || true
ls -ld "${WORK_ROOT}/conda/envs/RoboTwin" 2>/dev/null || true

echo
echo "===== GPU ====="
nvidia-smi || true

echo
echo "===== CUDA / NVCC ====="
which nvcc || true
nvcc --version || true
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDA_PATH=${CUDA_PATH:-}"

echo
echo "===== PYTHON / CONDA / UV ====="
which python || true
python --version || true
which python3 || true
python3 --version || true
which conda || true
conda --version || true
which uv || true
uv --version || true
uv cache dir 2>/dev/null || true

echo
echo "===== VULKAN ====="
which vulkaninfo || true
echo "VK_DRIVER_FILES=${VK_DRIVER_FILES:-}"
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-}"
unset DISPLAY
vulkaninfo --summary | sed -n '1,120p' || true

echo
echo "===== NETWORK PROBE ====="
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
timeout 15 curl -I -L https://github.com 2>/dev/null | head -n 6 || echo "github direct failed"
timeout 15 curl -I -L https://ghfast.top/https://github.com 2>/dev/null | head -n 6 || echo "ghfast failed"
timeout 15 curl -I -L https://pypi.org/simple/pip/ 2>/dev/null | head -n 6 || echo "pypi failed"
timeout 15 curl -I -L https://mirrors.aliyun.com/pypi/simple/pip/ 2>/dev/null | head -n 6 || echo "aliyun pypi failed"
timeout 15 curl -I -L https://huggingface.co 2>/dev/null | head -n 6 || echo "hf failed"
timeout 15 curl -I -L https://hf-mirror.com 2>/dev/null | head -n 6 || echo "hf mirror failed"

} 2>&1 | tee "${LOG_FILE}"

echo
echo "LOG_FILE=${LOG_FILE}"
```

Check

```Bash
grep -nE "Vulkan|GPU id|deviceName|ERROR|failed|No such|CUDA|uv|ghfast|pypi|huggingface|hf-mirror" \
  "${LOG_FILE}" \
  | sed -n '1,200p'
```



环境脚本；实测清华源快点

```Bash
# =========================
# Write RLinf env script
# =========================
cat > /root/autodl-tmp/rlinf_env.sh <<'EOF'
# =========================
# RLinf / RoboTwin OpenPI PPO env
# Source this file manually before RLinf work:
#   source /root/autodl-tmp/rlinf_env.sh
# =========================

# -------------------------
# Project layout
# -------------------------
export WORK_ROOT="/root/autodl-tmp"

export RLINF_ROOT="${WORK_ROOT}/RLinf"
export ROBOTWIN_PATH="${WORK_ROOT}/RoboTwin_RLinf"
export RLINF_MODEL_ROOT="${WORK_ROOT}/models/rlinf"

# -------------------------
# Cache / temp layout
# -------------------------
export CACHE_ROOT="${WORK_ROOT}/cache"
export TMPDIR="${WORK_ROOT}/tmp"

mkdir -p \
  "${RLINF_MODEL_ROOT}" \
  "${CACHE_ROOT}/uv" \
  "${CACHE_ROOT}/uv_python/install" \
  "${CACHE_ROOT}/uv_python/cache" \
  "${CACHE_ROOT}/pip" \
  "${CACHE_ROOT}/huggingface" \
  "${CACHE_ROOT}/huggingface/hub" \
  "${CACHE_ROOT}/root_cache" \
  "${CACHE_ROOT}/torch_extensions" \
  "${CACHE_ROOT}/cuda" \
  "${CACHE_ROOT}/rlinf_assets" \
  "${TMPDIR}" \
  "${WORK_ROOT}/setup_logs"

# -------------------------
# Generic cache redirection
# -------------------------
export XDG_CACHE_HOME="${CACHE_ROOT}/root_cache"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"

# -------------------------
# uv cache and network behavior
# -------------------------
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export UV_PYTHON_INSTALL_DIR="${CACHE_ROOT}/uv_python/install"
export UV_PYTHON_CACHE_DIR="${CACHE_ROOT}/uv_python/cache"
export UV_HTTP_TIMEOUT=120
export UV_HTTP_RETRIES=8

# 当前 AutoDL 机器实测清华源最快
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

# -------------------------
# pip fallback mirror
# -------------------------
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"

# -------------------------
# HuggingFace behavior
# -------------------------
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600

# -------------------------
# RLinf/OpenPI asset fallback directory
# Some RLinf asset scripts may use DOWNLOAD_DIR.
# This prevents extra assets from falling back to /root/.cache.
# -------------------------
export DOWNLOAD_DIR="${CACHE_ROOT}/rlinf_assets"

# -------------------------
# Runtime variables used later by RLinf RoboTwin
# -------------------------
export ROBOT_PLATFORM="ALOHA"

# Avoid repeatedly prepending the same RoboTwin path when sourcing multiple times.
case ":${PYTHONPATH:-}:" in
  *":${ROBOTWIN_PATH}:"*) ;;
  *) export PYTHONPATH="${ROBOTWIN_PATH}:${PYTHONPATH:-}" ;;
esac

# uv may be installed here by the installer.
case ":${PATH}:" in
  *":${HOME}/.local/bin:"*) ;;
  *) export PATH="${HOME}/.local/bin:${PATH}" ;;
esac

# -------------------------
# Optional CUDA toolkit path
# Only activates if nvcc exists. Current preflight shows nvcc is not in PATH,
# so this block is harmless unless a CUDA toolkit is present.
# -------------------------
for _cuda_dir in /usr/local/cuda /usr/local/cuda-13* /usr/local/cuda-12*; do
  if [ -x "${_cuda_dir}/bin/nvcc" ]; then
    export CUDA_HOME="${_cuda_dir}"
    export CUDA_PATH="${_cuda_dir}"

    case ":${PATH}:" in
      *":${_cuda_dir}/bin:"*) ;;
      *) export PATH="${_cuda_dir}/bin:${PATH}" ;;
    esac

    case ":${LD_LIBRARY_PATH:-}:" in
      *":${_cuda_dir}/lib64:"*) ;;
      *) export LD_LIBRARY_PATH="${_cuda_dir}/lib64:${LD_LIBRARY_PATH:-}" ;;
    esac

    break
  fi
done
unset _cuda_dir

# -------------------------
# Avoid stale proxy by default.
# If GitHub clone needs AutoDL acceleration, source /etc/network_turbo temporarily
# only around that command.
# -------------------------
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
EOF

# =========================
# Activate once in current shell
# =========================
source /root/autodl-tmp/rlinf_env.sh

echo "===== RLinf env ====="
echo "RLINF_ROOT=${RLINF_ROOT}"
echo "ROBOTWIN_PATH=${ROBOTWIN_PATH}"
echo "RLINF_MODEL_ROOT=${RLINF_MODEL_ROOT}"
echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR}"
echo "UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX}"
echo "PIP_INDEX_URL=${PIP_INDEX_URL}"
echo "HF_HOME=${HF_HOME}"
echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "TMPDIR=${TMPDIR}"
echo "DOWNLOAD_DIR=${DOWNLOAD_DIR}"
echo "ROBOT_PLATFORM=${ROBOT_PLATFORM}"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDA_PATH=${CUDA_PATH:-}"

echo
echo "===== paths ====="
ls -ld \
  "${RLINF_MODEL_ROOT}" \
  "${CACHE_ROOT}/uv" \
  "${CACHE_ROOT}/uv_python/install" \
  "${CACHE_ROOT}/huggingface" \
  "${CACHE_ROOT}/rlinf_assets" \
  "${TMPDIR}"

echo
echo "===== disk ====="
df -h / "${WORK_ROOT}"
```

Check

```Bash
source /root/autodl-tmp/rlinf_env.sh

echo "===== cache env check ====="
env | grep -E 'RLINF_ROOT|ROBOTWIN_PATH|RLINF_MODEL_ROOT|UV_|HF_|PIP_|TMPDIR|DOWNLOAD_DIR|ROBOT_PLATFORM|CUDA_HOME|CUDA_PATH' \
  | sort

echo
echo "===== root cache symlink ====="
ls -ld /root/.cache
readlink -f /root/.cache

echo
echo "===== no Motus touched ====="
ls -ld /root/autodl-tmp/RoboTwin
ls -ld /root/autodl-tmp/Motus
ls -ld /root/autodl-tmp/models/motus
```

使用：

```Bash
必须你显式执行：
source /root/autodl-tmp/rlinf_env.sh
作用范围是：**当前这个终端 shell，以及从这个 shell 启动的后续命令**。新开一个终端后，需要重新 source 一次。
```

每次新bash：

```Bash
source /root/autodl-tmp/rlinf_env.sh
```



#### 装rlinf

仓库

Check

```Bash
source /root/autodl-tmp/rlinf_env.sh

echo "===== env ====="
echo "RLINF_ROOT=${RLINF_ROOT}"
echo "ROBOTWIN_PATH=${ROBOTWIN_PATH}"
echo "UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX}"
echo "PIP_INDEX_URL=${PIP_INDEX_URL}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"

echo
echo "===== cuda ====="
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDA_PATH=${CUDA_PATH:-}"
which nvcc || true
nvcc --version || true

echo
echo "===== disk ====="
df -h / /root/autodl-tmp
```

Clone，顺利

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${WORK_ROOT}"

if [ -d "${RLINF_ROOT}/.git" ]; then
  echo "RLinf already exists: ${RLINF_ROOT}"
else
  git clone https://github.com/RLinf/RLinf.git "${RLINF_ROOT}"
fi

cd "${RLINF_ROOT}"

echo "===== repo ====="
pwd
git rev-parse --short HEAD
git status --short
```

Check

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"

echo "===== install help ====="
bash requirements/install.sh --help | sed -n '1,180p'

echo
echo "===== supported models/envs ====="
grep -n "SUPPORTED_MODELS" requirements/install.sh
grep -n "SUPPORTED_ENVS" requirements/install.sh

echo
echo "===== openpi / robotwin / mirror logic ====="
grep -n "install_openpi_model" requirements/install.sh
grep -n "install_robotwin_env" requirements/install.sh
grep -n "setup_mirror" requirements/install.sh
grep -n "ghfast" requirements/install.sh
```



装依赖；刚开始速度正常

```Bash
bash requirements/install.sh embodied --model openpi --env robotwin
```

问题：

```Bash
pytorch3d，， GitHub 连接超时
flash-attn 已经装成功了，RoboTwin 常规依赖也装了一大部分，失败点是 pytorch3d 这一步从 GitHub 拉源码超时
```

改脚本，备份

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"

# =========================
# Backup and create patched installer
# =========================
STAMP="$(date +%Y%m%d_%H%M%S)"

ORIG="requirements/install.sh"
BAK="requirements/install.sh.bak_${STAMP}"
PATCHED="requirements/install_no_pytorch3d.sh"

cp "${ORIG}" "${BAK}"
cp "${ORIG}" "${PATCHED}"

python - <<'PY'
from pathlib import Path

p = Path("requirements/install_no_pytorch3d.sh")
s = p.read_text()

old = '    uv pip install git+${GITHUB_PREFIX}https://github.com/facebookresearch/pytorch3d.git@v0.7.9  --no-build-isolation\n'

new = (
    '    echo "[install_no_pytorch3d] Skipping pytorch3d for robotwin env; install manually later if needed."\n'
    '    # SKIP_PYTORCH3D: original line disabled because GitHub fetch timed out on AutoDL.\n'
    '    # uv pip install git+${GITHUB_PREFIX}https://github.com/facebookresearch/pytorch3d.git@v0.7.9  --no-build-isolation\n'
)

if old not in s:
    raise SystemExit("Target pytorch3d install line not found. Do not continue.")

# Replace first occurrence only: this targets install_robotwin_env.
s = s.replace(old, new, 1)

p.write_text(s)
PY

chmod +x "${PATCHED}"

echo "===== backup ====="
ls -lh "${BAK}"

echo
echo "===== patched installer ====="
ls -lh "${PATCHED}"

echo
echo "===== check pytorch3d patch ====="
grep -nC 4 "pytorch3d" "${PATCHED}"
```

跑新脚本

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"

mkdir -p logs/setup

LOG="logs/setup/install_openpi_robotwin_no_pytorch3d_$(date +%Y%m%d_%H%M%S).log"

echo "LOG=${LOG}"

set -o pipefail

bash requirements/install_no_pytorch3d.sh embodied \
  --model openpi \
  --env robotwin \
  --no-root \
  --no-flash-attn \
  2>&1 | tee "${LOG}"

echo
echo "EXIT_CODE=${PIPESTATUS[0]}"
echo "LOG=${LOG}"
```

Check

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"
source .venv/bin/activate

echo "===== python ====="
which python
python --version
python -c "import sys; print(sys.executable)"

echo
echo "===== torch / cuda ====="
python - <<'PY'
import torch
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count =", torch.cuda.device_count())
    print("gpu0 =", torch.cuda.get_device_name(0))
PY

echo
echo "===== key imports ====="
python - <<'PY'
mods = [
    "openpi",
    "sapien",
    "mplib",
    "open3d",
    "warp",
    "curobo",
    "flash_attn",
]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "ok", getattr(mod, "__file__", ""))
    except Exception as e:
        print(m, "failed:", repr(e))

try:
    import pytorch3d
    print("pytorch3d ok", getattr(pytorch3d, "__file__", ""))
except Exception as e:
    print("pytorch3d optional missing:", repr(e))
PY

echo
echo "===== pip check ====="
python -m pip check || true
```

问题：

```Bash
flash-attn 不是 pytorch3d，后面 OpenPI/RLinf 模型侧更可能用到它。它之前已经成功装过一次，只是你后来用 --no-flash-attn 重跑时，uv sync 把它卸掉了，所以现在单独补回来。
```

补：

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"
source .venv/bin/activate

uv pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

python - <<'PY'
import flash_attn
print("flash_attn ok =", flash_attn.__file__)
PY
```

查

```Python
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"
source .venv/bin/activate

python - <<'PY'
mods = [
    "openpi",
    "sapien",
    "mplib",
    "open3d",
    "warp",
    "curobo",
    "flash_attn",
]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "ok", getattr(mod, "__file__", ""))
    except Exception as e:
        print(m, "failed:", repr(e))

try:
    import pytorch3d
    print("pytorch3d ok", getattr(pytorch3d, "__file__", ""))
except Exception as e:
    print("pytorch3d optional missing:", repr(e))
PY
```



#### 装rt2



Clone

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${WORK_ROOT}"

if [ -d "${ROBOTWIN_PATH}/.git" ]; then
  echo "RoboTwin_RLinf already exists: ${ROBOTWIN_PATH}"
else
  git clone https://github.com/RoboTwin-Platform/RoboTwin.git \
    -b RLinf_support \
    "${ROBOTWIN_PATH}"
fi

cd "${ROBOTWIN_PATH}"

echo "===== RoboTwin_RLinf repo ====="
pwd
git branch --show-current
git rev-parse --short HEAD
git status --short
ls -lah | sed -n '1,120p'
```

软链接asset

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${ROBOTWIN_PATH}"

echo "===== existing old assets ====="
du -sh /root/autodl-tmp/RoboTwin/assets 2>/dev/null || true
ls -ld /root/autodl-tmp/RoboTwin/assets 2>/dev/null || true

echo
echo "===== current RLinf assets ====="
ls -ld "${ROBOTWIN_PATH}/assets" 2>/dev/null || true

if [ ! -e "${ROBOTWIN_PATH}/assets" ]; then
  ln -s /root/autodl-tmp/RoboTwin/assets "${ROBOTWIN_PATH}/assets"
fi

echo
echo "===== after link ====="
ls -ld "${ROBOTWIN_PATH}/assets"
du -sh "${ROBOTWIN_PATH}/assets" 2>/dev/null || true
ls -lah "${ROBOTWIN_PATH}/assets" | sed -n '1,80p'
```

问题：

```Bash
RoboTwin_RLinf 克隆下来时本来就带了一个小的 assets/ 目录，所以这句判断没有触发：
if [ ! -e "${ROBOTWIN_PATH}/assets" ]; then
  ln -s /root/autodl-tmp/RoboTwin/assets "${ROBOTWIN_PATH}/assets"
fi
当前实际状态是：
/root/autodl-tmp/RoboTwin/assets        16G   # 旧 Motus/RoboTwin 已下载资产
/root/autodl-tmp/RoboTwin_RLinf/assets 5.8M  # RLinf_support 自带的小目录，不是完整资产

我建议用**稳妥官方路线**：保留 RoboTwin_RLinf/assets 这个小目录，然后直接运行：
bash script/_download_assets.sh
理由是：RLinf_support 分支自带的 assets/_download.py 和 assets/files 可能对应这个分支自己的资产下载逻辑；直接替换成旧 assets 虽然大概率能跑，但不如官方路径稳。你现在数据盘还有 1016G，重复下载/解压 16G 级别资产不是问题。
```

单独下载：

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${ROBOTWIN_PATH}"

echo "===== before assets download ====="
pwd
git branch --show-current
du -sh assets 2>/dev/null || true
ls -lah assets | sed -n '1,120p'

echo
echo "===== HF/cache env ====="
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "HF_HOME=${HF_HOME}"
echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "TMPDIR=${TMPDIR}"

echo
echo "===== run asset download ====="
bash script/_download_assets.sh

echo
echo "===== after assets download ====="
du -sh assets
find assets -maxdepth 2 -type d | sort | sed -n '1,120p'
```

速度正常，十几MB



#### 下模型



pi0

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"
source .venv/bin/activate

mkdir -p "${RLINF_MODEL_ROOT}"

echo "===== env ====="
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "HF_HOME=${HF_HOME}"
echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "RLINF_MODEL_ROOT=${RLINF_MODEL_ROOT}"

echo
echo "===== hf ====="
hf --version || true

echo
echo "===== download Pi0 SFT ====="
hf download RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle \
  --local-dir "${RLINF_MODEL_ROOT}/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"

echo
echo "===== downloaded ====="
du -sh "${RLINF_MODEL_ROOT}/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
find "${RLINF_MODEL_ROOT}/RLinf-Pi0-RoboTwin-SFT-adjust_bottle" \
  -maxdepth 2 \
  -type f \
  | sed -n '1,120p'
```

Check

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"

echo "===== Pi0 configs ====="
ls -lh examples/embodiment/config/robotwin_adjust_bottle_ppo_OpenPI.yaml
ls -lh examples/embodiment/config/robotwin_adjust_bottle_ppo_OpenPI_eval.yaml

echo
echo "===== key fields ====="
for f in \
  examples/embodiment/config/robotwin_adjust_bottle_ppo_OpenPI.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_ppo_OpenPI_eval.yaml
do
  echo
  echo "----- $f -----"
  grep -nE "model_path|ckpt_path|assets_path|only_eval|max_steps|total_num_envs|component_placement|center_crop|embodiment|collect_wrist_camera|save_interval|val_check_interval|eval_rollout_epoch|global_batch_size|micro_batch_size|group_size|gradient_checkpointing|enable_offload|config_name|add_value_head|num_action_chunks|action_dim" "$f" || true
done
```



成功，模型在：

```Bash
/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
```

7\.6G



### run

#### 评估



配参数

/root/autodl\-tmp/RLinf/examples/embodiment/config/robotwin\_adjust\_bottle\_ppo\_openpi\_eval\.yaml

复制yaml

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"

SRC="examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_eval.yaml"
DST="examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_eval_autodl_pi0.yaml"

cp "${SRC}" "${DST}"

echo "DST=${DST}"

grep -nE "model_path|ckpt_path|assets_path|only_eval|max_steps|total_num_envs|component_placement|center_crop|embodiment|collect_wrist_camera|save_interval|val_check_interval|eval_rollout_epoch|global_batch_size|micro_batch_size|group_size|gradient_checkpointing|enable_offload|config_name|add_value_head|num_action_chunks|action_dim|task_name|experiment_name" \
  "${DST}" \
  || true
```

/root/autodl\-tmp/RLinf/examples/embodiment/config/robotwin\_adjust\_bottle\_ppo\_openpi\_eval\_autodl\_pi0\.yaml



手动改：

```Bash
actor.model.model_path
/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle

env.train.assets_path
env.eval.assets_path
/root/autodl-tmp/RoboTwin_RLinf

runner.experiment_name
robotwin_ppo_openpi_eval_autodl

algorithm.eval_rollout_epoch
10
env.eval.total_num_envs
16

cluster.component_placement.actor, env, rollout
0-1

enable_offload
True （不变）

```



评估

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}

# 这两个通常脚本会设，但手动显式给出更稳
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

CONFIG_NAME="robotwin_adjust_bottle_ppo_openpi_eval_autodl_pi0"

echo "CONFIG_NAME=${CONFIG_NAME}"
echo "ROBOTWIN_PATH=${ROBOTWIN_PATH}"
echo "REPO_PATH=${REPO_PATH}"
echo "EMBODIED_PATH=${EMBODIED_PATH}"

bash examples/embodiment/eval_embodiment.sh "${CONFIG_NAME}"
```

成功开始，两张卡峰值显存都是20G左右，远没顶满

15min左右，挺快



产物

rlinf的日志目录：/root/autodl\-tmp/RLinf/logs

问题：视频有点快；TODO：排查问题，应该有参数

成功率75



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OGY4NjdiYTE0MzAyYzFhN2Q4MjliMGI0MDMyYTQ1MzlfMzFiMmMwNThmMjk4NmI5ZWY4ZjM1NTZlMDRjNmQ0ZWJfSUQ6NzY1MDg5NDk0NDMyODAyNzMyNl8xNzg1MTYwMjkyOjE3ODUyNDY2OTJfVjM)



#### 训练

改参数

复制到：/root/autodl\-tmp/RLinf/examples/embodiment/config/robotwin\_adjust\_bottle\_ppo\_openpi\_autodl\_pi0\.yaml

```Bash
actor.model.model_path
/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle

env.train.assets_path
env.eval.assets_path
/root/autodl-tmp/RoboTwin_RLinf

cluster.component_placement.actor, env, rollout
0-1

env.train.total_num_envs
32

env.eval.total_num_envs
8

algorithm.eval_rollout_epoch
4

actor.micro_batch_size
32

actor.global_batch_size
512

runner.max_steps
10

runner.val_check_interval
5

runner.save_interval
5
```



训练

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

mkdir -p logs/nohup

CONFIG_NAME="robotwin_adjust_bottle_ppo_openpi_autodl_pi0"
LOG="logs/nohup/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc "
set -o pipefail

source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:\${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash examples/embodiment/run_embodiment.sh ${CONFIG_NAME}
" > "${LOG}" 2>&1 &

PID=$!

echo "PID=${PID}"
echo "LOG=${LOG}"
```

\[1\] 65748

PID=65748

LOG=logs/nohup/robotwin\_adjust\_bottle\_ppo\_openpi\_autodl\_pi0\_train20\_20260613\_182713\.log

失败：

```Bash
杜撰了robotwin_adjust_bottle_ppo_openpi_autodl_pi0_train20
```

改回

\[1\] 66666

PID=66666

LOG=logs/nohup/robotwin\_adjust\_bottle\_ppo\_openpi\_autodl\_pi0\_20260613\_183552\.log



时间：4h；2300收

显存峰值42G



中断：CPU爆，要保存ckpt时

```Bash
它跑到了 Global Step 9/20，接近第 10 步验证阶段时，Ray 因为 CPU 内存超过阈值杀掉了一个 EnvWorker，导致训练崩掉。
```

降低 eval 阶段内存；调参

再放10步

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"
source .venv/bin/activate

CONFIG_NAME="robotwin_adjust_bottle_ppo_openpi_autodl_pi0"
CONFIG_FILE="${RLINF_ROOT}/examples/embodiment/config/${CONFIG_NAME}.yaml"

echo "CONFIG_FILE=${CONFIG_FILE}"
test -f "${CONFIG_FILE}" || {
  echo "ERROR: config file not found: ${CONFIG_FILE}"
  exit 1
}

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

mkdir -p logs/nohup

LOG="logs/nohup/${CONFIG_NAME}_retry_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc "
set -o pipefail

source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

CONFIG_NAME='${CONFIG_NAME}'
CONFIG_FILE='/root/autodl-tmp/RLinf/examples/embodiment/config/${CONFIG_NAME}.yaml'

echo \"CONFIG_NAME=\${CONFIG_NAME}\"
echo \"CONFIG_FILE=\${CONFIG_FILE}\"
test -f \"\${CONFIG_FILE}\" || {
  echo \"ERROR: config file not found: \${CONFIG_FILE}\"
  exit 1
}

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:\${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash examples/embodiment/run_embodiment.sh \${CONFIG_NAME}
" > "${LOG}" 2>&1 &

PID=$!

echo "PID=${PID}"
echo "LOG=${LOG}"
```

\[2\] 301309

PID=301309

LOG=logs/nohup/robotwin\_adjust\_bottle\_ppo\_openpi\_autodl\_pi0\_retry\_20260613\_211041\.log



又爆，不过ckpt存了

```Bash
训练确实跑起来了，也成功过了 step 5 的 eval 和 checkpoint 保存；但随后又因为 CPU 内存/RAM 超过 Ray 95% 阈值被杀。不是 GPU OOM。

崩在 step 5 保存之后，进入下一轮 rollout 时。日志中后面出现：
ValueError: Unsupported object type
Broadcast failed / connection closed
Ray OutOfMemoryError
这些 Unsupported object type 和 collective/broadcast 错误是**后果**，根因是 Ray OOM：
Memory was 228.07GB / 240.00GB = 95.03%
threshold = 95.00%
Ray 因为超过内存阈值杀 worker。Top memory users 里最主要的是：
304865 ray::EnvWorker.interact 86.70GB
304867 ray::EnvWorker.interact 86.13GB
304841 ray::EmbodiedFSDPActor.recv_rollout_trajectories 17.30GB
304834 ray::EmbodiedFSDPActor.recv_rollout_trajectories 17.00GB
所以根因仍然是 **EnvWorker.interact 的 RAM 过高**。两个 EnvWorker 加起来就 172GB，再加 actor/rollout/Ray 进程，直接顶到 228GB/240GB。
```

5步的ckpt：

```Bash
/root/autodl-tmp/RLinf/logs/20260613-21:10:41-robotwin_adjust_bottle_ppo_openpi_autodl_pi0/robotwin_ppo_openpi/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt
```



工具：显存监视

```Bash
cd /root/autodl-tmp/RLinf

mkdir -p logs/gpu_monitor

cat > logs/gpu_monitor/watch_gpu_mem.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./watch_gpu_mem.sh [INTERVAL_SECONDS]
#
# Example:
#   ./watch_gpu_mem.sh 2

INTERVAL="${1:-2}"
LOG_DIR="/root/autodl-tmp/RLinf/logs/gpu_monitor"
STAMP="$(date +%Y%m%d_%H%M%S)"

CSV="${LOG_DIR}/gpu_mem_${STAMP}.csv"
PEAK_TXT="${LOG_DIR}/gpu_mem_peak_${STAMP}.txt"
STDOUT_LOG="${LOG_DIR}/watch_stdout_${STAMP}.log"

mkdir -p "${LOG_DIR}"

trim() {
  local x="$1"
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  printf '%s' "$x"
}

declare -A PEAK_USED
declare -A PEAK_TIME
declare -A GPU_NAME
declare -A GPU_TOTAL

echo "timestamp,gpu_index,gpu_name,memory_used_mb,memory_total_mb,gpu_util_pct,power_w,temp_c,peak_memory_used_mb" > "${CSV}"

ln -sfn "${CSV}" "${LOG_DIR}/latest_gpu_mem.csv"
ln -sfn "${PEAK_TXT}" "${LOG_DIR}/latest_gpu_peak.txt"

echo "[watch_gpu_mem] started at $(date)"
echo "[watch_gpu_mem] interval=${INTERVAL}s"
echo "[watch_gpu_mem] csv=${CSV}"
echo "[watch_gpu_mem] peak=${PEAK_TXT}"
echo "[watch_gpu_mem] pid=$$"
echo "$$" > "${LOG_DIR}/watch_gpu_mem.pid"

trap 'echo "[watch_gpu_mem] stopped at $(date)"; exit 0' INT TERM

while true; do
  TS="$(date '+%F %T')"

  mapfile -t ROWS < <(
    nvidia-smi \
      --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits
  )

  for ROW in "${ROWS[@]}"; do
    IFS=',' read -r IDX NAME USED TOTAL UTIL POWER TEMP <<< "${ROW}"

    IDX="$(trim "${IDX}")"
    NAME="$(trim "${NAME}")"
    USED="$(trim "${USED}")"
    TOTAL="$(trim "${TOTAL}")"
    UTIL="$(trim "${UTIL}")"
    POWER="$(trim "${POWER}")"
    TEMP="$(trim "${TEMP}")"

    # nvidia-smi with nounits usually returns integers for memory.
    USED_INT="${USED%.*}"
    TOTAL_INT="${TOTAL%.*}"

    GPU_NAME["${IDX}"]="${NAME}"
    GPU_TOTAL["${IDX}"]="${TOTAL_INT}"

    OLD_PEAK="${PEAK_USED[${IDX}]:-0}"

    if [ "${USED_INT}" -gt "${OLD_PEAK}" ]; then
      PEAK_USED["${IDX}"]="${USED_INT}"
      PEAK_TIME["${IDX}"]="${TS}"
    fi

    CUR_PEAK="${PEAK_USED[${IDX}]:-0}"

    echo "${TS},${IDX},${NAME},${USED_INT},${TOTAL_INT},${UTIL},${POWER},${TEMP},${CUR_PEAK}" >> "${CSV}"
  done

  {
    echo "updated_at: ${TS}"
    echo "csv: ${CSV}"
    echo
    printf "%-8s %-24s %-14s %-14s %-22s\n" "gpu" "name" "total_mb" "peak_mb" "peak_time"
    printf "%-8s %-24s %-14s %-14s %-22s\n" "---" "----" "--------" "-------" "---------"

    for IDX in $(printf '%s\n' "${!PEAK_USED[@]}" | sort -n); do
      printf "%-8s %-24s %-14s %-14s %-22s\n" \
        "${IDX}" \
        "${GPU_NAME[${IDX}]}" \
        "${GPU_TOTAL[${IDX}]}" \
        "${PEAK_USED[${IDX}]}" \
        "${PEAK_TIME[${IDX}]}"
    done
  } > "${PEAK_TXT}"

  sleep "${INTERVAL}"
done
EOF

chmod +x logs/gpu_monitor/watch_gpu_mem.sh

ls -lh logs/gpu_monitor/watch_gpu_mem.sh
```

启动

```Bash
cd /root/autodl-tmp/RLinf

nohup logs/gpu_monitor/watch_gpu_mem.sh 2 \
  > logs/gpu_monitor/watch_gpu_mem_launcher_$(date +%Y%m%d_%H%M%S).log \
  2>&1 &

echo "MONITOR_PID=$!"
```



工具：内存显存监控

```Python
cd /root/autodl-tmp/RLinf

mkdir -p logs/resource_monitor

cat > logs/resource_monitor/watch_resources.py <<'PY'
#!/usr/bin/env python3
import argparse
import csv
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

RUNNING = True


def on_signal(signum, frame):
    global RUNNING
    RUNNING = False


signal.signal(signal.SIGINT, on_signal)
signal.signal(signal.SIGTERM, on_signal)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_text() -> str:
    return datetime.now().strftime("%F %T")


def force_symlink(src: Path, dst: Path):
    try:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
    except FileNotFoundError:
        pass
    dst.symlink_to(src)


def read_meminfo_mb():
    vals = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            key, rest = line.split(":", 1)
            vals[key] = int(rest.strip().split()[0]) // 1024

    mem_total = vals.get("MemTotal", 0)
    mem_available = vals.get("MemAvailable", 0)
    mem_used = max(mem_total - mem_available, 0)
    mem_used_pct = (mem_used / mem_total * 100.0) if mem_total else 0.0

    swap_total = vals.get("SwapTotal", 0)
    swap_free = vals.get("SwapFree", 0)
    swap_used = max(swap_total - swap_free, 0)
    swap_used_pct = (swap_used / swap_total * 100.0) if swap_total else 0.0

    return {
        "mem_total_mb": mem_total,
        "mem_used_mb": mem_used,
        "mem_available_mb": mem_available,
        "mem_used_pct": mem_used_pct,
        "swap_total_mb": swap_total,
        "swap_used_mb": swap_used,
        "swap_used_pct": swap_used_pct,
    }


def parse_int_or_none(x):
    x = str(x).strip()
    if not x or x in {"N/A", "[N/A]", "nan"}:
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def parse_float_or_none(x):
    x = str(x).strip()
    if not x or x in {"N/A", "[N/A]", "nan"}:
        return None
    try:
        return float(x)
    except Exception:
        return None


def read_gpu_rows():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        return [], f"nvidia-smi failed: {e}"

    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, mem_used, mem_total, util, power, temp = parts[:7]
        rows.append({
            "gpu_index": idx,
            "gpu_name": name,
            "gpu_mem_used_mb": parse_int_or_none(mem_used),
            "gpu_mem_total_mb": parse_int_or_none(mem_total),
            "gpu_util_pct": parse_int_or_none(util),
            "gpu_power_w": parse_float_or_none(power),
            "gpu_temp_c": parse_int_or_none(temp),
        })
    return rows, None


def read_top_processes(topn: int):
    cmd = ["ps", "-eo", "pid,ppid,rss,%mem,cmd", "--sort=-rss"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []

    procs = []
    for line in out.splitlines()[1: topn + 1]:
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, rss_kb, mem_pct, cmdline = parts
        try:
            rss_mb = int(rss_kb) / 1024.0
        except Exception:
            rss_mb = 0.0
        try:
            mem_pct_f = float(mem_pct)
        except Exception:
            mem_pct_f = 0.0
        procs.append({
            "pid": pid,
            "ppid": ppid,
            "rss_mb": rss_mb,
            "mem_pct": mem_pct_f,
            "cmd": cmdline,
        })
    return procs


def write_peak_file(path: Path, ts: str, mem, mem_peak, swap_peak, gpu_peaks, gpu_current, procs, gpu_error):
    tmp = path.with_suffix(path.suffix + ".tmp")

    ray_threshold_mb = int(mem["mem_total_mb"] * 0.95)

    lines = []
    lines.append(f"updated_at: {ts}")
    lines.append("")
    lines.append("===== SYSTEM RAM =====")
    lines.append(f"current_used_mb:       {mem['mem_used_mb']}")
    lines.append(f"current_available_mb:  {mem['mem_available_mb']}")
    lines.append(f"current_used_pct:      {mem['mem_used_pct']:.2f}%")
    lines.append(f"peak_used_mb:          {mem_peak['used_mb']}")
    lines.append(f"peak_used_pct:         {mem_peak['used_pct']:.2f}%")
    lines.append(f"peak_time:             {mem_peak['time']}")
    lines.append(f"ray_95pct_threshold:   {ray_threshold_mb} MB")
    if mem["mem_used_mb"] >= ray_threshold_mb:
        lines.append("WARNING: current RAM is above Ray default 95% kill threshold.")
    elif mem["mem_used_pct"] >= 90:
        lines.append("WARNING: current RAM is above 90%; Ray OOM risk is high.")

    lines.append("")
    lines.append("===== SWAP =====")
    lines.append(f"current_swap_used_mb:  {mem['swap_used_mb']}")
    lines.append(f"current_swap_used_pct: {mem['swap_used_pct']:.2f}%")
    lines.append(f"peak_swap_used_mb:     {swap_peak['used_mb']}")
    lines.append(f"peak_swap_time:        {swap_peak['time']}")

    lines.append("")
    lines.append("===== GPU MEMORY PEAKS =====")
    if gpu_error:
        lines.append(gpu_error)
    else:
        lines.append(f"{'gpu':<5} {'name':<26} {'current_mb':>12} {'total_mb':>10} {'peak_mb':>10} {'peak_pct':>9} {'peak_time':>22}")
        for idx in sorted(gpu_peaks.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            cur = gpu_current.get(idx, {})
            peak = gpu_peaks[idx]
            total = cur.get("gpu_mem_total_mb") or peak.get("total_mb") or 0
            current_used = cur.get("gpu_mem_used_mb")
            peak_used = peak.get("used_mb", 0)
            peak_pct = (peak_used / total * 100.0) if total else 0.0
            name = cur.get("gpu_name") or peak.get("name") or ""
            lines.append(
                f"{idx:<5} {name:<26} {str(current_used):>12} {total:>10} {peak_used:>10} {peak_pct:>8.2f}% {peak.get('time',''):>22}"
            )

    lines.append("")
    lines.append("===== TOP MEMORY PROCESSES CURRENT =====")
    lines.append(f"{'rank':<5} {'pid':>8} {'ppid':>8} {'rss_mb':>12} {'mem_pct':>8}  command")
    for rank, p in enumerate(procs, start=1):
        cmd = p["cmd"]
        if len(cmd) > 160:
            cmd = cmd[:157] + "..."
        lines.append(f"{rank:<5} {p['pid']:>8} {p['ppid']:>8} {p['rss_mb']:>12.1f} {p['mem_pct']:>7.2f}%  {cmd}")

    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--topn", type=int, default=15)
    parser.add_argument("--log-dir", type=str, default="/root/autodl-tmp/RLinf/logs/resource_monitor")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = now_stamp()

    system_csv = log_dir / f"system_mem_{stamp}.csv"
    gpu_csv = log_dir / f"gpu_mem_{stamp}.csv"
    proc_csv = log_dir / f"process_mem_{stamp}.csv"
    peak_txt = log_dir / f"resource_peak_{stamp}.txt"
    pid_file = log_dir / "watch_resources.pid"

    force_symlink(system_csv, log_dir / "latest_system_mem.csv")
    force_symlink(gpu_csv, log_dir / "latest_gpu_mem.csv")
    force_symlink(proc_csv, log_dir / "latest_process_mem.csv")
    force_symlink(peak_txt, log_dir / "latest_resource_peak.txt")

    pid_file.write_text(str(os.getpid()) + "\n")

    mem_peak = {"used_mb": 0, "used_pct": 0.0, "time": ""}
    swap_peak = {"used_mb": 0, "time": ""}
    gpu_peaks = {}
    gpu_current = {}

    with open(system_csv, "w", newline="", buffering=1) as sf, \
         open(gpu_csv, "w", newline="", buffering=1) as gf, \
         open(proc_csv, "w", newline="", buffering=1) as pf:

        system_writer = csv.DictWriter(sf, fieldnames=[
            "timestamp",
            "mem_total_mb",
            "mem_used_mb",
            "mem_available_mb",
            "mem_used_pct",
            "swap_total_mb",
            "swap_used_mb",
            "swap_used_pct",
            "peak_mem_used_mb",
            "peak_mem_used_pct",
            "peak_swap_used_mb",
        ])
        gpu_writer = csv.DictWriter(gf, fieldnames=[
            "timestamp",
            "gpu_index",
            "gpu_name",
            "gpu_mem_used_mb",
            "gpu_mem_total_mb",
            "gpu_mem_used_pct",
            "gpu_util_pct",
            "gpu_power_w",
            "gpu_temp_c",
            "peak_gpu_mem_used_mb",
            "peak_gpu_mem_used_pct",
            "peak_gpu_util_pct",
        ])
        proc_writer = csv.DictWriter(pf, fieldnames=[
            "timestamp",
            "rank",
            "pid",
            "ppid",
            "rss_mb",
            "mem_pct",
            "cmd",
        ])

        system_writer.writeheader()
        gpu_writer.writeheader()
        proc_writer.writeheader()

        print(f"[watch_resources] started at {now_text()}", flush=True)
        print(f"[watch_resources] interval={args.interval}s", flush=True)
        print(f"[watch_resources] system_csv={system_csv}", flush=True)
        print(f"[watch_resources] gpu_csv={gpu_csv}", flush=True)
        print(f"[watch_resources] proc_csv={proc_csv}", flush=True)
        print(f"[watch_resources] peak_txt={peak_txt}", flush=True)
        print(f"[watch_resources] pid={os.getpid()}", flush=True)

        while RUNNING:
            ts = now_text()

            mem = read_meminfo_mb()
            if mem["mem_used_mb"] > mem_peak["used_mb"]:
                mem_peak = {
                    "used_mb": mem["mem_used_mb"],
                    "used_pct": mem["mem_used_pct"],
                    "time": ts,
                }
            if mem["swap_used_mb"] > swap_peak["used_mb"]:
                swap_peak = {
                    "used_mb": mem["swap_used_mb"],
                    "time": ts,
                }

            system_writer.writerow({
                "timestamp": ts,
                **mem,
                "peak_mem_used_mb": mem_peak["used_mb"],
                "peak_mem_used_pct": f"{mem_peak['used_pct']:.4f}",
                "peak_swap_used_mb": swap_peak["used_mb"],
            })

            gpu_rows, gpu_error = read_gpu_rows()
            gpu_current = {}
            for row in gpu_rows:
                idx = str(row["gpu_index"])
                used = row["gpu_mem_used_mb"] or 0
                total = row["gpu_mem_total_mb"] or 0
                util = row["gpu_util_pct"] or 0

                gpu_current[idx] = row

                if idx not in gpu_peaks:
                    gpu_peaks[idx] = {
                        "name": row["gpu_name"],
                        "used_mb": used,
                        "total_mb": total,
                        "util_pct": util,
                        "time": ts,
                    }
                else:
                    if used > gpu_peaks[idx]["used_mb"]:
                        gpu_peaks[idx]["used_mb"] = used
                        gpu_peaks[idx]["total_mb"] = total
                        gpu_peaks[idx]["name"] = row["gpu_name"]
                        gpu_peaks[idx]["time"] = ts
                    if util > gpu_peaks[idx].get("util_pct", 0):
                        gpu_peaks[idx]["util_pct"] = util

                used_pct = (used / total * 100.0) if total else 0.0
                peak_used = gpu_peaks[idx]["used_mb"]
                peak_pct = (peak_used / total * 100.0) if total else 0.0

                gpu_writer.writerow({
                    "timestamp": ts,
                    **row,
                    "gpu_mem_used_pct": f"{used_pct:.4f}",
                    "peak_gpu_mem_used_mb": peak_used,
                    "peak_gpu_mem_used_pct": f"{peak_pct:.4f}",
                    "peak_gpu_util_pct": gpu_peaks[idx].get("util_pct", 0),
                })

            procs = read_top_processes(args.topn)
            for rank, proc in enumerate(procs, start=1):
                proc_writer.writerow({
                    "timestamp": ts,
                    "rank": rank,
                    **proc,
                    "rss_mb": f"{proc['rss_mb']:.2f}",
                    "mem_pct": f"{proc['mem_pct']:.4f}",
                })

            write_peak_file(
                peak_txt,
                ts,
                mem,
                mem_peak,
                swap_peak,
                gpu_peaks,
                gpu_current,
                procs,
                gpu_error,
            )

            time.sleep(args.interval)

    print(f"[watch_resources] stopped at {now_text()}", flush=True)


if __name__ == "__main__":
    main()
PY

chmod +x logs/resource_monitor/watch_resources.py

ls -lh logs/resource_monitor/watch_resources.py
```

启动

```Bash
cd /root/autodl-tmp/RLinf

mkdir -p logs/resource_monitor

nohup .venv/bin/python logs/resource_monitor/watch_resources.py \
  --interval 2 \
  --topn 15 \
  > logs/resource_monitor/watch_resources_launcher_$(date +%Y%m%d_%H%M%S).log \
  2>&1 &

echo "RESOURCE_MONITOR_PID=$!"
```

\[3\] 454564

RESOURCE\_MONITOR\_PID=454564



日志

\[metrics\.log\]

\[run\_embodiment\.log\]

```Bash
critic 指标：
Stepexplained_variancevalue_lossvalue_clip_ratio
1-0.1110.1170.093
2-0.0670.0870.000
30.0210.0740.000
40.0320.0690.000
50.1400.0680.000
这个是比较好的趋势：
value_loss 从 **0.117 降到 0.068**，说明 value head 在拟合 rollout return。 
explained_variance 从负数涨到 **0.140**，虽然还不高，但方向正确。 
value_clip_ratio 后面基本为 0，说明 critic update 没有大量触发 value clipping。 
所以 critic 没有发散，反而是在逐步变好
```

训练总体正常，方向对



TODO

分析日志，日志中有些报错，不知道是否影响；更细的记录，tensorboard？看看有啥有用的

训推中间都有报错，好像都不影响？curobo缺包之类的

调参顶显存；注意内存爆；分别受什么因素影响？



#### 评估



指定ckpt

```Bash
runner.ckpt_path
/root/autodl-tmp/RLinf/logs/20260613-21:10:41-robotwin_adjust_bottle_ppo_openpi_autodl_pi0/robotwin_ppo_openpi/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt

runner.logger.experiment_name
robotwin_ppo_openpi_eval_autodl_step5
```



评估

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"
source .venv/bin/activate

CONFIG_NAME="robotwin_adjust_bottle_ppo_openpi_eval_autodl_pi0"
CONFIG_FILE="${RLINF_ROOT}/examples/embodiment/config/${CONFIG_NAME}.yaml"

echo "CONFIG_NAME=${CONFIG_NAME}"
echo "CONFIG_FILE=${CONFIG_FILE}"

test -f "${CONFIG_FILE}" || {
  echo "ERROR: config file not found: ${CONFIG_FILE}"
  exit 1
}

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

mkdir -p logs/nohup

LOG="logs/nohup/${CONFIG_NAME}_step5_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc "
set -o pipefail

source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

CONFIG_NAME='${CONFIG_NAME}'
CONFIG_FILE='/root/autodl-tmp/RLinf/examples/embodiment/config/${CONFIG_NAME}.yaml'

echo \"CONFIG_NAME=\${CONFIG_NAME}\"
echo \"CONFIG_FILE=\${CONFIG_FILE}\"

test -f \"\${CONFIG_FILE}\" || {
  echo \"ERROR: config file not found: \${CONFIG_FILE}\"
  exit 1
}

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:\${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash examples/embodiment/eval_embodiment.sh \${CONFIG_NAME}
" > "${LOG}" 2>&1 &

PID=$!

echo "PID=${PID}"
echo "LOG=${LOG}"
```



83\.1%

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZTc4MTJhMWY5MjA2ZDhlYWZjYWQwYjBkZDNiNGIxMDJfYWUxMjgyNjIxYTdhOTU5ZDIyNzQwZWE2ZTc2N2Y4NzhfSUQ6NzY1MDg5NDcxNTQ5MTA0NDMyN18xNzg1MTYwMjkyOjE3ODUyNDY2OTJfVjM)



TODO

评估显存顶更多



