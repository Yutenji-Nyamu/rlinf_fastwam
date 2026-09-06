# Exp\_snd



### TODO



实例整体管理

换自回归wam？有rt2？RynnVLA\-002？libero？

换baseline？fastwam ？更快更省？5090？

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a22907e\-d914\-832b\-a939\-9dfb75da5f14



impl

tts：

Tts v8

Opd

m1

m2

找其他tta？

Exp

日志统一放



目前的patch文件：

\[deploy\_policy\.py\]

\[deploy\_policy\.yml\]

\[eval\.sh\]

\[eval\_policy\.py\]

\[motus\.py\]

\[gpu\_queue\_worker\.sh\]

\[wait\_gpu\_and\_run\.sh\]



1 motus tts

实验队列：

不同成功率任务的表现；找更高的看看

开发队列

问题：

confidently wrong；

失败执行的后期；可能和失败ep后期的震荡有关；如何针对处理失败后期的震荡？

没用的trick：视频等？

更多方法；

wam的加入

tta类？或者mg

？？：multi level tts？chunk\+v

可用性

操作

反代（需要？）和codex

codex桌面版连服务器？更多功能？

other

传参：推理时选择ckpt，还是改配置文件方便？

可视化

找：rw都可视化什么；tts等；例如tts中间如何work，如何观察rollout，更多现象

如何可视化所有成功失败chunk到一张图；如何追踪chunk和视频latent的成功失败；对齐长度？

打印更多值，曲线等；

实验可视化方案

稳定性

种子成功率统计、选择、固定？在eval\.py？

卡：

autodl，没事，克隆实例，占盘，关机；

监视2平台余额，实例





2 tts\-\>opd

**放实验，调参**

**m2**

找方案：5chat，code；实现多个版本；tta tmk？

峰值大概率在20步以内



\[deploy\_policy\.py\]

\[deploy\_policy\.yml\]

\[eval\.sh\]

\[motus\.py\]

\[deploy\_policy\.py\]

\[deploy\_policy\.yml\]

\[eval\.sh\]

\[eval\_policy\.py\]

\[motus\.py\]

3 rl

rlinf\+wam有没有？把motus缝过去？改一版无监督grpo?先看代码



### motus重新跑通

#### 算力



显存先2张80g

预计：

Action Expert 全参；冻结的具体实现，参考其他

训练：开t5复用；推理：显存吃紧就real\-world的复用缝进rt2推理



gpt：

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a06dabf\-fc80\-83ea\-9065\-caa2cbf9672b



参考

在compshare，用之前rt2镜像：[RoboTwin 2\.0 Compshare a800](https://my.feishu.cn/wiki/T5fewcixnimrfyk3u3LcStJwnwg)

motus安装参考之前a6000上；[Motus RoboTwin 2\.0 Evaluation](https://my.feishu.cn/wiki/SmuRwkVvpidltvkkT7KcVybOn5e)



复用之前rt2 act bbh镜像



#### 环境

装motus环境



备份环境

```Plain Text
conda create -n RoboTwin_before_motus --clone RoboTwin -y
```



路径设置：

```Plain Text
export ROBOTWIN_ROOT="/home/ubuntu/workspace/RoboTwin"
export MODEL_ROOT="/home/ubuntu/models/motus"
export CONDA_NAME="RoboTwin"

echo "$ROBOTWIN_ROOT"
echo "$MODEL_ROOT"
ls -lah "$ROBOTWIN_ROOT"
ls -lah "$ROBOTWIN_ROOT/policy"
```

需要？



克隆项目

```Bash
cd /home/ubuntu

if [ ! -d "/home/ubuntu/Motus" ]; then
  git clone https://github.com/thu-ml/Motus.git
fi

cp -r /home/ubuntu/Motus/inference/robotwin/Motus /home/ubuntu/workspace/RoboTwin/policy/

ls -lah /home/ubuntu/workspace/RoboTwin/policy/Motus | sed -n '1,120p'
```



motus依赖

```Plain Text
cd /home/ubuntu/workspace/RoboTwin/policy/Motus
conda activate RoboTwin

python -m pip install -U pip setuptools wheel ninja packaging psutil
python -m pip install flash_attn --no-build-isolation
python -m pip install -r requirements.txt
```

卡在编译比较久，几十分钟；正常？

中止



0516继续

机子不能在vscode直接连，一定要新开ssh，输入ip密码等



检查昨天调codex的代理是否残留

```Bash
echo "===== current proxy env ====="
env | grep -i proxy || true

echo "===== check 17897 listener ====="
ss -ltnp | grep 17897 || true
```

没有



继续装motus依赖

```Plain Text
cd /home/ubuntu/workspace/RoboTwin/policy/Motus
conda activate RoboTwin

python -m pip install -U pip setuptools wheel ninja packaging psutil

export MAX_JOBS=8
python -m pip install -v flash-attn --no-build-isolation

python -m pip install -r requirements.txt
```

\-v 能打印输出看编译进度

gpt：下完，正在编译



修环境（根据之前的跑通）

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus
conda activate RoboTwin

python -m pip install --force-reinstall \
  "numpy<2" \
  "transformers==4.57.1" \
  "tokenizers<0.23" \
  "huggingface-hub>=0.34.0,<1.0" \
  "setuptools==80.9.0" \
  wheel
```



遗留：

```Plain Text
异常是这一行：
diffusers 0.38.0 requires safetensors>=0.8.0-rc.0, but you have safetensors 0.7.0 which is incompatible.
这不是致命错误，pip 已经安装完成了；但为了避免后面 import diffusers 或 Motus 代码时出问题，建议现在补回 safetensors>=0.8.0rc0。不要再动 transformers。
执行：
python -m pip install --upgrade "safetensors>=0.8.0rc0"
```

后面报错再修



检查

```Java
cd /home/ubuntu/workspace/RoboTwin/policy/Motus
conda activate RoboTwin

python - <<'PY'
import sys
print("python =", sys.executable)

import torch
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count =", torch.cuda.device_count())
    print("gpu 0 =", torch.cuda.get_device_name(0))

import numpy
print("numpy =", numpy.__version__)

import transformers, tokenizers, huggingface_hub
print("transformers =", transformers.__version__)
print("tokenizers =", tokenizers.__version__)
print("huggingface_hub =", huggingface_hub.__version__)

from transformers import Qwen3VLForConditionalGeneration
print("Qwen3VL import ok")

import flash_attn
print("flash_attn ok =", flash_attn.__file__)

import sapien
print("sapien ok =", sapien.__file__)

import toppra
print("toppra ok =", toppra.__file__)

import mplib
print("mplib ok =", mplib.__file__)
PY
```

gpt：还行



目录和网络变量

```Bash
conda activate RoboTwin

export MODEL_ROOT="/home/ubuntu/models/motus"
mkdir -p "$MODEL_ROOT"
cd "$MODEL_ROOT"

env | grep -i proxy || true
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

export HF_ENDPOINT="https://hf-mirror.com"
```



#### 下模型

下载

```Python
hf download motus-robotics/Motus_robotwin2 \
  --local-dir "$MODEL_ROOT/Motus_robotwin2"

hf download motus-robotics/Motus_Wan2_2_5B_pretrain \
  --local-dir "$MODEL_ROOT/Motus_Wan2_2_5B_pretrain"

hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir "$MODEL_ROOT/Qwen3-VL-2B-Instruct"

hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir "$MODEL_ROOT/Wan2.2-TI2V-5B"
```

成功开始

vscode不要最小化，会断

gpt建议：后续大下载最好用 `tmux`

下t5的时候有点慢，一下几百kb，然后2MB

gpt：经检查，下载完



#### 改

配置

```YAML
# Motus Evaluation Path Configuration
# Paths needed for inference

# ============================================================================
# Core Paths - MUST MODIFY THESE
# ============================================================================
robotwin_root: "/home/ubuntu/workspace/RoboTwin"     # e.g., "/path/to/RoboTwin"
conda_env: "/home/ubuntu/miniconda3/envs/RoboTwin"         # e.g., "/path/to/miniconda3/envs/RoboTwin"
checkpoint_path: "/home/ubuntu/models/motus/Motus_robotwin2"   # (directory containing mp_rank_00_model_states.pt)

# Pretrained model paths (only for loading configs, not weights)
wan_path: "/home/ubuntu/models/motus/Wan2.2-TI2V-5B"          # e.g., "/path/to/Wan2.2-TI2V-5B"
vlm_path: "/home/ubuntu/models/motus/Qwen3-VL-2B-Instruct"    # e.g., "/path/to/Qwen3-VL-2B-Instruct"

# ============================================================================
# Optional Configuration
# ============================================================================
# GPU IDs (empty means auto-detect)
gpu_ids: [0]  # e.g., [0, 1, 2, 3, 4, 5, 6, 7]

# Task configuration
task_config: "demo_randomized"
seed: 42
tasks_file: "tasks_all.txt"
```



改eval\.sh

```Bash
#!/bin/bash
# Single task evaluation script for Motus policy on RoboTwin platform

# ============================================================================
# Single Task Configuration - MODIFY THESE
# ============================================================================

# 修改，使其能传入指定任务的参数
# TASK_NAME="click_alarmclock"  # Change this to the task you want to test
TASK_NAME="${1:-click_alarmclock}"

GPU_ID=0                       # GPU to use

# ============================================================================
# Script starts here
# ============================================================================
echo "Starting single task evaluation at $(date)"

# Get script directory (policy/Motus/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"

# ============================================================================
# Load Configuration from paths_config.yml
# ============================================================================
CONFIG_FILE="${POLICY_DIR}/paths_config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please create paths_config.yml with required paths."
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"

# Parse YAML (improved - remove comments and extra whitespace)
ROBOTWIN_ROOT=$(grep "^robotwin_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CONDA_ENV=$(grep "^conda_env:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CHECKPOINT_PATH=$(grep "^checkpoint_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
WAN_PATH=$(grep "^wan_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
VLM_PATH=$(grep "^vlm_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Optional configurations
TASK_CONFIG=$(grep "^task_config:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
SEED=$(grep "^seed:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Default values
TASK_CONFIG=${TASK_CONFIG:-"demo_randomized"}
SEED=${SEED:-"42"}
POLICY_NAME="Motus"

# ============================================================================
# Validation
# ============================================================================
if [ -z "$ROBOTWIN_ROOT" ]; then
    echo "Error: robotwin_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CONDA_ENV" ]; then
    echo "Error: conda_env is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: checkpoint_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$WAN_PATH" ]; then
    echo "Error: wan_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$VLM_PATH" ]; then
    echo "Error: vlm_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$ROBOTWIN_ROOT" ]; then
    echo "Error: RoboTwin root not found: $ROBOTWIN_ROOT"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$WAN_PATH" ]; then
    echo "Error: WAN path not found: $WAN_PATH"
    exit 1
fi

if [ ! -d "$VLM_PATH" ]; then
    echo "Error: VLM path not found: $VLM_PATH"
    exit 1
fi

cd "$ROBOTWIN_ROOT" || exit 1

# 处理conda报错
#######

# # Activate conda
# if ! command -v conda &> /dev/null; then
#     echo "Error: conda not found."
#     exit 1
# fi

# eval "$(conda shell.bash hook)"
# conda activate "$CONDA_ENV"

# Activate conda
# export PATH="/home/sumita-mana/anaconda3/bin:$PATH"
# source /home/sumita-mana/anaconda3/etc/profile.d/conda.sh

export PATH="/home/ubuntu/miniconda3/bin:$PATH"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found."
    exit 1
fi

conda activate "$CONDA_ENV"

#######

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

# Set environment
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create logs directory
LOG_DIR="${POLICY_DIR}/logs_single_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ckpt_setting="${CHECKPOINT_PATH}"
log_file="${LOG_DIR}/${TASK_NAME}.log"

echo ""
echo "================================================================"
echo "Single Task Evaluation Configuration"
echo "================================================================"
echo "Task Name:         $TASK_NAME"
echo "GPU:               $GPU_ID"
echo "----------------------------------------------------------------"
echo "RoboTwin Root:     $ROBOTWIN_ROOT"
echo "Policy Dir:        $POLICY_DIR"
echo "Checkpoint:        $CHECKPOINT_PATH"
echo "WAN Path:          $WAN_PATH"
echo "VLM Path:          $VLM_PATH"
echo "Task Config:       $TASK_CONFIG"
echo "Seed:              $SEED"
echo "Log File:          $log_file"
echo "================================================================"
echo ""

# Run evaluation with WAN_PATH passed as argument
echo "Starting evaluation..."

# 处理环境进入问题
#############

# PYTHONWARNINGS=ignore::UserWarning \
# python script/eval_policy.py \
#     --config "policy/${POLICY_NAME}/deploy_policy.yml" \
#     --overrides \
#     --task_name "${TASK_NAME}" \
#     --task_config "${TASK_CONFIG}" \
#     --ckpt_setting "${ckpt_setting}" \
#     --seed "${SEED}" \
#     --policy_name "${POLICY_NAME}" \
#     --log_dir "${LOG_DIR}" \
#     --wan_path "${WAN_PATH}" \
#     --vlm_path "${VLM_PATH}" \
#     2>&1 | tee "$log_file"

echo "Python executable: ${CONDA_ENV}/bin/python"
"${CONDA_ENV}/bin/python" - <<'PY'
import sys
print("sys.executable =", sys.executable)
import sapien
print("sapien ok =", sapien.__file__)
import importlib
m = importlib.import_module("sapien.core")
print("sapien.core ok =", m)
PY

PYTHONWARNINGS=ignore::UserWarning \
"${CONDA_ENV}/bin/python" script/eval_policy.py \
    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${SEED}" \
    --policy_name "${POLICY_NAME}" \
    --log_dir "${LOG_DIR}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}" \
    2>&1 | tee "$log_file"

#############

exit_code=${PIPESTATUS[0]}

echo ""
echo "================================================================"
if [ $exit_code -eq 0 ]; then
    echo "✅ Task $TASK_NAME completed successfully"
    echo "================================================================"
    exit 0
else
    echo "❌ Task $TASK_NAME failed with exit code $exit_code"
    echo "================================================================"
    echo "Log file: $log_file"
    exit 1
fi
```



检查脚本

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

echo "===== config ====="
cat paths_config.yml

echo "===== eval.sh key paths ====="
grep -nE "sumita|anaconda3|miniconda3|CONDA_ENV|bin/python|TASK_NAME|GPU_ID" eval.sh

echo "===== python import check ====="
/home/ubuntu/miniconda3/envs/RoboTwin/bin/python - <<'PY'
import sys
print("python =", sys.executable)

import torch
print("torch =", torch.__version__)
print("cuda =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu =", torch.cuda.get_device_name(0))

import transformers, numpy
print("transformers =", transformers.__version__)
print("numpy =", numpy.__version__)

from transformers import Qwen3VLForConditionalGeneration
print("Qwen3VL import ok")

import flash_attn
print("flash_attn ok")

import sapien
print("sapien ok =", sapien.__file__)
PY
```



#### 评估

```Plain Text
cd /home/ubuntu/workspace/RoboTwin/policy/Motus
bash eval.sh beat_block_hammer
```

跑通，显存占用39g



Bbh 89

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MzNhNDQzMDZjYTk5Y2JhZjdhYmI3MGNiZmI3YmUzNjdfZmRhZTA2MmE2N2RhMzVmZjBlODhmNDEyOWE1M2M4MDNfSUQ6NzY0MDQzODEzNjYxMDU1NzEwN18xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MTA4YWQ3OGE5ZmY4Mjg3ZTZhOGNlZjJhODU1MmJjYjJfYjZmNDY4ZWVmZjU4ZmNiNWQ0NDgxMzNmOTdkNTRjYWNfSUQ6NzY0MDQzODI2MTI4NjM0MTU3N18xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



Scan object，比较慢

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MzIxNjAwYTg0MGM4YzYwNzMzOTQ1YjUzMjA0MjdjN2ZfMzdmNDVmMjNlNzUxMzBhMGU4MzI4ZThiYmY5NTg3YTZfSUQ6NzY0MDQ1OTE3MDMzMjE1MDk2M18xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MWQwY2Y2ODNmNjQ5NTI3YmZmY2IxYzIxYzRhMTI2MmNfNzFlNjhiZmM4NzE1ZThkZGM3ZWI0OGQ2MmMyMmI4NzRfSUQ6NzY0MDQzODMzMzYzNDY0NDk1OF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



### autodl跑通motus



#### guide

chat

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a17a696\-2150\-832b\-9464\-d707b4add11a



迁移规则

```Bash
同区/同地区：可以比较方便迁移。
 推荐用 AutoDL 的「克隆实例」或者「跨实例拷贝数据」。同地区迁移时，AutoDL 官方说明是系统盘和数据盘都可以迁移；克隆实例会以当前系统盘为模板创建新实例，如果需要数据盘，要勾选“同时拷贝数据盘”。
 关机原实例。 
 在原实例「更多操作」里选择「克隆实例」。 
 创建新实例时选择同地区另一台有空卡的主机。 
 如果要带走 /root/autodl-tmp，必须勾选数据盘拷贝。
 
跨区/跨地区：不能直接像同区那样快速克隆数据盘。
 系统盘建议先保存成私有镜像，再在新地区用“我的镜像”创建实例；数据盘则走公网传输，官方说一般约 10MB/s 左右，小文件很多会慢数倍。可用 scp，或者更推荐先打包再通过 ssh | tar 传输，或者借助公网网盘。
 官方给的跨地区方法是从源实例向目标实例 scp。例如目标实例 SSH 是：
ssh -p 66666 root@region-3.autodl.com
那么在源实例执行：
scp -rP 66666 /root/autodl-tmp/xxx root@region-3.autodl.com:/root/autodl-tmp/
注意是大写 -P 指定端口。官方也提醒跨地区走公网，一般约 10MB/s，小文件多会慢很多。
对你这种 RoboTwin/Motus 环境，我更推荐打包流式传输：
cd /root/autodl-tmp
tar cf - workspace models datasets checkpoints | \
ssh -p 66666 root@region-3.autodl.com "cd /root/autodl-tmp && tar xf -"
如果要迁移整个数据盘：
cd /root/autodl-tmp
tar cf - * | \
ssh -p 66666 root@region-3.autodl.com "cd /root/autodl-tmp && tar xf -"
这个比直接 scp -r 大量小文件稳定很多。AutoDL 官方也给了类似的 tar | ssh | tar 方法
```



注意：

下模型前显式指向tmp

下hf前排查隐性占用，例如

```Bash
export HF_HOME=/root/autodl-tmp/hf_home
mkdir -p "$HF_HOME"
```

其他下载前也排查缓存占系统盘的问题

监视网页存储占用



这里是从头开始装



#### 基础环境

检查环境

```Bash
# =========================
# CONFIG
# =========================
WORK_ROOT="/root/autodl-tmp"
CONDA_PKGS_DIR="${WORK_ROOT}/conda/pkgs"
CONDA_ENVS_DIR="${WORK_ROOT}/conda/envs"
LOG_ROOT="${WORK_ROOT}/setup_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/stage1_check_${STAMP}.log"

# =========================
# 1. Prepare dirs
# =========================
mkdir -p "${WORK_ROOT}" "${CONDA_PKGS_DIR}" "${CONDA_ENVS_DIR}" "${LOG_ROOT}"

# =========================
# 2. Full environment check
# =========================
{
echo "===== TIME ====="
date

echo
echo "===== OS ====="
cat /etc/os-release || true
uname -a || true

echo
echo "===== GPU / Driver ====="
nvidia-smi || true
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"

echo
echo "===== Disk ====="
df -h
echo
du -sh /root 2>/dev/null || true
du -sh /root/autodl-tmp 2>/dev/null || true
du -sh /root/autodl-fs 2>/dev/null || true
du -sh /tmp 2>/dev/null || true

echo
echo "===== Python / Conda ====="
which python || true
python --version || true
which python3 || true
python3 --version || true
which conda || true
conda --version || true
echo "CONDA_EXE=${CONDA_EXE}"

echo
echo "===== Build / Runtime Tools ====="
which git || true
git --version || true
which curl || true
curl --version | head -n 1 || true
which wget || true
wget --version | head -n 1 || true
which unzip || true
unzip -v | head -n 1 || true
which ffmpeg || true
ffmpeg -version | head -n 1 || true

echo
echo "===== CUDA / Compiler ====="
which nvcc || true
nvcc --version || true

echo
echo "===== Vulkan ====="
which vulkaninfo || true
vulkaninfo --summary | sed -n '1,160p' || true
echo
echo "--- /etc/vulkan/icd.d ---"
ls -lah /etc/vulkan/icd.d/ 2>/dev/null || true
echo
echo "--- /usr/share/vulkan/icd.d ---"
ls -lah /usr/share/vulkan/icd.d/ 2>/dev/null || true
echo
echo "--- NVIDIA libs ---"
ldconfig -p | grep -E 'libGLX_nvidia|libEGL_nvidia|libvulkan' || true

echo
echo "===== Proxy / Network ====="
env | grep -i proxy || true
echo
echo "--- github probe ---"
curl -I -L https://github.com 2>/dev/null | head -n 5 || true
echo
echo "--- huggingface probe ---"
curl -I -L https://huggingface.co 2>/dev/null | head -n 5 || true

} 2>&1 | tee "${LOG_FILE}"

echo
echo "LOG_FILE=${LOG_FILE}"

# =========================
# CONFIG
# =========================
CHECK_DIRS="/root/autodl-tmp /root/autodl-fs /autodl-tmp /autodl-fs /autodl-pub /autodl-pub/data"

# =========================
# Check mounts
# =========================
echo "===== pwd / root listing ====="
pwd
ls -lah /root | sed -n '1,120p'

echo
echo "===== mount check ====="
for p in ${CHECK_DIRS}; do
  echo
  echo "----- ${p} -----"
  ls -ld "${p}" 2>/dev/null || true
  echo "realpath: $(readlink -f "${p}" 2>/dev/null || true)"
  df -h "${p}" 2>/dev/null || true
  findmnt -T "${p}" 2>/dev/null || true
  stat -c 'mount=%m type=%F' "${p}" 2>/dev/null || true
done

# =========================
# CONFIG
# =========================
CURL_TIMEOUT=15

# =========================
# Clear proxy first
# =========================
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

echo "===== proxy after unset ====="
env | grep -i proxy || true

echo
echo "===== direct github ====="
timeout ${CURL_TIMEOUT} curl -I -L https://github.com 2>/dev/null | head -n 8 || echo "github direct timeout/fail"

echo
echo "===== direct huggingface ====="
timeout ${CURL_TIMEOUT} curl -I -L https://huggingface.co 2>/dev/null | head -n 8 || echo "huggingface direct timeout/fail"

echo
echo "===== network_turbo test ====="
if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
  env | grep -i proxy || true

  echo
  echo "===== turbo github ====="
  timeout ${CURL_TIMEOUT} curl -I -L https://github.com 2>/dev/null | head -n 8 || echo "github turbo timeout/fail"

  echo
  echo "===== turbo huggingface ====="
  timeout ${CURL_TIMEOUT} curl -I -L https://huggingface.co 2>/dev/null | head -n 8 || echo "huggingface turbo timeout/fail"
else
  echo "/etc/network_turbo not found"
fi

# =========================
# CUDA toolkit check
# =========================
echo "===== CUDA env ====="
echo "CUDA_HOME=${CUDA_HOME}"
echo "CUDA_PATH=${CUDA_PATH}"
which nvcc || true
nvcc --version || true

echo
echo "===== /usr/local ====="
ls -lah /usr/local | sed -n '1,120p'

echo
echo "===== possible nvcc paths ====="
for x in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
  if [ -e "$x" ]; then
    echo "$x"
    "$x" --version
  fi
done

```



装，配路径，检查

```Markdown
# =========================
# CONFIG
# =========================
APT_PACKAGES="git git-lfs curl wget unzip ca-certificates build-essential python3-pip python3-venv python-is-python3 ffmpeg libvulkan1 mesa-vulkan-drivers vulkan-tools libegl1 libglew2.2 libglew-dev libglfw3 libgl1-mesa-glx libosmesa6"

# =========================
# Install base packages
# =========================
apt-get update
apt-get install -y ${APT_PACKAGES}

# =========================
# Verify tools
# =========================
echo "===== tools ====="
git --version
git lfs version
ffmpeg -version | head -n 1
unzip -v | head -n 1

echo
echo "===== vulkaninfo exists ====="
which vulkaninfo
vulkaninfo --summary | sed -n '1,160p' || true

# =========================
# CONFIG
# =========================
WORK_ROOT="/root/autodl-tmp"
CONDA_PKGS_DIR="${WORK_ROOT}/conda/pkgs"
CONDA_ENVS_DIR="${WORK_ROOT}/conda/envs"

# =========================
# Configure conda dirs and mirrors
# =========================
mkdir -p "${CONDA_PKGS_DIR}" "${CONDA_ENVS_DIR}"

cat > /root/.condarc <<EOF
channels:
  - defaults
show_channel_urls: true

default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

pkgs_dirs:
  - ${CONDA_PKGS_DIR}

envs_dirs:
  - ${CONDA_ENVS_DIR}
EOF

# =========================
# Verify
# =========================
cat /root/.condarc
conda info | sed -n '1,120p'

echo "===== default vulkan ====="
vulkaninfo --summary | sed -n '1,160p' || true

echo
echo "===== nvidia icd file ====="
cat /etc/vulkan/icd.d/nvidia_icd.json || true

echo "===== disk ====="
df -h / /root/autodl-tmp

echo
echo "===== conda ====="
conda info | sed -n '1,120p'

echo
echo "===== cuda ====="
which nvcc
nvcc --version

echo
echo "===== vulkan ====="
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES}"
echo "VK_DRIVER_FILES=${VK_DRIVER_FILES}"
vulkaninfo --summary | sed -n '1,160p'

echo
echo "===== network turbo status ====="
env | grep -i proxy || true
```



关学术加速，开hf镜像

```Bash
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
```



#### 装rt2

创建环境，clone

```Bash
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin

which python
python --version
conda info --envs

cd /root/autodl-tmp
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd /root/autodl-tmp/RoboTwin

cd /root/autodl-tmp/RoboTwin
conda activate RoboTwin
bash script/_install.sh
```

问题

```Bash
CondaError: Run 'conda init' before 'conda activate'
所以当时 conda activate RoboTwin 没成功
但紧接着 GitHub clone 失败了，RoboTwin 目录不存在
```

conda加载

```Markdown
# =========================
# CONFIG
# =========================
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="RoboTwin"

# =========================
# Activate conda properly
# =========================
source "${CONDA_SH}"
conda activate "${ENV_NAME}"

which python
python --version
echo "CONDA_PREFIX=${CONDA_PREFIX}"
```

开加速clone

```Markdown
# =========================
# CONFIG
# =========================
WORK_ROOT="/root/autodl-tmp"
REPO_URL="https://github.com/RoboTwin-Platform/RoboTwin.git"
REPO_DIR="${WORK_ROOT}/RoboTwin"

# =========================
# Clean failed clone
# =========================
cd "${WORK_ROOT}"
rm -rf "${REPO_DIR}"

# =========================
# Temporary network turbo for GitHub clone
# =========================
source /etc/network_turbo
git clone "${REPO_URL}" "${REPO_DIR}"

# =========================
# Clear proxy after clone
# =========================
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

# =========================
# Verify
# =========================
cd "${REPO_DIR}"
pwd
ls -lah | sed -n '1,80p'
```



进环境，检查

```Bash
cd /root/autodl-tmp/RoboTwin

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

echo "===== python ====="
which python
python --version
echo "CONDA_PREFIX=${CONDA_PREFIX}"

echo "===== proxy ====="
env | grep -i proxy || true

echo "===== disk ====="
df -h / /root/autodl-tmp
```



问题：装的中间cache可能挤爆系统盘

```Bash
目前哪些还可能落到系统盘
主要有四类。
1. pip cache
默认可能在：
/root/.cache/pip
这个在系统盘。bash script/_install.sh 里面如果调用 pip 下载大 wheel，cache 可能进这里。
2. Hugging Face cache
默认可能在：
/root/.cache/huggingface
这个也在系统盘。后面下载 RoboTwin assets 或 Motus 模型时尤其要注意。
3. pip build 临时目录
默认可能在：
/tmp
你的 /tmp 大概率属于 30G 系统盘。编译扩展包时，例如 flash-attn、pytorch3d、curobo，临时文件可能很大。
4. torch extension cache
默认可能在：
/root/.cache/torch_extensions
后面编译 CUDA extension 时可能用到。
所以你的担心是合理的。**现在在跑 bash script/_install.sh 之前，建议先做 cache 重定向。**
```

cache也进tmp

```Bash
# =========================
# CONFIG
# =========================
WORK_ROOT="/root/autodl-tmp"
CACHE_ROOT="${WORK_ROOT}/cache"
TMP_ROOT="${WORK_ROOT}/tmp"
ROBOTWIN_ENV="/root/autodl-tmp/conda/envs/RoboTwin"

# =========================
# 1. Create cache dirs on data disk
# =========================
mkdir -p \
  "${CACHE_ROOT}/root_cache" \
  "${CACHE_ROOT}/pip" \
  "${CACHE_ROOT}/huggingface" \
  "${CACHE_ROOT}/torch_extensions" \
  "${CACHE_ROOT}/cuda" \
  "${TMP_ROOT}"

# =========================
# 2. Move existing /root/.cache to data disk, then symlink
# =========================
if [ -e /root/.cache ] && [ ! -L /root/.cache ]; then
  cp -a /root/.cache/. "${CACHE_ROOT}/root_cache/" 2>/dev/null || true
  rm -rf /root/.cache
fi

ln -sfn "${CACHE_ROOT}/root_cache" /root/.cache

# =========================
# 3. Current shell cache env
# =========================
export XDG_CACHE_HOME="${CACHE_ROOT}/root_cache"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
export TMPDIR="${TMP_ROOT}"

# Keep your current HF mirror
export HF_ENDPOINT=https://hf-mirror.com

# =========================
# 4. Persist for RoboTwin conda env activation
# =========================
mkdir -p "${ROBOTWIN_ENV}/etc/conda/activate.d"

cat > "${ROBOTWIN_ENV}/etc/conda/activate.d/cache_dirs.sh" <<EOF
export XDG_CACHE_HOME="${CACHE_ROOT}/root_cache"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
export TMPDIR="${TMP_ROOT}"
export HF_ENDPOINT=https://hf-mirror.com
EOF

# =========================
# 5. Verify
# =========================
echo "===== cache env ====="
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "CUDA_CACHE_PATH=${CUDA_CACHE_PATH}"
echo "TMPDIR=${TMPDIR}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"

echo
echo "===== cache paths ====="
ls -ld /root/.cache "${CACHE_ROOT}" "${TMP_ROOT}"
df -h / /root/autodl-tmp
```

问题：

```Bash
cat > .../cache_dirs.sh <<EOF 附近的显示有些粘贴错乱，所以我建议先检查一下 activate.d 文件是否完整。
```

检查

```Bash
# =========================
# CONFIG
# =========================
ROBOTWIN_ENV="/root/autodl-tmp/conda/envs/RoboTwin"
CACHE_FILE="${ROBOTWIN_ENV}/etc/conda/activate.d/cache_dirs.sh"

# =========================
# Check cache activate file
# =========================
echo "===== cache file ====="
cat "${CACHE_FILE}"

echo
echo "===== current env ====="
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "CUDA_CACHE_PATH=${CUDA_CACHE_PATH}"
echo "TMPDIR=${TMPDIR}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"

# =========================
# CONFIG
# =========================
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="RoboTwin"

# =========================
# Reactivate env
# =========================
source "${CONDA_SH}"
conda deactivate 2>/dev/null || true
conda activate "${ENV_NAME}"

echo "===== python ====="
which python
python --version
echo "CONDA_PREFIX=${CONDA_PREFIX}"

echo
echo "===== cache ====="
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "TMPDIR=${TMPDIR}"

echo
echo "===== proxy ====="
env | grep -i proxy || true
```



装包

```Bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /root/autodl-tmp/RoboTwin

which python
python --version
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "TMPDIR=$TMPDIR"

bash script/_install.sh
```

问题：慢，几百kb

换源，清华

```Bash
rm -rf /root/autodl-tmp/tmp/pip-*

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /root/autodl-tmp/RoboTwin

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

echo "Python: $(which python)"
python --version
echo "PIP_INDEX_URL=$PIP_INDEX_URL"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "TMPDIR=$TMPDIR"

bash script/_install.sh
```

清华源快



新终端，固定清华源到环境

```Bash
cat > /root/autodl-tmp/conda/envs/RoboTwin/etc/conda/activate.d/pip_mirror.sh <<'EOF'
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
EOF

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

echo "PIP_INDEX_URL=$PIP_INDEX_URL"
echo "PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST"
python -m pip config list
```



rt2环境装好；

pytorch3d 失败，先不管



#### 下rt2 asset

检查

```Python
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /root/autodl-tmp/RoboTwin

python - <<'PY'
import sys
print("python =", sys.executable)

import torch
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu =", torch.cuda.get_device_name(0))

import torchvision
print("torchvision =", torchvision.__version__)

import numpy
print("numpy =", numpy.__version__)

import sapien
print("sapien ok =", sapien.__file__)

import mplib
print("mplib ok =", mplib.__file__)

import toppra
print("toppra ok =", toppra.__file__)

import open3d
print("open3d ok =", open3d.__version__)

try:
    import curobo
    print("curobo ok =", curobo.__file__)
except Exception as e:
    print("curobo import failed:", repr(e))

try:
    import pytorch3d
    print("pytorch3d ok")
except Exception as e:
    print("pytorch3d optional missing:", repr(e))
PY

python - <<'PY'
import os
import sapien

print("VK_ICD_FILENAMES =", os.environ.get("VK_ICD_FILENAMES"))
print("VK_DRIVER_FILES =", os.environ.get("VK_DRIVER_FILES"))

scene = sapien.Scene()
print("sapien scene ok")
PY
```



下

```Bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /root/autodl-tmp/RoboTwin

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600

bash script/_download_assets.sh
```

正常速度



#### act采训推



检查

```Bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /root/autodl-tmp/RoboTwin

echo "===== disk ====="
df -h / /root/autodl-tmp

echo
echo "===== assets size ====="
du -sh assets 2>/dev/null || true
du -sh assets/objects 2>/dev/null || true
du -sh assets/embodiments 2>/dev/null || true

echo
echo "===== key assets ====="
ls -lah assets | sed -n '1,80p'
ls -lah assets/objects | sed -n '1,80p'
ls -lah assets/embodiments | sed -n '1,80p'
```



采

```Bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /root/autodl-tmp/RoboTwin

bash collect_data.sh beat_block_hammer demo_clean 0
```

保存有点慢；可能是a800渲染慢的问题；实际上还好，等一下就行



装

```Bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /root/autodl-tmp/RoboTwin/policy/ACT

pip install pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 dm_control==1.0.14 opencv-python matplotlib einops packaging h5py ipython

cd detr
pip install -e .
cd ..
```



检查依赖

```Bash
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
unset DISPLAY

python - <<'PY'
from dm_control import suite
env = suite.load(domain_name="cartpole", task_name="swingup")
print("dm_control ok")
print(env.action_spec())
PY

mkdir -p /root/autodl-tmp/conda/envs/RoboTwin/etc/conda/activate.d

cat > /root/autodl-tmp/conda/envs/RoboTwin/etc/conda/activate.d/robotwin_act_env.sh <<'EOF'
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
unset DISPLAY
EOF
```



处理

```Bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /root/autodl-tmp/RoboTwin/policy/ACT

bash process_data.sh beat_block_hammer demo_clean 50
```



训

```Bash
bash train.sh beat_block_hammer demo_clean 50 0 0
```

要下resnet，很快

有点慢，比4090慢，两个半小时左右

训完



推

```Bash
bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
```

评估和之前一样比较慢；能推完一个任务产出视频



#### motus模型下载

act跑的同时



检查

```Bash
# =========================
# CONFIG
# =========================
MODEL_ROOT="/root/autodl-tmp/models/motus"
HF_CACHE_ROOT="/root/autodl-tmp/cache/huggingface"
TMP_ROOT="/root/autodl-tmp/tmp"

# =========================
# Activate env
# =========================
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

# =========================
# Paths and cache
# =========================
mkdir -p "$MODEL_ROOT" "$HF_CACHE_ROOT" "$TMP_ROOT"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$HF_CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
export TMPDIR="$TMP_ROOT"

# 不默认使用 AutoDL 学术加速
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

# =========================
# Check
# =========================
echo "python = $(which python)"
python --version
echo "MODEL_ROOT=$MODEL_ROOT"
echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "TMPDIR=$TMPDIR"
echo "HF_ENDPOINT=$HF_ENDPOINT"
env | grep -i proxy || true
df -h / /root/autodl-tmp

command -v hf || command -v huggingface-cli || true


```



以后要下载的新终端要做：

```Bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

export MODEL_ROOT="/root/autodl-tmp/models/motus"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/root/autodl-tmp/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/cache/huggingface/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
export TMPDIR="/root/autodl-tmp/tmp"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
```



下载

```Python
MODEL_ROOT="/root/autodl-tmp/models/motus"

huggingface-cli download motus-robotics/Motus_robotwin2 \
  --local-dir "$MODEL_ROOT/Motus_robotwin2"
  
MODEL_ROOT="/root/autodl-tmp/models/motus"

huggingface-cli download motus-robotics/Motus_Wan2_2_5B_pretrain \
  --local-dir "$MODEL_ROOT/Motus_Wan2_2_5B_pretrain"

MODEL_ROOT="/root/autodl-tmp/models/motus"

huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir "$MODEL_ROOT/Qwen3-VL-2B-Instruct"

MODEL_ROOT="/root/autodl-tmp/models/motus"

huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir "$MODEL_ROOT/Wan2.2-TI2V-5B"
```

速度正常

下载完



#### motus代码配置



clone

```Bash
cd /root/autodl-tmp
git clone https://github.com/thu-ml/Motus.git
```



复制代码到rt2目录下

```Bash
ROBOTWIN_ROOT="/root/autodl-tmp/RoboTwin"
MOTUS_REPO="/root/autodl-tmp/Motus"
TARGET="$ROBOTWIN_ROOT/policy/Motus"

cd "$ROBOTWIN_ROOT/policy"

if [ -d "$TARGET" ]; then
  mv "$TARGET" "${TARGET}_bak_$(date +%Y%m%d_%H%M%S)"
fi

cp -r "$MOTUS_REPO/inference/robotwin/Motus" "$TARGET"

ls -lah "$TARGET" | sed -n '1,120p'
```



改配置 /root/autodl\-tmp/RoboTwin/policy/Motus/paths\_config\.yml

```Bash
# Motus Evaluation Path Configuration
# Paths needed for inference

# ============================================================================
# Core Paths - MUST MODIFY THESE
# ============================================================================
robotwin_root: "/root/autodl-tmp/RoboTwin"     # e.g., "/path/to/RoboTwin"
conda_env: "/root/autodl-tmp/conda/envs/RoboTwin"         # e.g., "/path/to/miniconda3/envs/RoboTwin"
checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"   # (directory containing mp_rank_00_model_states.pt)

# Pretrained model paths (only for loading configs, not weights)
wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"          # e.g., "/path/to/Wan2.2-TI2V-5B"
vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"    # e.g., "/path/to/Qwen3-VL-2B-Instruct"

# ============================================================================
# Optional Configuration
# ============================================================================
# GPU IDs (empty means auto-detect)
gpu_ids: [0]  # e.g., [0, 1, 2, 3, 4, 5, 6, 7]

# Task configuration
task_config: "demo_randomized"
seed: 42
tasks_file: "tasks_all.txt"
```



#### motus环境

备份环境

检查

```Bash
source /root/miniconda3/etc/profile.d/conda.sh

echo "===== envs dirs ====="
conda info | grep -A3 "envs directories"

echo
echo "===== disk ====="
df -h / /root/autodl-tmp

echo
echo "===== current envs ====="
conda env list
```

执行克隆备份

```Bash
source /root/miniconda3/etc/profile.d/conda.sh

conda create -n RoboTwin_Backup --clone RoboTwin -y

echo
echo "===== envs ====="
conda env list | grep RoboTwin

echo
echo "===== size ====="
du -sh /root/autodl-tmp/conda/envs/RoboTwin
du -sh /root/autodl-tmp/conda/envs/RoboTwin_Backup

echo
echo "===== disk ====="
df -h / /root/autodl-tmp
```



检查

```Bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

echo "python = $(which python)"
python --version
echo "CONDA_PREFIX=$CONDA_PREFIX"

df -h / /root/autodl-tmp
conda env list | grep RoboTwin
```



改eval\.sh硬编码路径

检查

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

grep -nE "home/ubuntu|sumita|anaconda3|miniconda3|autodl-tmp|CONDA_ENV|bin/python|TASK_NAME|GPU_ID|conda.sh" eval.sh
```

备份sh

\[eval\.sh\]





装

```Bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /root/autodl-tmp/RoboTwin/policy/Motus

python -m pip install -U pip wheel ninja packaging psutil
python -m pip install --force-reinstall "setuptools==80.9.0"

cd /root/autodl-tmp/RoboTwin/policy/Motus

export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export MAX_JOBS=8

which nvcc
nvcc --version

python -m pip install -v flash-attn --no-build-isolation

cd /root/autodl-tmp/RoboTwin/policy/Motus

python -m pip install -r requirements.txt

cd /root/autodl-tmp/RoboTwin/policy/Motus

python -m pip install --force-reinstall \
  "numpy<2" \
  "transformers==4.57.1" \
  "tokenizers<0.23" \
  "huggingface-hub>=0.34.0,<1.0" \
  "setuptools==80.9.0" \
  wheel

python -m pip install --upgrade "safetensors>=0.8.0rc0"
```

下编译的东西卡了下；总体上很快完成了



潜在问题：

```Bash
当前组合是：
torch = 2.4.1+cu121
nvcc = 12.8
```

出问题再说



TODO



检查

```Java
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /root/autodl-tmp/RoboTwin/policy/Motus

python - <<'PY'
import sys
print("python =", sys.executable)

import torch
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count =", torch.cuda.device_count())
    print("gpu 0 =", torch.cuda.get_device_name(0))

import numpy
print("numpy =", numpy.__version__)

import transformers, tokenizers, huggingface_hub
print("transformers =", transformers.__version__)
print("tokenizers =", tokenizers.__version__)
print("huggingface_hub =", huggingface_hub.__version__)

from transformers import Qwen3VLForConditionalGeneration
print("Qwen3VL import ok")

import flash_attn
print("flash_attn ok =", flash_attn.__file__)

import diffusers
print("diffusers =", diffusers.__version__)

import safetensors
print("safetensors =", safetensors.__version__)

import sapien
print("sapien ok =", sapien.__file__)

import toppra
print("toppra ok =", toppra.__file__)

import mplib
print("mplib ok =", mplib.__file__)
PY

python -m pip check

cd /root/autodl-tmp/RoboTwin/policy/Motus

echo "===== paths_config.yml ====="
cat paths_config.yml

echo
echo "===== eval.sh key paths ====="
grep -nE "home/ubuntu|sumita|anaconda3|/root/miniconda3|CONDA_ENV|bin/python|TASK_NAME|GPU_ID|conda.sh" eval.sh

echo
echo "===== model key files ====="
ls -lh /root/autodl-tmp/models/motus/Motus_robotwin2/mp_rank_00_model_states.pt
ls -lh /root/autodl-tmp/models/motus/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
ls -lh /root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct/model.safetensors



```



#### motus评估



检查

```Bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /root/autodl-tmp/RoboTwin/policy/Motus

nvidia-smi

echo "===== key files ====="
ls -lh deploy_policy.yml
ls -lh paths_config.yml
ls -lh /root/autodl-tmp/RoboTwin/script/eval_policy.py
ls -lh /root/autodl-tmp/models/motus/Motus_robotwin2/mp_rank_00_model_states.pt
ls -lh /root/autodl-tmp/models/motus/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
ls -lh /root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct/model.safetensors
```



最短跑通

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus
bash eval.sh beat_block_hammer --gpu-id 0 --no-tts
```



### Test Time Scaling



#### v1

##### guide

先最简单方案tts

对每个要执行的action chunk

先采样8个action chunk

算两两间的L2距离

算每个chunk和其他7个的平均L2距离

取最小，执行



传参：（eval\.sh\-\>eval\_policy\.py\-\>deploy\_policy\.py）

指定有没有tts；

采样数



##### 实现

eval\.sh

```Bash
#!/bin/bash
# Single task evaluation script for Motus policy on RoboTwin platform

# ============================================================================
# Single Task Configuration - MODIFY THESE
# ============================================================================

# 修改，使其能传入指定任务的参数
# TASK_NAME="click_alarmclock"  # Change this to the task you want to test

# TASK_NAME="${1:-click_alarmclock}"

#tts add
TASK_NAME="${1:-click_alarmclock}"
if [ $# -gt 0 ]; then
    shift
fi

TTS_ENABLE=False
TTS_NUM_SAMPLES=8
TTS_LOG_ACTIONS=True
TTS_SAVE_FULL_ACTIONS=True

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tts)
            TTS_ENABLE=True
            shift
            ;;
        --no-tts)
            TTS_ENABLE=False
            shift
            ;;
        --tts-num-samples)
            TTS_NUM_SAMPLES="$2"
            shift 2
            ;;
        --tts-log-actions)
            TTS_LOG_ACTIONS="$2"
            shift 2
            ;;
        --tts-save-full-actions)
            TTS_SAVE_FULL_ACTIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

GPU_ID=0                       # GPU to use

# ============================================================================
# Script starts here
# ============================================================================
echo "Starting single task evaluation at $(date)"

# Get script directory (policy/Motus/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"

# ============================================================================
# Load Configuration from paths_config.yml
# ============================================================================
CONFIG_FILE="${POLICY_DIR}/paths_config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please create paths_config.yml with required paths."
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"

# Parse YAML (improved - remove comments and extra whitespace)
ROBOTWIN_ROOT=$(grep "^robotwin_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CONDA_ENV=$(grep "^conda_env:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CHECKPOINT_PATH=$(grep "^checkpoint_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
WAN_PATH=$(grep "^wan_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
VLM_PATH=$(grep "^vlm_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Optional configurations
TASK_CONFIG=$(grep "^task_config:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
SEED=$(grep "^seed:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Default values
TASK_CONFIG=${TASK_CONFIG:-"demo_randomized"}
SEED=${SEED:-"42"}
POLICY_NAME="Motus"

# ============================================================================
# Validation
# ============================================================================
if [ -z "$ROBOTWIN_ROOT" ]; then
    echo "Error: robotwin_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CONDA_ENV" ]; then
    echo "Error: conda_env is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: checkpoint_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$WAN_PATH" ]; then
    echo "Error: wan_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$VLM_PATH" ]; then
    echo "Error: vlm_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$ROBOTWIN_ROOT" ]; then
    echo "Error: RoboTwin root not found: $ROBOTWIN_ROOT"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$WAN_PATH" ]; then
    echo "Error: WAN path not found: $WAN_PATH"
    exit 1
fi

if [ ! -d "$VLM_PATH" ]; then
    echo "Error: VLM path not found: $VLM_PATH"
    exit 1
fi

cd "$ROBOTWIN_ROOT" || exit 1

# 处理conda报错
#######

# # Activate conda
# if ! command -v conda &> /dev/null; then
#     echo "Error: conda not found."
#     exit 1
# fi

# eval "$(conda shell.bash hook)"
# conda activate "$CONDA_ENV"

# Activate conda
# export PATH="/home/sumita-mana/anaconda3/bin:$PATH"
# source /home/sumita-mana/anaconda3/etc/profile.d/conda.sh

export PATH="/home/ubuntu/miniconda3/bin:$PATH"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found."
    exit 1
fi

conda activate "$CONDA_ENV"

#######

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

# Set environment
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create logs directory
LOG_DIR="${POLICY_DIR}/logs_single_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ckpt_setting="${CHECKPOINT_PATH}"
log_file="${LOG_DIR}/${TASK_NAME}.log"

echo ""
echo "================================================================"
echo "Single Task Evaluation Configuration"
echo "================================================================"
echo "Task Name:         $TASK_NAME"
echo "GPU:               $GPU_ID"
echo "----------------------------------------------------------------"
echo "RoboTwin Root:     $ROBOTWIN_ROOT"
echo "Policy Dir:        $POLICY_DIR"
echo "Checkpoint:        $CHECKPOINT_PATH"
echo "WAN Path:          $WAN_PATH"
echo "VLM Path:          $VLM_PATH"
echo "Task Config:       $TASK_CONFIG"
echo "Seed:              $SEED"
echo "Log File:          $log_file"
echo "TTS Enable:        $TTS_ENABLE" #tts add
echo "TTS Num Samples:   $TTS_NUM_SAMPLES" #tts add
echo "TTS Log Actions:   $TTS_LOG_ACTIONS" #tts add
echo "TTS Save Full:     $TTS_SAVE_FULL_ACTIONS" #tts add
echo "================================================================"
echo ""

# Run evaluation with WAN_PATH passed as argument
echo "Starting evaluation..."

# 处理环境进入问题
#############

# PYTHONWARNINGS=ignore::UserWarning \
# python script/eval_policy.py \
#     --config "policy/${POLICY_NAME}/deploy_policy.yml" \
#     --overrides \
#     --task_name "${TASK_NAME}" \
#     --task_config "${TASK_CONFIG}" \
#     --ckpt_setting "${ckpt_setting}" \
#     --seed "${SEED}" \
#     --policy_name "${POLICY_NAME}" \
#     --log_dir "${LOG_DIR}" \
#     --wan_path "${WAN_PATH}" \
#     --vlm_path "${VLM_PATH}" \
#     2>&1 | tee "$log_file"

echo "Python executable: ${CONDA_ENV}/bin/python"
"${CONDA_ENV}/bin/python" - <<'PY'
import sys
print("sys.executable =", sys.executable)
import sapien
print("sapien ok =", sapien.__file__)
import importlib
m = importlib.import_module("sapien.core")
print("sapien.core ok =", m)
PY

PYTHONWARNINGS=ignore::UserWarning \
"${CONDA_ENV}/bin/python" script/eval_policy.py \
    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${SEED}" \
    --policy_name "${POLICY_NAME}" \
    --log_dir "${LOG_DIR}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}" \
    --tts_enable "${TTS_ENABLE}" \
    --tts_num_samples "${TTS_NUM_SAMPLES}" \
    --tts_log_actions "${TTS_LOG_ACTIONS}" \
    --tts_save_full_actions "${TTS_SAVE_FULL_ACTIONS}" \
    2>&1 | tee "$log_file"

#############

exit_code=${PIPESTATUS[0]}

echo ""
echo "================================================================"
if [ $exit_code -eq 0 ]; then
    echo "✅ Task $TASK_NAME completed successfully"
    echo "================================================================"
    exit 0
else
    echo "❌ Task $TASK_NAME failed with exit code $exit_code"
    echo "================================================================"
    echo "Log file: $log_file"
    exit 1
fi
```

deploy\_policy\.py

```Python
# Motus Policy for RoboTwin

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add model paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

from models.motus import Motus, MotusConfig

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel
from utils.image_utils import resize_with_padding

logger = logging.getLogger(__name__)

# =========================
# Test-Time Scaling Defaults
# =========================
DEFAULT_TTS_ENABLE = False
DEFAULT_TTS_NUM_SAMPLES = 1
DEFAULT_TTS_LOG_ACTIONS = True
DEFAULT_TTS_SAVE_FULL_ACTIONS = True

# tts add
def _as_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in ["1", "true", "yes", "y", "on"]

class MotusPolicy:
    """
    Motus Policy wrapper for RoboTwin evaluation.
    Implements the joint video-action diffusion model for robotic control.
    """
    
    def __init__(
        self, 
        checkpoint_path: str, 
        config_path: str, 
        wan_path: str, 
        vlm_path: str, 
        device: str = "cuda", 
        log_dir: Optional[str] = None, 
        task_name: Optional[str] = None,
        tts_enable: bool = DEFAULT_TTS_ENABLE,
        tts_num_samples: int = DEFAULT_TTS_NUM_SAMPLES,
        tts_log_actions: bool = DEFAULT_TTS_LOG_ACTIONS,
        tts_save_full_actions: bool = DEFAULT_TTS_SAVE_FULL_ACTIONS,        
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
        
        # Initialize model WITHOUT loading pretrained backbones
        self.model = self._load_model()

        # Initialize T5 encoder for language embeddings (WAN text encoder)
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(self.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.wan_path, 'google', 'umt5-xxl'),
        )

        # Initialize VLM processor from vlm_path (for tokenization only, weights from checkpoint)
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)
        
        # Initialize observation cache
        self.obs_cache = deque(maxlen=1)
        self.action_cache = deque()
        
        # Model state
        self.current_state = None
        self.current_state_norm = None
        self.is_first_step = True
        self.prev_action = None

        # Load normalization stats
        self._load_normalization_stats()
        
        # Initialize image saving
        self.save_images = True
        
        # base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        # task_dir_name = task_name or os.environ.get('TASK_NAME') or "default_task"
        # self.save_dir = Path(base_log_dir) / "images" / task_dir_name
        # self.save_dir.mkdir(parents=True, exist_ok=True)
        # self.episode_count = 0
        # self.step_count = 0

        # logger.info("Motus Policy initialized successfully")
        
        
        # tts add
        
        base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        task_dir_name = task_name or os.environ.get('TASK_NAME') or "default_task"

        self.log_dir = Path(base_log_dir)
        self.task_dir_name = task_dir_name

        self.save_dir = self.log_dir / "images" / task_dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # self.tts_enable = bool(tts_enable)
        # self.tts_num_samples = max(1, int(tts_num_samples))
        # self.tts_log_actions = bool(tts_log_actions)
        # self.tts_save_full_actions = bool(tts_save_full_actions)
        
        self.tts_enable = _as_bool(tts_enable)
        self.tts_num_samples = max(1, int(tts_num_samples))
        self.tts_log_actions = _as_bool(tts_log_actions)
        self.tts_save_full_actions = _as_bool(tts_save_full_actions)
        
        self.tts_dir = self.log_dir / "tts" / task_dir_name

        if self.tts_enable and self.tts_save_full_actions:
            self.tts_dir.mkdir(parents=True, exist_ok=True)

        self.episode_count = 0
        self.step_count = 0

        print(
            f"[TTS] enable={self.tts_enable}, "
            f"num_samples={self.tts_num_samples}, "
            f"log_actions={self.tts_log_actions}, "
            f"save_full_actions={self.tts_save_full_actions}, "
            f"tts_dir={self.tts_dir}"
        )

        logger.info("Motus Policy initialized successfully")

    def set_instruction(self, instruction: str):
        """Set the current instruction for the policy."""
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def _load_model(self) -> Motus:
        """Load the Motus model without pretrained backbones, then load checkpoint."""
        logger.info(f"Initializing Motus model from config (no pretrained backbones)")

        config = self._create_model_config()
        
        # Initialize model from config WITHOUT loading pretrained weights
        model = Motus(config)
        model = model.to(self.device)
        
        # Load checkpoint weights
        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info("Model checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        model.eval()
        return model
    
    def _create_model_config(self) -> MotusConfig:
        """Create model configuration from yaml config - inference mode."""
        common = self.config_dict['common']
        model_cfg = self.config_dict['model']

        # Use paths passed to constructor
        vae_path = os.path.join(self.wan_path, "Wan2.2_VAE.pth")
        vlm_checkpoint_path = self.vlm_path

        hidden_size = model_cfg['action_expert']['hidden_size']
        ffn_multiplier = model_cfg['action_expert']['ffn_dim_multiplier']

        config = MotusConfig(
            # Paths for config loading only (no weights loaded)
            wan_checkpoint_path=self.wan_path,
            vae_path=vae_path,
            wan_config_path=self.wan_path,
            video_precision='bfloat16',
            vlm_checkpoint_path=vlm_checkpoint_path,
            
            # Understanding expert config
            und_expert_hidden_size=512,
            und_expert_ffn_dim_multiplier=4,
            und_expert_norm_eps=1e-5,
            und_layers_to_extract=None,
            vlm_adapter_input_dim=2048,
            vlm_adapter_projector_type="mlp3x_silu",
            
            # Model architecture
            num_layers=30,
            action_state_dim=common['state_dim'],
            action_dim=common['action_dim'],
            action_expert_dim=hidden_size,
            action_expert_ffn_dim_multiplier=ffn_multiplier,
            action_expert_norm_eps=1e-6,
            
            # Training config
            global_downsample_rate=common['global_downsample_rate'],
            video_action_freq_ratio=common['video_action_freq_ratio'],
            num_video_frames=common['num_video_frames'],
            video_loss_weight=1.0,
            action_loss_weight=1.0,
            
            # Inference config
            batch_size=1,
            video_height=common['video_height'],
            video_width=common['video_width'],
            
            # Don't load pretrained backbones - will load full model from checkpoint
            load_pretrained_backbones=False,
            training_mode='finetune',
        )

        return config
    
    def update_obs(self, observation: Dict[str, Any]):
        """Update observation cache with new observation."""
        # Extract visual observations
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                head_img = obs_data['head_camera']['rgb']
                left_img = obs_data['left_camera']['rgb']
                right_img = obs_data['right_camera']['rgb']
                
                left_img_resized = cv2.resize(left_img, (160, 120))
                right_img_resized = cv2.resize(right_img, (160, 120))
                bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
                image = np.concatenate([head_img, bottom_row], axis=0)
            else:
                raise ValueError("Missing camera data")
        elif 'head_camera' in observation:
            image = observation['head_camera']
        elif 'image' in observation:
            image = observation['image']
        else:
            raise ValueError("No visual observation found")

        target_size = (self.config_dict['common']['video_height'],
                      self.config_dict['common']['video_width'])

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image

        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)
        
        self.obs_cache.append(image_tensor.to(self.device))

        # Extract robot state
        state = observation['joint_action']['vector']

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()

        self.current_state = state_tensor.to(self.device)
        self.current_state_norm = self._normalize_actions(self.current_state).to(self.device)

    # tts add
    def _run_single_inference(self, current_frame, t5_list, vlm_inputs, num_inference_steps):
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame,
                state=self.current_state,
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs],
            )
        return predicted_frames, predicted_actions

    # tts add
    def _select_tts_medoid(self, actions_stack: torch.Tensor):
        # actions_stack: [N, H, D], already on CPU
        n = actions_stack.shape[0]
        flat_actions = actions_stack.reshape(n, -1)
        pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)
        avg_l2 = pairwise_l2.sum(dim=1) / (n - 1)
        best_idx = int(torch.argmin(avg_l2).item())
        return best_idx, pairwise_l2, avg_l2

    # tts add
    def _log_tts_result(self, actions_stack, pairwise_l2, avg_l2, best_idx):
        if not self.tts_log_actions:
            return

        avg_list = [round(float(x), 6) for x in avg_l2]
        print(
            f"[TTS] episode={self.episode_count} "
            f"step={self.step_count} "
            f"samples={actions_stack.shape[0]} "
            f"best={best_idx} "
            f"avg_l2={avg_list}"
        )

        pairwise_np = pairwise_l2.numpy()
        print(
            "[TTS] pairwise_l2=\n"
            + np.array2string(pairwise_np, precision=4, suppress_small=True)
        )

        if self.tts_save_full_actions:
            save_path = self.tts_dir / f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.npz"
            np.savez_compressed(
                save_path,
                actions=actions_stack.numpy(),
                pairwise_l2=pairwise_l2.numpy(),
                avg_l2=avg_l2.numpy(),
                best_idx=np.array(best_idx, dtype=np.int64),
            )
            print(f"[TTS] saved full action chunks to {save_path}")

    def get_action(self, instruction: str = None) -> List[np.ndarray]:
        """Get action predictions from the model."""
        if len(self.obs_cache) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")
        
        if self.current_state is None:
            raise ValueError("No robot state available. Call update_obs first.")
        
        current_frame = self.obs_cache[-1]

        # Encode instruction with T5
        scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                        "The aloha robot is currently performing the following task: ")
        instruction = f"{scene_prefix}{self.current_instruction}"
        t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        # Build VLM inputs
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(instruction, first_frame_pil)

        # # Run inference
        # num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']
        # with torch.no_grad():
        #     predicted_frames, predicted_actions = self.model.inference_step(
        #         first_frame=current_frame,
        #         state=self.current_state,
        #         num_inference_steps=num_inference_steps,
        #         language_embeddings=t5_list,
        #         vlm_inputs=[vlm_inputs],
        #     )
        
        # tts add 
        # Run inference
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']

        if self.tts_enable and self.tts_num_samples > 1:
            action_candidates = []
            frame_candidates = []

            for sample_idx in range(self.tts_num_samples):
                cand_frames, cand_actions = self._run_single_inference(
                    current_frame=current_frame,
                    t5_list=t5_list,
                    vlm_inputs=vlm_inputs,
                    num_inference_steps=num_inference_steps,
                )

                action_candidates.append(cand_actions.squeeze(0).detach().float().cpu())
                frame_candidates.append(cand_frames.detach().float().cpu() if cand_frames is not None else None)

            actions_stack = torch.stack(action_candidates, dim=0)
            best_idx, pairwise_l2, avg_l2 = self._select_tts_medoid(actions_stack)
            self._log_tts_result(actions_stack, pairwise_l2, avg_l2, best_idx)

            predicted_actions = actions_stack[best_idx].unsqueeze(0)
            predicted_frames = frame_candidates[best_idx]
        else:
            predicted_frames, predicted_actions = self._run_single_inference(
                current_frame=current_frame,
                t5_list=t5_list,
                vlm_inputs=vlm_inputs,
                num_inference_steps=num_inference_steps,
            )
        # Run inference end

        # Save frame grid
        if predicted_frames is not None:
            if predicted_frames.dim() == 5:
                if predicted_frames.shape[1] == 3:
                    predicted_frames_viz = predicted_frames.permute(0, 2, 1, 3, 4)
                else:
                    predicted_frames_viz = predicted_frames
                
                # condition_frame_viz = current_frame.squeeze(0)
                # predicted_frames_viz = predicted_frames_viz.squeeze(0)
                
                condition_frame_viz = current_frame.squeeze(0).detach().cpu()
                predicted_frames_viz = predicted_frames_viz.squeeze(0).detach().cpu()
                
                self._save_frame_grid(condition_frame_viz, predicted_frames_viz)
                self.step_count += 1

        actions_real = predicted_actions.squeeze(0).cpu().numpy()
        self.prev_action = actions_real[-1].copy()
        self.action_cache.extend(actions_real)

        return actions_real

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        """Convert [C, H, W] tensor to PIL Image."""
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Build VLM inputs."""
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': instruction},
                    {'type': 'image', 'image': image},
                ]
            }
        ]
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device), 
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    def _load_normalization_stats(self):
        """Load action normalization stats."""
        try:
            stat_path = Path(__file__).parent / 'utils' / 'stat.json'
            with open(stat_path, 'r') as f:
                stat_data = yaml.safe_load(f) if stat_path.suffix in ['.yml', '.yaml'] else None
        except Exception:
            stat_data = None
        if stat_data is None:
            import json as _json
            with open(Path(__file__).parent / 'utils' / 'stat.json', 'r') as f:
                stat_data = _json.load(f)

        stats = stat_data.get('robotwin2')
        if stats is None:
            raise ValueError('Normalization stats not found')
        self.action_min = torch.tensor(stats['min'], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(stats['max'], dtype=torch.float32, device=self.device)
        self.action_range = self.action_max - self.action_min

    def _normalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0,1]."""
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
        return norm.reshape(shape)

    def _denormalize_actions(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0,1]."""
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        denorm = y_flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return denorm.reshape(shape)
    
    def _create_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:
        """Create horizontal grid."""
        def tensor_to_numpy(tensor):
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            tensor = tensor.detach().cpu().float()
            tensor = torch.clamp(tensor, 0, 1)
            return (tensor.numpy() * 255).astype(np.uint8)
        
        condition_np = tensor_to_numpy(condition_frame)
        predicted_np = []
        num_pred_frames = predicted_frames.shape[0]
        for i in range(num_pred_frames):
            frame_np = tensor_to_numpy(predicted_frames[i])
            predicted_np.append(frame_np)
        
        while len(predicted_np) < 4:
            predicted_np.append(predicted_np[-1] if predicted_np else condition_np)
        
        all_frames = [condition_np] + predicted_np[:4]
        grid_image = np.concatenate(all_frames, axis=1)
        
        return Image.fromarray(grid_image)
    
    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        """Save frame grid to disk."""
        if not self.save_images:
            return
        
        try:
            grid_image = self._create_frame_grid(condition_frame, predicted_frames)
            filename = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.png"
            save_path = self.save_dir / filename
            grid_image.save(save_path)
            logger.info(f"Saved frame grid to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save frame grid: {e}")

def encode_obs(observation):
    """Post-Process Observation"""
    return observation

def get_model(usr_args):
    """
    Initialize Motus model.
    
    Args:
        usr_args: Arguments from eval script (must include wan_path and vlm_path)
    """
    checkpoint_path = usr_args.get('ckpt_setting')
    wan_path = usr_args.get('wan_path')  # Passed from eval.sh or auto_eval.sh
    vlm_path = usr_args.get('vlm_path')  # Passed from eval.sh or auto_eval.sh
    
    if not wan_path:
        raise ValueError("wan_path not provided in usr_args")
    
    if not vlm_path:
        raise ValueError("vlm_path not provided in usr_args")
    
    policy_dir = Path(__file__).parent
    config_path = policy_dir / "utils" / "robotwin.yml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # tts add
    tts_enable = _as_bool(usr_args.get("tts_enable", DEFAULT_TTS_ENABLE))
    tts_num_samples = int(usr_args.get("tts_num_samples", DEFAULT_TTS_NUM_SAMPLES))
    tts_log_actions = _as_bool(usr_args.get("tts_log_actions", DEFAULT_TTS_LOG_ACTIONS))
    tts_save_full_actions = _as_bool(usr_args.get("tts_save_full_actions", DEFAULT_TTS_SAVE_FULL_ACTIONS))
        
    # policy = MotusPolicy(
    #     checkpoint_path=checkpoint_path,
    #     wan_path=wan_path,
    #     vlm_path=vlm_path,
    #     config_path=str(config_path),
    #     device=device,
    #     log_dir=usr_args.get('log_dir'),
    #     task_name=usr_args.get('task_name')
    # )
    
    # tts add
    policy = MotusPolicy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        config_path=str(config_path),
        device=device,
        log_dir=usr_args.get('log_dir'),
        task_name=usr_args.get('task_name'),
        tts_enable=tts_enable,
        tts_num_samples=tts_num_samples,
        tts_log_actions=tts_log_actions,
        tts_save_full_actions=tts_save_full_actions,
    )
    
    return policy

def eval(TASK_ENV, model, observation):
    """Evaluation function."""
    obs = encode_obs(observation)
    
    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)

    actions = model.get_action()
    
    for action in actions:
        TASK_ENV.take_action(action, action_type='qpos')

def reset_model(model):  
    """Reset model cache at episode start."""
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.is_first_step = True
    model.prev_action = None
    model.episode_count += 1
    model.step_count = 0
    logger.info(f"Model reset completed for episode {model.episode_count}")
```



##### 评估

```Plain Text
cd /home/ubuntu/workspace/RoboTwin/policy/Motus
bash eval.sh scan_object --tts --tts-num-samples 8
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MzNiYjBmMjQ0MDMyMWJlOTljNzBkZjFmYTlkNDg5ZWFfOGJkYTRlZTI5OGU0YmMwMTQ5OGNmZjQ3YjhkYjI1ODFfSUQ6NzY0MDY3OTkzNTA0ODM4NzU0NV8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

两个点并不明显



实现两卡分别运行两个进程

改eval\.sh：

```Bash
#!/bin/bash
# Single task evaluation script for Motus policy on RoboTwin platform

# ============================================================================
# Single Task Configuration - MODIFY THESE
# ============================================================================

# 修改，使其能传入指定任务的参数
# TASK_NAME="click_alarmclock"  # Change this to the task you want to test

# TASK_NAME="${1:-click_alarmclock}"

#tts add
TASK_NAME="${1:-click_alarmclock}"
if [ $# -gt 0 ]; then
    shift
fi

GPU_ID=0
TTS_ENABLE=False
TTS_NUM_SAMPLES=8
TTS_LOG_ACTIONS=True
TTS_SAVE_FULL_ACTIONS=True

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --tts)
            TTS_ENABLE=True
            shift
            ;;
        --no-tts)
            TTS_ENABLE=False
            shift
            ;;
        --tts-num-samples)
            TTS_NUM_SAMPLES="$2"
            shift 2
            ;;
        --tts-log-actions)
            TTS_LOG_ACTIONS="$2"
            shift 2
            ;;
        --tts-save-full-actions)
            TTS_SAVE_FULL_ACTIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# GPU_ID=0                       # GPU to use

# ============================================================================
# Script starts here
# ============================================================================
echo "Starting single task evaluation at $(date)"

# Get script directory (policy/Motus/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"

# ============================================================================
# Load Configuration from paths_config.yml
# ============================================================================
CONFIG_FILE="${POLICY_DIR}/paths_config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please create paths_config.yml with required paths."
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"

# Parse YAML (improved - remove comments and extra whitespace)
ROBOTWIN_ROOT=$(grep "^robotwin_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CONDA_ENV=$(grep "^conda_env:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CHECKPOINT_PATH=$(grep "^checkpoint_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
WAN_PATH=$(grep "^wan_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
VLM_PATH=$(grep "^vlm_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Optional configurations
TASK_CONFIG=$(grep "^task_config:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
SEED=$(grep "^seed:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Default values
TASK_CONFIG=${TASK_CONFIG:-"demo_randomized"}
SEED=${SEED:-"42"}
POLICY_NAME="Motus"

# ============================================================================
# Validation
# ============================================================================
if [ -z "$ROBOTWIN_ROOT" ]; then
    echo "Error: robotwin_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CONDA_ENV" ]; then
    echo "Error: conda_env is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: checkpoint_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$WAN_PATH" ]; then
    echo "Error: wan_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$VLM_PATH" ]; then
    echo "Error: vlm_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$ROBOTWIN_ROOT" ]; then
    echo "Error: RoboTwin root not found: $ROBOTWIN_ROOT"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$WAN_PATH" ]; then
    echo "Error: WAN path not found: $WAN_PATH"
    exit 1
fi

if [ ! -d "$VLM_PATH" ]; then
    echo "Error: VLM path not found: $VLM_PATH"
    exit 1
fi

cd "$ROBOTWIN_ROOT" || exit 1

# 处理conda报错
#######

# # Activate conda
# if ! command -v conda &> /dev/null; then
#     echo "Error: conda not found."
#     exit 1
# fi

# eval "$(conda shell.bash hook)"
# conda activate "$CONDA_ENV"

# Activate conda
# export PATH="/home/sumita-mana/anaconda3/bin:$PATH"
# source /home/sumita-mana/anaconda3/etc/profile.d/conda.sh

export PATH="/home/ubuntu/miniconda3/bin:$PATH"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found."
    exit 1
fi

conda activate "$CONDA_ENV"

#######

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

# Set environment
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create logs directory
LOG_DIR="${POLICY_DIR}/logs_single_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ckpt_setting="${CHECKPOINT_PATH}"
log_file="${LOG_DIR}/${TASK_NAME}.log"

echo ""
echo "================================================================"
echo "Single Task Evaluation Configuration"
echo "================================================================"
echo "Task Name:         $TASK_NAME"
echo "GPU:               $GPU_ID"
echo "----------------------------------------------------------------"
echo "RoboTwin Root:     $ROBOTWIN_ROOT"
echo "Policy Dir:        $POLICY_DIR"
echo "Checkpoint:        $CHECKPOINT_PATH"
echo "WAN Path:          $WAN_PATH"
echo "VLM Path:          $VLM_PATH"
echo "Task Config:       $TASK_CONFIG"
echo "Seed:              $SEED"
echo "Log File:          $log_file"
echo "TTS Enable:        $TTS_ENABLE" #tts add
echo "TTS Num Samples:   $TTS_NUM_SAMPLES" #tts add
echo "TTS Log Actions:   $TTS_LOG_ACTIONS" #tts add
echo "TTS Save Full:     $TTS_SAVE_FULL_ACTIONS" #tts add
echo "================================================================"
echo ""

# Run evaluation with WAN_PATH passed as argument
echo "Starting evaluation..."

# 处理环境进入问题
#############

# PYTHONWARNINGS=ignore::UserWarning \
# python script/eval_policy.py \
#     --config "policy/${POLICY_NAME}/deploy_policy.yml" \
#     --overrides \
#     --task_name "${TASK_NAME}" \
#     --task_config "${TASK_CONFIG}" \
#     --ckpt_setting "${ckpt_setting}" \
#     --seed "${SEED}" \
#     --policy_name "${POLICY_NAME}" \
#     --log_dir "${LOG_DIR}" \
#     --wan_path "${WAN_PATH}" \
#     --vlm_path "${VLM_PATH}" \
#     2>&1 | tee "$log_file"

echo "Python executable: ${CONDA_ENV}/bin/python"
"${CONDA_ENV}/bin/python" - <<'PY'
import sys
print("sys.executable =", sys.executable)
import sapien
print("sapien ok =", sapien.__file__)
import importlib
m = importlib.import_module("sapien.core")
print("sapien.core ok =", m)
PY

PYTHONWARNINGS=ignore::UserWarning \
"${CONDA_ENV}/bin/python" script/eval_policy.py \
    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${SEED}" \
    --policy_name "${POLICY_NAME}" \
    --log_dir "${LOG_DIR}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}" \
    --tts_enable "${TTS_ENABLE}" \
    --tts_num_samples "${TTS_NUM_SAMPLES}" \
    --tts_log_actions "${TTS_LOG_ACTIONS}" \
    --tts_save_full_actions "${TTS_SAVE_FULL_ACTIONS}" \
    2>&1 | tee "$log_file"

#############

exit_code=${PIPESTATUS[0]}

echo ""
echo "================================================================"
if [ $exit_code -eq 0 ]; then
    echo "✅ Task $TASK_NAME completed successfully"
    echo "================================================================"
    exit 0
else
    echo "❌ Task $TASK_NAME failed with exit code $exit_code"
    echo "================================================================"
    echo "Log file: $log_file"
    exit 1
fi
```

能传参指定gpu id了



运行实验2：

```Plain Text
cd /home/ubuntu/workspace/RoboTwin/policy/Motus
bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 16
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=M2FmOTA1NThhNDhlMGRjY2FhOTkwMzc2N2MzOTkzNzFfMTNhNjdhZmM4ODJhMWFiZGMyMDc4NGJkNDU5YzQ2YjlfSUQ6NzY0MDc2NzMzODU4MDc0MTMwMF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

提高10个点



motus论文报告的结果

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YTBjOTlkZTg3OGI0Y2RlZDJjZWFmZmM3MDRjNDY3NDNfNjE1MzI5ZjY3N2YwY2M1ZGExNjcxOWRkMDM3OTNkNmJfSUQ6NzY0MDY4MTA1MzQwNDkyNDg3OV8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

put\_bottles\_dustbin



#### v2 



##### 开发

备份两个文件，ttsv1



可视化调整：终端输出干净点，另外保存能打开文本

改写日志的函数：

```Python
def _log_tts_result(self, actions_stack, pairwise_l2, avg_l2, best_idx):
        if not self.tts_log_actions:
            return

        self.tts_dir.mkdir(parents=True, exist_ok=True)

        avg_l2_np = avg_l2.numpy()
        pairwise_l2_np = pairwise_l2.numpy()
        actions_np = actions_stack.numpy()

        best_avg_l2 = float(avg_l2_np[best_idx])
        min_avg_l2 = float(avg_l2_np.min())
        max_avg_l2 = float(avg_l2_np.max())
        mean_avg_l2 = float(avg_l2_np.mean())

        avg_list = [round(float(x), 6) for x in avg_l2_np]

        print(
            f"[TTS] episode={self.episode_count} "
            f"step={self.step_count} "
            f"samples={actions_stack.shape[0]} "
            f"best={best_idx} "
            f"best_avg_l2={best_avg_l2:.6f} "
            f"min_avg_l2={min_avg_l2:.6f} "
            f"max_avg_l2={max_avg_l2:.6f}"
        )

        summary_path = self.tts_dir / "summary.csv"
        write_header = not summary_path.exists()

        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "episode",
                    "step",
                    "samples",
                    "best_idx",
                    "best_avg_l2",
                    "min_avg_l2",
                    "max_avg_l2",
                    "mean_avg_l2",
                    "avg_l2",
                    "npz_path",
                ])

            npz_path = ""
            if self.tts_save_full_actions:
                npz_path = str(self.tts_dir / f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.npz")

            writer.writerow([
                self.episode_count,
                self.step_count,
                actions_stack.shape[0],
                best_idx,
                f"{best_avg_l2:.8f}",
                f"{min_avg_l2:.8f}",
                f"{max_avg_l2:.8f}",
                f"{mean_avg_l2:.8f}",
                avg_list,
                npz_path,
            ])

        if self.tts_save_full_actions:
            save_path = self.tts_dir / f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.npz"
            np.savez_compressed(
                save_path,
                actions=actions_np,
                pairwise_l2=pairwise_l2_np,
                avg_l2=avg_l2_np,
                best_idx=np.array(best_idx, dtype=np.int64),
            )
            # print(f"[TTS] saved full action chunks to {save_path}")
```



先改：

添加聚类方法、方法选择，改日志保存

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a096cc2\-1ad0\-8333\-bf20\-62105a5fcfcd

再给2文件，重新调整实现：

随机采样：

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a09526a\-e7e4\-832c\-ae5f\-b43f5792f03a



##### 评估

指令：

朴素tts

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_global_medoid_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 16 \
  --tts-method global_medoid \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

聚类tts

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_keystone_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 0 --tts --tts-num-samples 8 \
  --tts-method keystone \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

Rank softmax：

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_rank_softmax_tau1_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 8 \
  --tts-method rank_softmax \
  --tts-tau 1.0 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```



运行：

```Plain Text
聚类
PID: 172357
LOG: logs/scan_object_tts_keystone_20260517_172735.log

随机
PID: 172655
LOG: logs/scan_object_tts_rank_softmax_tau1_20260517_172810.log
```



聚类：67\.2，更高

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MDUzOWVhNGUwMWU2ODRmODE4NmEzYmUwYzQ2MjE5YWRfMjgwNjk4MzNiYTg2YTY5YjUwMzEyYWZiYjVlZGQ4ODZfSUQ6NzY0MDg4NzI3NTQyNTA5MDc2OV8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



随机：53\.6，没变化

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZmJlM2ZhNTU3NDQ5YTU4NGNmZmM1MTUxNDkwZTcwYjVfNDc0ZDlhNDkwNDc3MzdiZWI4NmFkZWZjMDFmMTdhMmNfSUQ6NzY0MDg4NzM1MTU2ODQ1MjU4NF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



实验2，都n=16



聚类tts

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_keystone_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 0 --tts --tts-num-samples 16 \
  --tts-method keystone \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

PID: 205602

LOG: logs/scan\_object\_tts\_keystone\_20260517\_234935\.log

略高

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ODllN2Y3OGQ4N2FmOTZhNjZmYTI3YmUzNWIyOTBkY2JfMjFlYWJmMWM5ODQ0OTQyYTJiY2U5NDQ5MWZjMjViODVfSUQ6NzY0MTA4NjMwNDk0MzgwMzU4Ml8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



Rank softmax：

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_rank_softmax_tau1_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 16 \
  --tts-method rank_softmax \
  --tts-tau 1.0 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

PID: 205897

LOG: logs/scan\_object\_tts\_rank\_softmax\_tau1\_20260517\_234957\.log

更高

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OTM1NmFlODIyNDNlYmNmMDkwNjg4MzhmYjZjMjc1MmRfZWU2MjhmNTU0OGJjY2QwMzIzZThjYjFhNjY5MTVlMzlfSUQ6NzY0MTA4NDM3MDExOTU2MDQxMF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



#### v3



##### 实现

实现：L2 min \+ Cluster \+ Random；缝起来先跑通



eval\.sh

```Bash
#!/bin/bash
# Single task evaluation script for Motus policy on RoboTwin platform
# impl and exp of tts and opd

# ============================================================================
# Single Task Configuration - MODIFY THESE
# ============================================================================

#tts add
TASK_NAME="${1:-click_alarmclock}"
if [ $# -gt 0 ]; then
    shift
fi

# ttsv2 add
GPU_ID=0

# Test-time scaling
TTS_ENABLE=False
TTS_NUM_SAMPLES=8

# Selection method:
#   global_medoid:          average-L2/global-medoid selection
#   keystone:               KeyStone-style guard + kmeans + largest-cluster medoid
#   rank_softmax:           rank-based stochastic selection, P(i)=softmax(-rank_i/tau)
#   cluster_rank_softmax:   guard + kmeans largest-cluster + local rank-softmax
TTS_METHOD="global_medoid"

# KeyStone defaults
TTS_NUM_CLUSTERS=2

# tau meaning:
#   keystone:     unimodality guard threshold, default 0.3
#   rank_softmax: rank-softmax temperature, recommended 1.0
TTS_TAU=0.3

# Rank-softmax temperature for cluster_rank_softmax.
# Keep this separate from TTS_TAU, which is used as the unimodality guard.
TTS_RANK_TAU=1.0

TTS_KMEANS_ITERS=10

# Logging
TTS_LOG_ACTIONS=True

# Deprecated/no-op in the new Python implementation.
# 保留这个参数只是为了兼容旧命令；新版本不再保存 .npz。
TTS_SAVE_FULL_ACTIONS=False
# ttsv2 add end

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --tts)
            TTS_ENABLE=True
            shift
            ;;
        --no-tts)
            TTS_ENABLE=False
            shift
            ;;
        --tts-num-samples)
            TTS_NUM_SAMPLES="$2"
            shift 2
            ;;
        --tts-method)
            TTS_METHOD="$2"
            shift 2
            ;;
        --tts-num-clusters)
            TTS_NUM_CLUSTERS="$2"
            shift 2
            ;;
        --tts-tau)
            TTS_TAU="$2"
            shift 2
            ;;
        --tts-rank-tau)
            TTS_RANK_TAU="$2"
            shift 2
            ;;        
        --tts-kmeans-iters)
            TTS_KMEANS_ITERS="$2"
            shift 2
            ;;            
        --tts-log-actions)
            TTS_LOG_ACTIONS="$2"
            shift 2
            ;;
        --tts-save-full-actions)
            TTS_SAVE_FULL_ACTIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# GPU_ID=0                       # GPU to use

# ============================================================================
# Script starts here
# ============================================================================
echo "Starting single task evaluation at $(date)"

# Get script directory (policy/Motus/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"

# ============================================================================
# Load Configuration from paths_config.yml
# ============================================================================
CONFIG_FILE="${POLICY_DIR}/paths_config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please create paths_config.yml with required paths."
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"

# Parse YAML (improved - remove comments and extra whitespace)
ROBOTWIN_ROOT=$(grep "^robotwin_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CONDA_ENV=$(grep "^conda_env:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CHECKPOINT_PATH=$(grep "^checkpoint_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
WAN_PATH=$(grep "^wan_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
VLM_PATH=$(grep "^vlm_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Optional configurations
TASK_CONFIG=$(grep "^task_config:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
SEED=$(grep "^seed:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Default values
TASK_CONFIG=${TASK_CONFIG:-"demo_randomized"}
SEED=${SEED:-"42"}
POLICY_NAME="Motus"

# ============================================================================
# Validation
# ============================================================================
if [ -z "$ROBOTWIN_ROOT" ]; then
    echo "Error: robotwin_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CONDA_ENV" ]; then
    echo "Error: conda_env is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: checkpoint_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$WAN_PATH" ]; then
    echo "Error: wan_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$VLM_PATH" ]; then
    echo "Error: vlm_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$ROBOTWIN_ROOT" ]; then
    echo "Error: RoboTwin root not found: $ROBOTWIN_ROOT"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$WAN_PATH" ]; then
    echo "Error: WAN path not found: $WAN_PATH"
    exit 1
fi

if [ ! -d "$VLM_PATH" ]; then
    echo "Error: VLM path not found: $VLM_PATH"
    exit 1
fi

cd "$ROBOTWIN_ROOT" || exit 1

# 处理conda报错
#######

# # Activate conda
# if ! command -v conda &> /dev/null; then
#     echo "Error: conda not found."
#     exit 1
# fi

# eval "$(conda shell.bash hook)"
# conda activate "$CONDA_ENV"

# Activate conda
# export PATH="/home/sumita-mana/anaconda3/bin:$PATH"
# source /home/sumita-mana/anaconda3/etc/profile.d/conda.sh

export PATH="/home/ubuntu/miniconda3/bin:$PATH"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found."
    exit 1
fi

conda activate "$CONDA_ENV"

#######

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

# Set environment
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create logs directory
LOG_DIR="${POLICY_DIR}/logs_single_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ckpt_setting="${CHECKPOINT_PATH}"
log_file="${LOG_DIR}/${TASK_NAME}.log"

echo ""
echo "================================================================"
echo "Single Task Evaluation Configuration"
echo "================================================================"
echo "Task Name:         $TASK_NAME"
echo "GPU:               $GPU_ID"
echo "----------------------------------------------------------------"
echo "RoboTwin Root:     $ROBOTWIN_ROOT"
echo "Policy Dir:        $POLICY_DIR"
echo "Checkpoint:        $CHECKPOINT_PATH"
echo "WAN Path:          $WAN_PATH"
echo "VLM Path:          $VLM_PATH"
echo "Task Config:       $TASK_CONFIG"
echo "Seed:              $SEED"
echo "Log File:          $log_file"
# echo "TTS Enable:        $TTS_ENABLE" #tts add
# echo "TTS Num Samples:   $TTS_NUM_SAMPLES" #tts add
# echo "TTS Log Actions:   $TTS_LOG_ACTIONS" #tts add
# echo "TTS Save Full:     $TTS_SAVE_FULL_ACTIONS" #tts add
echo "TTS Enable:        $TTS_ENABLE"
echo "TTS Num Samples:   $TTS_NUM_SAMPLES"
echo "TTS Method:        $TTS_METHOD"
echo "TTS Num Clusters:  $TTS_NUM_CLUSTERS"
echo "TTS Tau:           $TTS_TAU"
echo "TTS Rank Tau:      $TTS_RANK_TAU"
echo "TTS KMeans Iters:  $TTS_KMEANS_ITERS"
echo "TTS Log Actions:   $TTS_LOG_ACTIONS"
echo "TTS Save Full:     $TTS_SAVE_FULL_ACTIONS (deprecated; npz disabled)"
echo "================================================================"
echo ""

# Run evaluation with WAN_PATH passed as argument
echo "Starting evaluation..."

# 处理环境进入问题
#############

echo "Python executable: ${CONDA_ENV}/bin/python"
"${CONDA_ENV}/bin/python" - <<'PY'
import sys
print("sys.executable =", sys.executable)
import sapien
print("sapien ok =", sapien.__file__)
import importlib
m = importlib.import_module("sapien.core")
print("sapien.core ok =", m)
PY

PYTHONWARNINGS=ignore::UserWarning \
"${CONDA_ENV}/bin/python" script/eval_policy.py \
    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${SEED}" \
    --policy_name "${POLICY_NAME}" \
    --log_dir "${LOG_DIR}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}" \
    --tts_enable "${TTS_ENABLE}" \
    --tts_num_samples "${TTS_NUM_SAMPLES}" \
    --tts_method "${TTS_METHOD}" \
    --tts_num_clusters "${TTS_NUM_CLUSTERS}" \
    --tts_tau "${TTS_TAU}" \
    --tts_rank_tau "${TTS_RANK_TAU}" \
    --tts_kmeans_iters "${TTS_KMEANS_ITERS}" \
    --tts_log_actions "${TTS_LOG_ACTIONS}" \
    --tts_save_full_actions "${TTS_SAVE_FULL_ACTIONS}" \
    2>&1 | tee "$log_file"

#############

exit_code=${PIPESTATUS[0]}

echo ""
echo "================================================================"
if [ $exit_code -eq 0 ]; then
    echo "✅ Task $TASK_NAME completed successfully"
    echo "================================================================"
    exit 0
else
    echo "❌ Task $TASK_NAME failed with exit code $exit_code"
    echo "================================================================"
    echo "Log file: $log_file"
    exit 1
fi
```



deploy\_policy\.py

```Python
# Motus Policy for RoboTwin
# impl and exp of tts and opd

# for save csv
import csv

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add model paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

from models.motus import Motus, MotusConfig

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel
from utils.image_utils import resize_with_padding

logger = logging.getLogger(__name__)

# =========================
# Test-Time Scaling Defaults
# =========================
DEFAULT_TTS_ENABLE = False
DEFAULT_TTS_NUM_SAMPLES = 8

# Methods:
#   global_medoid:          average-L2/global-medoid selection
#   keystone:               unimodality guard + kmeans + largest-cluster medoid
#   rank_softmax:           rank-based stochastic selection, P(i)=softmax(-rank_i/tau)
#   cluster_rank_softmax:   unimodality guard + kmeans largest-cluster + local rank-softmax
DEFAULT_TTS_METHOD = "global_medoid"

# TTS defaults
DEFAULT_TTS_NUM_CLUSTERS = 2

# tau meaning:
#   keystone:               unimodality guard threshold
#   cluster_rank_softmax:   unimodality guard threshold
#   rank_softmax:           rank-softmax temperature, kept for backward compatibility
DEFAULT_TTS_TAU = 0.3

# tts v3 add
# Separate rank-softmax temperature for cluster_rank_softmax.
# This avoids overloading DEFAULT_TTS_TAU, whose default is tuned as a guard threshold.
DEFAULT_TTS_RANK_TAU = 1.0

DEFAULT_TTS_KMEANS_ITERS = 10

# Logging
DEFAULT_TTS_LOG_ACTIONS = True

# Deprecated/no-op: kept only for backward compatibility with old scripts.
# New implementation does not save .npz files.
DEFAULT_TTS_SAVE_FULL_ACTIONS = False

# tts add
def _as_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in ["1", "true", "yes", "y", "on"]

class MotusPolicy:
    """
    Motus Policy wrapper for RoboTwin evaluation.
    Implements the joint video-action diffusion model for robotic control.
    """
    
    def __init__(
        self, 
        checkpoint_path: str, 
        config_path: str, 
        wan_path: str, 
        vlm_path: str, 
        device: str = "cuda", 
        log_dir: Optional[str] = None, 
        task_name: Optional[str] = None, 
        tts_enable: bool = DEFAULT_TTS_ENABLE,
        tts_num_samples: int = DEFAULT_TTS_NUM_SAMPLES,
        tts_method: str = DEFAULT_TTS_METHOD,
        tts_num_clusters: int = DEFAULT_TTS_NUM_CLUSTERS,
        tts_tau: float = DEFAULT_TTS_TAU,
        tts_rank_tau: float = DEFAULT_TTS_RANK_TAU,
        tts_kmeans_iters: int = DEFAULT_TTS_KMEANS_ITERS,
        tts_log_actions: bool = DEFAULT_TTS_LOG_ACTIONS,
        tts_save_full_actions: bool = DEFAULT_TTS_SAVE_FULL_ACTIONS,             
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
        
        # Initialize model WITHOUT loading pretrained backbones
        self.model = self._load_model()

        # Initialize T5 encoder for language embeddings (WAN text encoder)
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(self.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.wan_path, 'google', 'umt5-xxl'),
        )

        # Initialize VLM processor from vlm_path (for tokenization only, weights from checkpoint)
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)
        
        # Initialize observation cache
        self.obs_cache = deque(maxlen=1)
        self.action_cache = deque()
        
        # Model state
        self.current_state = None
        self.current_state_norm = None
        self.is_first_step = True
        self.prev_action = None

        # Load normalization stats
        self._load_normalization_stats()
        
        # Initialize image saving
        self.save_images = True
        
        
        # tts add
        
        base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        task_dir_name = task_name or os.environ.get('TASK_NAME') or "default_task"

        self.log_dir = Path(base_log_dir)
        self.task_dir_name = task_dir_name

        self.save_dir = self.log_dir / "images" / task_dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.tts_enable = _as_bool(tts_enable)
        self.tts_num_samples = max(1, int(tts_num_samples))

        self.tts_method = str(tts_method).strip().lower()
        
        allowed_tts_methods = {
            "global_medoid",
            "keystone",
            "rank_softmax",
            "cluster_rank_softmax",
        }
        
        if self.tts_method not in allowed_tts_methods:
            raise ValueError(
                f"Unknown tts_method={self.tts_method}. "
                f"Allowed methods: {sorted(allowed_tts_methods)}"
            )

        self.tts_num_clusters = max(1, int(tts_num_clusters))
        self.tts_tau = float(tts_tau)
        self.tts_rank_tau = float(tts_rank_tau)
        self.tts_kmeans_iters = max(1, int(tts_kmeans_iters))

        self.tts_log_actions = _as_bool(tts_log_actions)

        # Deprecated/no-op. Kept so old command lines still parse.
        self.tts_save_full_actions = _as_bool(tts_save_full_actions)

        self.tts_dir = self.log_dir / "tts" / task_dir_name

        if self.tts_enable and self.tts_log_actions:
            self.tts_dir.mkdir(parents=True, exist_ok=True)

        self.episode_count = 0
        self.step_count = 0

        print(
            f"[TTS] enable={self.tts_enable}, "
            f"num_samples={self.tts_num_samples}, "
            f"method={self.tts_method}, "
            f"num_clusters={self.tts_num_clusters}, "
            f"tau={self.tts_tau}, "
            f"rank_tau={self.tts_rank_tau}, "
            f"kmeans_iters={self.tts_kmeans_iters}, "
            f"log_actions={self.tts_log_actions}, "
            f"save_full_actions={self.tts_save_full_actions} (deprecated/no-op), "
            f"tts_dir={self.tts_dir}"
        )

        logger.info("Motus Policy initialized successfully")

    def set_instruction(self, instruction: str):
        """Set the current instruction for the policy."""
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def _load_model(self) -> Motus:
        """Load the Motus model without pretrained backbones, then load checkpoint."""
        logger.info(f"Initializing Motus model from config (no pretrained backbones)")

        config = self._create_model_config()
        
        # Initialize model from config WITHOUT loading pretrained weights
        model = Motus(config)
        model = model.to(self.device)
        
        # Load checkpoint weights
        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info("Model checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        model.eval()
        return model
    
    def _create_model_config(self) -> MotusConfig:
        """Create model configuration from yaml config - inference mode."""
        common = self.config_dict['common']
        model_cfg = self.config_dict['model']

        # Use paths passed to constructor
        vae_path = os.path.join(self.wan_path, "Wan2.2_VAE.pth")
        vlm_checkpoint_path = self.vlm_path

        hidden_size = model_cfg['action_expert']['hidden_size']
        ffn_multiplier = model_cfg['action_expert']['ffn_dim_multiplier']

        config = MotusConfig(
            # Paths for config loading only (no weights loaded)
            wan_checkpoint_path=self.wan_path,
            vae_path=vae_path,
            wan_config_path=self.wan_path,
            video_precision='bfloat16',
            vlm_checkpoint_path=vlm_checkpoint_path,
            
            # Understanding expert config
            und_expert_hidden_size=512,
            und_expert_ffn_dim_multiplier=4,
            und_expert_norm_eps=1e-5,
            und_layers_to_extract=None,
            vlm_adapter_input_dim=2048,
            vlm_adapter_projector_type="mlp3x_silu",
            
            # Model architecture
            num_layers=30,
            action_state_dim=common['state_dim'],
            action_dim=common['action_dim'],
            action_expert_dim=hidden_size,
            action_expert_ffn_dim_multiplier=ffn_multiplier,
            action_expert_norm_eps=1e-6,
            
            # Training config
            global_downsample_rate=common['global_downsample_rate'],
            video_action_freq_ratio=common['video_action_freq_ratio'],
            num_video_frames=common['num_video_frames'],
            video_loss_weight=1.0,
            action_loss_weight=1.0,
            
            # Inference config
            batch_size=1,
            video_height=common['video_height'],
            video_width=common['video_width'],
            
            # Don't load pretrained backbones - will load full model from checkpoint
            load_pretrained_backbones=False,
            training_mode='finetune',
        )

        return config
    
    def update_obs(self, observation: Dict[str, Any]):
        """Update observation cache with new observation."""
        # Extract visual observations
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                head_img = obs_data['head_camera']['rgb']
                left_img = obs_data['left_camera']['rgb']
                right_img = obs_data['right_camera']['rgb']
                
                left_img_resized = cv2.resize(left_img, (160, 120))
                right_img_resized = cv2.resize(right_img, (160, 120))
                bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
                image = np.concatenate([head_img, bottom_row], axis=0)
            else:
                raise ValueError("Missing camera data")
        elif 'head_camera' in observation:
            image = observation['head_camera']
        elif 'image' in observation:
            image = observation['image']
        else:
            raise ValueError("No visual observation found")

        target_size = (self.config_dict['common']['video_height'],
                      self.config_dict['common']['video_width'])

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image

        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)
        
        self.obs_cache.append(image_tensor.to(self.device))

        # Extract robot state
        state = observation['joint_action']['vector']

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()

        self.current_state = state_tensor.to(self.device)
        self.current_state_norm = self._normalize_actions(self.current_state).to(self.device)

    # tts add
    def _run_single_inference(self, current_frame, t5_list, vlm_inputs, num_inference_steps):
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame,
                state=self.current_state,
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs],
            )
        return predicted_frames, predicted_actions
    
    # tts v2 add
    def _select_tts_action(self, actions_stack: torch.Tensor):
        """
        Dispatch TTS selector according to self.tts_method.

        actions_stack: [K, H, D], CPU tensor.
        """
        if self.tts_method == "global_medoid":
            return self._select_tts_global_medoid(actions_stack)

        if self.tts_method == "keystone":
            return self._select_tts_keystone(actions_stack)  
        
        if self.tts_method == "rank_softmax":
            return self._select_tts_rank_softmax(actions_stack)

        if self.tts_method == "cluster_rank_softmax":
            return self._select_tts_cluster_rank_softmax(actions_stack)

        raise ValueError(f"Unknown tts_method={self.tts_method}")    

    # ttsv2 add
    def _build_tts_ranks(self, avg_l2: torch.Tensor):
        """
        Build ranks from avg_l2.

        avg_l2: lower is better.
        return:
            ranks: [K], rank 0 is best / smallest avg_l2.
            order: [K], candidate indices sorted by avg_l2 ascending.
        """
        order = torch.argsort(avg_l2, descending=False)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(len(avg_l2), dtype=order.dtype, device=avg_l2.device)
        return ranks, order

    # tts add
    def _select_tts_global_medoid(self, actions_stack: torch.Tensor):
        """
        Current baseline method:
        flatten each action chunk, compute pairwise L2, select the global medoid.
        """
        k = actions_stack.shape[0]
        flat_actions = actions_stack.reshape(k, -1)

        pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            best_idx = 0
        else:
            avg_l2 = pairwise_l2.sum(dim=1) / (k - 1)
            best_idx = int(torch.argmin(avg_l2).item())

        # ttsv2 add
        ranks, order = self._build_tts_ranks(avg_l2)
        selection_probs = torch.zeros(k, dtype=torch.float32, device=avg_l2.device)
        selection_probs[best_idx] = 1.0
        
        # ttsv2 add
        metrics = {
            "tts_method": "global_medoid",
            "selection_stage": "global_medoid",
            "global_medoid_idx": best_idx,
            "unimodal": True,
            "s_score": "",
            "num_clusters": 1,
            "selected_cluster": -1,
            "selected_cluster_size": k,
            "cluster_counts": "",
            "cluster_ids": "",
            "selected_idx": best_idx,
            "selected_rank": int(ranks[best_idx].item()),
            "selected_prob": "1.00000000",
            "rank_temperature": "",
            "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
            "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
        }

        return best_idx, pairwise_l2, avg_l2, metrics

    # ttsv2 add
    def _select_tts_rank_softmax(self, actions_stack: torch.Tensor):
        """
        Rank-softmax selector:
        1. Flatten each action chunk.
        2. Compute pairwise L2.
        3. Compute avg_l2 for each candidate.
        4. Rank candidates by avg_l2.
        5. Sample selected_idx from softmax(-rank / tau).

        tau is self.tts_tau for this method.
        """
        k = actions_stack.shape[0]
        flat_actions = actions_stack.reshape(k, -1)

        pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            selected_idx = 0
            global_medoid_idx = 0
            ranks = torch.zeros(k, dtype=torch.long, device=pairwise_l2.device)
            selection_probs = torch.ones(k, dtype=torch.float32, device=pairwise_l2.device)
            selected_rank = 0
            selected_prob = 1.0
        else:
            avg_l2 = pairwise_l2.sum(dim=1) / (k - 1)

            ranks, order = self._build_tts_ranks(avg_l2)
            global_medoid_idx = int(order[0].item())

            tau = max(1e-8, float(self.tts_tau))

            logits = -ranks.to(torch.float32) / tau
            selection_probs = torch.softmax(logits, dim=0)

            selected_idx = int(torch.multinomial(selection_probs, num_samples=1).item())
            selected_rank = int(ranks[selected_idx].item())
            selected_prob = float(selection_probs[selected_idx].item())

        metrics = {
            "tts_method": "rank_softmax",
            "selection_stage": "rank_softmax_sample",
            "global_medoid_idx": global_medoid_idx,
            "unimodal": "",
            "s_score": "",
            "num_clusters": 1,
            "selected_cluster": -1,
            "selected_cluster_size": k,
            "cluster_counts": "",
            "cluster_ids": "",
            "selected_idx": selected_idx,
            "selected_rank": selected_rank,
            "selected_prob": f"{selected_prob:.8f}",
            "rank_temperature": f"{float(self.tts_tau):.8f}",
            "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
            "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
        }

        return selected_idx, pairwise_l2, avg_l2, metrics    

    # ttsv3 add
    def _select_tts_cluster_rank_softmax(self, actions_stack: torch.Tensor):
        """
        L2 min + Cluster + Random selector.

        1. Compute global average-L2 medoid scores.
        2. Run the same unimodality guard used by KeyStone.
        3. If the guard says the samples are unimodal, sample from all candidates by rank-softmax.
        4. Otherwise, run deterministic k-means, keep the largest cluster, and sample only inside
        that cluster by local intra-cluster rank-softmax.

        self.tts_tau is the unimodality guard threshold.
        self.tts_rank_tau is the rank-softmax temperature.
        """
        k = actions_stack.shape[0]
        flat_actions = actions_stack.reshape(k, -1)

        pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            ranks = torch.zeros(k, dtype=torch.long, device=pairwise_l2.device)
            selection_probs = torch.ones(k, dtype=torch.float32, device=pairwise_l2.device)

            metrics = {
                "tts_method": "cluster_rank_softmax",
                "selection_stage": "single_sample",
                "global_medoid_idx": 0,
                "unimodal": True,
                "s_score": "",
                "num_clusters": 1,
                "selected_cluster": -1,
                "selected_cluster_size": k,
                "cluster_counts": "",
                "cluster_ids": "",
                "selected_idx": 0,
                "selected_rank": 0,
                "selected_prob": "1.00000000",
                "rank_temperature": f"{float(self.tts_rank_tau):.8f}",
                "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
                "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
            }
            return 0, pairwise_l2, avg_l2, metrics

        global_medoid_idx, avg_l2, s_score = self._compute_global_medoid_and_guard(
            flat_actions=flat_actions,
            pairwise_l2=pairwise_l2,
        )

        use_all_candidates = s_score < self.tts_tau or self.tts_num_clusters <= 1

        if use_all_candidates:
            candidate_indices = torch.arange(k, dtype=torch.long, device=avg_l2.device)
            local_scores = avg_l2

            selection_stage = "guard_rank_softmax" if s_score < self.tts_tau else "no_cluster_rank_softmax"
            unimodal = bool(s_score < self.tts_tau)
            num_clusters = 1
            selected_cluster = -1
            selected_cluster_size = k
            cluster_counts = ""
            cluster_ids = ""
        else:
            num_clusters = min(self.tts_num_clusters, k)
            cluster_ids_tensor = self._kmeans_small(
                flat_actions,
                num_clusters=num_clusters,
                max_iter=self.tts_kmeans_iters,
            )

            cluster_counts_tensor = torch.bincount(cluster_ids_tensor, minlength=num_clusters)
            selected_cluster = int(torch.argmax(cluster_counts_tensor).item())

            mask = cluster_ids_tensor == selected_cluster
            candidate_indices = mask.nonzero(as_tuple=True)[0]
            selected_cluster_size = int(cluster_counts_tensor[selected_cluster].item())

            sub_dists = pairwise_l2[mask][:, mask]
            if selected_cluster_size <= 1:
                local_scores = torch.zeros(
                    selected_cluster_size,
                    dtype=pairwise_l2.dtype,
                    device=pairwise_l2.device,
                )
            else:
                local_scores = sub_dists.sum(dim=1) / (selected_cluster_size - 1)

            selection_stage = "cluster_rank_softmax"
            unimodal = False
            cluster_counts = "|".join(map(str, cluster_counts_tensor.detach().cpu().tolist()))
            cluster_ids = "|".join(map(str, cluster_ids_tensor.detach().cpu().tolist()))

        local_ranks, _ = self._build_tts_ranks(local_scores)
        rank_tau = max(1e-8, float(self.tts_rank_tau))

        logits = -local_ranks.to(torch.float32) / rank_tau
        local_probs = torch.softmax(logits, dim=0)

        selected_local_idx = int(torch.multinomial(local_probs, num_samples=1).item())
        selected_idx = int(candidate_indices[selected_local_idx].item())
        selected_rank = int(local_ranks[selected_local_idx].item())
        selected_prob = float(local_probs[selected_local_idx].item())

        rank_ids = torch.full((k,), -1, dtype=torch.long, device=avg_l2.device)
        rank_ids[candidate_indices] = local_ranks

        selection_probs = torch.zeros(k, dtype=torch.float32, device=avg_l2.device)
        selection_probs[candidate_indices] = local_probs

        metrics = {
            "tts_method": "cluster_rank_softmax",
            "selection_stage": selection_stage,
            "global_medoid_idx": global_medoid_idx,
            "unimodal": unimodal,
            "s_score": f"{s_score:.8f}",
            "num_clusters": num_clusters,
            "selected_cluster": selected_cluster,
            "selected_cluster_size": selected_cluster_size,
            "cluster_counts": cluster_counts,
            "cluster_ids": cluster_ids,
            "selected_idx": selected_idx,
            "selected_rank": selected_rank,
            "selected_prob": f"{selected_prob:.8f}",
            "rank_temperature": f"{float(self.tts_rank_tau):.8f}",
            "rank_ids": "|".join(map(str, rank_ids.detach().cpu().tolist())),
            "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
        }

        return selected_idx, pairwise_l2, avg_l2, metrics

    # tts add
    def _compute_global_medoid_and_guard(
        self,
        flat_actions: torch.Tensor,
        pairwise_l2: torch.Tensor,
    ):
        """
        KeyStone-style unimodality guard.

        s_score = ||mean(candidate_chunks) - global_medoid|| / median_pairwise_distance

        If s_score < tau, treat candidates as one cluster and use global medoid.
        """
        k = flat_actions.shape[0]

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            return 0, avg_l2, 0.0

        avg_l2 = pairwise_l2.sum(dim=1) / (k - 1)
        global_medoid_idx = int(torch.argmin(avg_l2).item())

        iu = torch.triu_indices(k, k, offset=1, device=pairwise_l2.device)
        median_d = pairwise_l2[iu[0], iu[1]].median()

        eps = 1e-8
        s_score = (
            torch.norm(flat_actions.mean(dim=0) - flat_actions[global_medoid_idx], p=2)
            / (median_d + eps)
        )

        return global_medoid_idx, avg_l2, float(s_score.item())

    # tts add
    def _kmeans_small(
        self,
        x: torch.Tensor,
        num_clusters: int,
        max_iter: int,
    ):
        """
        Small deterministic k-means for K action chunks.

        x: [K, D], CPU tensor.
        return: labels [K].

        Initialization:
        - first center = global medoid
        - next centers = farthest-first
        This avoids consuming random seeds during evaluation.
        """
        k = x.shape[0]
        c = min(max(1, int(num_clusters)), k)

        if c == 1:
            return torch.zeros(k, dtype=torch.long, device=x.device)

        dists = torch.cdist(x, x, p=2)
        first_idx = int(torch.argmin(dists.sum(dim=1)).item())

        selected = [first_idx]
        centers = [x[first_idx].clone()]

        for _ in range(1, c):
            center_tensor = torch.stack(centers, dim=0)
            dist_to_centers = torch.cdist(x, center_tensor, p=2).min(dim=1).values

            for idx in selected:
                dist_to_centers[idx] = -1.0

            next_idx = int(torch.argmax(dist_to_centers).item())
            selected.append(next_idx)
            centers.append(x[next_idx].clone())

        centers = torch.stack(centers, dim=0)
        labels = torch.full((k,), -1, dtype=torch.long, device=x.device)

        for _ in range(max_iter):
            new_labels = torch.cdist(x, centers, p=2).argmin(dim=1)

            if torch.equal(new_labels, labels):
                break

            labels = new_labels

            for cid in range(c):
                mask = labels == cid
                if mask.any():
                    centers[cid] = x[mask].mean(dim=0)
                else:
                    # Empty cluster: re-seed with point farthest from current centers.
                    dist_to_centers = torch.cdist(x, centers, p=2).min(dim=1).values
                    centers[cid] = x[int(torch.argmax(dist_to_centers).item())].clone()

        return labels

    # ttsv2 add
    def _select_tts_keystone(self, actions_stack: torch.Tensor):
        """
        KeyStone-style selector:
        1. Compute global medoid.
        2. Compute unimodality score s_score.
        3. If s_score < tau: return global medoid.
        4. Else: k-means into C clusters.
        5. Select the largest cluster.
        6. Return the medoid inside the largest cluster.
        """
        k = actions_stack.shape[0]
        flat_actions = actions_stack.reshape(k, -1)

        pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            
            ranks = torch.zeros(k, dtype=torch.long, device=pairwise_l2.device)
            selection_probs = torch.ones(k, dtype=torch.float32, device=pairwise_l2.device)
            
            metrics = {
                "tts_method": "keystone",
                "selection_stage": "single_sample",
                "global_medoid_idx": 0,
                "unimodal": True,
                "s_score": "",
                "num_clusters": 1,
                "selected_cluster": -1,
                "selected_cluster_size": k,
                "cluster_counts": "",
                "cluster_ids": "",
                "selected_idx": 0,
                "selected_rank": 0,
                "selected_prob": "1.00000000",
                "rank_temperature": "",
                "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
                "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
            }
            return 0, pairwise_l2, avg_l2, metrics

        global_medoid_idx, avg_l2, s_score = self._compute_global_medoid_and_guard(
            flat_actions=flat_actions,
            pairwise_l2=pairwise_l2,
        )

        # Unimodality guard: if candidates look like one cluster,
        # do not force k-means to split them.
        if s_score < self.tts_tau or self.tts_num_clusters <= 1:
            ranks, order = self._build_tts_ranks(avg_l2)
            selection_probs = torch.zeros(k, dtype=torch.float32, device=avg_l2.device)
            selection_probs[global_medoid_idx] = 1.0
            
            metrics = {
                "tts_method": "keystone",
                "selection_stage": "guard_global_medoid",
                "global_medoid_idx": global_medoid_idx,
                "unimodal": True,
                "s_score": f"{s_score:.8f}",
                "num_clusters": 1,
                "selected_cluster": -1,
                "selected_cluster_size": k,
                "cluster_counts": "",
                "cluster_ids": "",
                "selected_idx": global_medoid_idx,
                "selected_rank": int(ranks[global_medoid_idx].item()),
                "selected_prob": "1.00000000",
                "rank_temperature": "",
                "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
                "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
            }
            return global_medoid_idx, pairwise_l2, avg_l2, metrics

        c = min(self.tts_num_clusters, k)
        cluster_ids = self._kmeans_small(
            flat_actions,
            num_clusters=c,
            max_iter=self.tts_kmeans_iters,
        )

        cluster_counts = torch.bincount(cluster_ids, minlength=c)
        selected_cluster = int(torch.argmax(cluster_counts).item())

        mask = cluster_ids == selected_cluster
        idxs = mask.nonzero(as_tuple=True)[0]

        # Medoid inside the largest cluster, not the centroid itself.
        sub_dists = pairwise_l2[mask][:, mask]
        local_idx = int(torch.argmin(sub_dists.sum(dim=1)).item())
        best_idx = int(idxs[local_idx].item())
        
        ranks, order = self._build_tts_ranks(avg_l2)
        selection_probs = torch.zeros(k, dtype=torch.float32, device=avg_l2.device)
        selection_probs[best_idx] = 1.0

        metrics = {
            "tts_method": "keystone",
            "selection_stage": "cluster_medoid",
            "global_medoid_idx": global_medoid_idx,
            "unimodal": False,
            "s_score": f"{s_score:.8f}",
            "num_clusters": c,
            "selected_cluster": selected_cluster,
            "selected_cluster_size": int(cluster_counts[selected_cluster].item()),
            "cluster_counts": "|".join(map(str, cluster_counts.detach().cpu().tolist())),
            "cluster_ids": "|".join(map(str, cluster_ids.detach().cpu().tolist())),
            "selected_idx": best_idx,
            "selected_rank": int(ranks[best_idx].item()),
            "selected_prob": "1.00000000",
            "rank_temperature": "",
            "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
            "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
        }

        return best_idx, pairwise_l2, avg_l2, metrics
    
    # ttsv2 add
    def _log_tts_result(self, actions_stack, pairwise_l2, avg_l2, selected_idx, metrics):
        """
        Log TTS selector information.

        Important:
        - Fixed CSV schema for all methods.
        - No .npz saving.
        - Unsupported fields for a method are filled with empty string or -1.
        """
        if not self.tts_log_actions:
            return

        self.tts_dir.mkdir(parents=True, exist_ok=True)

        avg_l2_np = avg_l2.detach().cpu().numpy()

        selected_avg_l2 = float(avg_l2_np[selected_idx])
        min_avg_l2 = float(avg_l2_np.min())
        max_avg_l2 = float(avg_l2_np.max())
        mean_avg_l2 = float(avg_l2_np.mean())

        avg_list = "|".join(f"{float(x):.6f}" for x in avg_l2_np)

        global_medoid_idx = metrics.get("global_medoid_idx", "")
        global_medoid_avg_l2 = ""
        if global_medoid_idx != "":
            global_medoid_avg_l2 = f"{float(avg_l2_np[int(global_medoid_idx)]):.8f}"

        print(
            f"[TTS] episode={self.episode_count} "
            f"step={self.step_count} "
            f"method={metrics.get('tts_method', self.tts_method)} "
            f"stage={metrics.get('selection_stage', '')} "
            f"samples={actions_stack.shape[0]} "
            f"C={self.tts_num_clusters} "
            f"tau={self.tts_tau} "
            f"rank_tau={metrics.get('rank_temperature', '')} "
            f"kmeans_iters={self.tts_kmeans_iters} "
            f"selected={selected_idx} "
            f"selected_rank={metrics.get('selected_rank', '')} "
            f"selected_prob={metrics.get('selected_prob', '')} "
            f"global_medoid={global_medoid_idx} "
            f"unimodal={metrics.get('unimodal', '')} "
            f"s_score={metrics.get('s_score', '')} "
            f"selected_cluster={metrics.get('selected_cluster', '')} "
            f"cluster_size={metrics.get('selected_cluster_size', '')} "
            f"cluster_counts={metrics.get('cluster_counts', '')} "
            f"selected_avg_l2={selected_avg_l2:.6f} "
            f"global_medoid_avg_l2={global_medoid_avg_l2} "
            f"min_avg_l2={min_avg_l2:.6f} "
            f"max_avg_l2={max_avg_l2:.6f}"
        )

        summary_path = self.tts_dir / "summary.csv"
        write_header = not summary_path.exists()

        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    "episode",
                    "step",
                    "samples",
                    "tts_method",
                    "selection_stage",

                    "selected_idx",
                    "selected_rank",
                    "selected_prob",

                    "global_medoid_idx",
                    "global_medoid_avg_l2",

                    "unimodal",
                    "s_score",
                    "num_clusters",
                    "tau",
                    "kmeans_iters",

                    "selected_cluster",
                    "selected_cluster_size",
                    "cluster_counts",
                    "cluster_ids",

                    "rank_temperature",
                    "rank_ids",
                    "selection_probs",

                    "selected_avg_l2",
                    "min_avg_l2",
                    "max_avg_l2",
                    "mean_avg_l2",
                    "avg_l2",
                ])

            writer.writerow([
                self.episode_count,
                self.step_count,
                actions_stack.shape[0],
                metrics.get("tts_method", self.tts_method),
                metrics.get("selection_stage", ""),

                selected_idx,
                metrics.get("selected_rank", ""),
                metrics.get("selected_prob", ""),

                global_medoid_idx,
                global_medoid_avg_l2,

                metrics.get("unimodal", ""),
                metrics.get("s_score", ""),
                metrics.get("num_clusters", self.tts_num_clusters),
                self.tts_tau,
                self.tts_kmeans_iters,

                metrics.get("selected_cluster", ""),
                metrics.get("selected_cluster_size", ""),
                metrics.get("cluster_counts", ""),
                metrics.get("cluster_ids", ""),

                metrics.get("rank_temperature", ""),
                metrics.get("rank_ids", ""),
                metrics.get("selection_probs", ""),

                f"{selected_avg_l2:.8f}",
                f"{min_avg_l2:.8f}",
                f"{max_avg_l2:.8f}",
                f"{mean_avg_l2:.8f}",
                avg_list,
            ])

    def get_action(self, instruction: str = None) -> List[np.ndarray]:
        """Get action predictions from the model."""
        if len(self.obs_cache) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")
        
        if self.current_state is None:
            raise ValueError("No robot state available. Call update_obs first.")
        
        current_frame = self.obs_cache[-1]

        # Encode instruction with T5
        scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                        "The aloha robot is currently performing the following task: ")
        instruction = f"{scene_prefix}{self.current_instruction}"
        t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        # Build VLM inputs
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(instruction, first_frame_pil)
        
        # tts add 
        # Run inference
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']

        if self.tts_enable and self.tts_num_samples > 1:
            action_candidates = []
            frame_candidates = []

            for sample_idx in range(self.tts_num_samples):
                cand_frames, cand_actions = self._run_single_inference(
                    current_frame=current_frame,
                    t5_list=t5_list,
                    vlm_inputs=vlm_inputs,
                    num_inference_steps=num_inference_steps,
                )

                action_candidates.append(cand_actions.squeeze(0).detach().float().cpu())
                frame_candidates.append(cand_frames.detach().float().cpu() if cand_frames is not None else None)
            
            actions_stack = torch.stack(action_candidates, dim=0)
            selected_idx, pairwise_l2, avg_l2, metrics = self._select_tts_action(actions_stack)
            self._log_tts_result(actions_stack, pairwise_l2, avg_l2, selected_idx, metrics)

            predicted_actions = actions_stack[selected_idx].unsqueeze(0)
            predicted_frames = frame_candidates[selected_idx]
        else:
            predicted_frames, predicted_actions = self._run_single_inference(
                current_frame=current_frame,
                t5_list=t5_list,
                vlm_inputs=vlm_inputs,
                num_inference_steps=num_inference_steps,
            )
        # Run inference end

        # Save frame grid
        if predicted_frames is not None:
            if predicted_frames.dim() == 5:
                if predicted_frames.shape[1] == 3:
                    predicted_frames_viz = predicted_frames.permute(0, 2, 1, 3, 4)
                else:
                    predicted_frames_viz = predicted_frames
                
                # condition_frame_viz = current_frame.squeeze(0)
                # predicted_frames_viz = predicted_frames_viz.squeeze(0)
                
                condition_frame_viz = current_frame.squeeze(0).detach().cpu()
                predicted_frames_viz = predicted_frames_viz.squeeze(0).detach().cpu()
                
                self._save_frame_grid(condition_frame_viz, predicted_frames_viz)
                self.step_count += 1

        actions_real = predicted_actions.squeeze(0).cpu().numpy()
        self.prev_action = actions_real[-1].copy()
        self.action_cache.extend(actions_real)

        return actions_real

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        """Convert [C, H, W] tensor to PIL Image."""
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Build VLM inputs."""
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': instruction},
                    {'type': 'image', 'image': image},
                ]
            }
        ]
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device), 
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    def _load_normalization_stats(self):
        """Load action normalization stats."""
        try:
            stat_path = Path(__file__).parent / 'utils' / 'stat.json'
            with open(stat_path, 'r') as f:
                stat_data = yaml.safe_load(f) if stat_path.suffix in ['.yml', '.yaml'] else None
        except Exception:
            stat_data = None
        if stat_data is None:
            import json as _json
            with open(Path(__file__).parent / 'utils' / 'stat.json', 'r') as f:
                stat_data = _json.load(f)

        stats = stat_data.get('robotwin2')
        if stats is None:
            raise ValueError('Normalization stats not found')
        self.action_min = torch.tensor(stats['min'], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(stats['max'], dtype=torch.float32, device=self.device)
        self.action_range = self.action_max - self.action_min

    def _normalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0,1]."""
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
        return norm.reshape(shape)

    def _denormalize_actions(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0,1]."""
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        denorm = y_flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return denorm.reshape(shape)
    
    def _create_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:
        """Create horizontal grid."""
        def tensor_to_numpy(tensor):
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            tensor = tensor.detach().cpu().float()
            tensor = torch.clamp(tensor, 0, 1)
            return (tensor.numpy() * 255).astype(np.uint8)
        
        condition_np = tensor_to_numpy(condition_frame)
        predicted_np = []
        num_pred_frames = predicted_frames.shape[0]
        for i in range(num_pred_frames):
            frame_np = tensor_to_numpy(predicted_frames[i])
            predicted_np.append(frame_np)
        
        while len(predicted_np) < 4:
            predicted_np.append(predicted_np[-1] if predicted_np else condition_np)
        
        all_frames = [condition_np] + predicted_np[:4]
        grid_image = np.concatenate(all_frames, axis=1)
        
        return Image.fromarray(grid_image)
    
    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        """Save frame grid to disk."""
        if not self.save_images:
            return
        
        try:
            grid_image = self._create_frame_grid(condition_frame, predicted_frames)
            filename = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.png"
            save_path = self.save_dir / filename
            grid_image.save(save_path)
            logger.info(f"Saved frame grid to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save frame grid: {e}")

def encode_obs(observation):
    """Post-Process Observation"""
    return observation

def get_model(usr_args):
    """
    Initialize Motus model.
    
    Args:
        usr_args: Arguments from eval script (must include wan_path and vlm_path)
    """
    checkpoint_path = usr_args.get('ckpt_setting')
    wan_path = usr_args.get('wan_path')  # Passed from eval.sh or auto_eval.sh
    vlm_path = usr_args.get('vlm_path')  # Passed from eval.sh or auto_eval.sh
    
    if not wan_path:
        raise ValueError("wan_path not provided in usr_args")
    
    if not vlm_path:
        raise ValueError("vlm_path not provided in usr_args")
    
    policy_dir = Path(__file__).parent
    config_path = policy_dir / "utils" / "robotwin.yml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # tts add
    tts_enable = _as_bool(usr_args.get("tts_enable", DEFAULT_TTS_ENABLE))
    tts_num_samples = int(usr_args.get("tts_num_samples", DEFAULT_TTS_NUM_SAMPLES))

    tts_method = str(usr_args.get("tts_method", DEFAULT_TTS_METHOD))
    tts_num_clusters = int(usr_args.get("tts_num_clusters", DEFAULT_TTS_NUM_CLUSTERS))
    tts_tau = float(usr_args.get("tts_tau", DEFAULT_TTS_TAU))
    tts_rank_tau = float(usr_args.get("tts_rank_tau", DEFAULT_TTS_RANK_TAU))
    tts_kmeans_iters = int(usr_args.get("tts_kmeans_iters", DEFAULT_TTS_KMEANS_ITERS))

    tts_log_actions = _as_bool(usr_args.get("tts_log_actions", DEFAULT_TTS_LOG_ACTIONS))

    # Deprecated/no-op; kept for old shell commands.
    tts_save_full_actions = _as_bool(
        usr_args.get("tts_save_full_actions", DEFAULT_TTS_SAVE_FULL_ACTIONS)
    )
    
    # tts add
    policy = MotusPolicy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        config_path=str(config_path),
        device=device,
        log_dir=usr_args.get('log_dir'),
        task_name=usr_args.get('task_name'),
        tts_enable=tts_enable,
        tts_num_samples=tts_num_samples,
        tts_method=tts_method,
        tts_num_clusters=tts_num_clusters,
        tts_tau=tts_tau,
        tts_rank_tau=tts_rank_tau,
        tts_kmeans_iters=tts_kmeans_iters,
        tts_log_actions=tts_log_actions,
        tts_save_full_actions=tts_save_full_actions,        
    )
    
    return policy

def eval(TASK_ENV, model, observation):
    """Evaluation function."""
    obs = encode_obs(observation)
    
    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)

    actions = model.get_action()
    
    for action in actions:
        TASK_ENV.take_action(action, action_type='qpos')

def reset_model(model):  
    """Reset model cache at episode start."""
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.is_first_step = True
    model.prev_action = None
    model.episode_count += 1
    model.step_count = 0
    logger.info(f"Model reset completed for episode {model.episode_count}")
```



##### 评估

实验：L2 min \+ Cluster \+ Random，16，8

先8，跑通再上16



8采样

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_cluster_rank_softmax_n8_tau03_ranktau1_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 0 --tts --tts-num-samples 8 \
  --tts-method cluster_rank_softmax \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-rank-tau 1.0 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 5298

PID: 5298

LOG: logs/scan\_object\_tts\_cluster\_rank\_softmax\_n8\_tau03\_ranktau1\_20260520\_213919\.log

62

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZDFkMDM4YTliNmYwODdmZTc3MWFhM2JlYWRhZjczNTJfZGJmOTY2Nzg4NGFhZThjMTA1OThmZjljYzU3N2ExZmZfSUQ6NzY0MjE4MTY4NDgxMzYxNDAwOF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



16采样

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_cluster_rank_softmax_n16_tau03_ranktau1_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 16 \
  --tts-method cluster_rank_softmax \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-rank-tau 1.0 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[2\] 6893

PID: 6893

LOG: logs/scan\_object\_tts\_cluster\_rank\_softmax\_n16\_tau03\_ranktau1\_20260520\_214503\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZWI4MzVhOGJhZTM4MjM0NDFhZjAyMDI4NjFhYTFhNmNfNTljNTgxMGM0YzdlNTNlNzg0NTM1M2I3YTE3ODZhYzdfSUQ6NzY0MjE2NTEwMDMwNTAwOTYzNl8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



#### v4 

##### 实现

/home/ubuntu/workspace/RoboTwin/policy/Motus/eval\.sh

```Bash
#!/bin/bash
# Single task evaluation script for Motus policy on RoboTwin platform
# impl and exp of tts and opd

# ============================================================================
# Single Task Configuration - MODIFY THESE
# ============================================================================

#tts add
TASK_NAME="${1:-click_alarmclock}"
if [ $# -gt 0 ]; then
    shift
fi

# ttsv2 add
GPU_ID=0

# Test-time scaling
TTS_ENABLE=False
TTS_NUM_SAMPLES=8

# Selection method:
#   global_medoid:          average-L2/global-medoid selection
#   keystone:               KeyStone-style guard + kmeans + largest-cluster medoid
#   rank_softmax:           rank-based stochastic selection, P(i)=softmax(-rank_i/tau)
#   cluster_rank_softmax:   guard + kmeans largest-cluster + local rank-softmax
TTS_METHOD="global_medoid"

# KeyStone defaults
TTS_NUM_CLUSTERS=2

# tau meaning:
#   keystone:     unimodality guard threshold, default 0.3
#   rank_softmax: rank-softmax temperature, recommended 1.0
TTS_TAU=0.3

# Rank-softmax temperature for cluster_rank_softmax.
# Keep this separate from TTS_TAU, which is used as the unimodality guard.
TTS_RANK_TAU=1.0

# Batched TTS candidate generation.
# 1 keeps the old sequential behavior.
TTS_BATCH_SIZE=1

# Whether to decode predicted videos during TTS candidate generation.
# False is faster and enough for action-only selectors.
TTS_DECODE_VIDEO=False

TTS_KMEANS_ITERS=10

# Logging
TTS_LOG_ACTIONS=True

# Deprecated/no-op in the new Python implementation.
# 保留这个参数只是为了兼容旧命令；新版本不再保存 .npz。
TTS_SAVE_FULL_ACTIONS=False
# ttsv2 add end

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --tts)
            TTS_ENABLE=True
            shift
            ;;
        --no-tts)
            TTS_ENABLE=False
            shift
            ;;
        --tts-num-samples)
            TTS_NUM_SAMPLES="$2"
            shift 2
            ;;
        --tts-method)
            TTS_METHOD="$2"
            shift 2
            ;;
        --tts-num-clusters)
            TTS_NUM_CLUSTERS="$2"
            shift 2
            ;;
        --tts-tau)
            TTS_TAU="$2"
            shift 2
            ;;
        --tts-rank-tau)
            TTS_RANK_TAU="$2"
            shift 2
            ;;      
        --tts-batch-size)
            TTS_BATCH_SIZE="$2"
            shift 2
            ;;
        --tts-decode-video)
            TTS_DECODE_VIDEO=True
            shift
            ;;
        --no-tts-decode-video)
            TTS_DECODE_VIDEO=False
            shift
            ;;  
        --tts-kmeans-iters)
            TTS_KMEANS_ITERS="$2"
            shift 2
            ;;            
        --tts-log-actions)
            TTS_LOG_ACTIONS="$2"
            shift 2
            ;;
        --tts-save-full-actions)
            TTS_SAVE_FULL_ACTIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# GPU_ID=0                       # GPU to use

# ============================================================================
# Script starts here
# ============================================================================
echo "Starting single task evaluation at $(date)"

# Get script directory (policy/Motus/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"

# ============================================================================
# Load Configuration from paths_config.yml
# ============================================================================
CONFIG_FILE="${POLICY_DIR}/paths_config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please create paths_config.yml with required paths."
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"

# Parse YAML (improved - remove comments and extra whitespace)
ROBOTWIN_ROOT=$(grep "^robotwin_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CONDA_ENV=$(grep "^conda_env:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CHECKPOINT_PATH=$(grep "^checkpoint_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
WAN_PATH=$(grep "^wan_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
VLM_PATH=$(grep "^vlm_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Optional configurations
TASK_CONFIG=$(grep "^task_config:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
SEED=$(grep "^seed:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Default values
TASK_CONFIG=${TASK_CONFIG:-"demo_randomized"}
SEED=${SEED:-"42"}
POLICY_NAME="Motus"

# ============================================================================
# Validation
# ============================================================================
if [ -z "$ROBOTWIN_ROOT" ]; then
    echo "Error: robotwin_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CONDA_ENV" ]; then
    echo "Error: conda_env is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: checkpoint_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$WAN_PATH" ]; then
    echo "Error: wan_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$VLM_PATH" ]; then
    echo "Error: vlm_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$ROBOTWIN_ROOT" ]; then
    echo "Error: RoboTwin root not found: $ROBOTWIN_ROOT"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$WAN_PATH" ]; then
    echo "Error: WAN path not found: $WAN_PATH"
    exit 1
fi

if [ ! -d "$VLM_PATH" ]; then
    echo "Error: VLM path not found: $VLM_PATH"
    exit 1
fi

cd "$ROBOTWIN_ROOT" || exit 1

# 处理conda报错
#######

# # Activate conda
# if ! command -v conda &> /dev/null; then
#     echo "Error: conda not found."
#     exit 1
# fi

# eval "$(conda shell.bash hook)"
# conda activate "$CONDA_ENV"

# Activate conda
# export PATH="/home/sumita-mana/anaconda3/bin:$PATH"
# source /home/sumita-mana/anaconda3/etc/profile.d/conda.sh

export PATH="/home/ubuntu/miniconda3/bin:$PATH"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found."
    exit 1
fi

conda activate "$CONDA_ENV"

#######

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

# Set environment
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create logs directory
LOG_DIR="${POLICY_DIR}/logs_single_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ckpt_setting="${CHECKPOINT_PATH}"
log_file="${LOG_DIR}/${TASK_NAME}.log"

echo ""
echo "================================================================"
echo "Single Task Evaluation Configuration"
echo "================================================================"
echo "Task Name:         $TASK_NAME"
echo "GPU:               $GPU_ID"
echo "----------------------------------------------------------------"
echo "RoboTwin Root:     $ROBOTWIN_ROOT"
echo "Policy Dir:        $POLICY_DIR"
echo "Checkpoint:        $CHECKPOINT_PATH"
echo "WAN Path:          $WAN_PATH"
echo "VLM Path:          $VLM_PATH"
echo "Task Config:       $TASK_CONFIG"
echo "Seed:              $SEED"
echo "Log File:          $log_file"
echo "TTS Enable:        $TTS_ENABLE"
echo "TTS Num Samples:   $TTS_NUM_SAMPLES"
echo "TTS Method:        $TTS_METHOD"
echo "TTS Num Clusters:  $TTS_NUM_CLUSTERS"
echo "TTS Tau:           $TTS_TAU"
echo "TTS Rank Tau:      $TTS_RANK_TAU"
echo "TTS Batch Size:    $TTS_BATCH_SIZE"
echo "TTS Decode Video:  $TTS_DECODE_VIDEO"
echo "TTS KMeans Iters:  $TTS_KMEANS_ITERS"
echo "TTS Log Actions:   $TTS_LOG_ACTIONS"
echo "TTS Save Full:     $TTS_SAVE_FULL_ACTIONS (deprecated; npz disabled)"
echo "================================================================"
echo ""

# Run evaluation with WAN_PATH passed as argument
echo "Starting evaluation..."

# 处理环境进入问题
#############

echo "Python executable: ${CONDA_ENV}/bin/python"
"${CONDA_ENV}/bin/python" - <<'PY'
import sys
print("sys.executable =", sys.executable)
import sapien
print("sapien ok =", sapien.__file__)
import importlib
m = importlib.import_module("sapien.core")
print("sapien.core ok =", m)
PY

PYTHONWARNINGS=ignore::UserWarning \
"${CONDA_ENV}/bin/python" script/eval_policy.py \
    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${SEED}" \
    --policy_name "${POLICY_NAME}" \
    --log_dir "${LOG_DIR}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}" \
    --tts_enable "${TTS_ENABLE}" \
    --tts_num_samples "${TTS_NUM_SAMPLES}" \
    --tts_method "${TTS_METHOD}" \
    --tts_num_clusters "${TTS_NUM_CLUSTERS}" \
    --tts_tau "${TTS_TAU}" \
    --tts_rank_tau "${TTS_RANK_TAU}" \
    --tts_batch_size "${TTS_BATCH_SIZE}" \
    --tts_decode_video "${TTS_DECODE_VIDEO}" \
    --tts_kmeans_iters "${TTS_KMEANS_ITERS}" \
    --tts_log_actions "${TTS_LOG_ACTIONS}" \
    --tts_save_full_actions "${TTS_SAVE_FULL_ACTIONS}" \
    2>&1 | tee "$log_file"

#############

exit_code=${PIPESTATUS[0]}

echo ""
echo "================================================================"
if [ $exit_code -eq 0 ]; then
    echo "✅ Task $TASK_NAME completed successfully"
    echo "================================================================"
    exit 0
else
    echo "❌ Task $TASK_NAME failed with exit code $exit_code"
    echo "================================================================"
    echo "Log file: $log_file"
    exit 1
fi
```

/home/ubuntu/workspace/RoboTwin/policy/Motus/deploy\_policy\.py

```Python
# Motus Policy for RoboTwin
# impl and exp of tts and opd

# for save csv
import csv

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add model paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

from models.motus import Motus, MotusConfig

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel
from utils.image_utils import resize_with_padding

logger = logging.getLogger(__name__)

# =========================
# Test-Time Scaling Defaults
# =========================
DEFAULT_TTS_ENABLE = False
DEFAULT_TTS_NUM_SAMPLES = 8

# Methods:
#   global_medoid:          average-L2/global-medoid selection
#   keystone:               unimodality guard + kmeans + largest-cluster medoid
#   rank_softmax:           rank-based stochastic selection, P(i)=softmax(-rank_i/tau)
#   cluster_rank_softmax:   unimodality guard + kmeans largest-cluster + local rank-softmax
DEFAULT_TTS_METHOD = "global_medoid"

# TTS defaults
DEFAULT_TTS_NUM_CLUSTERS = 2

# tau meaning:
#   keystone:               unimodality guard threshold
#   cluster_rank_softmax:   unimodality guard threshold
#   rank_softmax:           rank-softmax temperature, kept for backward compatibility
DEFAULT_TTS_TAU = 0.3

# tts v3 add
# Separate rank-softmax temperature for cluster_rank_softmax.
# This avoids overloading DEFAULT_TTS_TAU, whose default is tuned as a guard threshold.
DEFAULT_TTS_RANK_TAU = 1.0

# Batched candidate generation. 1 keeps the old sequential behavior.
DEFAULT_TTS_BATCH_SIZE = 1

# Decode predicted frames during TTS candidate generation.
# False is faster and enough for action-only selectors.
DEFAULT_TTS_DECODE_VIDEO = False

DEFAULT_TTS_KMEANS_ITERS = 10

# Logging
DEFAULT_TTS_LOG_ACTIONS = True

# Deprecated/no-op: kept only for backward compatibility with old scripts.
# New implementation does not save .npz files.
DEFAULT_TTS_SAVE_FULL_ACTIONS = False

# tts add
def _as_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in ["1", "true", "yes", "y", "on"]

class MotusPolicy:
    """
    Motus Policy wrapper for RoboTwin evaluation.
    Implements the joint video-action diffusion model for robotic control.
    """
    
    def __init__(
        self, 
        checkpoint_path: str, 
        config_path: str, 
        wan_path: str, 
        vlm_path: str, 
        device: str = "cuda", 
        log_dir: Optional[str] = None, 
        task_name: Optional[str] = None, 
        tts_enable: bool = DEFAULT_TTS_ENABLE,
        tts_num_samples: int = DEFAULT_TTS_NUM_SAMPLES,
        tts_method: str = DEFAULT_TTS_METHOD,
        tts_num_clusters: int = DEFAULT_TTS_NUM_CLUSTERS,
        tts_tau: float = DEFAULT_TTS_TAU,
        tts_rank_tau: float = DEFAULT_TTS_RANK_TAU,
        tts_batch_size: int = DEFAULT_TTS_BATCH_SIZE,
        tts_decode_video: bool = DEFAULT_TTS_DECODE_VIDEO,
        tts_kmeans_iters: int = DEFAULT_TTS_KMEANS_ITERS,
        tts_log_actions: bool = DEFAULT_TTS_LOG_ACTIONS,
        tts_save_full_actions: bool = DEFAULT_TTS_SAVE_FULL_ACTIONS,             
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
        
        # Initialize model WITHOUT loading pretrained backbones
        self.model = self._load_model()

        # Initialize T5 encoder for language embeddings (WAN text encoder)
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(self.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.wan_path, 'google', 'umt5-xxl'),
        )

        # Initialize VLM processor from vlm_path (for tokenization only, weights from checkpoint)
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)
        
        # Initialize observation cache
        self.obs_cache = deque(maxlen=1)
        self.action_cache = deque()
        
        # Model state
        self.current_state = None
        self.current_state_norm = None
        self.is_first_step = True
        self.prev_action = None

        # Load normalization stats
        self._load_normalization_stats()
        
        # Initialize image saving
        self.save_images = True
        
        
        # tts add
        
        base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        task_dir_name = task_name or os.environ.get('TASK_NAME') or "default_task"

        self.log_dir = Path(base_log_dir)
        self.task_dir_name = task_dir_name

        self.save_dir = self.log_dir / "images" / task_dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.tts_enable = _as_bool(tts_enable)
        self.tts_num_samples = max(1, int(tts_num_samples))

        self.tts_method = str(tts_method).strip().lower()
        
        allowed_tts_methods = {
            "global_medoid",
            "keystone",
            "rank_softmax",
            "cluster_rank_softmax",
        }
        
        if self.tts_method not in allowed_tts_methods:
            raise ValueError(
                f"Unknown tts_method={self.tts_method}. "
                f"Allowed methods: {sorted(allowed_tts_methods)}"
            )

        self.tts_num_clusters = max(1, int(tts_num_clusters))
        self.tts_tau = float(tts_tau)
        self.tts_rank_tau = float(tts_rank_tau)
        self.tts_batch_size = max(1, int(tts_batch_size))
        self.tts_decode_video = _as_bool(tts_decode_video)
        self.tts_kmeans_iters = max(1, int(tts_kmeans_iters))

        self.tts_log_actions = _as_bool(tts_log_actions)

        # Deprecated/no-op. Kept so old command lines still parse.
        self.tts_save_full_actions = _as_bool(tts_save_full_actions)

        self.tts_dir = self.log_dir / "tts" / task_dir_name

        if self.tts_enable and self.tts_log_actions:
            self.tts_dir.mkdir(parents=True, exist_ok=True)

        self.episode_count = 0
        self.step_count = 0

        print(
            f"[TTS] enable={self.tts_enable}, "
            f"num_samples={self.tts_num_samples}, "
            f"method={self.tts_method}, "
            f"num_clusters={self.tts_num_clusters}, "
            f"tau={self.tts_tau}, "
            f"rank_tau={self.tts_rank_tau}, "
            f"batch_size={self.tts_batch_size}, "
            f"decode_video={self.tts_decode_video}, "
            f"kmeans_iters={self.tts_kmeans_iters}, "
            f"log_actions={self.tts_log_actions}, "
            f"save_full_actions={self.tts_save_full_actions} (deprecated/no-op), "
            f"tts_dir={self.tts_dir}"
        )

        logger.info("Motus Policy initialized successfully")

    def set_instruction(self, instruction: str):
        """Set the current instruction for the policy."""
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def _load_model(self) -> Motus:
        """Load the Motus model without pretrained backbones, then load checkpoint."""
        logger.info(f"Initializing Motus model from config (no pretrained backbones)")

        config = self._create_model_config()
        
        # Initialize model from config WITHOUT loading pretrained weights
        model = Motus(config)
        model = model.to(self.device)
        
        # Load checkpoint weights
        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info("Model checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        model.eval()
        return model
    
    def _create_model_config(self) -> MotusConfig:
        """Create model configuration from yaml config - inference mode."""
        common = self.config_dict['common']
        model_cfg = self.config_dict['model']

        # Use paths passed to constructor
        vae_path = os.path.join(self.wan_path, "Wan2.2_VAE.pth")
        vlm_checkpoint_path = self.vlm_path

        hidden_size = model_cfg['action_expert']['hidden_size']
        ffn_multiplier = model_cfg['action_expert']['ffn_dim_multiplier']

        config = MotusConfig(
            # Paths for config loading only (no weights loaded)
            wan_checkpoint_path=self.wan_path,
            vae_path=vae_path,
            wan_config_path=self.wan_path,
            video_precision='bfloat16',
            vlm_checkpoint_path=vlm_checkpoint_path,
            
            # Understanding expert config
            und_expert_hidden_size=512,
            und_expert_ffn_dim_multiplier=4,
            und_expert_norm_eps=1e-5,
            und_layers_to_extract=None,
            vlm_adapter_input_dim=2048,
            vlm_adapter_projector_type="mlp3x_silu",
            
            # Model architecture
            num_layers=30,
            action_state_dim=common['state_dim'],
            action_dim=common['action_dim'],
            action_expert_dim=hidden_size,
            action_expert_ffn_dim_multiplier=ffn_multiplier,
            action_expert_norm_eps=1e-6,
            
            # Training config
            global_downsample_rate=common['global_downsample_rate'],
            video_action_freq_ratio=common['video_action_freq_ratio'],
            num_video_frames=common['num_video_frames'],
            video_loss_weight=1.0,
            action_loss_weight=1.0,
            
            # Inference config
            batch_size=1,
            video_height=common['video_height'],
            video_width=common['video_width'],
            
            # Don't load pretrained backbones - will load full model from checkpoint
            load_pretrained_backbones=False,
            training_mode='finetune',
        )

        return config
    
    def update_obs(self, observation: Dict[str, Any]):
        """Update observation cache with new observation."""
        # Extract visual observations
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                head_img = obs_data['head_camera']['rgb']
                left_img = obs_data['left_camera']['rgb']
                right_img = obs_data['right_camera']['rgb']
                
                left_img_resized = cv2.resize(left_img, (160, 120))
                right_img_resized = cv2.resize(right_img, (160, 120))
                bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
                image = np.concatenate([head_img, bottom_row], axis=0)
            else:
                raise ValueError("Missing camera data")
        elif 'head_camera' in observation:
            image = observation['head_camera']
        elif 'image' in observation:
            image = observation['image']
        else:
            raise ValueError("No visual observation found")

        target_size = (self.config_dict['common']['video_height'],
                      self.config_dict['common']['video_width'])

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image

        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)
        
        self.obs_cache.append(image_tensor.to(self.device))

        # Extract robot state
        state = observation['joint_action']['vector']

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()

        self.current_state = state_tensor.to(self.device)
        self.current_state_norm = self._normalize_actions(self.current_state).to(self.device)
    
    # tts add
    def _run_single_inference(
        self,
        current_frame,
        t5_list,
        vlm_inputs,
        num_inference_steps,
        decode_video=True,
    ):
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame,
                state=self.current_state,
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs],
                decode_video=decode_video,
            )
        return predicted_frames, predicted_actions

    # tts acceleration add
    def _run_batched_inference(
        self,
        current_frame,
        t5_list,
        vlm_inputs,
        num_inference_steps,
        batch_size,
        decode_video=False,
    ):
        current_frame_b = current_frame.repeat(batch_size, 1, 1, 1)
        state_b = self.current_state.repeat(batch_size, 1)

        # Motus inference_step accepts list-form per-sample language embeddings.
        t5_list_b = [t5_list[0] for _ in range(batch_size)]

        # Motus VLM path accepts a list of per-sample VLM input dicts and batches them internally.
        vlm_inputs_b = [vlm_inputs for _ in range(batch_size)]

        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame_b,
                state=state_b,
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list_b,
                vlm_inputs=vlm_inputs_b,
                decode_video=decode_video,
            )
        return predicted_frames, predicted_actions
    
    # tts v2 add
    def _select_tts_action(self, actions_stack: torch.Tensor):
        """
        Dispatch TTS selector according to self.tts_method.

        actions_stack: [K, H, D], CPU tensor.
        """
        if self.tts_method == "global_medoid":
            return self._select_tts_global_medoid(actions_stack)

        if self.tts_method == "keystone":
            return self._select_tts_keystone(actions_stack)  
        
        if self.tts_method == "rank_softmax":
            return self._select_tts_rank_softmax(actions_stack)

        if self.tts_method == "cluster_rank_softmax":
            return self._select_tts_cluster_rank_softmax(actions_stack)

        raise ValueError(f"Unknown tts_method={self.tts_method}")    

    # ttsv2 add
    def _build_tts_ranks(self, avg_l2: torch.Tensor):
        """
        Build ranks from avg_l2.

        avg_l2: lower is better.
        return:
            ranks: [K], rank 0 is best / smallest avg_l2.
            order: [K], candidate indices sorted by avg_l2 ascending.
        """
        order = torch.argsort(avg_l2, descending=False)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(len(avg_l2), dtype=order.dtype, device=avg_l2.device)
        return ranks, order

    # tts add
    def _select_tts_global_medoid(self, actions_stack: torch.Tensor):
        """
        Current baseline method:
        flatten each action chunk, compute pairwise L2, select the global medoid.
        """
        k = actions_stack.shape[0]
        flat_actions = actions_stack.reshape(k, -1)

        pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            best_idx = 0
        else:
            avg_l2 = pairwise_l2.sum(dim=1) / (k - 1)
            best_idx = int(torch.argmin(avg_l2).item())

        # ttsv2 add
        ranks, order = self._build_tts_ranks(avg_l2)
        selection_probs = torch.zeros(k, dtype=torch.float32, device=avg_l2.device)
        selection_probs[best_idx] = 1.0
        
        # ttsv2 add
        metrics = {
            "tts_method": "global_medoid",
            "selection_stage": "global_medoid",
            "global_medoid_idx": best_idx,
            "unimodal": True,
            "s_score": "",
            "num_clusters": 1,
            "selected_cluster": -1,
            "selected_cluster_size": k,
            "cluster_counts": "",
            "cluster_ids": "",
            "selected_idx": best_idx,
            "selected_rank": int(ranks[best_idx].item()),
            "selected_prob": "1.00000000",
            "rank_temperature": "",
            "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
            "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
        }

        return best_idx, pairwise_l2, avg_l2, metrics

    # ttsv2 add
    def _select_tts_rank_softmax(self, actions_stack: torch.Tensor):
        """
        Rank-softmax selector:
        1. Flatten each action chunk.
        2. Compute pairwise L2.
        3. Compute avg_l2 for each candidate.
        4. Rank candidates by avg_l2.
        5. Sample selected_idx from softmax(-rank / tau).

        tau is self.tts_tau for this method.
        """
        k = actions_stack.shape[0]
        flat_actions = actions_stack.reshape(k, -1)

        pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            selected_idx = 0
            global_medoid_idx = 0
            ranks = torch.zeros(k, dtype=torch.long, device=pairwise_l2.device)
            selection_probs = torch.ones(k, dtype=torch.float32, device=pairwise_l2.device)
            selected_rank = 0
            selected_prob = 1.0
        else:
            avg_l2 = pairwise_l2.sum(dim=1) / (k - 1)

            ranks, order = self._build_tts_ranks(avg_l2)
            global_medoid_idx = int(order[0].item())

            tau = max(1e-8, float(self.tts_tau))

            logits = -ranks.to(torch.float32) / tau
            selection_probs = torch.softmax(logits, dim=0)

            selected_idx = int(torch.multinomial(selection_probs, num_samples=1).item())
            selected_rank = int(ranks[selected_idx].item())
            selected_prob = float(selection_probs[selected_idx].item())

        metrics = {
            "tts_method": "rank_softmax",
            "selection_stage": "rank_softmax_sample",
            "global_medoid_idx": global_medoid_idx,
            "unimodal": "",
            "s_score": "",
            "num_clusters": 1,
            "selected_cluster": -1,
            "selected_cluster_size": k,
            "cluster_counts": "",
            "cluster_ids": "",
            "selected_idx": selected_idx,
            "selected_rank": selected_rank,
            "selected_prob": f"{selected_prob:.8f}",
            "rank_temperature": f"{float(self.tts_tau):.8f}",
            "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
            "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
        }

        return selected_idx, pairwise_l2, avg_l2, metrics    

    # ttsv3 add
    def _select_tts_cluster_rank_softmax(self, actions_stack: torch.Tensor):
        """
        L2 min + Cluster + Random selector.

        1. Compute global average-L2 medoid scores.
        2. Run the same unimodality guard used by KeyStone.
        3. If the guard says the samples are unimodal, sample from all candidates by rank-softmax.
        4. Otherwise, run deterministic k-means, keep the largest cluster, and sample only inside
        that cluster by local intra-cluster rank-softmax.

        self.tts_tau is the unimodality guard threshold.
        self.tts_rank_tau is the rank-softmax temperature.
        """
        k = actions_stack.shape[0]
        flat_actions = actions_stack.reshape(k, -1)

        pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            ranks = torch.zeros(k, dtype=torch.long, device=pairwise_l2.device)
            selection_probs = torch.ones(k, dtype=torch.float32, device=pairwise_l2.device)

            metrics = {
                "tts_method": "cluster_rank_softmax",
                "selection_stage": "single_sample",
                "global_medoid_idx": 0,
                "unimodal": True,
                "s_score": "",
                "num_clusters": 1,
                "selected_cluster": -1,
                "selected_cluster_size": k,
                "cluster_counts": "",
                "cluster_ids": "",
                "selected_idx": 0,
                "selected_rank": 0,
                "selected_prob": "1.00000000",
                "rank_temperature": f"{float(self.tts_rank_tau):.8f}",
                "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
                "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
            }
            return 0, pairwise_l2, avg_l2, metrics

        global_medoid_idx, avg_l2, s_score = self._compute_global_medoid_and_guard(
            flat_actions=flat_actions,
            pairwise_l2=pairwise_l2,
        )

        use_all_candidates = s_score < self.tts_tau or self.tts_num_clusters <= 1

        if use_all_candidates:
            candidate_indices = torch.arange(k, dtype=torch.long, device=avg_l2.device)
            local_scores = avg_l2

            selection_stage = "guard_rank_softmax" if s_score < self.tts_tau else "no_cluster_rank_softmax"
            unimodal = bool(s_score < self.tts_tau)
            num_clusters = 1
            selected_cluster = -1
            selected_cluster_size = k
            cluster_counts = ""
            cluster_ids = ""
        else:
            num_clusters = min(self.tts_num_clusters, k)
            cluster_ids_tensor = self._kmeans_small(
                flat_actions,
                num_clusters=num_clusters,
                max_iter=self.tts_kmeans_iters,
            )

            cluster_counts_tensor = torch.bincount(cluster_ids_tensor, minlength=num_clusters)
            selected_cluster = int(torch.argmax(cluster_counts_tensor).item())

            mask = cluster_ids_tensor == selected_cluster
            candidate_indices = mask.nonzero(as_tuple=True)[0]
            selected_cluster_size = int(cluster_counts_tensor[selected_cluster].item())

            sub_dists = pairwise_l2[mask][:, mask]
            if selected_cluster_size <= 1:
                local_scores = torch.zeros(
                    selected_cluster_size,
                    dtype=pairwise_l2.dtype,
                    device=pairwise_l2.device,
                )
            else:
                local_scores = sub_dists.sum(dim=1) / (selected_cluster_size - 1)

            selection_stage = "cluster_rank_softmax"
            unimodal = False
            cluster_counts = "|".join(map(str, cluster_counts_tensor.detach().cpu().tolist()))
            cluster_ids = "|".join(map(str, cluster_ids_tensor.detach().cpu().tolist()))

        local_ranks, _ = self._build_tts_ranks(local_scores)
        rank_tau = max(1e-8, float(self.tts_rank_tau))

        logits = -local_ranks.to(torch.float32) / rank_tau
        local_probs = torch.softmax(logits, dim=0)

        selected_local_idx = int(torch.multinomial(local_probs, num_samples=1).item())
        selected_idx = int(candidate_indices[selected_local_idx].item())
        selected_rank = int(local_ranks[selected_local_idx].item())
        selected_prob = float(local_probs[selected_local_idx].item())

        rank_ids = torch.full((k,), -1, dtype=torch.long, device=avg_l2.device)
        rank_ids[candidate_indices] = local_ranks

        selection_probs = torch.zeros(k, dtype=torch.float32, device=avg_l2.device)
        selection_probs[candidate_indices] = local_probs

        metrics = {
            "tts_method": "cluster_rank_softmax",
            "selection_stage": selection_stage,
            "global_medoid_idx": global_medoid_idx,
            "unimodal": unimodal,
            "s_score": f"{s_score:.8f}",
            "num_clusters": num_clusters,
            "selected_cluster": selected_cluster,
            "selected_cluster_size": selected_cluster_size,
            "cluster_counts": cluster_counts,
            "cluster_ids": cluster_ids,
            "selected_idx": selected_idx,
            "selected_rank": selected_rank,
            "selected_prob": f"{selected_prob:.8f}",
            "rank_temperature": f"{float(self.tts_rank_tau):.8f}",
            "rank_ids": "|".join(map(str, rank_ids.detach().cpu().tolist())),
            "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
        }

        return selected_idx, pairwise_l2, avg_l2, metrics

    # tts add
    def _compute_global_medoid_and_guard(
        self,
        flat_actions: torch.Tensor,
        pairwise_l2: torch.Tensor,
    ):
        """
        KeyStone-style unimodality guard.

        s_score = ||mean(candidate_chunks) - global_medoid|| / median_pairwise_distance

        If s_score < tau, treat candidates as one cluster and use global medoid.
        """
        k = flat_actions.shape[0]

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            return 0, avg_l2, 0.0

        avg_l2 = pairwise_l2.sum(dim=1) / (k - 1)
        global_medoid_idx = int(torch.argmin(avg_l2).item())

        iu = torch.triu_indices(k, k, offset=1, device=pairwise_l2.device)
        median_d = pairwise_l2[iu[0], iu[1]].median()

        eps = 1e-8
        s_score = (
            torch.norm(flat_actions.mean(dim=0) - flat_actions[global_medoid_idx], p=2)
            / (median_d + eps)
        )

        return global_medoid_idx, avg_l2, float(s_score.item())

    # tts add
    def _kmeans_small(
        self,
        x: torch.Tensor,
        num_clusters: int,
        max_iter: int,
    ):
        """
        Small deterministic k-means for K action chunks.

        x: [K, D], CPU tensor.
        return: labels [K].

        Initialization:
        - first center = global medoid
        - next centers = farthest-first
        This avoids consuming random seeds during evaluation.
        """
        k = x.shape[0]
        c = min(max(1, int(num_clusters)), k)

        if c == 1:
            return torch.zeros(k, dtype=torch.long, device=x.device)

        dists = torch.cdist(x, x, p=2)
        first_idx = int(torch.argmin(dists.sum(dim=1)).item())

        selected = [first_idx]
        centers = [x[first_idx].clone()]

        for _ in range(1, c):
            center_tensor = torch.stack(centers, dim=0)
            dist_to_centers = torch.cdist(x, center_tensor, p=2).min(dim=1).values

            for idx in selected:
                dist_to_centers[idx] = -1.0

            next_idx = int(torch.argmax(dist_to_centers).item())
            selected.append(next_idx)
            centers.append(x[next_idx].clone())

        centers = torch.stack(centers, dim=0)
        labels = torch.full((k,), -1, dtype=torch.long, device=x.device)

        for _ in range(max_iter):
            new_labels = torch.cdist(x, centers, p=2).argmin(dim=1)

            if torch.equal(new_labels, labels):
                break

            labels = new_labels

            for cid in range(c):
                mask = labels == cid
                if mask.any():
                    centers[cid] = x[mask].mean(dim=0)
                else:
                    # Empty cluster: re-seed with point farthest from current centers.
                    dist_to_centers = torch.cdist(x, centers, p=2).min(dim=1).values
                    centers[cid] = x[int(torch.argmax(dist_to_centers).item())].clone()

        return labels

    # ttsv2 add
    def _select_tts_keystone(self, actions_stack: torch.Tensor):
        """
        KeyStone-style selector:
        1. Compute global medoid.
        2. Compute unimodality score s_score.
        3. If s_score < tau: return global medoid.
        4. Else: k-means into C clusters.
        5. Select the largest cluster.
        6. Return the medoid inside the largest cluster.
        """
        k = actions_stack.shape[0]
        flat_actions = actions_stack.reshape(k, -1)

        pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            
            ranks = torch.zeros(k, dtype=torch.long, device=pairwise_l2.device)
            selection_probs = torch.ones(k, dtype=torch.float32, device=pairwise_l2.device)
            
            metrics = {
                "tts_method": "keystone",
                "selection_stage": "single_sample",
                "global_medoid_idx": 0,
                "unimodal": True,
                "s_score": "",
                "num_clusters": 1,
                "selected_cluster": -1,
                "selected_cluster_size": k,
                "cluster_counts": "",
                "cluster_ids": "",
                "selected_idx": 0,
                "selected_rank": 0,
                "selected_prob": "1.00000000",
                "rank_temperature": "",
                "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
                "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
            }
            return 0, pairwise_l2, avg_l2, metrics

        global_medoid_idx, avg_l2, s_score = self._compute_global_medoid_and_guard(
            flat_actions=flat_actions,
            pairwise_l2=pairwise_l2,
        )

        # Unimodality guard: if candidates look like one cluster,
        # do not force k-means to split them.
        if s_score < self.tts_tau or self.tts_num_clusters <= 1:
            ranks, order = self._build_tts_ranks(avg_l2)
            selection_probs = torch.zeros(k, dtype=torch.float32, device=avg_l2.device)
            selection_probs[global_medoid_idx] = 1.0
            
            metrics = {
                "tts_method": "keystone",
                "selection_stage": "guard_global_medoid",
                "global_medoid_idx": global_medoid_idx,
                "unimodal": True,
                "s_score": f"{s_score:.8f}",
                "num_clusters": 1,
                "selected_cluster": -1,
                "selected_cluster_size": k,
                "cluster_counts": "",
                "cluster_ids": "",
                "selected_idx": global_medoid_idx,
                "selected_rank": int(ranks[global_medoid_idx].item()),
                "selected_prob": "1.00000000",
                "rank_temperature": "",
                "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
                "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
            }
            return global_medoid_idx, pairwise_l2, avg_l2, metrics

        c = min(self.tts_num_clusters, k)
        cluster_ids = self._kmeans_small(
            flat_actions,
            num_clusters=c,
            max_iter=self.tts_kmeans_iters,
        )

        cluster_counts = torch.bincount(cluster_ids, minlength=c)
        selected_cluster = int(torch.argmax(cluster_counts).item())

        mask = cluster_ids == selected_cluster
        idxs = mask.nonzero(as_tuple=True)[0]

        # Medoid inside the largest cluster, not the centroid itself.
        sub_dists = pairwise_l2[mask][:, mask]
        local_idx = int(torch.argmin(sub_dists.sum(dim=1)).item())
        best_idx = int(idxs[local_idx].item())
        
        ranks, order = self._build_tts_ranks(avg_l2)
        selection_probs = torch.zeros(k, dtype=torch.float32, device=avg_l2.device)
        selection_probs[best_idx] = 1.0

        metrics = {
            "tts_method": "keystone",
            "selection_stage": "cluster_medoid",
            "global_medoid_idx": global_medoid_idx,
            "unimodal": False,
            "s_score": f"{s_score:.8f}",
            "num_clusters": c,
            "selected_cluster": selected_cluster,
            "selected_cluster_size": int(cluster_counts[selected_cluster].item()),
            "cluster_counts": "|".join(map(str, cluster_counts.detach().cpu().tolist())),
            "cluster_ids": "|".join(map(str, cluster_ids.detach().cpu().tolist())),
            "selected_idx": best_idx,
            "selected_rank": int(ranks[best_idx].item()),
            "selected_prob": "1.00000000",
            "rank_temperature": "",
            "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
            "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
        }

        return best_idx, pairwise_l2, avg_l2, metrics
    
    # ttsv2 add
    def _log_tts_result(self, actions_stack, pairwise_l2, avg_l2, selected_idx, metrics):
        """
        Log TTS selector information.

        Important:
        - Fixed CSV schema for all methods.
        - No .npz saving.
        - Unsupported fields for a method are filled with empty string or -1.
        """
        if not self.tts_log_actions:
            return

        self.tts_dir.mkdir(parents=True, exist_ok=True)

        avg_l2_np = avg_l2.detach().cpu().numpy()

        selected_avg_l2 = float(avg_l2_np[selected_idx])
        min_avg_l2 = float(avg_l2_np.min())
        max_avg_l2 = float(avg_l2_np.max())
        mean_avg_l2 = float(avg_l2_np.mean())

        avg_list = "|".join(f"{float(x):.6f}" for x in avg_l2_np)

        global_medoid_idx = metrics.get("global_medoid_idx", "")
        global_medoid_avg_l2 = ""
        if global_medoid_idx != "":
            global_medoid_avg_l2 = f"{float(avg_l2_np[int(global_medoid_idx)]):.8f}"

        print(
            f"[TTS] episode={self.episode_count} "
            f"step={self.step_count} "
            f"method={metrics.get('tts_method', self.tts_method)} "
            f"stage={metrics.get('selection_stage', '')} "
            f"samples={actions_stack.shape[0]} "
            f"C={self.tts_num_clusters} "
            f"tau={self.tts_tau} "
            f"rank_tau={metrics.get('rank_temperature', '')} "
            f"kmeans_iters={self.tts_kmeans_iters} "
            f"selected={selected_idx} "
            f"selected_rank={metrics.get('selected_rank', '')} "
            f"selected_prob={metrics.get('selected_prob', '')} "
            f"global_medoid={global_medoid_idx} "
            f"unimodal={metrics.get('unimodal', '')} "
            f"s_score={metrics.get('s_score', '')} "
            f"selected_cluster={metrics.get('selected_cluster', '')} "
            f"cluster_size={metrics.get('selected_cluster_size', '')} "
            f"cluster_counts={metrics.get('cluster_counts', '')} "
            f"selected_avg_l2={selected_avg_l2:.6f} "
            f"global_medoid_avg_l2={global_medoid_avg_l2} "
            f"min_avg_l2={min_avg_l2:.6f} "
            f"max_avg_l2={max_avg_l2:.6f}"
        )

        summary_path = self.tts_dir / "summary.csv"
        write_header = not summary_path.exists()

        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    "episode",
                    "step",
                    "samples",
                    "tts_method",
                    "selection_stage",

                    "selected_idx",
                    "selected_rank",
                    "selected_prob",

                    "global_medoid_idx",
                    "global_medoid_avg_l2",

                    "unimodal",
                    "s_score",
                    "num_clusters",
                    "tau",
                    "kmeans_iters",

                    "selected_cluster",
                    "selected_cluster_size",
                    "cluster_counts",
                    "cluster_ids",

                    "rank_temperature",
                    "rank_ids",
                    "selection_probs",

                    "selected_avg_l2",
                    "min_avg_l2",
                    "max_avg_l2",
                    "mean_avg_l2",
                    "avg_l2",
                ])

            writer.writerow([
                self.episode_count,
                self.step_count,
                actions_stack.shape[0],
                metrics.get("tts_method", self.tts_method),
                metrics.get("selection_stage", ""),

                selected_idx,
                metrics.get("selected_rank", ""),
                metrics.get("selected_prob", ""),

                global_medoid_idx,
                global_medoid_avg_l2,

                metrics.get("unimodal", ""),
                metrics.get("s_score", ""),
                metrics.get("num_clusters", self.tts_num_clusters),
                self.tts_tau,
                self.tts_kmeans_iters,

                metrics.get("selected_cluster", ""),
                metrics.get("selected_cluster_size", ""),
                metrics.get("cluster_counts", ""),
                metrics.get("cluster_ids", ""),

                metrics.get("rank_temperature", ""),
                metrics.get("rank_ids", ""),
                metrics.get("selection_probs", ""),

                f"{selected_avg_l2:.8f}",
                f"{min_avg_l2:.8f}",
                f"{max_avg_l2:.8f}",
                f"{mean_avg_l2:.8f}",
                avg_list,
            ])

    def get_action(self, instruction: str = None) -> List[np.ndarray]:
        """Get action predictions from the model."""
        if len(self.obs_cache) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")
        
        if self.current_state is None:
            raise ValueError("No robot state available. Call update_obs first.")
        
        current_frame = self.obs_cache[-1]

        # Encode instruction with T5
        scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                        "The aloha robot is currently performing the following task: ")
        instruction = f"{scene_prefix}{self.current_instruction}"
        t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        # Build VLM inputs
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(instruction, first_frame_pil)
        
        # tts add 
        # Run inference
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']

        if self.tts_enable and self.tts_num_samples > 1:
            action_candidates = []
            frame_candidates = []

            remaining = self.tts_num_samples
            while remaining > 0:
                cur_bs = min(self.tts_batch_size, remaining)

                cand_frames_b, cand_actions_b = self._run_batched_inference(
                    current_frame=current_frame,
                    t5_list=t5_list,
                    vlm_inputs=vlm_inputs,
                    num_inference_steps=num_inference_steps,
                    batch_size=cur_bs,
                    decode_video=self.tts_decode_video,
                )

                for local_idx in range(cur_bs):
                    action_candidates.append(cand_actions_b[local_idx].detach().float().cpu())
                    if cand_frames_b is None:
                        frame_candidates.append(None)
                    else:
                        frame_candidates.append(cand_frames_b[local_idx:local_idx + 1].detach().float().cpu())

                remaining -= cur_bs
            
            actions_stack = torch.stack(action_candidates, dim=0)
            selected_idx, pairwise_l2, avg_l2, metrics = self._select_tts_action(actions_stack)
            self._log_tts_result(actions_stack, pairwise_l2, avg_l2, selected_idx, metrics)

            predicted_actions = actions_stack[selected_idx].unsqueeze(0)
            predicted_frames = frame_candidates[selected_idx]
        else:
            predicted_frames, predicted_actions = self._run_single_inference(
                current_frame=current_frame,
                t5_list=t5_list,
                vlm_inputs=vlm_inputs,
                num_inference_steps=num_inference_steps,
                decode_video=True,
            )
        # Run inference end
        
        # Save frame grid
        if predicted_frames is not None:
            if predicted_frames.dim() == 5:
                if predicted_frames.shape[1] == 3:
                    predicted_frames_viz = predicted_frames.permute(0, 2, 1, 3, 4)
                else:
                    predicted_frames_viz = predicted_frames

                condition_frame_viz = current_frame.squeeze(0).detach().cpu()
                predicted_frames_viz = predicted_frames_viz.squeeze(0).detach().cpu()

                self._save_frame_grid(condition_frame_viz, predicted_frames_viz)

        self.step_count += 1

        actions_real = predicted_actions.squeeze(0).cpu().numpy()
        self.prev_action = actions_real[-1].copy()
        self.action_cache.extend(actions_real)

        return actions_real

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        """Convert [C, H, W] tensor to PIL Image."""
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Build VLM inputs."""
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': instruction},
                    {'type': 'image', 'image': image},
                ]
            }
        ]
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device), 
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    def _load_normalization_stats(self):
        """Load action normalization stats."""
        try:
            stat_path = Path(__file__).parent / 'utils' / 'stat.json'
            with open(stat_path, 'r') as f:
                stat_data = yaml.safe_load(f) if stat_path.suffix in ['.yml', '.yaml'] else None
        except Exception:
            stat_data = None
        if stat_data is None:
            import json as _json
            with open(Path(__file__).parent / 'utils' / 'stat.json', 'r') as f:
                stat_data = _json.load(f)

        stats = stat_data.get('robotwin2')
        if stats is None:
            raise ValueError('Normalization stats not found')
        self.action_min = torch.tensor(stats['min'], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(stats['max'], dtype=torch.float32, device=self.device)
        self.action_range = self.action_max - self.action_min

    def _normalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0,1]."""
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
        return norm.reshape(shape)

    def _denormalize_actions(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0,1]."""
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        denorm = y_flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return denorm.reshape(shape)
    
    def _create_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:
        """Create horizontal grid."""
        def tensor_to_numpy(tensor):
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            tensor = tensor.detach().cpu().float()
            tensor = torch.clamp(tensor, 0, 1)
            return (tensor.numpy() * 255).astype(np.uint8)
        
        condition_np = tensor_to_numpy(condition_frame)
        predicted_np = []
        num_pred_frames = predicted_frames.shape[0]
        for i in range(num_pred_frames):
            frame_np = tensor_to_numpy(predicted_frames[i])
            predicted_np.append(frame_np)
        
        while len(predicted_np) < 4:
            predicted_np.append(predicted_np[-1] if predicted_np else condition_np)
        
        all_frames = [condition_np] + predicted_np[:4]
        grid_image = np.concatenate(all_frames, axis=1)
        
        return Image.fromarray(grid_image)
    
    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        """Save frame grid to disk."""
        if not self.save_images:
            return
        
        try:
            grid_image = self._create_frame_grid(condition_frame, predicted_frames)
            filename = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.png"
            save_path = self.save_dir / filename
            grid_image.save(save_path)
            logger.info(f"Saved frame grid to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save frame grid: {e}")

def encode_obs(observation):
    """Post-Process Observation"""
    return observation

def get_model(usr_args):
    """
    Initialize Motus model.
    
    Args:
        usr_args: Arguments from eval script (must include wan_path and vlm_path)
    """
    checkpoint_path = usr_args.get('ckpt_setting')
    wan_path = usr_args.get('wan_path')  # Passed from eval.sh or auto_eval.sh
    vlm_path = usr_args.get('vlm_path')  # Passed from eval.sh or auto_eval.sh
    
    if not wan_path:
        raise ValueError("wan_path not provided in usr_args")
    
    if not vlm_path:
        raise ValueError("vlm_path not provided in usr_args")
    
    policy_dir = Path(__file__).parent
    config_path = policy_dir / "utils" / "robotwin.yml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # tts add
    tts_enable = _as_bool(usr_args.get("tts_enable", DEFAULT_TTS_ENABLE))
    tts_num_samples = int(usr_args.get("tts_num_samples", DEFAULT_TTS_NUM_SAMPLES))

    tts_method = str(usr_args.get("tts_method", DEFAULT_TTS_METHOD))
    tts_num_clusters = int(usr_args.get("tts_num_clusters", DEFAULT_TTS_NUM_CLUSTERS))
    tts_tau = float(usr_args.get("tts_tau", DEFAULT_TTS_TAU))
    tts_rank_tau = float(usr_args.get("tts_rank_tau", DEFAULT_TTS_RANK_TAU))
    tts_batch_size = int(usr_args.get("tts_batch_size", DEFAULT_TTS_BATCH_SIZE))
    tts_decode_video = _as_bool(usr_args.get("tts_decode_video", DEFAULT_TTS_DECODE_VIDEO))
    tts_kmeans_iters = int(usr_args.get("tts_kmeans_iters", DEFAULT_TTS_KMEANS_ITERS))

    tts_log_actions = _as_bool(usr_args.get("tts_log_actions", DEFAULT_TTS_LOG_ACTIONS))

    # Deprecated/no-op; kept for old shell commands.
    tts_save_full_actions = _as_bool(
        usr_args.get("tts_save_full_actions", DEFAULT_TTS_SAVE_FULL_ACTIONS)
    )
    
    policy = MotusPolicy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        config_path=str(config_path),
        device=device,
        log_dir=usr_args.get('log_dir'),
        task_name=usr_args.get('task_name'),
        tts_enable=tts_enable,
        tts_num_samples=tts_num_samples,
        tts_method=tts_method,
        tts_num_clusters=tts_num_clusters,
        tts_tau=tts_tau,
        tts_rank_tau=tts_rank_tau,
        tts_batch_size=tts_batch_size,
        tts_decode_video=tts_decode_video,
        tts_kmeans_iters=tts_kmeans_iters,
        tts_log_actions=tts_log_actions,
        tts_save_full_actions=tts_save_full_actions,
    )
    
    return policy

def eval(TASK_ENV, model, observation):
    """Evaluation function."""
    obs = encode_obs(observation)
    
    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)

    actions = model.get_action()
    
    for action in actions:
        TASK_ENV.take_action(action, action_type='qpos')

def reset_model(model):  
    """Reset model cache at episode start."""
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.is_first_step = True
    model.prev_action = None
    model.episode_count += 1
    model.step_count = 0
    logger.info(f"Model reset completed for episode {model.episode_count}")
```

/home/ubuntu/workspace/RoboTwin/policy/Motus/models/motus\.py

```Python
# Motus - Modular Architecture
# Three-modal UniDiffuser: Video Model (WAN) + Action Expert + Understanding Expert
# Implements MoT (Mixture of Tokens) architecture with unified attention

import sys
import json
import torch
import logging
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

BAK_ROOT = str((Path(__file__).parent.parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from utils.common import get_t_distribution
from wan.modules.model import sinusoidal_embedding_1d
from transformers import Qwen3VLForConditionalGeneration, AutoConfig
from .wan_model import WanVideoModel
from .action_expert import ActionExpert, ActionExpertConfig
from .und_expert import UndExpert, UndExpertConfig
# Add Flow-Matching schedulers
from wan.utils.fm import FlowMatchScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)

@dataclass 
class MotusConfig:
    """Configuration for Motus."""
    # Video model settings
    wan_checkpoint_path: str = "/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B"
    vae_path: str = "/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    wan_config_path: str = "/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B"
    video_precision: str = "bfloat16"

    # VLM settings
    vlm_checkpoint_path: str = "/share/home/bhz/pretrained_models/Qwen3-VL-2B-Instruct"
    
    # Understanding Expert settings - configurable from yaml
    und_expert_hidden_size: int = 512        # Understanding expert hidden dimension
    und_expert_ffn_dim_multiplier: int = 4   # Understanding expert FFN dimension multiplier
    und_expert_norm_eps: float = 1e-5        # Understanding expert layer norm epsilon
    und_layers_to_extract: List[int] = None  # Which VLM layers to extract from
    
    # VLM adapter settings for understanding expert
    vlm_adapter_input_dim: int = 2048        # VLM feature dimension (input)
    vlm_adapter_projector_type: str = "mlp3x_silu"  # VLM adapter type

    # Action expert settings  
    num_layers: int = 30 
    action_state_dim: int = 14
    action_dim: int = 14
    action_expert_dim: int = 1024           # Configurable hidden dimension
    action_expert_ffn_dim_multiplier: int = 4  # FFN dimension multiplier
    action_expert_norm_eps: float = 1e-6    # Layer norm epsilon for Action Expert

    # Sampling settings
    global_downsample_rate: int = 3     # Global downsampling rate
    video_action_freq_ratio: int = 4    # Video:Action frequency ratio
    num_video_frames: int = 4           # Number of video frames to predict
    
    # Video dimensions
    video_height: int = 512             # Input video height
    video_width: int = 512              # Input video width
    
    # Training settings
    batch_size: int = 8

    # Training mode
    training_mode: str = 'finetune'  # 'pretrain' or 'finetune'

    # Loss weights
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0

    # Control whether to load pretrained WAN/VLM backbones.
    # None = default behavior (load), False = skip loading (init from config only)
    load_pretrained_backbones: Optional[bool] = None

    def __post_init__(self):
        """Calculate derived parameters."""
        # Action chunk size is determined by global downsample rate and frequency ratio
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        
        # Default understanding layers to extract from (if not specified)
        if self.und_layers_to_extract is None:
            # Extract from all layers for comprehensive understanding
            self.und_layers_to_extract = list(range(self.num_layers))

class VideoModule(nn.Module):
    """Video processing module - handles WAN + T5 operations."""

    def __init__(self, video_model, dtype, device, grid_sizes):
        super().__init__()
        self.video_model = video_model
        self.dtype = dtype
        self.device = device
        self.grid_sizes = grid_sizes

    def prepare_input(self, noisy_video_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare video tokens from pre-processed noisy latent."""
        # Through patch_embedding: 48 -> 3072 channels
        video_patched = self.video_model.wan_model.patch_embedding(noisy_video_latent)

        # Flatten and convert to tokens
        video_features = video_patched.flatten(2).transpose(1, 2)

        # Calculate sequence length and padding
        # seq_lens = torch.tensor([u.size(1) for u in video_tokens_list], dtype=torch.long, device=self.device)
        # seq_len = seq_lens.max().item()

        # Concatenate with padding
        # video_tokens = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) 
        #     for u in video_tokens_list
        # ])

        # return video_tokens

        return video_features

    def preprocess_t5_embeddings(self, language_embeddings) -> torch.Tensor:
        """Pre-process T5 embeddings once for all layers."""
        # Handle both old format (List[torch.Tensor]) and new format (torch.Tensor)
        if isinstance(language_embeddings, list):
            # Old format: List[torch.Tensor] - do padding
            text_len = self.video_model.wan_model.text_len  # 512
            padded_embeddings = []

            for emb in language_embeddings:
                if emb.shape[0] <= text_len:
                    padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
                else:
                    padded = emb[:text_len]
                padded_embeddings.append(padded)

            t5_context_raw = torch.stack(padded_embeddings, dim=0)
        else:
            # New format: torch.Tensor [B, seq_len, dim] - already padded by collate_fn
            t5_context_raw = language_embeddings
        
        # Convert via text_embedding layer (4096 -> 3072)
        t5_context = self.video_model.wan_model.text_embedding(t5_context_raw)

        return t5_context

    def get_time_embedding(self, t_video: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get WAN's time embedding using WAN's own weights."""
        if t_video.dim() == 1:
            t_video = t_video.unsqueeze(1).expand(t_video.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t_video.size(0)
            t_flat = t_video.flatten()
            
            t_emb = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.video_model.wan_model.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )
            t_emb_proj = self.video_model.wan_model.time_projection(t_emb).unflatten(2, (6, 3072))
            assert t_emb.dtype == torch.float32 and t_emb_proj.dtype == torch.float32
            
        return t_emb, t_emb_proj

    def process_cross_attention(self, video_tokens: torch.Tensor, video_adaln_params: torch.Tensor, 
                               layer_idx: int, processed_t5_context: torch.Tensor) -> torch.Tensor:
        """Process WAN cross attention with pre-processed T5 context."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        context_lens = None  # WAN uses None for fixed-length context
        cross_out = wan_layer.cross_attn(wan_layer.norm3(video_tokens), processed_t5_context, context_lens)
        return video_tokens + cross_out
    
    def compute_adaln_modulation(self, video_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for WAN (6 components)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                wan_layer.modulation.unsqueeze(0)
                + video_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, video_tokens: torch.Tensor, video_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process WAN FFN with proper AdaLN modulation."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        
        # AdaLN params
        v_mod = video_adaln_modulation

        # WAN FFN with AdaLN (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = wan_layer.norm2(video_tokens).float() * (1 + v_mod[4].squeeze(2)) + v_mod[3].squeeze(2)
        ffn_out = wan_layer.ffn(ffn_input)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            return video_tokens + ffn_out * v_mod[5].squeeze(2)

    def apply_output_head(self, video_tokens: torch.Tensor, video_time_emb: torch.Tensor) -> torch.Tensor:
        """Apply WAN's head + unpatchify for final video output."""
        x = self.video_model.wan_model.head(video_tokens, video_time_emb)
        x = self.video_model.wan_model.unpatchify(x, self.grid_sizes)
        return torch.stack([u.float() for u in x], dim=0)

    def process_joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_adaln_modulation: tuple,
        action_adaln_modulation: tuple,
        layer_idx: int,
        action_block: nn.Module,
        und_tokens: torch.Tensor,
        und_block: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trimodal joint self-attention: WAN + Action + Understanding via WAN self-attn (MoT)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]

        # AdaLN params (already computed)
        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation

        # Pre-attn normalization with AdaLN
        norm_video = wan_layer.norm1(video_tokens).float() * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)
        norm_action = action_block.norm1(action_tokens) * (1 + a_mod[1].squeeze(2)) + a_mod[0].squeeze(2)

        # Get dimensions
        B, L_v, C = norm_video.shape
        L_a = norm_action.shape[1]
        n = self.video_model.wan_model.num_heads
        d = C // n

        # Action heads for WAN space (1024 -> 24*128)
        a_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_action, action_block.wan_action_qkv)
        a_q_h, a_k_h, a_v_h = a_qkv[0], a_qkv[1], a_qkv[2]
        a_q = action_block.wan_action_norm_q(a_q_h.flatten(-2)).view(B, L_a, n, d)
        a_k = action_block.wan_action_norm_k(a_k_h.flatten(-2)).view(B, L_a, n, d)
        a_v = a_v_h.view(B, L_a, n, d)

        # Understanding Expert processing
        norm_und = und_block.norm1(und_tokens)
        L_u = norm_und.shape[1]
        
        # Understanding Expert heads for WAN space (2048 -> 24*128)
        u_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_und, und_block.wan_und_qkv)
        u_q_h, u_k_h, u_v_h = u_qkv[0], u_qkv[1], u_qkv[2]
        u_q = und_block.wan_und_norm_q(u_q_h.flatten(-2)).view(B, L_u, n, d)
        u_k = und_block.wan_und_norm_k(u_k_h.flatten(-2)).view(B, L_u, n, d)
        u_v = u_v_h.view(B, L_u, n, d)

        # Meta info for WAN attention
        seq_lens = torch.full((B,), L_v + L_a + L_u, dtype=torch.long, device=self.device)
        freqs = self.video_model.wan_model.freqs
        if freqs.device != self.device:
            freqs = freqs.to(self.device)

        # Call WAN self-attn with trimodal MoT
        y, action_out_h, und_out_h = wan_layer.self_attn(
            norm_video, seq_lens, self.grid_sizes, freqs,
            action_q=a_q, action_k=a_k, action_v=a_v,
            und_q=u_q, und_k=u_k, und_v=u_v
        )
        
        # Project Understanding Expert output
        und_out = und_block.wan_und_o(und_out_h.flatten(2))

        # Project back and residual connections
        action_out = action_block.wan_action_o(action_out_h.flatten(2))
        video_tokens = video_tokens + y * v_mod[2].squeeze(2)
        action_tokens = action_tokens + action_out * a_mod[2].squeeze(2)
        und_tokens = und_tokens + und_out  # Regular residual connection

        return video_tokens, action_tokens, und_tokens

class UndModule(nn.Module):
    """Understanding module - handles VLM with understanding queries and Understanding Expert."""

    def __init__(self, vlm_model, und_expert, config, dtype, device):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        
        # VLM model reference
        self.vlm_model = vlm_model
        
        # Understanding Expert reference
        self.und_expert = und_expert
        
    def extract_und_features(
        self,
        vlm_inputs
    ) -> torch.Tensor:
        """Extract understanding features from VLM last layer."""
        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs['input_ids'].shape[0]

        # Returns: inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = self._process_vlm_inputs_to_tokens(vlm_inputs, B)

        # Forward through VLM with proper attention_mask and DeepStack features
        vlm_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': None,
            'use_cache': False,
            'output_attentions': False,
            'output_hidden_states': True,
            'return_dict': True
        }

        # Add DeepStack parameters for Qwen3-VL
        if visual_pos_masks is not None:
            vlm_kwargs['visual_pos_masks'] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds

        with torch.no_grad():
            vlm_output = self.vlm_model.model.language_model(**vlm_kwargs)

        # Extract last layer features directly
        last_layer_features = vlm_output.hidden_states[-1]  # [B, seq_len, vlm_dim]

        # [B, seq_len, vlm_dim] -> [B, seq_len, und_dim]
        adapted_features = self.und_expert.vlm_adapter(last_layer_features)

        return adapted_features
        
    def _process_vlm_inputs_to_tokens(self, vlm_inputs, B: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list], torch.Tensor]:
        """Convert VLM inputs to tokens.

        Returns:
            Tuple of (inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids)
        """
        # Handle both old format (List[Dict]) and new format (Dict[str, Tensor])
        if isinstance(vlm_inputs, list):
            # Old format: List[Dict] - do padding and batching
            input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
            attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
            pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
            image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]

            # Pad input_ids and attention_mask to same length
            max_seq_len = max(ids.shape[1] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_masks = []
            
            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[1] < max_seq_len:
                    padding_size = max_seq_len - ids.shape[1]
                    # Pad input_ids with zeros
                    id_padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, id_padding], dim=1)
                    # Pad attention_mask with zeros (padding tokens should be ignored)
                    mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                    padded_mask = torch.cat([mask, mask_padding], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)

            # Batch process
            input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.device)
            attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(self.device)
            pixel_values_batch = torch.cat([pv.to(self.device) for pv in pixel_values_list], dim=0)
            image_grid_thw_batch = torch.cat([igt.to(self.device) for igt in image_grid_thw_list], dim=0)
        else:
            # New format: Dict[str, Tensor] - already batched and padded by collate_fn
            input_ids_batch = vlm_inputs['input_ids'].to(self.device)
            attention_mask_batch = vlm_inputs['attention_mask'].to(self.device)
            pixel_values_batch = vlm_inputs['pixel_values'].to(self.device)
            image_grid_thw_batch = vlm_inputs['image_grid_thw'].to(self.device)

        # Get input embeddings
        inputs_embeds = self.vlm_model.get_input_embeddings()(input_ids_batch)

        # Process images - handle different return formats between Qwen2.5-VL and Qwen3-VL
        image_embeds, deepstack_image_embeds = self.vlm_model.get_image_features(pixel_values_batch, image_grid_thw_batch)

        image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.dtype)

        # Insert image embeddings
        image_mask, _ = self.vlm_model.model.get_placeholder_mask(
            input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        visual_pos_masks = image_mask[..., 0]  # [B, seq_len] - visual positions only

        # Compute position_ids (position_ids remains as original: [3, B, seq_len])
        # Qwen3-VL get_rope_index has different signature: (input_ids, image_grid_thw, video_grid_thw, attention_mask)
        position_ids, _rope_deltas = self.vlm_model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None,  # No video in current implementation
            attention_mask=attention_mask_batch
        )

        return inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids
    
    def process_ffn(self, und_tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Process Understanding Expert FFN with regular LayerNorm."""
        block = self.und_expert.blocks[layer_idx]
        
        # Pre-norm for FFN (regular LayerNorm)
        ffn_input = block.norm2(und_tokens)
        ffn_output = block.ffn(ffn_input)
        
        # FFN residual connection
        und_tokens = und_tokens + ffn_output
        
        return und_tokens

class ActionModule(nn.Module):
    """Action processing module - handles Action Expert + joint attentions + masks."""
    
    def __init__(self, action_expert: ActionExpert, config, video_model, vlm_model, dtype, device):
        super().__init__()
        self.action_expert = action_expert
        self.config = config
        self.video_model = video_model  # For accessing WAN weights
        self.vlm_model = vlm_model      # For accessing VLM weights
        self.dtype = dtype
        self.device = device
    
    def get_time_embedding(self, t: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action time embedding."""
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()
            
            # Create sinusoidal embedding (same pattern as VideoModule)
            a_e = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(self.action_expert.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )  # [B, seq_len, freq_dim]
            
            # Project to AdaLN parameters (6 params: 3 for WAN-Action joint attn + 3 for FFN)
            a_e0 = self.action_expert.time_projection(a_e).unflatten(2, (6, self.config.action_expert_dim))  # [B, seq_len, 6, dim]
            
            assert a_e.dtype == torch.float32 and a_e0.dtype == torch.float32

        return a_e, a_e0  # (basic_emb, adaln_params)

    def compute_adaln_modulation(self, action_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for 6 components (3 for WAN-Action joint attn + 3 for FFN)."""
        action_layer = self.action_expert.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                action_layer.modulation.unsqueeze(0)
                + action_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, action_tokens: torch.Tensor, action_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process Action Expert FFN with AdaLN modulation."""
        action_block = self.action_expert.blocks[layer_idx]

        # AdaLN params
        a_mod = action_adaln_modulation

        # Apply FFN with AdaLN modulation (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = action_block.norm2(action_tokens).float() * (1 + a_mod[4].squeeze(2)) + a_mod[3].squeeze(2)
        ffn_out = action_block.ffn(ffn_input)
        
        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_tokens = action_tokens + ffn_out * a_mod[5].squeeze(2)
        return action_tokens

class Motus(nn.Module):
    """
    Modular Three-modal UniDiffuser with VGM, VLM, and Action modules.
    """

    def __init__(self, config: MotusConfig):
        super().__init__()
        self.config = config

        # Set unified data type for the model
        self.dtype = torch.bfloat16

        # Decide whether to load pretrained backbones
        load_backbones = True if config.load_pretrained_backbones is None else bool(config.load_pretrained_backbones)

        # Initialize video model (WAN)
        logger.info("Initializing WAN video model...")
        if load_backbones:
            self.video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path,
                config_path=config.wan_config_path,
                precision=config.video_precision
            )
        else:
            self.video_model = WanVideoModel.from_config(
                config_path=config.wan_config_path,
                vae_path=config.vae_path,
                device="cuda",
                precision=config.video_precision
            )

        # Initialize VLM (frozen)
        logger.info("Initializing VLM (frozen)...")
        if load_backbones:
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.vlm_checkpoint_path,
                dtype=self.dtype,
                device_map="cuda",
                trust_remote_code=True
            )
        else:
            vlm_cfg = AutoConfig.from_pretrained(config.vlm_checkpoint_path, trust_remote_code=True)
            self.vlm_model = Qwen3VLForConditionalGeneration._from_config(vlm_cfg, torch_dtype=self.dtype)
            self.vlm_model.to(device="cuda", dtype=self.dtype)

        # Freeze VLM parameters
        for param in self.vlm_model.parameters():
            param.requires_grad = False
        logger.info("VLM parameters frozen")

        # Keep VLM complete (do not truncate)
        logger.info(f"VLM kept complete with {len(self.vlm_model.model.language_model.layers)} layers")

        # Get WAN and VLM configurations directly
        wan_dim = getattr(self.video_model.wan_model.config, 'dim', 3072)
        wan_num_heads = getattr(self.video_model.wan_model.config, 'num_heads', 24)
        wan_head_dim = wan_dim // wan_num_heads

        vlm_dim = self.vlm_model.config.text_config.hidden_size
        vlm_num_heads = self.vlm_model.config.text_config.num_attention_heads
        vlm_num_kv_heads = getattr(self.vlm_model.config.text_config if hasattr(self.vlm_model.config, 'text_config') else self.vlm_model.config, 'num_key_value_heads', vlm_num_heads)
        vlm_num_hidden_layers  = self.vlm_model.config.text_config.num_hidden_layers
        vlm_head_dim = vlm_dim // vlm_num_heads

        logger.info(f"Model configurations:")
        logger.info(f"  WAN: {wan_num_heads} heads × {wan_head_dim} head_dim = {wan_dim}D")
        logger.info(f"  VLM: {vlm_num_heads} Q heads, {vlm_num_kv_heads} KV heads × {vlm_head_dim} head_dim = {vlm_dim}D")

        # Create config dictionaries for ActionExpert
        wan_config = {
            'dim': wan_dim,
            'num_heads': wan_num_heads, 
            'head_dim': wan_head_dim
        }
        vlm_config = {
            'hidden_size': vlm_dim,
            'num_attention_heads': vlm_num_heads,
            'num_key_value_heads': vlm_num_kv_heads,
            'head_dim': vlm_head_dim,
            'num_hidden_layers': vlm_num_hidden_layers,
        }

        # Initialize action expert with unified configs
        logger.info("Initializing Action Expert...")

        # Determine chunk_size based on training mode
        if config.training_mode == 'pretrain':
            action_chunk_size_for_expert = config.action_chunk_size
        else:
            action_chunk_size_for_expert = config.action_chunk_size + 1  # include state token

        # Configure registers by mode: no registers in pretrain, keep default (e.g., 4) in finetune
        num_registers = 0 if config.training_mode == 'pretrain' else 4

        action_config = ActionExpertConfig(
            dim=config.action_expert_dim,
            ffn_dim=config.action_expert_dim * config.action_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            state_dim=config.action_state_dim,
            action_dim=config.action_dim,
            chunk_size=action_chunk_size_for_expert,
            num_registers=num_registers,
            video_feature_dim=wan_dim,
            causal=False,
            eps=config.action_expert_norm_eps,
            training_mode=config.training_mode,
        )

        self.action_expert = ActionExpert(action_config, wan_config)

        # Initialize Understanding Expert
        logger.info("Initializing Understanding Expert...")
        und_config = UndExpertConfig(
            dim=config.und_expert_hidden_size,
            ffn_dim=config.und_expert_hidden_size * config.und_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            vlm_input_dim=config.vlm_adapter_input_dim,
            vlm_projector_type=config.vlm_adapter_projector_type,
            eps=config.und_expert_norm_eps,
        )
        
        self.und_expert = UndExpert(und_config, wan_config, vlm_config)

        # Move models to device
        self.device = next(self.video_model.parameters()).device
        self.action_expert.to(device=self.device, dtype=self.dtype)
        self.und_expert.to(device=self.device, dtype=self.dtype)
        
        # Set time embedding layers to float32 for numerical stability
        self.action_expert.time_embedding.to(dtype=torch.float32)
        self.action_expert.time_projection.to(dtype=torch.float32)

        # Pre-compute grid_sizes for training batch size
        lat_T = 1 + config.num_video_frames // 4
        lat_H = config.video_height // 32
        lat_W = config.video_width // 32
        batch_size = config.batch_size
        self.grid_sizes = torch.tensor(
            [lat_T, lat_H, lat_W], 
            dtype=torch.long, 
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)  # [batch_size, 3] - pre-expanded
        
        logger.info(f"Pre-computed grid_sizes: T={lat_T}, H={lat_H}, W={lat_W}")

        # Initialize modular components
        self.video_module = VideoModule(self.video_model, self.dtype, self.device, self.grid_sizes)
        self.und_module = UndModule(self.vlm_model, self.und_expert, self.config, self.dtype, self.device)
        self.action_module = ActionModule(self.action_expert, self.config, self.video_model, self.vlm_model, self.dtype, self.device)

        # Initialize t distributions from config
        time_dist_config = getattr(config, 'time_distribution', {})
        model_config = {
            'timestep_sample_method': time_dist_config.get('timestep_sample_method', 'logit_normal'),
            'sigmoid_scale': time_dist_config.get('sigmoid_scale', 1.0),
            'min_t': time_dist_config.get('min_t', 0.0),
            'max_t': time_dist_config.get('max_t', 1.0)
        }

        # Flow-Matching scheduler for training (video branch only)
        try:
            self.fm_train_scheduler = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode to build per-timestep weights (if used)
            self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)
            logger.info("Initialized FlowMatchScheduler for training (video)")
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler: {e}")

        # Flow-Matching scheduler for training (action branch)
        try:
            self.fm_train_scheduler_action = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode for action as well
            self.fm_train_scheduler_action.set_timesteps(num_inference_steps=1000, training=True)
            logger.info("Initialized FlowMatchScheduler for training (action)")
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler for action: {e}")

        # Log parameter counts
        self.log_parameter_counts()

    def log_parameter_counts(self):
        """Log detailed parameter counts for each component."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        video_params = sum(p.numel() for p in self.video_model.parameters())
        action_params = sum(p.numel() for p in self.action_expert.parameters())
        vlm_params = sum(p.numel() for p in self.vlm_model.parameters())
        und_params = sum(p.numel() for p in self.und_expert.parameters())

        logger.info(f"Motus parameter breakdown:")
        logger.info(f"  Total parameters: {total_params / 1e9:.2f}B")
        logger.info(f"  Trainable parameters: {trainable_params / 1e9:.2f}B")
        logger.info(f"  Video Model (WAN): {video_params / 1e9:.2f}B")
        logger.info(f"  Action Expert: {action_params / 1e6:.1f}M")
        logger.info(f"  VLM (frozen): {vlm_params / 1e9:.2f}B")
        logger.info(f"  Und Expert: {und_params / 1e6:.1f}M")

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        """Load model checkpoint."""
        # Handle directory path
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            path = str(checkpoint_file)
    
        # Load state dict
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['module']  
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Checkpoint loaded from {path}: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
        
        # Return additional state
        additional_state = {k: v for k, v in checkpoint.items() 
                          if k not in ['module', 'config']}
        return additional_state

    def load_pretrain_weights(self, path: str) -> None:
        """Load weights from a pretrain checkpoint when current mode is finetune.

        Skips layers that depend on state vs action-only differences:
          - action_expert.input_encoder.*
          - action_expert.decoder.*
        """
        if self.config.training_mode != 'finetune':
            raise ValueError("load_pretrain_weights should be called only in finetune mode")
        # Handle directory path (align with load_checkpoint style)
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "pytorch_model" / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            path = str(checkpoint_file)

        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('module', checkpoint)
        filtered = {}
        for k, v in state_dict.items():
            if ('action_expert.input_encoder' in k or 'action_expert.decoder' in k):
                continue
            filtered[k] = v
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded pretrain weights (filtered). Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def training_step(
        self,
        first_frame: torch.Tensor,         # [B, C, H, W] - first frame
        video_frames: torch.Tensor,       # [B, num_frames, C, H, W] - target frames
        state: torch.Tensor = None,       # [B, state_dim] - robot state
        actions: torch.Tensor = None,     # [B, chunk_size, action_dim] - actions
        language_embeddings: Optional[List[torch.Tensor]] = None,  # Pre-encoded T5 embeddings for WAN
        vlm_inputs: Optional[List] = None,  # Complete VLM inputs from dataset
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        UniDiffuser training step with three modalities.
        
        Args:
            first_frame: First video frame for Teacher Forcing
            video_frames: Target video frames
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state
            actions: Target action sequence
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            return_dict: Whether to return detailed outputs
            
        Returns:
            Dictionary containing losses and metrics
        """
        B = video_frames.shape[0]

        # 1. Video pipeline
        # Normalize/format
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [B, C, 1, H, W]
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)  # [B, C, num_frames, H, W]
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)  # [B, C, frames+1, H, W]

        # Encode video using VAE
        with torch.no_grad():
            clean_full_latent = self.video_model.encode_video(full_video.to(self.dtype))  # [B, 48, latent_frames, H', W']
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))  # [B, 48, 1, H', W']

        # Flow-Matching noise mixture
        timestep_id = torch.randint(0, self.fm_train_scheduler.num_train_timesteps, (B,))
        # Scalar timesteps (0..num_train_timesteps) for time embedding
        video_t_embed = self.fm_train_scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)  # [B]
        # Sigma for noise mixture
        sigma = self.fm_train_scheduler.sigmas[timestep_id].to(dtype=self.dtype, device=self.device).view(B, 1, 1, 1, 1)
        video_noise = torch.randn_like(clean_full_latent, dtype=self.dtype)
        noisy_video_latent = clean_full_latent * (1 - sigma) + video_noise * sigma
        # Teacher Forcing on the first frame
        noisy_video_latent[:, :, 0:1] = condition_frame_latent
        # Flow-Matching target: noise - clean
        video_target = video_noise - clean_full_latent
        video_target[:, :, 0:1] = 0

        # Latent to Tokens
        video_tokens = self.video_module.prepare_input(noisy_video_latent.to(self.dtype))

        # 2. Action pipeline 
        timestep_id_action = torch.randint(0, self.fm_train_scheduler_action.num_train_timesteps, (B,))
        # Discrete timesteps for time embedding (0..num_train_timesteps)
        action_t_embed = self.fm_train_scheduler_action.timesteps[timestep_id_action].to(dtype=self.dtype, device=self.device)  # [B]
        # Sigma for action noise mixture
        sigma_action = self.fm_train_scheduler_action.sigmas[timestep_id_action].to(dtype=self.dtype, device=self.device).view(B, 1, 1)
        action_noise = torch.randn_like(actions, dtype=self.dtype)
        noisy_actions = actions * (1 - sigma_action) + action_noise * sigma_action
        action_target = action_noise - actions

        # Encode Action Chunk with optional Registers
        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
        else:
            registers = None
        if self.config.training_mode == 'pretrain':
            action_tokens = self.action_expert.input_encoder(None, noisy_actions, registers)
        else:
            state_tokens = state.unsqueeze(1).to(self.dtype)
            action_tokens = self.action_expert.input_encoder(state_tokens, noisy_actions, registers)

        und_tokens = self.und_module.extract_und_features(vlm_inputs)  # [B, seq_len, und_dim]

        # Time embeddings
        # Use scheduler-provided timesteps (0..num_train_timesteps) for WAN/action time embeddings
        video_head_time_emb, video_adaln_params  = self.video_module.get_time_embedding(video_t_embed, video_tokens.shape[1])
        action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_embed, action_tokens.shape[1])

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. MoT forward
        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            # Process through 30 layers - modality-grouped execution
            for layer_idx in range(self.config.num_layers):
                # Compute AdaLN modulation once per layer using pre-computed parameters
                video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                
                # Trimodal MoT: WAN + Action + Understanding Expert joint attention
                video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                    video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, 
                    self.action_expert.blocks[layer_idx],
                    und_tokens, self.und_expert.blocks[layer_idx]
                )

                # WAN cross
                video_tokens = self.video_module.process_cross_attention(video_tokens, video_adaln_params, layer_idx, processed_t5_context)

                # FFNs: WAN, Action, Understanding (each processes their own FFN)
                video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)
                
        
            # 4. Heads + Losses
            video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
            action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
            up_len = action_pred_full.shape[1] - self.action_expert.config.num_registers
            # Slice predicted actions depending on mode
            if self.config.training_mode == 'pretrain':
                action_pred = action_pred_full[:, :up_len, :]
            else:
                action_pred = action_pred_full[:, 1:up_len, :]

            # Video loss (mask the first frame)
            video_pred_masked = video_pred.clone()
            video_pred_masked[:, :, 0:1] = 0
            video_loss = torch.nn.functional.mse_loss(video_pred_masked, video_target, reduction='mean')
        
            # Action loss
            action_loss = torch.nn.functional.mse_loss(action_pred, action_target, reduction='mean')

        total_loss = (
            self.config.video_loss_weight * video_loss +
            self.config.action_loss_weight * action_loss
        )
        
        if return_dict:
            return {
                'total_loss': total_loss,
                'video_loss': video_loss,
                'action_loss': action_loss,
                'video_timestep_mean': sigma.float().mean().item(),
                'action_timestep_mean': sigma_action.float().mean().item(),
            }

    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        decode_video: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction.
        
        Args:
            first_frame: Initial frame [B, C, H, W]
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state [B, state_dim]
            num_inference_steps: Number of denoising steps
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            
        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """

        B = first_frame.shape[0]

        # TTS batched inference may use a different runtime batch size from config.batch_size.
        # Keep grid_sizes aligned with the actual runtime B used by WAN self-attention/unpatchify.
        if self.grid_sizes.shape[0] != B:
            lat_T = 1 + self.config.num_video_frames // 4
            lat_H = self.config.video_height // 32
            lat_W = self.config.video_width // 32
            grid_sizes = torch.tensor(
                [lat_T, lat_H, lat_W],
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0).expand(B, -1)
            self.grid_sizes = grid_sizes
            self.video_module.grid_sizes = grid_sizes

        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
                
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        # 1. Video/Action latents init
        # Condition frame encode
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [B, C', 1, H', W']

        # Init video/action latents
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = condition_frame_latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)

        # 2. Understanding Expert features and T5 context
        # Extract understanding features from VLM
        und_tokens = self.und_module.extract_und_features(vlm_inputs)

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Denoising loop: from noise (t=1) to clean (t=0)
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)
        for i in range(num_inference_steps):
            # Timesteps
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            video_t_scaled = (t * 1000).expand(B).to(self.dtype)
            action_t_scaled = (t * 1000).expand(B).to(self.dtype)

            # Tokens with Registers
            video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
            state_tokens = state.unsqueeze(1).to(self.dtype)
            # Expand registers for batch
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
            action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)

            # Note: Understanding tokens already extracted before the loop, will be updated in joint attention
            und_tokens = self.und_module.extract_und_features(vlm_inputs)  # [B, num_queries * num_layers, und_dim]

            
            # Trimodal MoT forward - joint denoising for WAN, Action, Understanding
            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                # Time embeddings
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(video_t_scaled, video_tokens.shape[1])
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_scaled, action_tokens.shape[1])

                # Process through all layers - trimodal denoising of WAN, Action, Understanding
                for layer_idx in range(self.config.num_layers):
                    # Compute AdaLN modulation using pre-computed parameters
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                    
                    # Trimodal joint attention: WAN + Action + Understanding
                    video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                        video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, 
                        self.action_expert.blocks[layer_idx],
                        und_tokens, self.und_expert.blocks[layer_idx]
                    )

                    # WAN cross-attention with T5 embeddings 
                    video_tokens = self.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context
                    )

                    # FFNs: WAN, Action, Understanding
                    video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                    und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                # Heads (velocities)
                video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
                # Use decoder with all tokens (including registers)
                action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                # Extract middle action chunk (skip first state token and last register tokens)
                action_velocity = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]

                # Euler integration
                video_latent = video_latent + video_velocity * dt
                action_latent = action_latent + action_velocity * dt

                # Teacher Forcing
                video_latent[:, :, 0:1] = condition_frame_latent
    
        # 4. Decode outputs
        predicted_frames = None
        if decode_video:
            with torch.no_grad():
                decoded_frames = self.video_model.decode_video(video_latent)
            predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = action_latent.float()  # [B, action_chunk_size, 14]

        return predicted_frames, predicted_actions

def test_motus():
    """Test the complete model."""
    print("Testing Motus...")

    config = MotusConfig()

    try:
        model = Motus(config)
        print("Model created successfully")

        # Test parameter counting
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e9:.2f}B")

    except Exception as e:
        print(f"Model creation failed: {e}")
        print("This is expected without actual pretrained weights")

if __name__ == "__main__":
    test_motus()
```



##### 评估

batch 4 ，gpu0，同任务，L2 min \+ Random，32采样

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_rank_softmax_n32_b4_tau1_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 0 --tts --tts-num-samples 32 \
  --tts-method rank_softmax \
  --tts-tau 1.0 \
  --tts-batch-size 4 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 68794

PID: 68794

LOG: logs/scan\_object\_tts\_rank\_softmax\_n32\_b4\_tau1\_nodecode\_20260521\_104322\.log



监视显存的脚本：

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

cat > monitor_gpu_peak.sh <<'EOF'
#!/usr/bin/env bash

GPU_ID="${1:-0}"
INTERVAL="${2:-1}"
OUT_DIR="${3:-logs}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/gpu${GPU_ID}_mem_$(date +%Y%m%d_%H%M%S).csv"

echo "timestamp,gpu_index,memory_used_mib,gpu_util_percent,peak_memory_used_mib" > "$LOG"

peak=0

echo "Monitoring GPU $GPU_ID every ${INTERVAL}s"
echo "Log: $LOG"

while true; do
    line=$(nvidia-smi \
        --id="$GPU_ID" \
        --query-gpu=timestamp,index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits)

    mem=$(echo "$line" | awk -F', ' '{print $3}')

    if [ "$mem" -gt "$peak" ]; then
        peak="$mem"
    fi

    echo "$line,$peak" >> "$LOG"
    sleep "$INTERVAL"
done
EOF

chmod +x monitor_gpu_peak.sh
```

启动脚本：

```Plain Text
cd /home/ubuntu/workspace/RoboTwin/policy/Motus
nohup ./monitor_gpu_peak.sh 0 1 logs > logs/gpu0_monitor_stdout.log 2>&1 &
echo "MONITOR_PID: $!"
```

\[2\] 72630

MONITOR\_PID: 72630

显存反而从原来41G到39G；是不解码视频的原因



GPU1，batch16

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_keystone_n32_b16_tau03_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 32 \
  --tts-method keystone \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[3\] 85687

PID: 85687

LOG: logs/scan\_object\_tts\_keystone\_n32\_b16\_tau03\_nodecode\_20260521\_112950\.log

监视显存：

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

nohup ./monitor_gpu_peak.sh 1 1 logs > logs/gpu1_monitor_stdout.log 2>&1 &

echo "MONITOR_PID: $!"
```



全断掉，都改b32顶满

m2

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_keystone_n32_b32_tau03_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 0 --tts --tts-num-samples 32 \
  --tts-method keystone \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-batch-size 32 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[5\] 101215

PID: 101215

LOG: logs/scan\_object\_tts\_keystone\_n32\_b32\_tau03\_nodecode\_20260521\_114910\.log

m3

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_rank_softmax_n32_b32_tau1_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 32 \
  --tts-method rank_softmax \
  --tts-tau 1.0 \
  --tts-batch-size 32 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[6\] 101805

PID: 101805

LOG: logs/scan\_object\_tts\_rank\_softmax\_n32\_b32\_tau1\_nodecode\_20260521\_114937\.log



显存峰值缓慢上涨，来到49G左右





n=4 b=4

m2

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_keystone_n4_b4_tau03_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 0 --tts --tts-num-samples 4 \
  --tts-method keystone \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-batch-size 4 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[5\] 319520

PID: 319520

LOG: logs/scan\_object\_tts\_keystone\_n4\_b4\_tau03\_nodecode\_20260521\_164954\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MDIzY2JiYWU2YzlkNTcxMTBjMzc3ZDMwMjgwMTUwODlfODJjYTlmYWE1MTY2MDkwMGQ5ZjZlOTFlYjhiMThlNDBfSUQ6NzY0MjM1MzY5MTU4MzkyNTQ1OF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

m3

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_rank_softmax_n4_b4_tau1_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 4 \
  --tts-method rank_softmax \
  --tts-tau 1.0 \
  --tts-batch-size 4 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[6\] 319721

PID: 319721

LOG: logs/scan\_object\_tts\_rank\_softmax\_n4\_b4\_tau1\_nodecode\_20260521\_164959\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MGIyNTZmNzU1Y2YzOTU0NWE3ZTJjN2NkNThiMDg1NTVfOTQzZjIzNzU2YWU4NmNjYmFjY2E4NDI0ZWY2ODcxODlfSUQ6NzY0MjM1Mzc3MTUxMDUzMzA1Ml8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



#### v5



##### 实现

L2 min \+ 视频信息，先rank再混合，2种混合方式

\[deploy\_policy\.py\]

\[eval\.sh\]

\[motus\.py\]



##### 实验

W

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_video_rank_weighted_borda_n8_b8_w05_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 0 --tts --tts-num-samples 8 \
  --tts-method video_rank_fusion \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 0.5 \
  --tts-batch-size 8 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 572407

PID: 572407

LOG: logs/scan\_object\_tts\_video\_rank\_weighted\_borda\_n8\_b8\_w05\_latent\_nodecode\_20260521\_225210\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=M2RkMThjZDkwYTkwM2M3MzhmMmYzNDczOGEwM2E2YjBfMzY0MGIzOTI1YzFkYTQwNTE0ZDBhNTYyYTBlZmYxYjNfSUQ6NzY0MjUxMjExNTEzMTQ1MjYwNl8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

Rrf

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_video_rank_rrf_n8_b8_w05_k1_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 8 \
  --tts-method video_rank_fusion \
  --tts-video-feature latent \
  --tts-rank-fusion-method rrf \
  --tts-video-weight 0.5 \
  --tts-rrf-k 1 \
  --tts-batch-size 8 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[2\] 572830

PID: 572830

LOG: logs/scan\_object\_tts\_video\_rank\_rrf\_n8\_b8\_w05\_k1\_latent\_nodecode\_20260521\_225224\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZDU0ZWJjMDA3NzA5NzhmZGU0MDRjY2U3MTI2MWVkZWNfOGFhOGFjMzQyZTU5YTI0NjlkNjllMDUwMThiNjJjNzFfSUQ6NzY0MjUxMTU5NTc0MjU2MzUzOF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



实现：放后续进程，但等GPU空闲才开始跑；嵌套脚本

脚本：

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

cat > wait_gpu_and_run.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG
# =========================
GPU_ID="$1"
THRESHOLD_MB=5
STABLE_SECONDS=10
CHECK_INTERVAL=1

shift
RUN_CMD="$*"

echo "[queue] GPU_ID=${GPU_ID}"
echo "[queue] THRESHOLD_MB=${THRESHOLD_MB}"
echo "[queue] STABLE_SECONDS=${STABLE_SECONDS}"
echo "[queue] RUN_CMD=${RUN_CMD}"
echo "[queue] start waiting at $(date)"

stable_count=0

while true; do
    used_mb=$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')

    if [ -z "$used_mb" ]; then
        used_mb=999999
    fi

    echo "[queue] $(date '+%F %T') gpu=${GPU_ID} used_mb=${used_mb} stable_count=${stable_count}/${STABLE_SECONDS}"

    if [ "$used_mb" -lt "$THRESHOLD_MB" ]; then
        stable_count=$((stable_count + CHECK_INTERVAL))
    else
        stable_count=0
    fi

    if [ "$stable_count" -ge "$STABLE_SECONDS" ]; then
        echo "[queue] GPU ${GPU_ID} has been idle for ${STABLE_SECONDS}s. Launching at $(date)"
        bash -lc "$RUN_CMD"
        echo "[queue] submitted command at $(date)"
        exit 0
    fi

    sleep "$CHECK_INTERVAL"
done
SH

chmod +x wait_gpu_and_run.sh
```

等gpu的进程，w

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

QUEUE_LOG=logs/queue_scan_object_video_rank_weighted_borda_gpu0_n4_b4_$(date +%Y%m%d_%H%M%S).log

nohup ./wait_gpu_and_run.sh 0 '
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_video_rank_weighted_borda_n4_b4_w05_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 0 --tts --tts-num-samples 4 \
  --tts-method video_rank_fusion \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 0.5 \
  --tts-batch-size 4 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
' > "$QUEUE_LOG" 2>&1 &

echo "QUEUE_PID: $!"
echo "QUEUE_LOG: $QUEUE_LOG"
```

\[3\] 604907

QUEUE\_PID: 604907

QUEUE\_LOG: logs/queue\_scan\_object\_video\_rank\_weighted\_borda\_gpu0\_n4\_b4\_20260521\_232815\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NTJlNWRkMDNlYzE3NmNjOWE4ZWYxNTk2ZmI4MDc0MDBfZjczMjc0ZjIxZTk4MTFkMjQyNTFmNTIzMmY2MWM0ZGFfSUQ6NzY0MjUxMjAxNTk3Mzk2MDY2MV8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

等gpu的进程，rrf

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

QUEUE_LOG=logs/queue_scan_object_video_rank_rrf_gpu1_n4_b4_$(date +%Y%m%d_%H%M%S).log

nohup ./wait_gpu_and_run.sh 1 '
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_video_rank_rrf_n4_b4_w05_k1_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 4 \
  --tts-method video_rank_fusion \
  --tts-video-feature latent \
  --tts-rank-fusion-method rrf \
  --tts-video-weight 0.5 \
  --tts-rrf-k 1 \
  --tts-batch-size 4 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
' > "$QUEUE_LOG" 2>&1 &

echo "QUEUE_PID: $!"
echo "QUEUE_LOG: $QUEUE_LOG"
```

\[4\] 605285

QUEUE\_PID: 605285

QUEUE\_LOG: logs/queue\_scan\_object\_video\_rank\_rrf\_gpu1\_n4\_b4\_20260521\_232829\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YzYyNjFiMDc4NzdlYjU5OTgyM2JlMTE4NzI1YTBiMmFfZjViNDhiMWU3MDUzNzJlNGM2OTI2MzkwMWE5M2YxOWZfSUQ6NzY0MjUxMTQzODQxNDA1NjQxNF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



n16b16

w

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_video_rank_weighted_borda_n16_b16_w05_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 0 --tts --tts-num-samples 16 \
  --tts-method video_rank_fusion \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 0.5 \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 1139433

PID: 1139433

LOG: logs/scan\_object\_tts\_video\_rank\_weighted\_borda\_n16\_b16\_w05\_latent\_nodecode\_20260522\_085743\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZjgyNTczZmMxMWM5OWIzYTlhMWNlN2IxYTkwNjFiNjZfMDE2YmVjNzczMTMzNDI1OThmYTE4N2E5ZTM3NDlmODVfSUQ6NzY0MjYxMDM4ODQxNjg0MjY4NF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

rrf

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_video_rank_rrf_n16_b16_w05_k1_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 16 \
  --tts-method video_rank_fusion \
  --tts-video-feature latent \
  --tts-rank-fusion-method rrf \
  --tts-video-weight 0.5 \
  --tts-rrf-k 1 \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[2\] 1139602

PID: 1139602

LOG: logs/scan\_object\_tts\_video\_rank\_rrf\_n16\_b16\_w05\_k1\_latent\_nodecode\_20260522\_085747\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YzI5ZWU2ZDdmZDE0NGI1YTMwMzNkNjc1MWUzNGJkNjFfMmZjNmFmNmUxNDdkMmRlMTIxYjdkYmRiZTI3YzhmNzdfSUQ6NzY0MjYxMDQ4NDQ5NTc0ODAzN18xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



#### v6

先备份



##### 实现

融合\+聚类/随机/门控

\[deploy\_policy\.py\]

\[eval\.sh\]

\[motus\.py\]

\[tts\_log\_visualizer\_zh\_v5\.py\]



##### 实验

给建议参数，和之前一致，或者根据之前反馈调整

融合\+聚类，n8

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_video_cluster_fusion_n8_b8_c2_tau03_w05_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 0 --tts --tts-num-samples 8 \
  --tts-method video_cluster_fusion \
  --tts-video \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 0.5 \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  --tts-batch-size 8 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 1438877

PID: 1438877

LOG: logs/scan\_object\_tts\_video\_cluster\_fusion\_n8\_b8\_c2\_tau03\_w05\_latent\_nodecode\_20260522\_160727\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NmQ1OTMzNGE3YTRjNjRlNDliMWJjNDgxMjcwMjBjN2JfZTNjYTBkMWNiMGE4ZjhlYmI4MGEzZjk2NDAzZTc5ZjZfSUQ6NzY0MjY5OTcyMTExMDMwOTg0MV8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

融合\+随机，n8

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_video_fusion_rank_softmax_n8_b8_ranktau1_w05_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 8 \
  --tts-method video_fusion_rank_softmax \
  --tts-video \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 0.5 \
  --tts-rank-tau 1.0 \
  --tts-batch-size 8 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[2\] 1439292

PID: 1439292

LOG: logs/scan\_object\_tts\_video\_fusion\_rank\_softmax\_n8\_b8\_ranktau1\_w05\_latent\_nodecode\_20260522\_160740\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YWUyZDJlNzc2NDc0ODJkNTM1MzlmZDJkZWJhYWQwYzlfMTY1OGNmOTIzZjk5OTc1MmMyZDM4Mjk4MmI2NjRiMDhfSUQ6NzY0MjY5OTg0MzYwOTM0OTA2MV8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



l2 fusion gate 4

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/scan_object_tts_video_gated_fusion_n4_b4_w05_wlow0_sp03_dr2_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 0 --tts --tts-num-samples 4 \
  --tts-method video_gated_fusion \
  --tts-video \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 0.5 \
  --tts-video-weight-low 0.0 \
  --tts-gate-spearman-thresh 0.3 \
  --tts-gate-distance-ratio-thresh 2.0 \
  --tts-batch-size 4 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 2556

PID: 2556

LOG: logs/scan\_object\_tts\_video\_gated\_fusion\_n4\_b4\_w05\_wlow0\_sp03\_dr2\_latent\_nodecode\_20260524\_091945\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZTQ1NDMwMzMxZDVhODRiODcyZGNjYTJlYjZjOWZhZDlfZTU1ZGQ4NGYxMDY5ODIxNjc4ZWJkMzBmZWUyOThjYjhfSUQ6NzY0MzMxMzQzOTU2MTA1OTI3MV8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

l2 fusion gate 8

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus
mkdir -p logs

LOG=logs/scan_object_tts_video_gated_fusion_n8_b8_w05_wlow0_sp03_dr2_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh scan_object --gpu-id 1 --tts --tts-num-samples 8 \
  --tts-method video_gated_fusion \
  --tts-video \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 0.5 \
  --tts-video-weight-low 0.0 \
  --tts-gate-spearman-thresh 0.3 \
  --tts-gate-distance-ratio-thresh 2.0 \
  --tts-batch-size 8 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[2\] 2873

PID: 2873

LOG: logs/scan\_object\_tts\_video\_gated\_fusion\_n8\_b8\_w05\_wlow0\_sp03\_dr2\_latent\_nodecode\_20260524\_092006\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ODBkYmMxMmMwOWM2YzBjYWYxMTZhYjdjY2VhY2MzMTZfYmJjMjIyZGU4YTQ2NjI0MGQ5OGJlOTQ2NDAwMmQ0OTlfSUQ6NzY0MzMyMDM2OTYyNTE4OTU2NF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



put\_bottles\_dustbin base

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/put_bottles_dustbin_baseline_no_tts_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh put_bottles_dustbin --gpu-id 0 --no-tts \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 37587

PID: 37587

LOG: logs/put\_bottles\_dustbin\_baseline\_no\_tts\_20260524\_132232\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NzdjYjhkZGJhYjIyNDA2YjljZmEyNTdlOWZkYWUwOWZfZDg4YmU2MzRlMTVjYjhmYjZkN2I0ODRmNWViYjkwMTdfSUQ6NzY0MzQwNjQ5MjYxNTQxMjY2OF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

put\_bottles\_dustbin L2 min \+ 加权 Fusion：N=4，B=4，video weight=1

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

LOG=logs/put_bottles_dustbin_tts_video_rank_fusion_n4_b4_w1_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh put_bottles_dustbin --gpu-id 1 --tts --tts-num-samples 4 \
  --tts-method video_rank_fusion \
  --tts-video \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 1.0 \
  --tts-batch-size 4 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[2\] 37776

PID: 37776

LOG: logs/put\_bottles\_dustbin\_tts\_video\_rank\_fusion\_n4\_b4\_w1\_latent\_nodecode\_20260524\_132241\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=Zjc5YjQ2ZDYyNjcyMjU2MTYzYmY0Y2QxOTkzMjA5YmJfYjYwNzQwOGY3NWI1NzM1MDc1ZTQ5ZmE1Yjg2ODk5MDNfSUQ6NzY0MzQwNzgyNjE2NTE1NzA3NF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



##### 日志分析



l2 fusion cluster 8

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

python tts_log_visualizer_zh_v5.py \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs_single_20260522_160728/tts/scan_object/summary.csv \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs/scan_object_tts_video_cluster_fusion_n8_b8_c2_tau03_w05_latent_nodecode_20260522_160727.log
```

K=8 时，cluster 通常会把候选池缩到 6 或 7 个，不是特别激进

它几乎总是强制做 cluster。这个可能是 cluster 方法没有明显收益的一个原因：**它可能过度相信 action\-space cluster，把一些 video 可能想纠偏的候选提前剪掉了。**

如果后面还想继续试 cluster，我更建议调高 guard threshold，例如：



l2 fusion rand 8

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

python tts_log_visualizer_zh_v5.py \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs_single_20260522_160741/tts/scan_object/summary.csv \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs/scan_object_tts_video_fusion_rank_softmax_n8_b8_ranktau1_w05_latent_nodecode_20260522_160740.log
```



最重要的发现：Spearman 和 distance ratio 确实是有效信号

两个实验都显示同一个趋势：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjYwMzUyNTVhZjM0OGFkNWFmMDJmYjA3OGUyOGFhY2JfMTk3OTFlNWVmYWQ1OWIzYzliOTE0MWRhZDZlZmM4ZTVfSUQ6NzY0MjcwMzgxOTEyNTYyMzc3N18xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)

说明：

成功时：action rank 和 video rank 更一致，video latent 更稳定
失败时：action/video 更冲突，video latent 更分散，而且 action/video margin 反而更大



l2 fusion gate 4

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

conda activate RoboTwin

python tts_log_visualizer_zh_v5.py \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs_single_20260524_091945/tts/scan_object/summary.csv \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs/scan_object_tts_video_gated_fusion_n4_b4_w05_wlow0_sp03_dr2_latent_nodecode_20260524_091945.log
```

当前 video fusion 实际上非常保守，最终策略几乎还是 action\-only medoid。视频 latent 有信号，但还没有真正强干预动作选择。

Spearman 中位数是 0\.4，正相关比例约 62\.6%，说明 video latent rank 和 action rank 有一定一致性。但是一致性还不强，不能直接把 video rank 当成可靠 verifier。

98% 的 TTS 决策最终仍然选择 action branch 的第一名，也就是 action medoid / action best。

当前 `w=0.5` 偏弱。更接近等权的应该是 `w=1.0`，因为当前公式里 action 的隐含权重是 1。

Gate 确实在筛掉两类不可靠情况：

action rank 和 video rank 排序相反或接近无关； 

video latent 距离尺度相对 action 距离尺度过大。

问题在于：**筛出来之后，video weight 又太弱，所以 pass 之后也很少真正改变选择。**



l2 fusion gate 8

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

conda activate RoboTwin

python tts_log_visualizer_zh_v5.py \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs_single_20260524_092006/tts/scan_object/summary.csv \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs/scan_object_tts_video_gated_fusion_n8_b8_w05_wlow0_sp03_dr2_latent_nodecode_20260524_092006.log
```

N=8 以后候选更多，第一名和第二名之间更容易接近，所以 action margin 和 video margin 都下降。候选更多也让 action/video 的 top1 精确一致更难，所以 `rank_agree` 和 `selected_video_rank=0` 会下降。

N=8 也给 video 更多机会，因为 video rank 的跨度从 0 到 7，比 N=4 的 0 到 3 更大。在公式：

Gate 做了两件事：

当 action/video rank 一致性差时，直接退回 action\-only； 

当 action/video rank 一致性较高时，video 确实开始影响选择。

Gate 之后的 video 权重还是偏弱。即使在 gate pass 的 544 行里，也有 433 行仍然选择 action rank 0。也就是说：

gate pass 以后，只有 20\.4% 真正改变 action\-only 选择。



```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

conda activate RoboTwin

python tts_log_visualizer_zh_v5.py \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs_single_20260524_132241/tts/put_bottles_dustbin/summary.csv \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs/put_bottles_dustbin_tts_video_rank_fusion_n4_b4_w1_latent_nodecode_20260524_132241.log
```



**video latent rank fusion 有价值，但 ****`N=4 + video_weight=1.0 + no gate`**** 太激进。**

具体风险是：

3770. action/video best 只有 41% 一致； 

3771. fused margin 经常很小或打平； 

3772. N=4 下 Spearman 很粗，取值只有离散档位； 

3773. video latent 的距离尺度和 action 不一致； 

3774. failure 后期会出现高 margin 的错误动作，不能把这种“高置信”直接当成好信号



#### v7  \(working……\)

##### 实现

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a1292d8\-8e80\-8326\-a6ef\-377dcbf511fd

l2\+fusion\+gate\+cluster\+random

\[deploy\_policy\.py\]

\[eval\.sh\]

\[motus\.py\]

\[tts\_log\_visualizer\_zh\_v6\.py\]





队列实现

使用：

```Bash
核心规则：
1. 你把任务脚本放进 pending/gpu0 或 pending/gpu1。
2. worker 按文件名排序取最早的 .sh。
3. 取到后先 mv 到 running，避免重复执行。
4. 等 GPU 空闲。
5. 用 nohup 启动这个任务，并 wait 它结束。
6. 成功则移到 done，失败则移到 failed。
7. 然后继续取下一个。
重要约定：**任务文件内部不要再写 nohup ... &**。任务文件里写前台命令即可，比如 bash eval.sh ... > "$LOG" 2>&1。外层 worker 会负责 nohup 和等待。这样才能保证真正串行。

3. 如何添加一个任务
关键点：**先写到 tmp 文件，再 mv 到 pending**。这样可以避免 worker 读到写了一半的文件。
示例：往 GPU 0 队列添加 N=16 keystone 任务
cd /root/autodl-tmp/RoboTwin/policy/Motus

mkdir -p gpu_queue/tmp gpu_queue/pending/gpu0

JOB_TMP=gpu_queue/tmp/001_turn_switch_keystone_n16_c2_tau05.sh
JOB_DST=gpu_queue/pending/gpu0/001_turn_switch_keystone_n16_c2_tau05.sh

cat > "$JOB_TMP" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

TASK=turn_switch
GPU_ID=0

LOG=logs/${TASK}_tts_keystone_n16_b16_c2_tau05_nodecode_$(date +%Y%m%d_%H%M%S).log

bash eval.sh ${TASK} --gpu-id ${GPU_ID} --tts --tts-num-samples 16 \
  --tts-method keystone \
  --tts-num-clusters 2 \
  --tts-tau 0.5 \
  --tts-kmeans-iters 10 \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  > "$LOG" 2>&1

echo "LOG: $LOG"
SH

chmod +x "$JOB_TMP"
mv "$JOB_TMP" "$JOB_DST"

echo "submitted: $JOB_DST"

4. 如何一次放很多任务
你可以按文件名前缀控制顺序：
001_turn_switch_xxx.sh
002_scan_object_xxx.sh
003_put_bottles_xxx.sh

6. 如何停止队列但不杀当前任务
如果你只想“不再启动新任务”，但当前已经跑起来的任务继续跑：
cd /root/autodl-tmp/RoboTwin/policy/Motus

touch gpu_queue/STOP_gpu0
touch gpu_queue/STOP_gpu1
worker 会在当前任务结束后看到 STOP 文件，然后退出，不再启动后续任务。
恢复队列：
rm -f gpu_queue/STOP_gpu0 gpu_queue/STOP_gpu1
然后重新启动 worker。
```

脚本：

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

cat > gpu_queue_worker.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG
# =========================
GPU_ID="${1:?Usage: ./gpu_queue_worker.sh GPU_ID}"

QUEUE_ROOT="/root/autodl-tmp/RoboTwin/policy/Motus/gpu_queue"

THRESHOLD_MB=100
STABLE_SECONDS=20
CHECK_INTERVAL=2
NO_JOB_SLEEP=5

# 0: keep waiting for new jobs
# 1: exit when queue is empty
EXIT_WHEN_EMPTY=0

PENDING_DIR="${QUEUE_ROOT}/pending/gpu${GPU_ID}"
RUNNING_DIR="${QUEUE_ROOT}/running/gpu${GPU_ID}"
DONE_DIR="${QUEUE_ROOT}/done/gpu${GPU_ID}"
FAILED_DIR="${QUEUE_ROOT}/failed/gpu${GPU_ID}"
LOG_DIR="${QUEUE_ROOT}/logs/gpu${GPU_ID}"
LOCK_ROOT="${QUEUE_ROOT}/locks"
LOCK_DIR="${LOCK_ROOT}/gpu${GPU_ID}.lock"
STOP_FILE="${QUEUE_ROOT}/STOP_gpu${GPU_ID}"

# =========================
# Setup
# =========================
mkdir -p "${PENDING_DIR}" "${RUNNING_DIR}" "${DONE_DIR}" "${FAILED_DIR}" "${LOG_DIR}" "${LOCK_ROOT}"

acquire_lock() {
    while true; do
        if mkdir "${LOCK_DIR}" 2>/dev/null; then
            echo "$$" > "${LOCK_DIR}/pid"
            trap 'rm -rf "${LOCK_DIR}"' EXIT
            return 0
        fi

        old_pid="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
        if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
            echo "[worker] another worker is already running for gpu${GPU_ID}: pid=${old_pid}"
            exit 1
        fi

        echo "[worker] removing stale lock: ${LOCK_DIR}"
        rm -rf "${LOCK_DIR}"
    done
}

get_used_mb() {
    local used_mb
    used_mb="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9')"
    if [ -z "${used_mb}" ]; then
        used_mb=999999
    fi
    echo "${used_mb}"
}

wait_gpu_idle() {
    local stable_count=0
    local used_mb

    echo "[worker] waiting for gpu${GPU_ID} idle at $(date)"
    while true; do
        used_mb="$(get_used_mb)"

        echo "[worker] $(date '+%F %T') gpu=${GPU_ID} used_mb=${used_mb} stable_count=${stable_count}/${STABLE_SECONDS}"

        if [ "${used_mb}" -lt "${THRESHOLD_MB}" ]; then
            stable_count=$((stable_count + CHECK_INTERVAL))
        else
            stable_count=0
        fi

        if [ "${stable_count}" -ge "${STABLE_SECONDS}" ]; then
            echo "[worker] gpu${GPU_ID} idle for ${STABLE_SECONDS}s"
            return 0
        fi

        sleep "${CHECK_INTERVAL}"
    done
}

next_job() {
    find "${PENDING_DIR}" -maxdepth 1 -type f -name "*.sh" | sort | head -n 1
}

run_job() {
    local job="$1"
    local base
    local stamp
    local running_job
    local job_log
    local child_pid
    local status

    base="$(basename "${job}")"
    stamp="$(date +%Y%m%d_%H%M%S)"
    running_job="${RUNNING_DIR}/${stamp}_${base}"
    job_log="${LOG_DIR}/${stamp}_${base%.sh}.log"

    if ! mv "${job}" "${running_job}" 2>/dev/null; then
        echo "[worker] failed to claim job, maybe another worker took it: ${job}"
        return 0
    fi

    chmod +x "${running_job}"

    echo "[worker] claimed job: ${running_job}"
    echo "[worker] job log: ${job_log}"

    wait_gpu_idle

    echo "[worker] launching job at $(date)"
    nohup bash "${running_job}" > "${job_log}" 2>&1 &
    child_pid="$!"
    echo "${child_pid}" > "${running_job}.pid"

    echo "[worker] child pid: ${child_pid}"

    if wait "${child_pid}"; then
        status=0
    else
        status="$?"
    fi

    rm -f "${running_job}.pid"

    if [ "${status}" -eq 0 ]; then
        echo "[worker] job succeeded: ${running_job}"
        mv "${running_job}" "${DONE_DIR}/${stamp}_${base}"
    else
        echo "[worker] job failed with status=${status}: ${running_job}"
        mv "${running_job}" "${FAILED_DIR}/${stamp}_${base}"
    fi

    echo "[worker] job finished at $(date)"
}

# =========================
# Main
# =========================
acquire_lock

echo "[worker] start at $(date)"
echo "[worker] GPU_ID=${GPU_ID}"
echo "[worker] QUEUE_ROOT=${QUEUE_ROOT}"
echo "[worker] PENDING_DIR=${PENDING_DIR}"
echo "[worker] THRESHOLD_MB=${THRESHOLD_MB}"
echo "[worker] STABLE_SECONDS=${STABLE_SECONDS}"
echo "[worker] EXIT_WHEN_EMPTY=${EXIT_WHEN_EMPTY}"
echo "[worker] STOP_FILE=${STOP_FILE}"

while true; do
    if [ -f "${STOP_FILE}" ]; then
        echo "[worker] stop file found: ${STOP_FILE}"
        echo "[worker] exiting without launching more jobs"
        exit 0
    fi

    job="$(next_job || true)"

    if [ -z "${job}" ]; then
        echo "[worker] no pending job at $(date)"
        if [ "${EXIT_WHEN_EMPTY}" -eq 1 ]; then
            echo "[worker] queue empty, exiting"
            exit 0
        fi
        sleep "${NO_JOB_SLEEP}"
        continue
    fi

    run_job "${job}"
done
SH

chmod +x gpu_queue_worker.sh
mkdir -p gpu_queue/pending/gpu0 gpu_queue/pending/gpu1 gpu_queue/logs/gpu0 gpu_queue/logs/gpu1

ls -lh gpu_queue_worker.sh
```

启动两个队列：

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p gpu_queue/logs

WORKER_LOG=gpu_queue/logs/worker_gpu0_$(date +%Y%m%d_%H%M%S).log
nohup ./gpu_queue_worker.sh 0 > "$WORKER_LOG" 2>&1 &
echo "WORKER_GPU0_PID: $!"
echo "WORKER_GPU0_LOG: $WORKER_LOG"

WORKER_LOG=gpu_queue/logs/worker_gpu1_$(date +%Y%m%d_%H%M%S).log
nohup ./gpu_queue_worker.sh 1 > "$WORKER_LOG" 2>&1 &
echo "WORKER_GPU1_PID: $!"
echo "WORKER_GPU1_LOG: $WORKER_LOG"
```

\[1\] 320139

WORKER\_GPU0\_PID: 320139

WORKER\_GPU0\_LOG: gpu\_queue/logs/worker\_gpu0\_20260529\_153032\.log

\[2\] 320141

WORKER\_GPU1\_PID: 320141

WORKER\_GPU1\_LOG: gpu\_queue/logs/worker\_gpu1\_20260529\_153032\.log



##### 实验



###### all

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

# =========================
# N=8, GPU 0
# =========================
LOG=logs/put_bottles_dustbin_tts_video_all_fusion_n8_b8_w1_wlow0_sp03_dr2_c2_tau03_rt1_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh put_bottles_dustbin --gpu-id 0 --tts --tts-num-samples 8 \
  --tts-method video_all_fusion \
  --tts-video \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 1 \
  --tts-video-weight-low 0.0 \
  --tts-gate-spearman-thresh 0.3 \
  --tts-gate-distance-ratio-thresh 2.0 \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-rank-tau 1.0 \
  --tts-kmeans-iters 10 \
  --tts-batch-size 8 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID N8: $!"
echo "LOG N8: $LOG"


# =========================
# N=16, GPU 1
# =========================
LOG=logs/put_bottles_dustbin_tts_video_all_fusion_n16_b16_w1_wlow0_sp03_dr2_c2_tau03_rt1_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh put_bottles_dustbin --gpu-id 1 --tts --tts-num-samples 16 \
  --tts-method video_all_fusion \
  --tts-video \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 1 \
  --tts-video-weight-low 0.0 \
  --tts-gate-spearman-thresh 0.3 \
  --tts-gate-distance-ratio-thresh 2.0 \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-rank-tau 1.0 \
  --tts-kmeans-iters 10 \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID N16: $!"
echo "LOG N16: $LOG"
```

\[1\] 69899

PID N8: 69899

LOG N8: logs/put\_bottles\_dustbin\_tts\_video\_all\_fusion\_n8\_b8\_w1\_wlow0\_sp03\_dr2\_c2\_tau03\_rt1\_latent\_nodecode\_20260524\_185254\.log

\[2\] 69901

PID N16: 69901

LOG N16: logs/put\_bottles\_dustbin\_tts\_video\_all\_fusion\_n16\_b16\_w1\_wlow0\_sp03\_dr2\_c2\_tau03\_rt1\_latent\_nodecode\_20260524\_185254\.log

太长，先中断



###### Ts all

turn\_switch任务

```Bash
# =========================
# 2. Baseline no TTS, GPU 0
# =========================
BASE_LOG=logs/${TASK}_baseline_no_tts_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh ${TASK} --gpu-id ${BASE_GPU} --no-tts \
  > "$BASE_LOG" 2>&1 &

echo "PID BASE: $!"
echo "LOG BASE: $BASE_LOG"


# =========================
# 3. Latest all fusion, N=8, GPU 1
# =========================
TTS_LOG=logs/${TASK}_tts_video_all_fusion_n8_b8_w1_wlow0_sp03_dr2_c2_tau03_rt1_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh ${TASK} --gpu-id ${TTS_GPU} --tts --tts-num-samples 8 \
  --tts-method video_all_fusion \
  --tts-video \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 1 \
  --tts-video-weight-low 0.0 \
  --tts-gate-spearman-thresh 0.3 \
  --tts-gate-distance-ratio-thresh 2.0 \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-rank-tau 1.0 \
  --tts-kmeans-iters 10 \
  --tts-batch-size 8 \
  --no-tts-decode-video \
  > "$TTS_LOG" 2>&1 &

echo "PID TTS N8: $!"
echo "LOG TTS N8: $TTS_LOG"
```

\[1\] 73530

PID BASE: 73530

LOG BASE: logs/turn\_switch\_baseline\_no\_tts\_20260524\_193357\.log

\[2\] 73532

PID TTS N8: 73532

LOG TTS N8: logs/turn\_switch\_tts\_video\_all\_fusion\_n8\_b8\_w1\_wlow0\_sp03\_dr2\_c2\_tau03\_rt1\_latent\_nodecode\_20260524\_193357\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NTFkOTU2YWQ5ODk5MDJmMmM0MzQwMmYzOGZjZjYxMzFfZWNlZTNiZTE1ZjUzNzc0M2I0MTczNmMwMzhmZjAwZGJfSUQ6NzY0MzQ1MDA3MzM1MjQ3MzU0M18xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MDJhYmEyZGI2NWFjNDJlNzEwOGMyM2NjNzhiNzIzYWZfOTljNWQxMGFmNWE5YjVhZTg2M2MwZTQyZDc2N2Y4ZTFfSUQ6NzY0MzQ1MDAxNTAzMzQ5NDcxOV8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



调参

```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus
mkdir -p logs

LOG=logs/turn_switch_tts_video_all_fusion_n8_b8_w05_wlow01_sp03_dr3_c2_tau03_rt07_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh turn_switch --gpu-id 0 --tts --tts-num-samples 8 \
  --tts-method video_all_fusion \
  --tts-video \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 0.5 \
  --tts-video-weight-low 0.1 \
  --tts-gate-spearman-thresh 0.3 \
  --tts-gate-distance-ratio-thresh 3.0 \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-rank-tau 0.7 \
  --tts-batch-size 8 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```



Autodl snd1 参数2

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

LOG=logs/turn_switch_tts_video_all_fusion_n8_b8_w05_wlow01_sp03_dr3_c2_tau03_rt07_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh turn_switch --gpu-id 0 --tts --tts-num-samples 8 \
  --tts-method video_all_fusion \
  --tts-video \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 0.5 \
  --tts-video-weight-low 0.1 \
  --tts-gate-spearman-thresh 0.3 \
  --tts-gate-distance-ratio-thresh 3.0 \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-rank-tau 0.7 \
  --tts-batch-size 8 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 1690

PID: 1690

LOG: logs/turn\_switch\_tts\_video\_all\_fusion\_n8\_b8\_w05\_wlow01\_sp03\_dr3\_c2\_tau03\_rt07\_latent\_nodecode\_20260528\_174911\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NGI0ZWQyMDhiMDg4NDNmZWQwZmU1NWEwNTY2NWU1OGJfYjEwNTNjMGQwYjA4YmY1ZjU2NmU4MzNjMjQxMGFjMTRfSUQ6NzY0NDk2NjYzMzQzNzUzMTMxNF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



Autodl snd2 参数2

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

LOG=logs/turn_switch_tts_video_all_fusion_n16_b16_w05_wlow01_sp03_dr3_c2_tau03_rt07_latent_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh turn_switch --gpu-id 0 --tts --tts-num-samples 16 \
  --tts-method video_all_fusion \
  --tts-video \
  --tts-video-feature latent \
  --tts-rank-fusion-method weighted_borda \
  --tts-video-weight 0.5 \
  --tts-video-weight-low 0.1 \
  --tts-gate-spearman-thresh 0.3 \
  --tts-gate-distance-ratio-thresh 3.0 \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-rank-tau 0.7 \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 2372

PID: 2372

LOG: logs/turn\_switch\_tts\_video\_all\_fusion\_n16\_b16\_w05\_wlow01\_sp03\_dr3\_c2\_tau03\_rt07\_latent\_nodecode\_20260528\_180427\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YzRlYjQxOWExMzJiNDljYjBhNDE4MDYxNzU1MDI3YmNfOWM5NWNmYmE1YjJiZGU5MDU4OTBlYWJjMGMwZWRlZmFfSUQ6NzY0NDk2NjY5MDE2NTQ2MDE4M18xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



###### l2

snd2 0 l2 min 4

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

TASK=turn_switch
GPU_ID=0

LOG=logs/${TASK}_tts_global_medoid_n4_b4_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh ${TASK} --gpu-id ${GPU_ID} --tts --tts-num-samples 4 \
  --tts-method global_medoid \
  --tts-batch-size 4 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 1893

PID: 1893

LOG: logs/turn\_switch\_tts\_global\_medoid\_n4\_b4\_nodecode\_20260528\_235146\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MjFjMWM4NDUzZmNlYzc2NjcxZmQyMzNhZmQ5MDg2ZjlfYzMwYmMwYjllY2QzZDg4NDg3MjQwMmEwMzMzN2JjZThfSUQ6NzY0NTEzNDk2MjIzNjYwNzQ1OF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



snd2 1 l2 min 8

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

TASK=turn_switch
GPU_ID=1

LOG=logs/${TASK}_tts_global_medoid_n8_b8_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh ${TASK} --gpu-id ${GPU_ID} --tts --tts-num-samples 8 \
  --tts-method global_medoid \
  --tts-batch-size 8 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[2\] 2183

PID: 2183

LOG: logs/turn\_switch\_tts\_global\_medoid\_n8\_b8\_nodecode\_20260528\_235159\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YWU0ZjI2ZmFiYWQyMGE0NjUxYTE2ZGU0YjZkMGY4MzlfYzlkMjVkZTA0MjE0NWU1MWM5MTUwOTc2YjdjYmMyNTlfSUQ6NzY0NTEzNTAwOTUzNTQ5NTExMF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



autodl上实现队列

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

cat > wait_gpu_and_run.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG
# =========================
GPU_ID="$1"
THRESHOLD_MB=100
STABLE_SECONDS=20
CHECK_INTERVAL=2

shift
RUN_CMD="$*"

echo "[queue] GPU_ID=${GPU_ID}"
echo "[queue] THRESHOLD_MB=${THRESHOLD_MB}"
echo "[queue] STABLE_SECONDS=${STABLE_SECONDS}"
echo "[queue] RUN_CMD=${RUN_CMD}"
echo "[queue] start waiting at $(date)"

stable_count=0

while true; do
    used_mb=$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9')

    if [ -z "$used_mb" ]; then
        used_mb=999999
    fi

    echo "[queue] $(date '+%F %T') gpu=${GPU_ID} used_mb=${used_mb} stable_count=${stable_count}/${STABLE_SECONDS}"

    if [ "$used_mb" -lt "$THRESHOLD_MB" ]; then
        stable_count=$((stable_count + CHECK_INTERVAL))
    else
        stable_count=0
    fi

    if [ "$stable_count" -ge "$STABLE_SECONDS" ]; then
        echo "[queue] GPU ${GPU_ID} idle for ${STABLE_SECONDS}s. Launching at $(date)"
        bash -lc "$RUN_CMD"
        echo "[queue] submitted command at $(date)"
        exit 0
    fi

    sleep "$CHECK_INTERVAL"
done
SH

chmod +x wait_gpu_and_run.sh
mkdir -p logs

ls -lh wait_gpu_and_run.sh
```



放等待任务

16

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

QUEUE_LOG=logs/queue_turn_switch_global_medoid_n16_b16_gpu0_$(date +%Y%m%d_%H%M%S).log

nohup ./wait_gpu_and_run.sh 0 '
cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

TASK=turn_switch
GPU_ID=0

LOG=logs/${TASK}_tts_global_medoid_n16_b16_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh ${TASK} --gpu-id ${GPU_ID} --tts --tts-num-samples 16 \
  --tts-method global_medoid \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
' > "$QUEUE_LOG" 2>&1 &

echo "QUEUE_PID N16: $!"
echo "QUEUE_LOG N16: $QUEUE_LOG"
```

\[3\] 5279

QUEUE\_PID N16: 5279

QUEUE\_LOG N16: logs/queue\_turn\_switch\_global\_medoid\_n16\_b16\_gpu0\_20260528\_235756\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OGU5YzVjYWQ3ZDhkMjFkMjIxMzlhOWM3OWExOGMwMWNfNDI3M2Y2Y2FjOWVlNzM0MDk2YzQwNzY5MzE2Y2QyNTFfSUQ6NzY0NTEzNTA0NzYzMzEyODY1M18xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



32

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

QUEUE_LOG=logs/queue_turn_switch_global_medoid_n32_b32_gpu1_$(date +%Y%m%d_%H%M%S).log

nohup ./wait_gpu_and_run.sh 1 '
cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

TASK=turn_switch
GPU_ID=1

LOG=logs/${TASK}_tts_global_medoid_n32_b32_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh ${TASK} --gpu-id ${GPU_ID} --tts --tts-num-samples 32 \
  --tts-method global_medoid \
  --tts-batch-size 32 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
' > "$QUEUE_LOG" 2>&1 &

echo "QUEUE_PID N32: $!"
echo "QUEUE_LOG N32: $QUEUE_LOG"
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NDMxOTMyN2QzOGYwYzdjY2Q1NWMxYjM3NGFjM2I4NDhfOTlhZDY4ZDQ1NTNkMzk0Y2QwNmM0MTIzOGUyY2Y3YjhfSUQ6NzY0NTEzNTA3MzQ2NTI0MDc2NV8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



###### keystone



4

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

TASK=turn_switch
GPU_ID=0

LOG=logs/${TASK}_tts_keystone_n4_b4_c2_tau03_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh ${TASK} --gpu-id ${GPU_ID} --tts --tts-num-samples 4 \
  --tts-method keystone \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  --tts-batch-size 4 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID N4: $!"
echo "LOG N4: $LOG"
```

\[1\] 165645

PID N4: 165645

LOG N4: logs/turn\_switch\_tts\_keystone\_n4\_b4\_c2\_tau03\_nodecode\_20260529\_102641\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MjNhYWI4MzZlZGRmYzVhNjg3NTU0MjEwYmYxNTk5MDdfYjViNGUxMmQ0ZTczZTJiMzkxZDU5NTA2YWI5ZDExMDlfSUQ6NzY0NTE4Njc4NTEzMTkwODI5Nl8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



8

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

TASK=turn_switch
GPU_ID=1

LOG=logs/${TASK}_tts_keystone_n8_b8_c2_tau03_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh ${TASK} --gpu-id ${GPU_ID} --tts --tts-num-samples 8 \
  --tts-method keystone \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  --tts-batch-size 8 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID N8: $!"
echo "LOG N8: $LOG"
```

\[2\] 165880

PID N8: 165880

LOG N8: logs/turn\_switch\_tts\_keystone\_n8\_b8\_c2\_tau03\_nodecode\_20260529\_102652\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MGI0MTE2NGMxMmI1NDBlNDhjODc5MmUyMTJmMDcyM2ZfNzhmNjJjNWQyMzc0OTZhNjkzYmVlODUyMDM2ZWQ0ZWFfSUQ6NzY0NTE4NjgxODU5MzY4ODc4MV8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



16

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

QUEUE_LOG=logs/queue_turn_switch_keystone_n16_b16_c2_tau03_gpu0_$(date +%Y%m%d_%H%M%S).log

nohup ./wait_gpu_and_run.sh 0 '
cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

TASK=turn_switch
GPU_ID=0

LOG=logs/${TASK}_tts_keystone_n16_b16_c2_tau03_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh ${TASK} --gpu-id ${GPU_ID} --tts --tts-num-samples 16 \
  --tts-method keystone \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID N16: $!"
echo "LOG N16: $LOG"
' > "$QUEUE_LOG" 2>&1 &

echo "QUEUE_PID N16: $!"
echo "QUEUE_LOG N16: $QUEUE_LOG"
```

\[3\] 166399

QUEUE\_PID N16: 166399

QUEUE\_LOG N16: logs/queue\_turn\_switch\_keystone\_n16\_b16\_c2\_tau03\_gpu0\_20260529\_102710\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ODM4OTViNTM1ZWIxNDNlODM2ODJkODk1ZmNmZGIyN2NfYjU0MjFmYjkwMmE0YzNiMmJhN2EzZmY2Y2UxMDRiZTZfSUQ6NzY0NTE4OTQ1MDkwNTE5MzQ0NF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



32

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

QUEUE_LOG=logs/queue_turn_switch_keystone_n32_b32_c2_tau03_gpu1_$(date +%Y%m%d_%H%M%S).log

nohup ./wait_gpu_and_run.sh 1 '
cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

TASK=turn_switch
GPU_ID=1

LOG=logs/${TASK}_tts_keystone_n32_b32_c2_tau03_nodecode_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh ${TASK} --gpu-id ${GPU_ID} --tts --tts-num-samples 32 \
  --tts-method keystone \
  --tts-num-clusters 2 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  --tts-batch-size 32 \
  --no-tts-decode-video \
  > "$LOG" 2>&1 &

echo "PID N32: $!"
echo "LOG N32: $LOG"
' > "$QUEUE_LOG" 2>&1 &

echo "QUEUE_PID N32: $!"
echo "QUEUE_LOG N32: $QUEUE_LOG"
```

\[4\] 166889

QUEUE\_PID N32: 166889

QUEUE\_LOG N32: logs/queue\_turn\_switch\_keystone\_n32\_b32\_c2\_tau03\_gpu1\_20260529\_102728\.log

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MmViMDk4YzlhZTFhOTJkMDJiNjJkNDM4N2I5YmYzMWJfMGU4NWUyZTgxYzhjZWVmNGU4ZDdiNGQ5OTNmZTA4ZDdfSUQ6NzY0NTIyNjQ2MTE1MTk1NjE3MV8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



###### Keystone quene



GPU0:

001  Cluster2 tau0\.5 n=4

002  Cluster2 tau0\.4 n=16

003  Cluster3 tau0\.3 n=16



GPU1:

001  Cluster2 tau0\.5 n=8

002  Cluster2 tau0\.4 n=8

003  Cluster3 tau0\.3 n=8



放队列

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

# =========================
# CONFIG
# =========================
QUEUE_ROOT="/root/autodl-tmp/RoboTwin/policy/Motus/gpu_queue"
BATCH="$(date +%Y%m%d_%H%M%S)"

mkdir -p \
  "${QUEUE_ROOT}/tmp" \
  "${QUEUE_ROOT}/pending/gpu0" \
  "${QUEUE_ROOT}/pending/gpu1" \
  "${QUEUE_ROOT}/logs/gpu0" \
  "${QUEUE_ROOT}/logs/gpu1"

# 确保 STOP 文件不存在，否则 worker 会直接退出
rm -f "${QUEUE_ROOT}/STOP_gpu0" "${QUEUE_ROOT}/STOP_gpu1"

# 确认 worker 脚本存在
if [ ! -x ./gpu_queue_worker.sh ]; then
  echo "ERROR: ./gpu_queue_worker.sh not found or not executable"
  echo "先创建 gpu_queue_worker.sh，再提交队列任务。"
  exit 1
fi

# =========================
# Start workers
# 如果对应 GPU 已经有 worker，脚本自身的 lock 会阻止重复 worker。
# =========================
WORKER_LOG="${QUEUE_ROOT}/logs/gpu0/worker_gpu0_${BATCH}.log"
nohup ./gpu_queue_worker.sh 0 > "$WORKER_LOG" 2>&1 &
echo "WORKER_GPU0_PID: $!"
echo "WORKER_GPU0_LOG: $WORKER_LOG"

WORKER_LOG="${QUEUE_ROOT}/logs/gpu1/worker_gpu1_${BATCH}.log"
nohup ./gpu_queue_worker.sh 1 > "$WORKER_LOG" 2>&1 &
echo "WORKER_GPU1_PID: $!"
echo "WORKER_GPU1_LOG: $WORKER_LOG"

# =========================
# Helper: submit one keystone job
# =========================
submit_keystone_job() {
  local GPU_ID="$1"
  local N="$2"
  local C="$3"
  local TAU_LABEL="$4"
  local TAU_VALUE="$5"
  local ORDER="$6"

  local JOB_NAME="${BATCH}_${ORDER}_turn_switch_keystone_n${N}_b${N}_c${C}_tau${TAU_LABEL}_gpu${GPU_ID}.sh"
  local JOB_TMP="${QUEUE_ROOT}/tmp/${JOB_NAME}"
  local JOB_DST="${QUEUE_ROOT}/pending/gpu${GPU_ID}/${JOB_NAME}"

  cat > "$JOB_TMP" <<SH
#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

TASK=turn_switch
GPU_ID=${GPU_ID}

LOG=logs/\${TASK}_tts_keystone_n${N}_b${N}_c${C}_tau${TAU_LABEL}_nodecode_\$(date +%Y%m%d_%H%M%S).log

bash eval.sh \${TASK} --gpu-id \${GPU_ID} --tts --tts-num-samples ${N} \\
  --tts-method keystone \\
  --tts-num-clusters ${C} \\
  --tts-tau ${TAU_VALUE} \\
  --tts-kmeans-iters 10 \\
  --tts-batch-size ${N} \\
  --no-tts-decode-video \\
  > "\$LOG" 2>&1

echo "LOG: \$LOG"
SH

  chmod +x "$JOB_TMP"
  mv "$JOB_TMP" "$JOB_DST"
  echo "submitted: $JOB_DST"
}

# =========================
# Submit GPU0 jobs
# =========================
submit_keystone_job 0 4  2 05 0.5 001
submit_keystone_job 0 16 2 04 0.4 002
submit_keystone_job 0 16 3 03 0.3 003

# =========================
# Submit GPU1 jobs
# =========================
submit_keystone_job 1 8 2 05 0.5 001
submit_keystone_job 1 8 2 04 0.4 002
submit_keystone_job 1 8 3 03 0.3 003

echo
echo "===== pending gpu0 ====="
find "${QUEUE_ROOT}/pending/gpu0" -maxdepth 1 -type f | sort

echo
echo "===== pending gpu1 ====="
find "${QUEUE_ROOT}/pending/gpu1" -maxdepth 1 -type f | sort
```





##### 日志



```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

conda activate RoboTwin

python tts_log_visualizer_zh_v6.py \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs_single_20260524_193358/tts/turn_switch/summary.csv \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs/turn_switch_tts_video_all_fusion_n8_b8_w1_wlow0_sp03_dr2_c2_tau03_rt1_latent_nodecode_20260524_193357.log
```

实际运行时大部分时间并没有真正用 video fusion，而是很快退回 action\-only。

video gate 大部分时间是关的

video pass 几乎集中在前两个 chunk：

实际行为是：

前 1 到 2 个 chunk：有机会用 video/action fusion
之后：几乎退化成 action\-only cluster \+ rank\_softmax

gate 为什么大部分时间 fallback，真正卡住的是 **distance ratio**。多数时候 video latent 的候选分散程度大约是 action 的 3 倍，gate 认为 video signal 不可靠，于是直接关掉 video branch

cluster 没有过度剪枝，反而偏弱；最大 cluster 经常有 7 个候选。



turn\_switch 全 参数2 8 16

也许太多方法效果不好，因为实际上信息没啥用，拉低了；筛有用的方法

```Bash
最大问题：video gate 大多数时候没有真正通过
你这组参数是：
video_weight = 0.5
video_weight_low = 0.1
gate_spearman_thresh = 0.3
gate_distance_ratio_thresh = 3.0
rank_tau = 0.7
但实际统计是：
指标N=8N=16
gate pass 比例**26.5%25.5%**
fallback 比例**73.5%74.5%**
平均 effective video weight**0.2060.202**
这说明大部分 TTS step 都在 fallback，用的是 video_weight_low=0.1，不是你设想中的 video_weight=0.5。所以这个实验名义上是 video fusion，实际大部分时候是 **弱视频权重的 action dominated fusion**。

两个实验的 action/video ranking agreement 都偏低：
指标N=8N=16
action best 和 video best 一致比例**17.7%11.3%**
rank Spearman 均值**0.2560.283**
video/action distance ratio 均值**2.903.08**
解释：
rank_agree 很低，说明视频 latent 认为最好的候选，通常不是 action consensus 认为最好的候选。 
Spearman 只有 0.25 到 0.28，说明两个排序只有弱正相关。

N=16 不仅没有更常选中 action best，也没有更常选中 video best。它只是把候选空间扩大了，但 fusion/rank-softmax 没能更准确地挑出好候选。

cluster 之后还保留了大多数候选。也就是说，video_all_fusion 里的 cluster gate 没有强力过滤掉坏候选。N=16 时，最大的 cluster 仍然平均保留 13 个左右，基本不是“从 16 里筛少数优质样本”，而是“从 16 里保留大部分再做 rank-softmax”。
```



turn\_switch l2 min cluster  4 8 16 32

采样越大，聚类要越多？

```Bash
3. stage 分布说明了什么
指标N=4N=8
cluster_medoid730 / 751 = **97.2%**693 / 730 = **94.9%**
guard_global_medoid21 / 751 = **2.8%**37 / 730 = **5.1%**
这说明 tau=0.3 很严格，大部分 step 都被判定为“非单峰”，于是进入 k-means cluster 分支。也就是说，这组 keystone 基本不是“偶尔聚类”，而是**几乎每次都聚类**。

N=8 时平均保留候选比例：
selected_cluster_size / samples = 0.785
也就是平均保留约 79% 候选。最常见的模式是 7|1，即 7 个候选在大 cluster，1 个候选被视为 outlier。
所以 **N=8 的 keystone 主要是在过滤少数 outlier，而不是从多个强模式中选择一个模式。** 这解释了为什么它和 global medoid N=8 很接近，甚至略低。

也就是说，即使 keystone 改选了非 global medoid，通常也只是从全局 L2 第一名换到非常接近的候选。这说明 cluster 没有在 L2 层面做很激进的选择。
这也是为什么它没有明显提升：**它改变了选择，但改变幅度通常很小。**

10. 一个值得注意的异常：首步 L2 有时极大
在 N=8 里有一些 step 0 的 L2 非常大，比如 selected avg L2 到 3 到 4 以上，而多数行中位数只有 0.21 左右。这些大值主要出现在 episode 的第 0 个 chunk。类似现象在 N=4 也存在。
这说明初始状态或初始视觉条件下，模型的候选动作分布有时会非常分散。它不一定导致失败，有些高 L2 首步 episode 仍然成功。所以这更像是任务初始状态多样性或候选分布尺度变化，而不是简单错误信号。
后续如果做更细分析，可以把 step=0 和 step>0 分开统计，不要混在一起看全局 L2 均值。

这次 N=16 的分布是：
stage次数比例
cluster_medoid661**94.2%**
guard_global_medoid41**5.8%**
这和 N=8 很接近。说明在 tau=0.3 下，**绝大多数 step 都会进入聚类分支**。tau=0.3 仍然是偏激进的聚类设置。
```



```Bash
实验配置成功率CSV 行数
N=4keystone, C=2, tau=0.5**80/100 = 80.0%**695
N=8keystone, C=2, tau=0.5**74/100 = 74.0%**833

这很关键。tau=0.5 对 N=8 来说太保守了，绝大多数 step 都不聚类，直接退回 global medoid。N=4 还保留了接近一半的 cluster 分支，所以它仍然有比较明显的 keystone 行为。
```



```Bash
2. tau=0.4 让 Keystone 变得更保守
Keystone 的逻辑是：
s_score < tau  -> guard_global_medoid，不聚类，直接选全局 L2 medoid
s_score >= tau -> cluster_medoid，做 k-means 后选最大 cluster 内 medoid
这次 stage 分布：
实验guard_global_medoidcluster_medoid
N=8 tau0.4**28.7%**71.3%
N=16 tau0.4**39.1%**60.9%

N=16 的 cluster 平均保留 88.5% 的候选，说明它主要还是剔除少数 outlier，而不是强模式选择。同时它的 selected_minus_min 极小，说明即使进入 cluster，最终选出来的 action 也几乎等于 global medoid。

对 N=8
tau0.3: 79%
tau0.4: 78%
tau0.5: 74%
趋势很清楚：**tau 变大后更保守，但成功率下降。**
N=8 下，cluster 分支似乎是有用的。tau0.5 让 79.5% step 直接 guard，效果最差；tau0.4 比 tau0.5 好，但仍不如 tau0.3。
对 N=16
tau0.3: 80%
tau0.4: 76%
N=16 下 tau0.4 明显不好。它让 39% step 退回 global medoid，同时 cluster 又主要是弱 outlier filtering，不足以带来更强选择。
```



#### v8

Tts 的 hack 的讨论，两个对话，通读

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a05ac17\-1bf4\-83ab\-b9b4\-5b0df5eded1f



tta类调研



? 如何处理错误共识？

```Plain Text
video_margin 成功 vs 失败
失败 episode 的 video_margin 反而更高。
这和 action_margin 类似，可能说明失败状态下视频分支形成了更“明确”的共识，但这个共识不一定对任务成功有帮助。换句话说，video_margin 大不一定代表可靠，它可能只是 confidently wrong。
```



和任务等价的扰动一致；扰动后也加入

```SQL
8. Co-rewarding 启发：多视角语言一致性可以作为 TTS 分数
Co-rewarding 的核心问题是 self-rewarding 容易陷入“单视角自洽幻觉”，所以它引入 data-side 的语义相近问题互证，以及 model-side 的 slowly-updated reference teacher。
放到 Motus，我们可以做 **cross-view prompt consistency**：
对同一个任务构造多个等价 prompt：
l(1),l(2),…,l(M)l^{(1)}, l^{(2)}, \dots, l^{(M)}l(1),l(2),…,l(M)
例如：
 原始 task instruction； 
 去掉冗余 scene prefix 的 instruction； 
 保留对象和目标，但改写句式； 
 masked / weak instruction 版本。 
如果这些 prompt 语义等价，那么动作和视频预测应该相近：
Sicross=−min⁡m[∥ϕA(Ai(0))−ϕA(Amedoid(m))∥22+λV∥ϕV(Vi(0))−ϕV(Vmedoid(m))∥22]S_i^{\text{cross}} = - \min_m \left[ \| \phi_A(A_i^{(0)})-\phi_A(A_{\text{medoid}}^{(m)}) \|_2^2 + \lambda_V \| \phi_V(V_i^{(0)})-\phi_V(V_{\text{medoid}}^{(m)}) \|_2^2 \right]Sicross=−mmin[∥ϕA(Ai(0))−ϕA(Amedoid(m))∥22+λV∥ϕV(Vi(0))−ϕV(Vmedoid(m))∥22]
也就是说，正常 prompt 下的候选，如果和 paraphrase prompt 下的候选族完全不一致，就降低分数。
这个方法和 MG-Select 的差别是：
MG-Select 偏好 **正常条件相对 masked reference 有足够偏离**；
Co-rewarding-style consistency 偏好 **语义等价条件之间保持一致**。
它们不矛盾。一个看“任务信息是否产生足够条件化”，一个看“等价任务表达是否稳定”。
但这个方向要谨慎。机器人 prompt 中空间词、对象词、动作词不能乱改，否则所谓 paraphrase 会改变任务语义。第一版只改 scene prefix 或句式，不改 object、relation、goal。
```



选择平滑去噪

```SQL
6. Self-Verifying / CoVo 方向：利用去噪轨迹，而不是只看最终 action
Motus 的 inference_step() 现在每个 step 都有：
xt+1V=xtV+vtVΔtx^V_{t+1}=x^V_t+v^V_t\Delta txt+1V=xtV+vtVΔtxt+1A=xtA+vtAΔtx^A_{t+1}=x^A_t+v^A_t\Delta txt+1A=xtA+vtAΔt
代码里就是：
video_latent = video_latent + video_velocity * dt
action_latent = action_latent + action_velocity * dt
这给我们一个很好的插入点。
可以加三个 trace-level 分数。
6.1 Vector-field consistency
一个好候选的 denoising trajectory 不应该剧烈抖动：
Sifield=−∑t∥vi,t+1A−vi,tA∥22−λV∑t∥vi,t+1V−vi,tV∥22S_i^{\text{field}} = - \sum_t \left\| v^A_{i,t+1}-v^A_{i,t} \right\|_2^2 - \lambda_V \sum_t \left\| v^V_{i,t+1}-v^V_{i,t} \right\|_2^2Sifield=−t∑vi,t+1A−vi,tA22−λVt∑vi,t+1V−vi,tV22
这对应“去噪方向稳定”。
6.2 Predicted-clean consistency
Flow matching 里可以从中间状态估计 clean endpoint。具体形式要按 Motus 的时间方向校准，但概念上是：
x^i,t0=g(xi,t,vi,t,t)\hat x^0_{i,t} = g(x_{i,t}, v_{i,t}, t)x^i,t0=g(xi,t,vi,t,t)
然后要求不同 step 估计出的 x^0\hat x^0x^0 一致：
Siclean=−∑t∥x^i,t+10−x^i,t0∥22S_i^{\text{clean}} = - \sum_t \left\| \hat x^0_{i,t+1} - \hat x^0_{i,t} \right\|_2^2Siclean=−t∑x^i,t+10−x^i,t022
这个就是 Self-Verifying 思路的 Motus 版本：不用外部 verifier，而是看模型自己的 denoising / flow trajectory 是否自洽。Self-Verifying 论文明确把 diffusion model 自身看作 distribution estimator，并用中间 denoising step 的信号来选择样本，不过该 OpenReview 版本是 withdrawn submission，所以我们可以借鉴思想，但论文引用时要谨慎。
6.3 CoVo-style consistency plus volatility
CoVo 的思想不是只看最终答案，而是看 trajectory 的一致性和波动性。放到 Motus，可以写成：
Ci=−1N−1∑j≠i∥ϕτ(ξi)−ϕτ(ξj)∥22C_i = - \frac{1}{N-1} \sum_{j\ne i} \left\| \phi_\tau(\xi_i)-\phi_\tau(\xi_j) \right\|_2^2Ci=−N−11j=i∑∥ϕτ(ξi)−ϕτ(ξj)∥22Voli=Vart(ψi,t)Vol_i = \text{Var}_t(\psi_{i,t})Voli=Vart(ψi,t)SiCoVo=Ci−β∣Voli−medianj(Volj)∣S_i^{\text{CoVo}} = C_i - \beta |Vol_i-\text{median}_j(Vol_j)|SiCoVo=Ci−β∣Voli−medianj(Volj)∣
也就是：候选轨迹要和多数候选一致，但内部也不能完全僵死或剧烈乱跳。对机器人控制，我不建议直接加 curiosity bonus 到执行策略里，因为探索性可能导致危险动作。curiosity 更适合离线生成 OPD 数据时保留多样样本。
```



3 其他融合trick

```Plain Text
第二，video latent 不应该无条件强行加入。更合理的是 **confidence-gated fusion**：
 如果 action_margin 很大，说明 action medoid 很确定，少让 video 覆盖。 
 如果 action_margin 很小，而 video_margin 较大，说明动作候选难分，但视频候选有共识，可以允许 video 覆盖。 
 如果 rank_spearman 很低甚至为负，说明 action 和 video 排序冲突，应该降低 video weight 或 fallback 到 action medoid。 
 如果 distance_ratio_video_over_action 极大，说明 video latent 很分散，可能是困难状态或视频预测不稳，也不应该盲目加权。
 
 distance_ratio_video_over_action 成功 vs 失败
失败 episode 的 ratio 高于成功 episode。
这比较有价值：失败时 video latent 相对 action 更分散，说明视频分支在失败场景下可能更不稳定。它可以作为 gating signal：当 ratio 过高时，不要过度相信 video rank。
```



https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a0ef226\-402c\-832c\-a690\-4320ca8292e3 

整理未选方案，concat，token等，后面再说。



更多方法
（目的是更高的点数；也许不太重要，是为了方法好看）
https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a0d86dc\-6a48\-832f\-8ebc\-9771dc88b9ac

L2 min \+ mask输入

应该也只改get\_action？

怎么mask？

去噪轨迹

motus\.py返回更多

检查deploy的所有相关调用级联修改？

具体实现，参考2rw？



更多加速？

```Plain Text
1 缓存 und_tokens，不要每个 denoise step 重跑 VLM
4.2 当前最大可疑浪费：VLM/understanding features 每个 denoise step 都重算
在 inference_step 里，你先在 loop 外算了一次：
und_tokens = self.und_module.extract_und_features(vlm_inputs)
但进入每个 denoise step 后，又重新算：
und_tokens = self.und_module.extract_und_features(vlm_inputs)
这意味着如果 num_inference_steps=50，同一个 observation、同一个 instruction、同一个 image 的 VLM features 可能被算 50 次。batch=16 时，还在 batch 维度上重复了 16 份相同图像和文本。这个非常可能是当前最大加速点之一。
更关键的是，VLM input batch 构造会把 list 里的 input ids、attention mask、pixel values、image grid 拼成 batch，然后走 get_image_features 和 language model，这对 identical copies 来说是明显重复计算。

2 TTS batch 下只 encode 一次 condition frame latent，再 expand 到 B
4.3 condition frame VAE encode 也被重复 B 次
TTS batch 里 current_frame_b = current_frame.repeat(batch_size, ...)，然后 inference_step 对这个 B=16 的 first frame 做 VAE encode。条件帧完全一样，因此这也是重复计算。

3 TTS batch 下 T5 context 只算一次，再 expand 到 B

4 用 torch.inference_mode() 替代 torch.no_grad()
具体看看有没有风险？
torch.inference_mode() 是比 torch.no_grad() 更强一点的推理上下文。no_grad() 只是不记录 autograd 梯度；inference_mode() 还会禁用 view tracking 和 version counter bumps，所以理论上推理会更省一点开销。PyTorch 官方说明它比 no_grad 少一些额外开销，但也更严格
Motus 里有一些地方虽然是推理，但会做 tensor view、autocast、VAE encode/decode、Qwen VLM 的 hidden states 提取。inference_mode() 通常没问题，但它比 no_grad() 更严格，万一某个模块依赖 version counter 或 view 行为，可能出现隐蔽问题。


```



#### 总表



\(NOW\)

turn\_switch 400 tts

||p|1|4|8|16|32|64|
|---|---|---|---|---|---|---|---|
|base||75||||||
|L2 min|||74|80|79|77||
|L2 min \+ Cluster|c2 t03||77|79|80|74||
||t05||80|74|x|x||
||t04|||78|76|x||
||c3|||75|78|x||
||c4||||81|||
||c5||||79|p1||
|L2 min \+ Cluster \+ ref|r01 k16 c4||||72|||
|L2 min \+ Fusion \+ Cluster \+ Random \+ Gate||||76||||
||p2|||75|74|||
||p3|||||||
|v level \+ l2 min|clean s5||78|||||
||noise s5||78|74||||
||metric s5|||77||||
||noise s5 g0005||75|72|75|81||
||metric s5 g0005||79|78|75|74||
||noise s2 g0005||76|72|74|79||
||metric s2 g0005||78|79|71|76||
|V level \+ l2 min \+ cluster|noise s5 c2||80|76|75|77|72|
||noise s2 c2||72|76|78|75||
||noise s5 c3|||77||||
||noise s2 c3||||75|||
||noise s5 c4||||73|||
||noise s2 c4||||76|||
||noise4 s2 c4||||77|||
||noise4 s5 c4||||78|||
||noise6 s5 c4||||80|||
||noise6 s2 c4||||72|||
||noise6 s5 c5||||77|||
||noise8 s5 c4||||77|||
||noise8 s5 c5||||80|||
||noise10 s5 c4||||78|||
|V level \+ l2 min \+ cluster \+ ref|noise s5 w05|||||74|67|
||noise s2 c2 w02|||||74||
||noise4 s2 c4 w02||||74|||
||noise4 s2 c4 w01||||72|||



turn\_switch 400 after chunk opd

|||10|20|30|40|50|60|
|---|---|---|---|---|---|---|---|
|base|75|||||||
|l2 min n8|80|||||||
|v1 p1|||||||73|
|v1 p2||77|74|75|72|73|73|

|||2|4|6|8|10|12|14|16|18|20|22|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|base|75||||||||||||
|l2 min cluster n16 c4|81||||||||||||
|v2||75|80|77|73|75|80|79|76|83|79|76|

|||2|6|10|14|18|22|26|30|34|38|42|46|50|54|58|62|66|70|74|78|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|base|75|||||||||||||||||||||
|l2 min cluster n16 c4|81|||||||||||||||||||||
|v3 p1||78|78|76|75|71|79|74|73|77|78|73|74|74|72|77|72|75|74|73|75|





scan\_object 500

||para||4 \(b4\)|8|8 \(b8\)|16|16 \(b16\)|32 \(b32\)|
|---|---|---|---|---|---|---|---|---|
|base||54|||||||
|L2 min||||56||63|||
|L2 min \+ Cluster|||57|67\.2 \(45/67\)||55\.4 \(36/65\)||53\.2 \(25/47\)|
|L2 min \+ Random|||57|53\.6 \(30/56\)||68\.4 \(54/79\)||54 \(27/50\)|
|L2 min \+ Cluster \+ Random||||62||56\.5 \(39/69\)|||
|L2 min \+ Fusion rrf|v 0\.5, k1||57||60||52||
||v 1, k2||||todo||||
|L2 min \+ Fusion w|v 0\.25||||todo||||
||v 0\.5||62||59||54||
||v 1||todo||||||
|L2 min \+ Fusion w \+ Cluster|tau 0\.3||||54||||
||tau 0\.4||||todo||||
|L2 min \+ Fusion w \+ Random|||||58||||
|L2 min \+ Fusion w \+ Gate|v 0\.5||58||60||||
||v 1||||||todo1||



put\_bottles\_dustbin 1700

|||1|4|8|16|||
|---|---|---|---|---|---|---|---|
|base||80||||||
|L2 min \+ Fusion|w 1||81|||||
||w 0\.5 |||||||
|L2 min \+ Fusion \+ Cluster \+ Random \+ Gate||||todo|todo|||







place\_can\_basket 700



place\_a2b\_left 400



move\_can\_pot 400



handover\_block 800



### OPD



#### m1



##### v1



###### 实现

\[deploy\_policy\.py\]

\[deploy\_policy\.yml\]

\[eval\.sh\]

\[eval\_policy\.py\]

\[motus\.py\]



在ttsv7之后



chat

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a182b6f\-ff60\-832f\-bfd1\-d39ebe9ec8bb

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZjRmZmRiNjA4YjdjYmJhNTFhZGI2ZTUxOWI5YmExZWFfYTQxZDZkZTI3MzU4Zjg3ODY3NDU4ZjRjY2JjM2IwZDVfSUQ6NzY0NTI2OTYxNzYwNjUyODE5MF8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)



报错：低级传参问题

\[deploy\_policy\.py\]

报错：模型加载设备问题

```Python
这次报错很清楚：**OPD 训练已经开始了，但 motus.py::action_opd_loss() 里用 CUDA 上的 index 去索引 CPU 上的 scheduler tensor。**
具体位置是：
timestep_id_action = torch.randint(..., device=self.device)
action_t_embed = self.fm_train_scheduler_action.timesteps[timestep_id_action]
self.fm_train_scheduler_action.timesteps 很可能在 CPU，而 timestep_id_action 在 CUDA，所以 PyTorch 报：
indices should be either on cpu or on the same device as the indexed tensor (cpu)
官方 training_step() 里的写法其实是 CPU index，然后再把选出来的 timestep/sigma .to(device)，也就是：
timestep_id_action = torch.randint(0, ..., (B,))
action_t_embed = timesteps[timestep_id_action].to(device)
我已经按这个修了 motus.py。同时我也修了 eval_policy.py 的 eval(value) 问题，因为你日志里这个现象：
raw opd_video_context="<module 'random' ...>"
根因就是 eval_policy.py 里对 override value 做了 eval(value)，random 被解析成了 Python 的 random 模块，而不是字符串。之前 fallback 能绕过去，但根因最好也修掉。
```

\[motus\.py\]

\[eval\_policy\.py\]



###### 实验



e1 参数1

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

LOG=logs/turn_switch_tts_global_medoid_opd_n8_b8_lr1e-6_u16_bs1_a005_sel0_beta1_full10_gpu0_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh turn_switch \
  --gpu-id 0 \
  --tts \
  --tts-num-samples 8 \
  --tts-batch-size 8 \
  --tts-method global_medoid \
  --opd \
  --opd-lr 1e-6 \
  --opd-update-steps 16 \
  --opd-batch-size 1 \
  --opd-alpha-nonwinner 0.05 \
  --opd-selected-weight 0.0 \
  --opd-beta 1.0 \
  --opd-video-context random \
  --opd-save-action-every-success 0 \
  --opd-save-full-every-success 10 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 466010

PID: 466010

LOG: logs/turn\_switch\_tts\_global\_medoid\_opd\_n8\_b8\_lr1e\-6\_u16\_bs1\_a005\_sel0\_beta1\_full10\_gpu0\_20260529\_204252\.log



batch8 等 参数2

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

LOG=logs/turn_switch_tts_global_medoid_opd_n8_b8_lr1e-6_u16_bs1_a005_sel0_beta1_full10_gpu1_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh turn_switch \
  --gpu-id 1 \
  --tts \
  --tts-num-samples 8 \
  --tts-batch-size 8 \
  --tts-method global_medoid \
  --opd \
  --opd-lr 5e-7 \
  --opd-update-steps 16 \
  --opd-batch-size 8 \
  --opd-alpha-nonwinner 0.05 \
  --opd-selected-weight 0.1 \
  --opd-beta 1.0 \
  --opd-video-context random \
  --opd-save-action-every-success 0 \
  --opd-save-full-every-success 10 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[2\] 477710

PID: 477710

LOG: logs/turn\_switch\_tts\_global\_medoid\_opd\_n8\_b8\_lr1e\-6\_u16\_bs1\_a005\_sel0\_beta1\_full10\_gpu1\_20260529\_212910\.log

（这里日志名字的参数忘记改，看时间）

b8，63g左右



放队列推理 gpu0；推训出来的ckpt

\[1\] 531532

WORKER\_GPU0\_PID: 531532

WORKER\_GPU0\_LOG: gpu\_queue/logs/worker\_gpu0\_20260529\_235028\.log

```Bash

cd /root/autodl-tmp/RoboTwin/policy/Motus

mkdir -p gpu_queue/tmp gpu_queue/pending/gpu0 logs

CKPTS=(
"/root/autodl-tmp/RoboTwin/policy/Motus/logs_single_20260529_212911/opd/turn_switch/full_ckpts/full_ep0015_succ0010_opd000160"
"/root/autodl-tmp/RoboTwin/policy/Motus/logs_single_20260529_212911/opd/turn_switch/full_ckpts/full_ep0028_succ0020_opd000320"
"/root/autodl-tmp/RoboTwin/policy/Motus/logs_single_20260529_212911/opd/turn_switch/full_ckpts/full_ep0044_succ0030_opd000480"
"/root/autodl-tmp/RoboTwin/policy/Motus/logs_single_20260529_212911/opd/turn_switch/full_ckpts/full_ep0055_succ0040_opd000640"
"/root/autodl-tmp/RoboTwin/policy/Motus/logs_single_20260529_212911/opd/turn_switch/full_ckpts/full_ep0069_succ0050_opd000800"
"/root/autodl-tmp/RoboTwin/policy/Motus/logs_single_20260529_212911/opd/turn_switch/full_ckpts/full_ep0085_succ0060_opd000960"
"/root/autodl-tmp/RoboTwin/policy/Motus/logs_single_20260529_204252/opd/turn_switch/full_ckpts/full_ep0086_succ0060_opd000960"
)

IDX=100
for CKPT in "${CKPTS[@]}"; do
    if [ ! -d "$CKPT" ]; then
        echo "[skip] checkpoint dir not found: $CKPT"
        continue
    fi

    TAG="$(basename "$CKPT")"
    JOB_TMP="gpu_queue/tmp/${IDX}_turn_switch_no_tts_eval_${TAG}.sh"
    JOB_DST="gpu_queue/pending/gpu0/${IDX}_turn_switch_no_tts_eval_${TAG}.sh"

    cat > "$JOB_TMP" <<SH
#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

TASK=turn_switch
GPU_ID=0
CKPT="${CKPT}"
TAG="${TAG}"

LOG=logs/\${TASK}_no_tts_eval_\${TAG}_gpu\${GPU_ID}_\$(date +%Y%m%d_%H%M%S).log

bash eval.sh \${TASK} \\
  --gpu-id \${GPU_ID} \\
  --checkpoint-path "\${CKPT}" \\
  --no-tts \\
  > "\$LOG" 2>&1

echo "CKPT: \$CKPT"
echo "LOG: \$LOG"
SH

    chmod +x "$JOB_TMP"
    mv "$JOB_TMP" "$JOB_DST"
    echo "submitted: $JOB_DST"

    IDX=$((IDX + 1))
done

cd /root/autodl-tmp/RoboTwin/policy/Motus
rm -f gpu_queue/STOP_gpu0

cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p gpu_queue/logs

WORKER_LOG=gpu_queue/logs/worker_gpu0_$(date +%Y%m%d_%H%M%S).log
nohup ./gpu_queue_worker.sh 0 > "$WORKER_LOG" 2>&1 &

echo "WORKER_GPU0_PID: $!"
echo "WORKER_GPU0_LOG: $WORKER_LOG"
```

放队列推理 gpu1；tts

\[2\] 535799

WORKER\_GPU1\_PID: 535799

WORKER\_GPU1\_LOG: gpu\_queue/logs/worker\_gpu1\_20260530\_000413\.log

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

mkdir -p gpu_queue/tmp gpu_queue/pending/gpu1 logs

# =========================
# Submit 3 Keystone jobs to GPU1
# =========================

JOBS=(
"16 4 0.3"
"16 5 0.3"
"32 5 0.3"
)

IDX=200
for SPEC in "${JOBS[@]}"; do
    read -r NUM_SAMPLES NUM_CLUSTERS TAU <<< "$SPEC"

    JOB_NAME="${IDX}_turn_switch_keystone_n${NUM_SAMPLES}_b${NUM_SAMPLES}_c${NUM_CLUSTERS}_tau03_nodecode.sh"
    JOB_TMP="gpu_queue/tmp/${JOB_NAME}"
    JOB_DST="gpu_queue/pending/gpu1/${JOB_NAME}"

    cat > "$JOB_TMP" <<SH
#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

TASK=turn_switch
GPU_ID=1

LOG=logs/\${TASK}_tts_keystone_n${NUM_SAMPLES}_b${NUM_SAMPLES}_c${NUM_CLUSTERS}_tau03_nodecode_gpu\${GPU_ID}_\$(date +%Y%m%d_%H%M%S).log

bash eval.sh \${TASK} \\
  --gpu-id \${GPU_ID} \\
  --tts \\
  --tts-num-samples ${NUM_SAMPLES} \\
  --tts-method keystone \\
  --tts-num-clusters ${NUM_CLUSTERS} \\
  --tts-tau ${TAU} \\
  --tts-kmeans-iters 10 \\
  --tts-batch-size ${NUM_SAMPLES} \\
  --no-tts-decode-video \\
  > "\$LOG" 2>&1

echo "LOG: \$LOG"
SH

    chmod +x "$JOB_TMP"
    mv "$JOB_TMP" "$JOB_DST"
    echo "submitted: $JOB_DST"

    IDX=$((IDX + 1))
done

# =========================
# Restart GPU1 queue worker
# =========================

rm -f gpu_queue/STOP_gpu1

mkdir -p gpu_queue/logs

WORKER_LOG=gpu_queue/logs/worker_gpu1_$(date +%Y%m%d_%H%M%S).log
nohup ./gpu_queue_worker.sh 1 > "$WORKER_LOG" 2>&1 &

echo "WORKER_GPU1_PID: $!"
echo "WORKER_GPU1_LOG: $WORKER_LOG"
```



如何推理

```Bash
用训练后的 full checkpoint 推理
假设保存出了：
/path/to/RoboTwin/policy/Motus/logs_single_xxx/opd/<task>/full_ckpts/full_ep0005_succ0005_opd000080
可以直接：
bash eval.sh <task_name> \
  --gpu-id 0 \
  --checkpoint-path /path/to/RoboTwin/policy/Motus/logs_single_xxx/opd/<task>/full_ckpts/full_ep0005_succ0005_opd000080 \
  --no-tts
或者继续带 TTS：
bash eval.sh <task_name> \
  --gpu-id 0 \
  --checkpoint-path /path/to/full_ep0005_succ0005_opd000080 \
  --tts \
  --tts-num-samples 8 \
  --tts-method global_medoid
```



todo

峰值大概率在20步以内；传参控制整体的步数？

100步够？更多？改eval\_policy\.py？

ckpt太大：只存action expert如何加载推理？

文件太大，跑通后拆分文件为多个py？

存储整体管理；转生到更大的机子？

可调参数：

```Bash
--opd-lr
学得更快，或者破坏
保守：5e-7
默认：1e-6
激进：2e-6

--opd-update-steps
信息被更充分利用
8
16

--opd-alpha-nonwinner
更强地把其他 noise branch 拉向 winner
保守：0.01
默认：0.05
较强：0.1
激进：0.25

--opd-selected-weight
但它并非完全没意义。因为 OPD 训练是随机 timestep flow matching，不是复现完整推理轨迹。selected branch 也可以作为一个“正样本 anchor”，帮助模型保持 winner 附近的速度场一致。
0.0：
0.1：
  给 winner branch 一个弱 anchor
  可能更稳
1.0：
  selected 和 nonwinner 都训
  更像普通 pseudo-label flow matching
  
--opd-beta
beta=1.0：
  最强 all-noise-to-winner
  方法最清楚
  也最容易 collapse
beta=0.5：
  把 nonwinner 往 winner 拉一半
  更像 soft distillation
  更稳但效果可能弱
beta=0.25：
  很保守

--opd-grad-clip
调小：
  更稳
  训练更保守
调大：
  允许更强更新
  但更容易不稳定


```



##### v2

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a1c06ef\-92b0\-8327\-9943\-6ac245266074



实验

放

```Bash
(base) root@autodl-container-nekaqbwt43-6ce5babb:~/autodl-tmp/RoboTwin/policy/Motus# cd /root/autodl-tmp/RoboTwin/policy/Motus

cat > gpu_queue/pending/gpu1/732_turn_switch_opd_chunk_k16_c4_ref_r01_eval30_save2.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RoboTwin/policy/Motus

bash eval.sh turn_switch --gpu-id 1 \
  --tts \
  --tts-level chunk \
  --tts-method keystone \
  --tts-num-samples 16 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --tts-ref \
  --tts-ref-num-samples 4 \
  --tts-ref-weight 0.1 \
  --tts-ref-distance-agg mean \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  --eval-test-num 30 \
  --opd-enable \
  --opd-save-action-every-success 2 \
  --opd-save-full-every-success 0
EOF

chmod +x gpu_queue/pending/gpu0/731_turn_switch_chunk_k16_c4_ref_r01_eval100.sh
chmod +x gpu_queue/pending/gpu1/732_turn_switch_opd_chunk_k16_c4_ref_r01_eval30_save2.sh

echo "GPU0 job:"
sed -n '1,80p' gpu_queue/pending/gpu0/731_turn_switch_chunk_k16_c4_ref_r01_eval100.sh

echo "GPU1 job:"
sed -n '1,120p' gpu_queue/pending/gpu1/732_turn_switch_opd_chunk_k16_c4_ref_r01_eval30_save2.sh
GPU0 job:
#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RoboTwin/policy/Motus

bash eval.sh turn_switch --gpu-id 0 \
  --tts \
  --tts-level chunk \
  --tts-method keystone \
  --tts-num-samples 16 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --tts-ref \
  --tts-ref-num-samples 4 \
  --tts-ref-weight 0.1 \
  --tts-ref-distance-agg mean \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  --eval-test-num 100
GPU1 job:
#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RoboTwin/policy/Motus

bash eval.sh turn_switch --gpu-id 1 \
  --tts \
  --tts-level chunk \
  --tts-method keystone \
  --tts-num-samples 16 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --tts-ref \
  --tts-ref-num-samples 4 \
  --tts-ref-weight 0.1 \
  --tts-ref-distance-agg mean \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  --eval-test-num 30 \
  --opd-enable \
  --opd-save-action-every-success 2 \
  --opd-save-full-every-success 0
(base) root@autodl-container-nekaqbwt43-6ce5babb:~/autodl-tmp/RoboTwin/policy/Motus# 
```

放

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

# =========================
# CONFIG
# =========================
MOTUS_DIR=/root/autodl-tmp/RoboTwin/policy/Motus
QUEUE_ROOT=${MOTUS_DIR}/gpu_queue
P0=${QUEUE_ROOT}/pending/gpu0
P1=${QUEUE_ROOT}/pending/gpu1
CKPT_DIR=${MOTUS_DIR}/logs_single_20260601_082941/opd_checkpoints/turn_switch

mkdir -p "$P0" "$P1"

make_job() {
  local path="$1"
  local cmd="$2"
  local tmp="${path}.tmp.$$"

  cat > "$tmp" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}
${cmd}
EOF

  chmod +x "$tmp"
  mv -f "$tmp" "$path"
}

# =========================
# GPU0: 734, 737 + odd OPD ckpts
# =========================

make_job "$P0/734_turn_switch_vtts_k16_s5_c4_noise8_cluster.sh" \
"bash eval.sh turn_switch --gpu-id 0 \
  --tts \
  --tts-level velocity \
  --tts-velocity-num-samples 16 \
  --tts-velocity-stride 5 \
  --tts-velocity-select action_keystone \
  --tts-velocity-perturb-scope vlm_only \
  --tts-velocity-perturb-policy noise_weak \
  --tts-velocity-noise-std 0.031372549019607843 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  --tts-log-actions \
  --eval-test-num 100"

make_job "$P0/737_turn_switch_vtts_k16_s5_c5_noise8_cluster.sh" \
"bash eval.sh turn_switch --gpu-id 0 \
  --tts \
  --tts-level velocity \
  --tts-velocity-num-samples 16 \
  --tts-velocity-stride 5 \
  --tts-velocity-select action_keystone \
  --tts-velocity-perturb-scope vlm_only \
  --tts-velocity-perturb-policy noise_weak \
  --tts-velocity-noise-std 0.031372549019607843 \
  --tts-num-clusters 5 \
  --tts-tau 0.3 \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  --tts-log-actions \
  --eval-test-num 100"

make_job "$P0/751_turn_switch_no_tts_action_expert_20260601_082941_succ0002_ep0004_eval100.sh" \
"bash eval.sh turn_switch --gpu-id 0 --no-tts --action-ckpt-path ${CKPT_DIR}/action_expert_20260601_082941_succ0002_ep0004.pt --eval-test-num 100"

make_job "$P0/753_turn_switch_no_tts_action_expert_20260601_082941_succ0006_ep0009_eval100.sh" \
"bash eval.sh turn_switch --gpu-id 0 --no-tts --action-ckpt-path ${CKPT_DIR}/action_expert_20260601_082941_succ0006_ep0009.pt --eval-test-num 100"

make_job "$P0/755_turn_switch_no_tts_action_expert_20260601_082941_succ0010_ep0015_eval100.sh" \
"bash eval.sh turn_switch --gpu-id 0 --no-tts --action-ckpt-path ${CKPT_DIR}/action_expert_20260601_082941_succ0010_ep0015.pt --eval-test-num 100"

make_job "$P0/757_turn_switch_no_tts_action_expert_20260601_082941_succ0014_ep0019_eval100.sh" \
"bash eval.sh turn_switch --gpu-id 0 --no-tts --action-ckpt-path ${CKPT_DIR}/action_expert_20260601_082941_succ0014_ep0019.pt --eval-test-num 100"

make_job "$P0/759_turn_switch_no_tts_action_expert_20260601_082941_succ0018_ep0023_eval100.sh" \
"bash eval.sh turn_switch --gpu-id 0 --no-tts --action-ckpt-path ${CKPT_DIR}/action_expert_20260601_082941_succ0018_ep0023.pt --eval-test-num 100"

make_job "$P0/761_turn_switch_no_tts_action_expert_20260601_082941_succ0022_ep0030_eval100.sh" \
"bash eval.sh turn_switch --gpu-id 0 --no-tts --action-ckpt-path ${CKPT_DIR}/action_expert_20260601_082941_succ0022_ep0030.pt --eval-test-num 100"

# =========================
# GPU1: 735, 736 + even OPD ckpts
# =========================

make_job "$P1/735_turn_switch_vtts_k16_s5_c4_noise10_cluster.sh" \
"bash eval.sh turn_switch --gpu-id 1 \
  --tts \
  --tts-level velocity \
  --tts-velocity-num-samples 16 \
  --tts-velocity-stride 5 \
  --tts-velocity-select action_keystone \
  --tts-velocity-perturb-scope vlm_only \
  --tts-velocity-perturb-policy noise_weak \
  --tts-velocity-noise-std 0.039215686274509804 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  --tts-log-actions \
  --eval-test-num 100"

make_job "$P1/736_turn_switch_vtts_k16_s5_c5_noise6_cluster.sh" \
"bash eval.sh turn_switch --gpu-id 1 \
  --tts \
  --tts-level velocity \
  --tts-velocity-num-samples 16 \
  --tts-velocity-stride 5 \
  --tts-velocity-select action_keystone \
  --tts-velocity-perturb-scope vlm_only \
  --tts-velocity-perturb-policy noise_weak \
  --tts-velocity-noise-std 0.02352941176470588 \
  --tts-num-clusters 5 \
  --tts-tau 0.3 \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  --tts-log-actions \
  --eval-test-num 100"

make_job "$P1/752_turn_switch_no_tts_action_expert_20260601_082941_succ0004_ep0007_eval100.sh" \
"bash eval.sh turn_switch --gpu-id 1 --no-tts --action-ckpt-path ${CKPT_DIR}/action_expert_20260601_082941_succ0004_ep0007.pt --eval-test-num 100"

make_job "$P1/754_turn_switch_no_tts_action_expert_20260601_082941_succ0008_ep0012_eval100.sh" \
"bash eval.sh turn_switch --gpu-id 1 --no-tts --action-ckpt-path ${CKPT_DIR}/action_expert_20260601_082941_succ0008_ep0012.pt --eval-test-num 100"

make_job "$P1/756_turn_switch_no_tts_action_expert_20260601_082941_succ0012_ep0017_eval100.sh" \
"bash eval.sh turn_switch --gpu-id 1 --no-tts --action-ckpt-path ${CKPT_DIR}/action_expert_20260601_082941_succ0012_ep0017.pt --eval-test-num 100"

make_job "$P1/758_turn_switch_no_tts_action_expert_20260601_082941_succ0016_ep0021_eval100.sh" \
"bash eval.sh turn_switch --gpu-id 1 --no-tts --action-ckpt-path ${CKPT_DIR}/action_expert_20260601_082941_succ0016_ep0021.pt --eval-test-num 100"

make_job "$P1/760_turn_switch_no_tts_action_expert_20260601_082941_succ0020_ep0028_eval100.sh" \
"bash eval.sh turn_switch --gpu-id 1 --no-tts --action-ckpt-path ${CKPT_DIR}/action_expert_20260601_082941_succ0020_ep0028.pt --eval-test-num 100"

echo "GPU0 pending:"
ls -1 "$P0"/73*.sh "$P0"/75*.sh "$P0"/76*.sh 2>/dev/null | sort

echo "GPU1 pending:"
ls -1 "$P1"/73*.sh "$P1"/75*.sh "$P1"/76*.sh 2>/dev/null | sort
```



##### v3

\[deploy\_policy\.py\]

\[deploy\_policy\.yml\]

\[eval\.sh\]

\[eval\_policy\.py\]

\[motus\.py\]



实验1

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

mkdir -p logs

LOG=logs/turn_switch_chunk_gkd_k16_c4_noref_lr1e-6_u8_bs2_eval30_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh turn_switch \
  --gpu-id 0 \
  --visible-gpus 0,1 \
  --tts \
  --tts-level chunk \
  --tts-method keystone \
  --tts-num-samples 16 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --no-tts-ref \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  --eval-test-num 100 \
  --opd-train \
  --opd-mode chunk_gkd \
  --opd-lr 1e-6 \
  --opd-update-steps 8 \
  --opd-batch-size 2 \
  --opd-mar-mode base_velocity \
  --opd-mar-weight 0.1 \
  --opd-mar-device cuda:1 \
  --opd-save-action-every-success 2 \
  --opd-save-full-every-success 0 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 2807

PID: 2807

LOG: logs/turn\_switch\_chunk\_gkd\_k16\_c4\_noref\_lr1e\-6\_u8\_bs2\_eval30\_20260602\_152538\.log

报错

```Bash
这里的问题是：**它用 config 里的 video_height/video_width 去手工构造 latent 尺寸，但 condition_frame_latent 是 VAE 根据实际 first_frame 编码出来的尺寸。** 这两个不一定一致，所以炸了。
对比一下 velocity-level inference 的实现，它是先 encode condition frame，再直接从 condition_frame_latent.shape 取 H_latent, W_latent 来初始化 video_latent，这是正确写法。
所以 chunk_gkd 应该照着 velocity/inference 那套写法改。
```

\[motus\.py\]

再跑

```SQL
cd /root/autodl-tmp/RoboTwin/policy/Motus

mkdir -p logs

LOG=logs/turn_switch_chunk_gkd_k16_c4_noref_lr1e-6_u8_bs2_eval100_fixlatent_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh turn_switch \
  --gpu-id 0 \
  --visible-gpus 0,1 \
  --tts \
  --tts-level chunk \
  --tts-method keystone \
  --tts-num-samples 16 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --no-tts-ref \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  --eval-test-num 100 \
  --opd-train \
  --opd-mode chunk_gkd \
  --opd-lr 1e-6 \
  --opd-update-steps 8 \
  --opd-batch-size 2 \
  --opd-mar-mode base_velocity \
  --opd-mar-weight 0.1 \
  --opd-mar-device cuda:1 \
  --opd-save-action-every-success 2 \
  --opd-save-full-every-success 0 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 39374

PID: 39374

LOG: logs/turn\_switch\_chunk\_gkd\_k16\_c4\_noref\_lr1e\-6\_u8\_bs2\_eval100\_fixlatent\_20260602\_161146\.log

报错：

```Python
现在 action_endpoint_gkd_step() 里有这几行：
timestep_id = torch.randint(
    0,
    self.fm_train_scheduler.num_train_timesteps,
    (B,),
    device=self.device,
)
video_t_embed = self.fm_train_scheduler.timesteps[timestep_id].to(
    dtype=self.dtype,
    device=self.device,
)
这里 timestep_id 在 CUDA 上，但 self.fm_train_scheduler.timesteps 在 CPU 上，所以 PyTorch 不允许用 CUDA index 去索引 CPU tensor。action 分支也有同样问题：
timestep_id_action = torch.randint(
    0,
    self.fm_train_scheduler_action.num_train_timesteps,
    (B,),
    device=self.device,
)
action_t_embed = self.fm_train_scheduler_action.timesteps[timestep_id_action]
sigma_action = self.fm_train_scheduler_action.sigmas[timestep_id_action]
也就是说，**不是 GPU 不够，也不是参数问题，也不是 --visible-gpus 0,1 的问题。** 是 chunk_gkd 训练函数里 timestep index 的 device 放错了。当前代码确实是在 _run_chunk_gkd_updates() 里调用 action_endpoint_gkd_step()，而 action_endpoint_gkd_step() 当前正是用 CUDA index 去索引 scheduler 的 CPU tensor
```

跑

```SQL
cd /root/autodl-tmp/RoboTwin/policy/Motus

mkdir -p logs

LOG=logs/turn_switch_chunk_gkd_k16_c4_noref_lr1e-6_u8_bs2_eval100_fixlatent_fixtimestep_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh turn_switch \
  --gpu-id 0 \
  --visible-gpus 0,1 \
  --tts \
  --tts-level chunk \
  --tts-method keystone \
  --tts-num-samples 16 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --no-tts-ref \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  --eval-test-num 100 \
  --opd-train \
  --opd-mode chunk_gkd \
  --opd-lr 1e-6 \
  --opd-update-steps 8 \
  --opd-batch-size 2 \
  --opd-mar-mode base_velocity \
  --opd-mar-weight 0.1 \
  --opd-mar-device cuda:1 \
  --opd-save-action-every-success 2 \
  --opd-save-full-every-success 0 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 60215

PID: 60215

LOG: logs/turn\_switch\_chunk\_gkd\_k16\_c4\_noref\_lr1e\-6\_u8\_bs2\_eval100\_fixlatent\_fixtimestep\_20260602\_162807\.log

报错

```Bash
现在报错：
RuntimeError: The size of tensor a (360) must match the size of tensor b (1440)
含义是：
actual video token length = 360
RoPE freq_grid length     = 1440
也就是 WAN 的 self-attention RoPE 认为 video token grid 是 1440 个位置，但实际 video_tokens 只有 360 个。
根因在这里：prepare_input() 不是直接把 VAE latent flatten，而是先过 patch_embedding，再 flatten 成 video tokens。也就是说 RoPE 的 grid_sizes 应该对应 **patch embedding 后的 token grid**，不是 VAE latent 的原始 H/W。
你现在的 action_endpoint_gkd_step() 里是：
grid_sizes = torch.tensor(
    [lat_T, H_latent, W_latent],
    ...
)
这里 H_latent, W_latent 是 VAE latent 尺寸，比如 24 和 20，于是 RoPE grid 变成：
lat_T * H_latent * W_latent = 3 * 24 * 20 = 1440
但 patch_embedding 后 spatial 下采样了 2 倍，实际 video token grid 应该是：
3 * 12 * 10 = 360
所以正确修法是：**video_latent 用 VAE latent H/W，但 grid_sizes 用 H_latent // 2 和 W_latent // 2**
```

修

\[motus\.py\]

再跑

```SQL
cd /root/autodl-tmp/RoboTwin/policy/Motus

mkdir -p logs

LOG=logs/turn_switch_chunk_gkd_k16_c4_noref_lr1e-6_u8_bs2_eval100_fixlatent_fixtimestep_fixgrid_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh turn_switch \
  --gpu-id 0 \
  --visible-gpus 0,1 \
  --tts \
  --tts-level chunk \
  --tts-method keystone \
  --tts-num-samples 16 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --no-tts-ref \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  --eval-test-num 100 \
  --opd-train \
  --opd-mode chunk_gkd \
  --opd-lr 1e-6 \
  --opd-update-steps 8 \
  --opd-batch-size 2 \
  --opd-mar-mode base_velocity \
  --opd-mar-weight 0.1 \
  --opd-mar-device cuda:1 \
  --opd-save-action-every-success 2 \
  --opd-save-full-every-success 0 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```

\[1\] 107020

PID: 107020

LOG: logs/turn\_switch\_chunk\_gkd\_k16\_c4\_noref\_lr1e\-6\_u8\_bs2\_eval100\_fixlatent\_fixtimestep\_fixgrid\_20260602\_170538\.log

跑通

放评估实验：

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

#################### CONFIG ####################
TASK=turn_switch
EVAL_TEST_NUM=100

CKPT_DIR="/root/autodl-tmp/RoboTwin/policy/Motus/logs_single_20260602_170538/opd_checkpoints/turn_switch"
MOTUS_DIR="/root/autodl-tmp/RoboTwin/policy/Motus"
QUEUE_ROOT="${MOTUS_DIR}/gpu_queue"

JOB_PREFIX="eval_notts_ckpt20_even"
STAMP="$(date +%Y%m%d_%H%M%S)"
################################################

mkdir -p \
  "${QUEUE_ROOT}/pending/gpu0" \
  "${QUEUE_ROOT}/pending/gpu1" \
  "${QUEUE_ROOT}/old_pending_${JOB_PREFIX}_${STAMP}"

# 备份旧 pending，避免混入旧任务。
find \
  "${QUEUE_ROOT}/pending/gpu0" \
  "${QUEUE_ROOT}/pending/gpu1" \
  -maxdepth 1 \
  -type f \
  -name "*.sh" \
  -print \
  -exec mv {} "${QUEUE_ROOT}/old_pending_${JOB_PREFIX}_${STAMP}/" \; \
  2>/dev/null || true

# 允许 worker 取任务。
rm -f \
  "${QUEUE_ROOT}/STOP_gpu0" \
  "${QUEUE_ROOT}/STOP_gpu1"

export \
  TASK \
  EVAL_TEST_NUM \
  CKPT_DIR \
  MOTUS_DIR \
  QUEUE_ROOT \
  JOB_PREFIX

python - <<'PY'
import os
from pathlib import Path

TASK = os.environ["TASK"]
EVAL_TEST_NUM = int(os.environ["EVAL_TEST_NUM"])

CKPT_DIR = Path(os.environ["CKPT_DIR"])
MOTUS_DIR = Path(os.environ["MOTUS_DIR"])
QUEUE_ROOT = Path(os.environ["QUEUE_ROOT"])
JOB_PREFIX = os.environ["JOB_PREFIX"]

ckpts = sorted(CKPT_DIR.glob("action_expert_*.pt"))
selected = ckpts[::2]

if len(ckpts) != 40:
    raise SystemExit(f"Expected 40 ckpts, got {len(ckpts)} from {CKPT_DIR}")

if len(selected) != 20:
    raise SystemExit(f"Expected 20 selected ckpts, got {len(selected)}")

for rank, ckpt in enumerate(selected):
    gpu = 0 if rank < 10 else 1
    pending_dir = QUEUE_ROOT / "pending" / f"gpu{gpu}"
    pending_dir.mkdir(parents=True, exist_ok=True)

    job_id = 900 + rank
    job_path = pending_dir / f"{job_id:03d}_{TASK}_{JOB_PREFIX}_{ckpt.stem}.sh"

    job_path.write_text(f"""#!/usr/bin/env bash
set -euo pipefail

cd "{MOTUS_DIR}"

bash eval.sh {TASK} \\
  --gpu-id {gpu} \\
  --action-ckpt "{ckpt}" \\
  --no-tts \\
  --eval-test-num {EVAL_TEST_NUM} \\
  --no-opd-train \\
  --no-opd-enable
""")
    job_path.chmod(0o755)

    print(f"{rank + 1:02d} gpu{gpu} {ckpt.name}")

print()
print("created gpu0 jobs:", len(list((QUEUE_ROOT / "pending" / "gpu0").glob("*.sh"))))
print("created gpu1 jobs:", len(list((QUEUE_ROOT / "pending" / "gpu1").glob("*.sh"))))
PY
```





（now）

保留类似grpo的参考模型，约束更新方向；如何实现？（zhp）

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a1d2b50\-54e4\-8330\-b375\-d62c01a84def



来自Bon的trick

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a1d31d2\-41bc\-832a\-a9c3\-090a67ec71b8



##### v4

c\_gkd没replay buffer

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a1f8e29\-2dec\-832b\-bbd7\-47b63dfffc38



Distill rw doc整个过一遍，更多工程trick

聚合更多开发点



调新参数训练：

scan\_object 上更显著些？

参数更小，更稳，更大batch，崩塌更晚到来

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

mkdir -p logs

LOG=logs/turn_switch_chunk_gkd_k16_c4_noref_lr1e-6_u4_bs8_buf128_mar03_eval100_$(date +%Y%m%d_%H%M%S).log

nohup bash eval.sh turn_switch \
  --gpu-id 0 \
  --visible-gpus 0,1 \
  --tts \
  --tts-level chunk \
  --tts-method keystone \
  --tts-num-samples 16 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --no-tts-ref \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  --eval-test-num 100 \
  --opd-train \
  --opd-mode chunk_gkd \
  --opd-lr 1e-6 \
  --opd-update-steps 4 \
  --opd-batch-size 16 \
  --opd-buffer-size 128 \
  --opd-mar-mode base_velocity \
  --opd-mar-weight 0.3 \
  --opd-mar-device cuda:1 \
  --opd-save-action-every-success 2 \
  --opd-save-full-every-success 0 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "LOG: $LOG"
```



v\_gkd再说



#### m2

##### v1

实现：

\[deploy\_policy\.py\]

\[deploy\_policy\.yml\]

\[eval\.sh\]

\[motus\.py\]



实验

启动队列1

队列2：

GPU0: noise\_weak,       K = 4 / 8 / 16 / 32, stride = 5 / 2

GPU1: photometric\_weak, K = 4 / 8 / 16 / 32, stride = 5 / 2



日志

```Bash
2.1 noise_weak：K=32 明显最好

2.2 photometric_weak：K=4/K=8 更好，K=16/K=32 变差

2.3 stride=2 没有系统收益
2 denoise_step=8 的扰动差异最大
按外层 log 汇总，stride=2 下各 denoise step 的平均 pairwise 差异大概是：
denoise_steppairwise meanfallback 触发率
00.04461.59%
20.04320.33%
40.04550.73%
60.05101.90%
8**0.064610.71%**
这很重要。说明越到后面的 denoise step，扰动对 velocity/action 的影响越大，尤其 step=8。gate=0.005 主要也是在 step=8 起作用。
这带来一个判断：
 stride=2 不是完全没用，它确实看到了更剧烈的中后段 velocity 差异。 
 但这些差异未必是好差异，可能会带来更多错误选择。
```



Chat

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a16b566\-0888\-832b\-aef7\-bdec5520c403



扰动rw

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a1932e2\-e2bc\-8328\-b0c2\-ca20e026535b



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZWQ1ZDFjMDc0NjQyZjVmN2NkZDdmNWExMTAyZjBmNWNfMWY2NWFjZjgzMjdjODdiYWMxNWNhMWNkMzRhYTU5NTBfSUQ6NzY0MzQyMjcxMTAyODY4MTkzOV8xNzg1MTYwMzAzOjE3ODUyNDY3MDNfVjM)





##### v2

实现

\[eval\.sh\]

\[motus\.py\]

\[deploy\_policy\.py\]

\[deploy\_policy\.yml\]



v1结论：

```Bash
noise是主线
noise_weak 主线往更大 K 搜。当前 K=32 有意义，后面可以继续看 K=48 / 64，甚至先做少数组 K=64。 
stride=1 需要测几组。虽然这批 stride=2 没有稳定收益，但你的判断是合理的：需要极端密集选择做验证，不能只凭 stride=2 否定。 
两条主线分开调：
noise_weak 走大 K，加密 denoise step 的方向。
photometric_weak 走较小 K，例如 K4/K8，再测 stride2/stride1。
```

先试noise，小步长，因为后面的去噪步差异大，所有要聚类



实验1

```Bash
GPU0:
  601: K32, stride 5, cluster only
  602: K32, stride 5, cluster + reference
  603: K32, stride 2, cluster only

GPU1:
  604: K64, stride 5, cluster only
  605: K64, stride 5, cluster + reference
  606: K32, stride 2, cluster + reference
```

放

```SQL
(base) root@autodl-container-nekaqbwt43-6ce5babb:~/autodl-tmp/RoboTwin/policy/Motus# cd /root/autodl-tmp/RoboTwin/policy/Motus

TASK=turn_switch
MOTUS_DIR=/root/autodl-tmp/RoboTwin/policy/Motus
QUEUE_ROOT=${MOTUS_DIR}/gpu_queue
P0=${QUEUE_ROOT}/pending/gpu0
P1=${QUEUE_ROOT}/pending/gpu1

mkdir -p "$P0" "$P1"

cat > ${P0}/601_${TASK}_vtts_k32_s5_noise_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 0 \\
  --tts \\
  --tts-level velocity \\
  --tts-velocity-num-samples 32 \\
  --tts-velocity-select action_keystone \\
  --tts-velocity-perturb-scope vlm_only \\
  --tts-velocity-perturb-policy noise_weak \\
  --tts-velocity-stride 5 \\
  --tts-num-clusters 2 \\
  --tts-tau 0.3 \\
  --tts-kmeans-iters 10 \\
  --tts-velocity-include-original \\
  --tts-velocity-diversity-eps 1e-6 \\
  --tts-velocity-max-original-gap-ratio 0.5 \\
  --tts-velocity-save-perturb-images \\
  --tts-velocity-perturb-image-max 1 \\
  --tts-velocity-perturb-image-mode max_delta \\
  --no-tts-decode-video
EOF

cat > ${P0}/602_${TASK}_vtts_k32_s5_noise_cluster_ref_w05.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 0 \\
  --tts \\
  --tts-level velocity \\
  --tts-velocity-num-samples 32 \\
  --tts-velocity-select action_keystone \\
  --tts-velocity-perturb-scope vlm_only \\
  --tts-velocity-perturb-policy noise_weak \\
  --tts-velocity-stride 5 \\
  --tts-num-clusters 2 \\
  --tts-tau 0.3 \\
  --tts-kmeans-iters 10 \\
  --tts-velocity-include-original \\
  --tts-velocity-diversity-eps 1e-6 \\
  --tts-velocity-max-original-gap-ratio 0.5 \\
  --tts-velocity-save-perturb-images \\
  --tts-velocity-perturb-image-max 1 \\
  --tts-velocity-perturb-image-mode max_delta \\
  --tts-ref \\
  --tts-ref-weight 0.5 \\
  --tts-ref-instruction "Perform the Task" \\
  --no-tts-decode-video
EOF

cat > ${P0}/603_${TASK}_vtts_k32_s2_noise_cluster.sh <<EOF
ls -lh ${P1}ng gpu1]"ASK}_vtts_k32_s2_noise_cluster_ref_w05.sh<EOF
[bash syntax check]
[pending gpu0]
total 12K
-rwxr-xr-x 1 root root 684 May 31 13:16 601_turn_switch_vtts_k32_s5_noise_cluster.sh
-rwxr-xr-x 1 root root 768 May 31 13:16 602_turn_switch_vtts_k32_s5_noise_cluster_ref_w05.sh
-rwxr-xr-x 1 root root 684 May 31 13:16 603_turn_switch_vtts_k32_s2_noise_cluster.sh
[pending gpu1]
total 12K
-rwxr-xr-x 1 root root 684 May 31 13:16 604_turn_switch_vtts_k64_s5_noise_cluster.sh
-rwxr-xr-x 1 root root 768 May 31 13:16 605_turn_switch_vtts_k64_s5_noise_cluster_ref_w05.sh
-rwxr-xr-x 1 root root 768 May 31 13:16 606_turn_switch_vtts_k32_s2_noise_cluster_ref_w05.sh
(base) root@autodl-container-nekaqbwt43-6ce5babb:~/autodl-tmp/RoboTwin/policy/Motus# 
```

日志换地方

```Bash
为什么 /logs 看不到
你以前手动跑的时候，经常是自己写：
LOG=logs/xxx_$(date ...).log
nohup bash eval.sh ... > "$LOG" 2>&1 &
所以外层日志在：
/root/autodl-tmp/RoboTwin/policy/Motus/logs/
```

日志

```YAML
但 C2 基本只是切掉小 outlier
这点非常明显。以 C2 为例：
601: K32 cluster-only
selection_stage:
  cluster_medoid: 1607 / 1610
  guard_global_medoid: 3 / 1610
也就是说 tau=0.3 下，unimodality guard 几乎不生效，几乎每次都进 k-means。
但实际 cluster split 很不均衡：
selected_cluster_size median = 28 / 32
minority cluster median = 4 / 32
常见 cluster_counts:
  31|1
  30|2
  29|3
  28|4
604: K64 cluster-only
selected_cluster_size median = 54 / 64
minority cluster median = 10 / 64
常见 cluster_counts:
  63|1
  62|2
  61|3
  60|4
```



实验2

放

```SQL
cd /root/autodl-tmp/RoboTwin/policy/Motus

TASK=turn_switch
MOTUS_DIR=/root/autodl-tmp/RoboTwin/policy/Motus
QUEUE_ROOT=${MOTUS_DIR}/gpu_queue
P0=${QUEUE_ROOT}/pending/gpu0
P1=${QUEUE_ROOT}/pending/gpu1

mkdir -p "$P0" "$P1"

cat > ${P0}/701_${TASK}_vtts_k4_s5_c2_noise_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 0 \\
  --tts \\
  --tts-level velocity \\
  --tts-velocity-num-samples 4 \\
  --tts-velocity-select action_keystone \\
  --tts-velocity-perturb-scope vlm_only \\
  --tts-velocity-perturb-policy noise_weak \\
  --tts-velocity-stride 5 \\
  --tts-num-clusters 2 \\
  --tts-tau 0.3 \\
  --tts-kmeans-iters 10 \\
  --tts-velocity-include-original \\
  --tts-velocity-diversity-eps 1e-6 \\
  --tts-velocity-max-original-gap-ratio 0.5 \\
  --tts-velocity-save-perturb-images \\
  --tts-velocity-perturb-image-max 1 \\
  --tts-velocity-perturb-image-mode max_delta \\
  --no-tts-ref \\
  --no-tts-decode-video
EOF

cat > ${P1}/702_${TASK}_vtts_k4_s2_c2_noise_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 1 \\
  --tts \\
  --tts-level velocity \\
  --tts-velocity-num-samples 4 \\
  --tts-velocity-select action_keystone \\
  --tts-velocity-perturb-scope vlm_only \\
  --tts-velocity-perturb-policy noise_weak \\
  --tts-velocity-stride 2 \\
  --tts-num-clusters 2 \\
  --tts-tau 0.3 \\
  --tts-kmeans-iters 10 \\
  --tts-velocity-include-original \\
  --tts-velocity-diversity-eps 1e-6 \\
  --tts-velocity-max-original-gap-ratio 0.5 \\
  --tts-velocity-save-perturb-images \\
  --tts-velocity-perturb-image-max 1 \\
  --tts-velocity-perturb-image-mode max_delta \\
  --no-tts-ref \\
  --no-tts-decode-video
EOF

cat > ${P1}/703_${TASK}_vtts_k8_s5_c2_noise_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 1 \\
  --tts \\
  --tts-level velocity \\
  --tts-velocity-num-samples 8 \\
  --tts-velocity-select action_keystone \\
  --tts-velocity-perturb-scope vlm_only \\
  --tts-velocity-perturb-policy noise_weak \\
  --tts-velocity-stride 5 \\
  --tts-num-clusters 2 \\
  --tts-tau 0.3 \\
  --tts-kmeans-iters 10 \\
  --tts-velocity-include-original \\
  --tts-velocity-diversity-eps 1e-6 \\
  --tts-velocity-max-original-gap-ratio 0.5 \\
  --tts-velocity-save-perturb-images \\
  --tts-velocity-perturb-image-max 1 \\
  --tts-velocity-perturb-image-mode max_delta \\
  --no-tts-ref \\
  --no-tts-decode-video
EOF

cat > ${P0}/704_${TASK}_vtts_k8_s2_c2_noise_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 0 \\
  --tts \\
  --tts-level velocity \\
  --tts-velocity-num-samples 8 \\
  --tts-velocity-select action_keystone \\
  --tts-velocity-perturb-scope vlm_only \\
  --tts-velocity-perturb-policy noise_weak \\
  --tts-velocity-stride 2 \\
  --tts-num-clusters 2 \\
  --tts-tau 0.3 \\
  --tts-kmeans-iters 10 \\
  --tts-velocity-include-original \\
  --tts-velocity-diversity-eps 1e-6 \\
  --tts-velocity-max-original-gap-ratio 0.5 \\
  --tts-velocity-save-perturb-images \\
  --tts-velocity-perturb-image-max 1 \\
  --tts-velocity-perturb-image-mode max_delta \\
  --no-tts-ref \\
  --no-tts-decode-video
EOF

cat > ${P0}/705_${TASK}_vtts_k16_s5_c2_noise_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 0 \\
  --tts \\
  --tts-level velocity \\
  --tts-velocity-num-samples 16 \\
  --tts-velocity-select action_keystone \\
  --tts-velocity-perturb-scope vlm_only \\
  --tts-velocity-perturb-policy noise_weak \\
  --tts-velocity-stride 5 \\
  --tts-num-clusters 2 \\
  --tts-tau 0.3 \\
  --tts-kmeans-iters 10 \\
  --tts-velocity-include-original \\
  --tts-velocity-diversity-eps 1e-6 \\
  --tts-velocity-max-original-gap-ratio 0.5 \\
  --tts-velocity-save-perturb-images \\
  --tts-velocity-perturb-image-max 1 \\
  --tts-velocity-perturb-image-mode max_delta \\
  --no-tts-ref \\
  --no-tts-decode-video
EOF

cat > ${P1}/706_${TASK}_vtts_k16_s2_c2_noise_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 1 \\
  --tts \\
  --tts-level velocity \\
  --tts-velocity-num-samples 16 \\
  --tts-velocity-select action_keystone \\
  --tts-velocity-perturb-scope vlm_only \\
  --tts-velocity-perturb-policy noise_weak \\
  --tts-velocity-stride 2 \\
  --tts-num-clusters 2 \\
  --tts-tau 0.3 \\
  --tts-kmeans-iters 10 \\
  --tts-velocity-include-original \\
  --tts-velocity-diversity-eps 1e-6 \\
  --tts-velocity-max-original-gap-ratio 0.5 \\
  --tts-velocity-save-perturb-images \\
  --tts-velocity-perturb-image-max 1 \\
  --tts-velocity-perturb-image-mode max_delta \\
  --no-tts-ref \\
  --no-tts-decode-video
EOF

cat > ${P1}/707_${TASK}_vtts_k16_s5_c3_noise_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 1 \\
  --tts \\
  --tts-level velocity \\
  --tts-velocity-num-samples 16 \\
  --tts-velocity-select action_keystone \\
  --tts-velocity-perturb-scope vlm_only \\
  --tts-velocity-perturb-policy noise_weak \\
  --tts-velocity-stride 5 \\
  --tts-num-clusters 3 \\
  --tts-tau 0.3 \\
  --tts-kmeans-iters 10 \\
  --tts-velocity-include-original \\
  --tts-velocity-diversity-eps 1e-6 \\
  --tts-velocity-max-original-gap-ratio 0.5 \\
  --tts-velocity-save-perturb-images \\
  --tts-velocity-perturb-image-max 1 \\
  --tts-velocity-perturb-image-mode max_delta \\
  --no-tts-ref \\
  --no-tts-decode-video
EOF

cat > ${P0}/708_${TASK}_vtts_k16_s2_c3_noise_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 0 \\
  --tts \\
  --tts-level velocity \\
  --tts-velocity-num-samples 16 \\
  --tts-velocity-select action_keystone \\
  --tts-velocity-perturb-scope vlm_only \\
  --tts-velocity-perturb-policy noise_weak \\
  --tts-velocity-stride 2 \\
  --tts-num-clusters 3 \\
  --tts-tau 0.3 \\
  --tts-kmeans-iters 10 \\
  --tts-velocity-include-original \\
  --tts-velocity-diversity-eps 1e-6 \\
  --tts-velocity-max-original-gap-ratio 0.5 \\
  --tts-velocity-save-perturb-images \\
  --tts-velocity-perturb-image-max 1 \\
  --tts-velocity-perturb-image-mode max_delta \\
  --no-tts-ref \\
  --no-tts-decode-video
EOF

chmod +x ${P0}/701_${TASK}_vtts_k4_s5_c2_noise_cluster.sh
chmod +x ${P1}/702_${TASK}_vtts_k4_s2_c2_noise_cluster.sh
chmod +x ${P1}/703_${TASK}_vtts_k8_s5_c2_noise_cluster.sh
chmod +x ${P0}/704_${TASK}_vtts_k8_s2_c2_noise_cluster.sh
chmod +x ${P0}/705_${TASK}_vtts_k16_s5_c2_noise_cluster.sh
chmod +x ${P1}/706_${TASK}_vtts_k16_s2_c2_noise_cluster.sh
chmod +x ${P1}/707_${TASK}_vtts_k16_s5_c3_noise_cluster.sh
chmod +x ${P0}/708_${TASK}_vtts_k16_s2_c3_noise_cluster.sh

echo "[bash syntax check]"
bash -n ${P0}/701_${TASK}_vtts_k4_s5_c2_noise_cluster.sh
bash -n ${P1}/702_${TASK}_vtts_k4_s2_c2_noise_cluster.sh
bash -n ${P1}/703_${TASK}_vtts_k8_s5_c2_noise_cluster.sh
bash -n ${P0}/704_${TASK}_vtts_k8_s2_c2_noise_cluster.sh
bash -n ${P0}/705_${TASK}_vtts_k16_s5_c2_noise_cluster.sh
bash -n ${P1}/706_${TASK}_vtts_k16_s2_c2_noise_cluster.sh
bash -n ${P1}/707_${TASK}_vtts_k16_s5_c3_noise_cluster.sh
bash -n ${P0}/708_${TASK}_vtts_k16_s2_c3_noise_cluster.sh

echo "[pending gpu0]"
ls -lh ${P0} | tail -20

echo "[pending gpu1]"
ls -lh ${P1} | tail -20
```

再放

```SQL
cd /root/autodl-tmp/RoboTwin/policy/Motus

TASK=turn_switch
MOTUS_DIR=/root/autodl-tmp/RoboTwin/policy/Motus
QUEUE_ROOT=${MOTUS_DIR}/gpu_queue
P0=${QUEUE_ROOT}/pending/gpu0

mkdir -p "$P0"

cat > ${P0}/709_${TASK}_vtts_k16_s5_c4_noise_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 0 \\
  --tts \\
  --tts-level velocity \\
  --tts-velocity-num-samples 16 \\
  --tts-velocity-select action_keystone \\
  --tts-velocity-perturb-scope vlm_only \\
  --tts-velocity-perturb-policy noise_weak \\
  --tts-velocity-stride 5 \\
  --tts-num-clusters 4 \\
  --tts-tau 0.3 \\
  --tts-kmeans-iters 10 \\
  --tts-velocity-include-original \\
  --tts-velocity-diversity-eps 1e-6 \\
  --tts-velocity-max-original-gap-ratio 0.5 \\
  --tts-velocity-save-perturb-images \\
  --tts-velocity-perturb-image-max 1 \\
  --tts-velocity-perturb-image-mode max_delta \\
  --no-tts-ref \\
  --no-tts-decode-video
EOF

cat > ${P0}/710_${TASK}_vtts_k16_s2_c4_noise_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 0 \\
  --tts \\
  --tts-level velocity \\
  --tts-velocity-num-samples 16 \\
  --tts-velocity-select action_keystone \\
  --tts-velocity-perturb-scope vlm_only \\
  --tts-velocity-perturb-policy noise_weak \\
  --tts-velocity-stride 2 \\
  --tts-num-clusters 4 \\
  --tts-tau 0.3 \\
  --tts-kmeans-iters 10 \\
  --tts-velocity-include-original \\
  --tts-velocity-diversity-eps 1e-6 \\
  --tts-velocity-max-original-gap-ratio 0.5 \\
  --tts-velocity-save-perturb-images \\
  --tts-velocity-perturb-image-max 1 \\
  --tts-velocity-perturb-image-mode max_delta \\
  --no-tts-ref \\
  --no-tts-decode-video
EOF

chmod +x ${P0}/709_${TASK}_vtts_k16_s5_c4_noise_cluster.sh
chmod +x ${P0}/710_${TASK}_vtts_k16_s2_c4_noise_cluster.sh

echo "[bash syntax check]"
bash -n ${P0}/709_${TASK}_vtts_k16_s5_c4_noise_cluster.sh
bash -n ${P0}/710_${TASK}_vtts_k16_s2_c4_noise_cluster.sh

echo "[pending gpu0]"
ls -lh ${P0} | tail -30
```

再放

```SQL
cd /root/autodl-tmp/RoboTwin/policy/Motus

TASK=turn_switch
MOTUS_DIR=/root/autodl-tmp/RoboTwin/policy/Motus
QUEUE_ROOT=${MOTUS_DIR}/gpu_queue
P0=${QUEUE_ROOT}/pending/gpu0
P1=${QUEUE_ROOT}/pending/gpu1

mkdir -p "$P0" "$P1"

cat > ${P0}/711_${TASK}_vtts_k16_s5_c4_noise4_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 0 \
  --tts \
  --tts-level velocity \
  --tts-velocity-num-samples 16 \
  --tts-velocity-select action_keystone \
  --tts-velocity-perturb-scope vlm_only \
  --tts-velocity-perturb-policy noise_weak \
  --tts-velocity-noise-std 0.01568627450980392 \
  --tts-velocity-stride 5 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  --tts-velocity-include-original \
  --tts-velocity-diversity-eps 1e-6 \
  --tts-velocity-max-original-gap-ratio 0.5 \
  --tts-velocity-save-perturb-images \
  --tts-velocity-perturb-image-max 1 \
  --tts-velocity-perturb-image-mode max_delta \
  --no-tts-ref \
  --no-tts-decode-video
EOF

cat > ${P0}/712_${TASK}_vtts_k16_s2_c4_noise4_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 0 \
  --tts \
  --tts-level velocity \
  --tts-velocity-num-samples 16 \
  --tts-velocity-select action_keystone \
  --tts-velocity-perturb-scope vlm_only \
  --tts-velocity-perturb-policy noise_weak \
  --tts-velocity-noise-std 0.01568627450980392 \
  --tts-velocity-stride 2 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  --tts-velocity-include-original \
  --tts-velocity-diversity-eps 1e-6 \
  --tts-velocity-max-original-gap-ratio 0.5 \
  --tts-velocity-save-perturb-images \
  --tts-velocity-perturb-image-max 1 \
  --tts-velocity-perturb-image-mode max_delta \
  --no-tts-ref \
  --no-tts-decode-video
EOF

cat > ${P0}/713_${TASK}_vtts_k16_s2_c4_noise4_cluster_ref_w02.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 0 \
  --tts \
  --tts-level velocity \
  --tts-velocity-num-samples 16 \
  --tts-velocity-select action_keystone \
  --tts-velocity-perturb-scope vlm_only \
  --tts-velocity-perturb-policy noise_weak \
  --tts-velocity-noise-std 0.01568627450980392 \
  --tts-velocity-stride 2 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  --tts-velocity-include-original \
  --tts-velocity-diversity-eps 1e-6 \
  --tts-velocity-max-original-gap-ratio 0.5 \
  --tts-velocity-save-perturb-images \
  --tts-velocity-perturb-image-max 1 \
  --tts-velocity-perturb-image-mode max_delta \
  --tts-ref \
  --tts-ref-weight 0.2 \
  --tts-ref-instruction "Perform the Task" \
  --no-tts-decode-video
EOF

cat > ${P1}/714_${TASK}_vtts_k16_s5_c4_noise6_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 1 \
  --tts \
  --tts-level velocity \
  --tts-velocity-num-samples 16 \
  --tts-velocity-select action_keystone \
  --tts-velocity-perturb-scope vlm_only \
  --tts-velocity-perturb-policy noise_weak \
  --tts-velocity-noise-std 0.02352941176470588 \
  --tts-velocity-stride 5 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  --tts-velocity-include-original \
  --tts-velocity-diversity-eps 1e-6 \
  --tts-velocity-max-original-gap-ratio 0.5 \
  --tts-velocity-save-perturb-images \
  --tts-velocity-perturb-image-max 1 \
  --tts-velocity-perturb-image-mode max_delta \
  --no-tts-ref \
  --no-tts-decode-video
EOF

cat > ${P1}/715_${TASK}_vtts_k16_s2_c4_noise6_cluster.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 1 \
  --tts \
  --tts-level velocity \
  --tts-velocity-num-samples 16 \
  --tts-velocity-select action_keystone \
  --tts-velocity-perturb-scope vlm_only \
  --tts-velocity-perturb-policy noise_weak \
  --tts-velocity-noise-std 0.02352941176470588 \
  --tts-velocity-stride 2 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  --tts-velocity-include-original \
  --tts-velocity-diversity-eps 1e-6 \
  --tts-velocity-max-original-gap-ratio 0.5 \
  --tts-velocity-save-perturb-images \
  --tts-velocity-perturb-image-max 1 \
  --tts-velocity-perturb-image-mode max_delta \
  --no-tts-ref \
  --no-tts-decode-video
EOF

cat > ${P1}/716_${TASK}_vtts_k16_s2_c4_noise4_cluster_ref_w01.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${MOTUS_DIR}

bash eval.sh ${TASK} --gpu-id 1 \
  --tts \
  --tts-level velocity \
  --tts-velocity-num-samples 16 \
  --tts-velocity-select action_keystone \
  --tts-velocity-perturb-scope vlm_only \
  --tts-velocity-perturb-policy noise_weak \
  --tts-velocity-noise-std 0.01568627450980392 \
  --tts-velocity-stride 2 \
  --tts-num-clusters 4 \
  --tts-tau 0.3 \
  --tts-kmeans-iters 10 \
  --tts-velocity-include-original \
  --tts-velocity-diversity-eps 1e-6 \
  --tts-velocity-max-original-gap-ratio 0.5 \
  --tts-velocity-save-perturb-images \
  --tts-velocity-perturb-image-max 1 \
  --tts-velocity-perturb-image-mode max_delta \
  --tts-ref \
  --tts-ref-weight 0.1 \
  --tts-ref-instruction "Perform the Task" \
  --no-tts-decode-video
EOF

chmod +x ${P0}/711_${TASK}_vtts_k16_s5_c4_noise4_cluster.sh
chmod +x ${P0}/712_${TASK}_vtts_k16_s2_c4_noise4_cluster.sh
chmod +x ${P0}/713_${TASK}_vtts_k16_s2_c4_noise4_cluster_ref_w02.sh
chmod +x ${P1}/714_${TASK}_vtts_k16_s5_c4_noise6_cluster.sh
chmod +x ${P1}/715_${TASK}_vtts_k16_s2_c4_noise6_cluster.sh
chmod +x ${P1}/716_${TASK}_vtts_k16_s2_c4_noise4_cluster_ref_w01.sh

echo "[bash syntax check]"
bash -n ${P0}/711_${TASK}_vtts_k16_s5_c4_noise4_cluster.sh
bash -n ${P0}/712_${TASK}_vtts_k16_s2_c4_noise4_cluster.sh
bash -n ${P0}/713_${TASK}_vtts_k16_s2_c4_noise4_cluster_ref_w02.sh
bash -n ${P1}/714_${TASK}_vtts_k16_s5_c4_noise6_cluster.sh
bash -n ${P1}/715_${TASK}_vtts_k16_s2_c4_noise6_cluster.sh
bash -n ${P1}/716_${TASK}_vtts_k16_s2_c4_noise4_cluster_ref_w01.sh

echo "[pending gpu0]"
ls -lh ${P0} | tail -20

echo "[pending gpu1]"
ls -lh ${P1} | tail -20
```



##### v3

Flow opd

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a1d2a4a\-ff40\-8328\-a738\-46b88b852e33



##### v4

flowopd后，opd都能用，更多收集；

可视化，类似rl，reward之类的？参考opd，rl，grpo，tts等



#### m3

tree

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a16b7bc\-f718\-832f\-80dd\-8846fe811fea

？相比rl grpo的优势在哪



#### m4

如何向wam注入信息使其效果更好，充当opsd的老师？

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a1595a2\-f024\-8332\-a3b1\-914b04537b8f



#### m5

```Bash
6. DSRL / 初始噪声选择策略
这条线我们先保存，暂时不作为主讨论线。
核心思想：
不改 Motus 权重
训练一个小策略网络
根据 observation/state/instruction 输出更好的 initial action noise
再交给 frozen Motus denoise
对应：
ϵa=πϕ(o,s,ℓ)\epsilon_a = \pi_\phi(o,s,\ell)ϵa=πϕ(o,s,ℓ)a=Φθ(ϵa,o,s,ℓ)a=\Phi_\theta(\epsilon_a,o,s,\ell)a=Φθ(ϵa,o,s,ℓ)
优点：
 正面承认 TTS 收益来自“好初始噪声选择”。 
 不需要训练 Motus 主体。 
 工程上可能更稳。 
缺点：
 推理时多一个 noise actor。 
 不是真正把能力完全内化进 Motus。 
 受 frozen Motus action support 限制。 
当前决定：
先记录，后面可能重启。
现在优先讨论更细粒度 denoising-step TTS。
```



### 日志分析



#### 06022152

```Bash
OPD 也确实触发了：**80 个成功 episode，每个 episode 做 8 次 update，所以 80 × 8 = 640 条 update**。checkpoint 每 2 个 success 存一次，最后存到：

第一，buffer_size 在 update log 里全部显示为：
current_episode_only
虽然 config 里写了 OPD Buffer Size: 64，但这次实际更新看起来**没有使用 replay buffer**，而是只用当前成功 episode 的数据。
第二，大部分成功 episode 只有 2 条 record，所以训练信号很少：
episode_recordsupdate 行数
2600
38
424
248
也就是说，绝大多数 update 都是在同一个成功 episode 的 2 条 chunk record 上做 8 步小更新。这会比较高方差，也容易 episode-specific overfit。

loss 没有明显随 update_idx 下降。按每 80 个 update 分段，均值在 0.02 到 0.06 之间波动，没有单调下降趋势。
每个 episode 内 8 次 update 也不是稳定下降：
现象比例
最后一次 loss 小于第一次 loss45%
最后一次 student_target_l2 小于第一次46.25%
这说明当前 OPD 更像是在做 **small online nudging**，而不是稳定的 supervised convergence。不能说坏，因为 rollout 分布在变，batch/record 也不完全可比；但它也说明现在的训练日志还不能证明 student 学得越来越好。

7. MAR anchor 太弱
你这次用了：
OPD MAR Mode: base_velocity
OPD MAR Weight: 0.1

0.1 × loss_mar / loss 平均只有 **0.8%**，中位数只有 **0.27%**。所以现在 MAR 基本只是轻微正则，不是强 anchor。
如果你担心 online distillation 把模型带偏，现在这个 MAR 权重可能不够。可以试：
设置建议
保守opd_mar_weight=0.3
中等opd_mar_weight=0.5
强 anchoropd_mar_weight=1.0
我不建议马上把 LR 大幅拉高，同时又保持 MAR=0.1。那样更容易把 action expert 往少量 winner record 过拟合。

你日志里最危险的字段是：
buffer=current_episode_only
虽然配置写了：
OPD Buffer Size: 64
但实际训练没有用历史成功 records。建议你下一轮第一优先级不是改 LR，而是确认代码里是否真的启用 replay buffer。
目标应该是：
当前成功 episode 的 records
+
过去若干成功 episode 的 records
一起采样训练
推荐参数：
--opd-buffer-size 128 \
--opd-batch-size 8 \
--opd-update-steps 4
或者更强一点：
--opd-buffer-size 256 \
--opd-batch-size 8 \
--opd-update-steps 8
```



#### 06011846

```Bash
4. velocity selector 内部确实在工作，但有一个明显现象
四个 velocity CSV 都显示：
denoise_step 只有 0 和 5，说明每个 chunk 内只在两个 denoising step 做 velocity TTS。 
selection_stage 几乎全是 cluster_medoid。 
selected_rank 全部是 0，说明 selector 每次都选了当前内部 ranking 下的第一名。 
fallback_reason 全为空，说明没有触发异常 fallback。 
guard_global_medoid 很少： 
 735：5/1590 
 736：2/1724 
 734：1/1648 
 737：3/1584 
这说明 tau=0.3 下，系统几乎总是认为候选是“足够多峰”的，然后进入 cluster_medoid，而不是走 unimodal guard。
更关键的是：**原始未扰动 candidate 几乎从来没被选中。**
```



#### 06011024

sota：K16, stride5, C4, noise6, no\-ref；noise继续增加

opd的各个ckpyno tts的评估下



```Bash
最值得继续保留的配置是 714：
K16, stride5, C4好
ref没用

4.2 seed 难度分层明显
这些实验大部分使用高度重叠的 seed。对 16 个 velocity-level 方法共有的 90 个 seed 做统计：
 41 个 seed：所有方法都成功。 
 7 个 seed：所有方法都失败。 
 13 个 seed：只有 1 到 5 个方法能成功，属于真正区分方法的 hard-ish seeds。 
全失败 seed 包括：
4300011, 4300045, 4300068, 4300102, 4300116, 4300127, 4300148
这几个 seed 很适合作为失败案例池，后面可以专门可视化。它们可能不是简单调 K、stride、noise 就能解决，可能需要动作策略本身、任务状态识别或更强的 selection signal。

第一，**stride5 的优势不只是省计算，还可能避免了 late denoise step 的过强干预。**
 在 stride2 里会选择 denoise step 0,2,4,6,8，CSV 显示 pairwise diversity 在后段明显变大，尤其 denoise_step=8 最大。noise6 + stride2 会在这种高差异后段继续强插选择，结果 715 掉到 72%。而 noise6 + stride5 只看 0,5，714 反而到 80%。
 
 CSV 里能看到 ref 不是完全没影响，它确实改变了选择，但改变方向不稳定。ref0.2 时，selected 从 global medoid 偏离的比例从 no-ref 的约 14% 到 16% 提高到 28.7%，但成功率下降。这说明 ref 现在更像是在引入额外偏置，而不是可靠的成功信号。
 
selected_cluster_size / candidate_pool_size：cluster 不是均匀分簇，而是一个 dominant mode 加若干小簇
K16 时，selected cluster 的典型大小大概是：
Cselected cluster share 中位数含义
C214/16 或 15/16，约 0.88一个巨大主簇，少量 outlier
C312/16，约 0.75主簇仍然很大
C410/16，约 0.62主簇变小，但仍是 dominant mode
这说明当前 kmeans clustering 的作用不是发现几个均衡的多模态动作方案，而是：
大主簇 + 若干小 outlier cluster

所以 C 的含义更像是“把 outlier 从主簇中剥离出去的强度”。C4 会让主簇更小，也更激进。
 这解释了为什么低噪声下 C4 不一定好：
K16 s5 C2 n2 = 75%
K16 s5 C3 n2 = 77%
K16 s5 C4 n2 = 73%
但当噪声提高后，C4 反而变好：
K16 s5 C4 n2 = 73%
K16 s5 C4 n4 = 78%
K16 s5 C4 n6 = 80%

结论是：**ref signal 目前弱，而且和成功不稳定对齐**

K16 下 ref_rank 平均 6 到 7，基本接近中间位置，不是 ref 最喜欢的 candidate。也就是说 fusion 仍然主要被 consensus/cluster 约束，ref 只是轻微改变排序。

4.2 但 ref 又足够改变一些选择，造成负收益

ref0.2 明显增加了偏离 global medoid 的比例，但成功率从 77% 掉到 74%。
 所以当前 ref 不是完全没作用，而是**有作用但方向不可靠**
 
 
```



#### 脚本



队列实现

使用：

```Bash
核心规则：
1. 你把任务脚本放进 pending/gpu0 或 pending/gpu1。
2. worker 按文件名排序取最早的 .sh。
3. 取到后先 mv 到 running，避免重复执行。
4. 等 GPU 空闲。
5. 用 nohup 启动这个任务，并 wait 它结束。
6. 成功则移到 done，失败则移到 failed。
7. 然后继续取下一个。
重要约定：**任务文件内部不要再写 nohup ... &**。任务文件里写前台命令即可，比如 bash eval.sh ... > "$LOG" 2>&1。外层 worker 会负责 nohup 和等待。这样才能保证真正串行。

3. 如何添加一个任务
关键点：**先写到 tmp 文件，再 mv 到 pending**。这样可以避免 worker 读到写了一半的文件。
示例：往 GPU 0 队列添加 N=16 keystone 任务
cd /root/autodl-tmp/RoboTwin/policy/Motus

mkdir -p gpu_queue/tmp gpu_queue/pending/gpu0

JOB_TMP=gpu_queue/tmp/001_turn_switch_keystone_n16_c2_tau05.sh
JOB_DST=gpu_queue/pending/gpu0/001_turn_switch_keystone_n16_c2_tau05.sh

cat > "$JOB_TMP" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p logs

TASK=turn_switch
GPU_ID=0

LOG=logs/${TASK}_tts_keystone_n16_b16_c2_tau05_nodecode_$(date +%Y%m%d_%H%M%S).log

bash eval.sh ${TASK} --gpu-id ${GPU_ID} --tts --tts-num-samples 16 \
  --tts-method keystone \
  --tts-num-clusters 2 \
  --tts-tau 0.5 \
  --tts-kmeans-iters 10 \
  --tts-batch-size 16 \
  --no-tts-decode-video \
  > "$LOG" 2>&1

echo "LOG: $LOG"
SH

chmod +x "$JOB_TMP"
mv "$JOB_TMP" "$JOB_DST"

echo "submitted: $JOB_DST"

4. 如何一次放很多任务
你可以按文件名前缀控制顺序：
001_turn_switch_xxx.sh
002_scan_object_xxx.sh
003_put_bottles_xxx.sh

6. 如何停止队列但不杀当前任务
如果你只想“不再启动新任务”，但当前已经跑起来的任务继续跑：
cd /root/autodl-tmp/RoboTwin/policy/Motus

touch gpu_queue/STOP_gpu0
touch gpu_queue/STOP_gpu1
worker 会在当前任务结束后看到 STOP 文件，然后退出，不再启动后续任务。
恢复队列：
rm -f gpu_queue/STOP_gpu0 gpu_queue/STOP_gpu1
然后重新启动 worker。
```

脚本：

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

cat > gpu_queue_worker.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG
# =========================
GPU_ID="${1:?Usage: ./gpu_queue_worker.sh GPU_ID}"

QUEUE_ROOT="/root/autodl-tmp/RoboTwin/policy/Motus/gpu_queue"

THRESHOLD_MB=100
STABLE_SECONDS=20
CHECK_INTERVAL=2
NO_JOB_SLEEP=5

# 0: keep waiting for new jobs
# 1: exit when queue is empty
EXIT_WHEN_EMPTY=0

PENDING_DIR="${QUEUE_ROOT}/pending/gpu${GPU_ID}"
RUNNING_DIR="${QUEUE_ROOT}/running/gpu${GPU_ID}"
DONE_DIR="${QUEUE_ROOT}/done/gpu${GPU_ID}"
FAILED_DIR="${QUEUE_ROOT}/failed/gpu${GPU_ID}"
LOG_DIR="${QUEUE_ROOT}/logs/gpu${GPU_ID}"
LOCK_ROOT="${QUEUE_ROOT}/locks"
LOCK_DIR="${LOCK_ROOT}/gpu${GPU_ID}.lock"
STOP_FILE="${QUEUE_ROOT}/STOP_gpu${GPU_ID}"

# =========================
# Setup
# =========================
mkdir -p "${PENDING_DIR}" "${RUNNING_DIR}" "${DONE_DIR}" "${FAILED_DIR}" "${LOG_DIR}" "${LOCK_ROOT}"

acquire_lock() {
    while true; do
        if mkdir "${LOCK_DIR}" 2>/dev/null; then
            echo "$$" > "${LOCK_DIR}/pid"
            trap 'rm -rf "${LOCK_DIR}"' EXIT
            return 0
        fi

        old_pid="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
        if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
            echo "[worker] another worker is already running for gpu${GPU_ID}: pid=${old_pid}"
            exit 1
        fi

        echo "[worker] removing stale lock: ${LOCK_DIR}"
        rm -rf "${LOCK_DIR}"
    done
}

get_used_mb() {
    local used_mb
    used_mb="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9')"
    if [ -z "${used_mb}" ]; then
        used_mb=999999
    fi
    echo "${used_mb}"
}

wait_gpu_idle() {
    local stable_count=0
    local used_mb

    echo "[worker] waiting for gpu${GPU_ID} idle at $(date)"
    while true; do
        used_mb="$(get_used_mb)"

        echo "[worker] $(date '+%F %T') gpu=${GPU_ID} used_mb=${used_mb} stable_count=${stable_count}/${STABLE_SECONDS}"

        if [ "${used_mb}" -lt "${THRESHOLD_MB}" ]; then
            stable_count=$((stable_count + CHECK_INTERVAL))
        else
            stable_count=0
        fi

        if [ "${stable_count}" -ge "${STABLE_SECONDS}" ]; then
            echo "[worker] gpu${GPU_ID} idle for ${STABLE_SECONDS}s"
            return 0
        fi

        sleep "${CHECK_INTERVAL}"
    done
}

next_job() {
    find "${PENDING_DIR}" -maxdepth 1 -type f -name "*.sh" | sort | head -n 1
}

run_job() {
    local job="$1"
    local base
    local stamp
    local running_job
    local job_log
    local child_pid
    local status

    base="$(basename "${job}")"
    stamp="$(date +%Y%m%d_%H%M%S)"
    running_job="${RUNNING_DIR}/${stamp}_${base}"
    job_log="${LOG_DIR}/${stamp}_${base%.sh}.log"

    if ! mv "${job}" "${running_job}" 2>/dev/null; then
        echo "[worker] failed to claim job, maybe another worker took it: ${job}"
        return 0
    fi

    chmod +x "${running_job}"

    echo "[worker] claimed job: ${running_job}"
    echo "[worker] job log: ${job_log}"

    wait_gpu_idle

    echo "[worker] launching job at $(date)"
    nohup bash "${running_job}" > "${job_log}" 2>&1 &
    child_pid="$!"
    echo "${child_pid}" > "${running_job}.pid"

    echo "[worker] child pid: ${child_pid}"

    if wait "${child_pid}"; then
        status=0
    else
        status="$?"
    fi

    rm -f "${running_job}.pid"

    if [ "${status}" -eq 0 ]; then
        echo "[worker] job succeeded: ${running_job}"
        mv "${running_job}" "${DONE_DIR}/${stamp}_${base}"
    else
        echo "[worker] job failed with status=${status}: ${running_job}"
        mv "${running_job}" "${FAILED_DIR}/${stamp}_${base}"
    fi

    echo "[worker] job finished at $(date)"
}

# =========================
# Main
# =========================
acquire_lock

echo "[worker] start at $(date)"
echo "[worker] GPU_ID=${GPU_ID}"
echo "[worker] QUEUE_ROOT=${QUEUE_ROOT}"
echo "[worker] PENDING_DIR=${PENDING_DIR}"
echo "[worker] THRESHOLD_MB=${THRESHOLD_MB}"
echo "[worker] STABLE_SECONDS=${STABLE_SECONDS}"
echo "[worker] EXIT_WHEN_EMPTY=${EXIT_WHEN_EMPTY}"
echo "[worker] STOP_FILE=${STOP_FILE}"

while true; do
    if [ -f "${STOP_FILE}" ]; then
        echo "[worker] stop file found: ${STOP_FILE}"
        echo "[worker] exiting without launching more jobs"
        exit 0
    fi

    job="$(next_job || true)"

    if [ -z "${job}" ]; then
        echo "[worker] no pending job at $(date)"
        if [ "${EXIT_WHEN_EMPTY}" -eq 1 ]; then
            echo "[worker] queue empty, exiting"
            exit 0
        fi
        sleep "${NO_JOB_SLEEP}"
        continue
    fi

    run_job "${job}"
done
SH

chmod +x gpu_queue_worker.sh
mkdir -p gpu_queue/pending/gpu0 gpu_queue/pending/gpu1 gpu_queue/logs/gpu0 gpu_queue/logs/gpu1

ls -lh gpu_queue_worker.sh
```

启动两个队列：

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus
mkdir -p gpu_queue/logs

WORKER_LOG=gpu_queue/logs/worker_gpu0_$(date +%Y%m%d_%H%M%S).log
nohup ./gpu_queue_worker.sh 0 > "$WORKER_LOG" 2>&1 &
echo "WORKER_GPU0_PID: $!"
echo "WORKER_GPU0_LOG: $WORKER_LOG"

WORKER_LOG=gpu_queue/logs/worker_gpu1_$(date +%Y%m%d_%H%M%S).log
nohup ./gpu_queue_worker.sh 1 > "$WORKER_LOG" 2>&1 &
echo "WORKER_GPU1_PID: $!"
echo "WORKER_GPU1_LOG: $WORKER_LOG"
```



分析流程：本地可视化，日志和图都给gpt，获得结论等



脚本v1

\[tts\_log\_visualizer\.py\]

使用：

```Plain Text
python tts_log_visualizer.py \
  --csv /path/to/summary.csv \
  --log /path/to/outer.log \
  --name my_run \
  --out /path/to/tts_report
```



v2

加中文注释

\[tts\_log\_visualizer\_zh\.py\]

```Plain Text
python tts_log_visualizer_zh.py \
  --csv /path/to/summary.csv \
  --log /path/to/outer.log \
  --name my_run \
  --out /path/to/tts_report_zh
```



v3

\[tts\_log\_visualizer\_zh\_v2\.py\]

调整输出目录，中文渲染，指令等

运行：

```Plain Text
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

python tts_log_visualizer_zh_v2.py \
  /path/to/summary.csv \
  /path/to/outer.log
```

装中文字体

```Plain Text
sudo apt-get update
sudo apt-get install -y fonts-noto-cjk fonts-wqy-microhei
fc-cache -fv
rm -rf ~/.cache/matplotlib
```



v4

\[tts\_log\_visualizer\_zh\_v4\.py\]



#### 05312144



```Bash
实验核心配置referenceCSB 触发 denoise step成功率CSV 行数
605_turn_switch_vtts_k64_s5_noise_cluster_ref_w05velocity TTS, K=64, stride=5, c=2, tau=0.3, noise_weak, vlm_only开启，text_mask, weight=0.50, 567/1002014
603_turn_switch_vtts_k32_s2_noise_clustervelocity TTS, K=32, stride=2, c=2, tau=0.3, noise_weak, vlm_only关闭0, 2, 4, 6, 875/1004105
701_turn_switch_vtts_k4_s5_c2_noise_clustervelocity TTS, K=4, stride=5, c=2, tau=0.3, noise_weak, vlm_only关闭0, 580/1001400

，c=2 大多数时候不是发现两个稳定动作模式，而是形成：
一个大簇 + 一个小 outlier 簇

tau 是 0.3，但绝大多数 s_score 都远高于 0.3。所以 guard_global_medoid 只触发了 3 到 4 次，几乎没有调控作用。


```



#### rad n4b4



```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

python tts_log_visualizer_zh_v4.py \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs_single_20260521_164959/tts/scan_object/summary.csv \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs/scan_object_tts_rank_softmax_n4_b4_tau1_nodecode_20260521_164959.log
```



164959



#### l2\+fusion w, n4b4

\[summary\_024039\.csv\]

\[scan\_object\_tts\_video\_rank\_weighted\_borda\_n4\_b4\_w05\_latent\_nodecode\_20260522\_024038\.log\]

video\_weight=0\.5 让 action rank 仍然占主导

#### l2\+fusion rrf, n4b4

RRF 当前参数几乎完全退化成 action\-only。



#### l2\+fusion w, n8b8



这次 weighted Borda `K=8, w=0.5` 有 31\.8% action override。

weighted Borda 当前偏激进，video latent 经常改变 action 选择。



```Bash
cd /home/ubuntu/workspace/RoboTwin/policy/Motus

python tts_log_visualizer_zh_v4.py \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs_single_20260521_225211/tts/scan_object/summary.csv \
  /home/ubuntu/workspace/RoboTwin/policy/Motus/logs/scan_object_tts_video_rank_weighted_borda_n8_b8_w05_latent_nodecode_20260521_225210.log
```



225211



#### l2\+fusion rrf, n8b8

结论： K 从 4 增大到 8 后，video rank 有更大空间让 action rank 1 的候选翻盘。

覆写率 7\.1



可视化

```Plain Text
python tts_log_visualizer_zh.py \
  --csv /home/ubuntu/workspace/RoboTwin/policy/Motus/logs_single_20260521_225224/tts/scan_object/summary.csv  \
  --log /home/ubuntu/workspace/RoboTwin/policy/Motus/logs/scan_object_tts_video_rank_rrf_n8_b8_w05_k1_latent_nodecode_20260521_225224.log \
  --name video_rank_rrf_n8_b8_w05_k1 \
  --out /home/ubuntu/workspace/RoboTwin/policy/Motus/plot
```



225224



### Git



尝试直接终端登录

#### 查

排查机子状态

文件大小：

```Bash
cd /home/ubuntu/workspace

echo "===== total workspace size ====="
du -sh .

echo "===== top-level directories ====="
du -h --max-depth=1 . | sort -h

echo "===== largest files ====="
find . -type f -printf '%s %p\n' | sort -nr | head -50 | awk '{printf "%.2f GB  %s\n", $1/1024/1024/1024, $2}'

echo "===== possible model/checkpoint files ====="
find . -type f \( \
  -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.safetensors" -o \
  -name "*.bin" -o -name "*.onnx" -o -name "*.pkl" -o -name "*.npz" -o -name "*.h5" \
\) -printf '%s %p\n' | sort -nr | head -100 | awk '{printf "%.2f GB  %s\n", $1/1024/1024/1024, $2}'

echo "===== possible media/log files ====="
find . -type f \( \
  -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.mp4" -o -name "*.avi" -o \
  -name "*.log" -o -name "*.npy" -o -name "*.npz" \
\) -printf '%s %p\n' | sort -nr | head -100 | awk '{printf "%.2f GB  %s\n", $1/1024/1024/1024, $2}'
```

Git

```Plain Text
cd /home/ubuntu/workspace

echo "===== existing git repos ====="
find . -name .git -type d -prune -print
```



查git远端

```Plain Text
cd /home/ubuntu/workspace/RoboTwin

git status -sb
git remote -v
```



#### ignore

备份原本ignore，加规则，检查规则，尝试add以检查

```Plain Text
cd /home/ubuntu/workspace/RoboTwin
cp .gitignore .gitignore.bak.$(date +%Y%m%d_%H%M%S)

cat >> .gitignore <<'EOF'

# =====================================================================
# Local generated files from this machine
# =====================================================================

# All log directories and log files, including logs_single_*
**/logs*/
*.log

# ACT generated data and checkpoints
policy/ACT/processed_data/
policy/ACT/act_ckpt/

# Large generated file types observed locally
*.ckpt
*.hdf5
*.npz

# Keep Motus source code under policy/Motus/models
!policy/Motus/models/
!policy/Motus/models/**
EOF

cd /home/ubuntu/workspace/RoboTwin

git check-ignore -v policy/ACT/processed_data/sim-beat_block_hammer/demo_clean-50/episode_1.hdf5 || true
git check-ignore -v policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50/policy_best.ckpt || true
git check-ignore -v policy/Motus/logs_single_20260516_222557/scan_object.log || true
git check-ignore -v policy/Motus/logs_single_20260516_222557/tts/scan_object/episode_0023_step_0008.npz || true
git check-ignore -v policy/Motus/models/motus.py || true

cd /home/ubuntu/workspace/RoboTwin
git add -n . | head -200
```



add并检查

```Python
git add -A

python - <<'PY'
import os
import subprocess

files = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines()
big = []
for f in files:
    if os.path.isfile(f):
        s = os.path.getsize(f)
        if s > 50 * 1024 * 1024:
            big.append((s, f))

for s, f in sorted(big, reverse=True):
    print(f"{s/1024/1024:.1f} MB\t{f}")

print("staged files >50MB:", len(big))
PY
```



问题：

```Plain Text
你把这行误写进了 .gitignore： 
cat >> .gitignore <<'EOF'

把 .gitignore.bak.20260517_125334 也暂存了。这个备份文件不需要进仓库。 
policy/Motus/models/__pycache__/*.pyc 被暂存了。这个不该进 Git。原因是我们加了： 
!policy/Motus/models/**
```

修

```Bash
cd /home/ubuntu/workspace/RoboTwin

# 1. 用刚才备份恢复干净的官方 .gitignore
LATEST_BAK=$(ls -t .gitignore.bak.* | head -1)
cp "$LATEST_BAK" .gitignore

# 2. 重新追加最小必要忽略规则
cat >> .gitignore <<'EOF'

# =====================================================================
# Local generated files from this machine
# =====================================================================

# All log directories and log files, including logs_single_*
**/logs*/
*.log

# ACT generated data and checkpoints
policy/ACT/processed_data/
policy/ACT/act_ckpt/

# Large generated file types observed locally
*.ckpt
*.hdf5
*.npz

# Keep Motus source code under policy/Motus/models
!policy/Motus/models/
!policy/Motus/models/**

# But still ignore Python cache files under Motus
policy/Motus/**/__pycache__/
policy/Motus/**/*.pyc
EOF

# 3. 删除本地 .gitignore 备份文件，不提交它
git restore --staged .gitignore.bak.* 2>/dev/null || true
rm -f .gitignore.bak.*

# 4. 从暂存区移除 pycache，不删工作区代码
git rm --cached -r --ignore-unmatch policy/Motus/models/__pycache__ >/dev/null 2>&1 || true

# 5. 重新暂存
git add -A

# 6. 再检查是否还有大文件
python - <<'PY'
import os
import subprocess

files = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines()
big = []
for f in files:
    if os.path.isfile(f):
        s = os.path.getsize(f)
        if s > 50 * 1024 * 1024:
            big.append((s, f))

for s, f in sorted(big, reverse=True):
    print(f"{s/1024/1024:.1f} MB\t{f}")

print("staged files >50MB:", len(big))
PY

# 7. 看看现在准备提交什么
git status --short
```



#### 交



本地交，移除rt2官方远端

```Plain Text
git config user.name "Yutenji-Nyamu"
git config user.email "Yutenji-Nyamu@users.noreply.github.com"
git commit -m "Add Motus TTS evaluation support"

git remote rename origin upstream
git remote -v
```



检查本地

```Plain Text
cd /home/ubuntu/workspace/RoboTwin

git status -sb
git remote -v
git log --oneline -1
```



登录

```Plain Text
which gh && gh --version

sudo apt update
sudo apt install -y gh

gh auth login
```

交互

```Plain Text
GitHub.com
HTTPS
Yes, authenticate Git with your GitHub credentials
Login with a web browser
```

检查

```Plain Text
gh auth status
```



创建仓库

```Plain Text
cd /home/ubuntu/workspace/RoboTwin

gh repo create RoboTwin-Motus-TTS \
  --private \
  --source=. \
  --remote=origin
  
git remote -v
  
git branch --show-current

git push -u origin main
```

gpt：成功

网页看得见



改成公开

```Plain Text
cd /home/ubuntu/workspace/RoboTwin

gh repo edit Yutenji-Nyamu/RoboTwin-Motus-TTS \
  --visibility public \
  --accept-visibility-change-consequences

gh repo view Yutenji-Nyamu/RoboTwin-Motus-TTS --json visibility,url
```

报错

```Plain Text
1. 你的 gh 版本比较旧：gh 2.4.0
2. 这个版本不支持 --accept-visibility-change-consequences
3. 这个版本的 gh repo view 也没有 visibility 字段，只能看 isPrivate
```

修

```Plain Text
cd /home/ubuntu/workspace/RoboTwin

gh repo edit Yutenji-Nyamu/RoboTwin-Motus-TTS --visibility public

gh repo view Yutenji-Nyamu/RoboTwin-Motus-TTS --json isPrivate,url
```

成功



