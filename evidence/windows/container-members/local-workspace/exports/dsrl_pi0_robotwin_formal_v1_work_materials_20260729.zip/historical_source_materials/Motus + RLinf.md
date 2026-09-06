# Motus \+ RLinf



### guide

从跑通 RLinf \+ RoboTwin 2\.0 \+ pi0 \+ PPO 和 Motus 训推的机子：[Openpi \+ PPO AutoDL A800](https://my.feishu.cn/wiki/HkMUwQDcOikfygkTVm2cBsnEnVe?from=from_copylink)





### 1 拉新rlinf，跑训推



#### 环境和代码



清理ray

```Bash
source /root/autodl-tmp/rlinf_env.sh

echo "===== stop ray ====="
ray stop -f || true

echo "===== optional kill old RLinf python processes ====="
pkill -f "train_embodied_agent.py" || true
pkill -f "eval_embodied_agent.py" || true
pkill -f "run_embodiment.sh" || true
pkill -f "eval_embodiment.sh" || true
pkill -f "evaluations/run_eval.sh" || true

echo "===== gpu check ====="
nvidia-smi
```



把旧 RLinf 目录改名保留；重新 clone 一个干净 RLinf 到原路径

```Bash
/root/autodl-tmp/RLinf_old_时间戳   # 旧代码，保底留着
/root/autodl-tmp/RLinf              # 新云端代码，继续用原路径
```

？？？检查看大小

```Bash
source /root/autodl-tmp/rlinf_env.sh

cd /root/autodl-tmp

STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -d /root/autodl-tmp/RLinf ]; then
  mv /root/autodl-tmp/RLinf "/root/autodl-tmp/RLinf_old_${STAMP}"
  echo "OLD_RLINF=/root/autodl-tmp/RLinf_old_${STAMP}"
fi

git clone https://github.com/RLinf/RLinf.git /root/autodl-tmp/RLinf

cd /root/autodl-tmp/RLinf

echo "===== new RLinf ====="
pwd
git rev-parse --short HEAD
git status --short
```



\.venv 复用：复制旧 \.venv 到新 RLinf

```Bash
旧环境已经装过 OpenPI、RoboTwin 依赖、flash-attn 等，直接复制过来能省很多时间。复制后再把新仓库注册成 editable，避免 .venv 里 editable path 指向旧仓库。
```

环境复制：

```Bash
source /root/autodl-tmp/rlinf_env.sh

OLD_RLINF="$(ls -dt /root/autodl-tmp/RLinf_old_* | head -n 1)"
NEW_RLINF="/root/autodl-tmp/RLinf"

echo "OLD_RLINF=${OLD_RLINF}"
echo "NEW_RLINF=${NEW_RLINF}"

if [ -d "${OLD_RLINF}/.venv" ] && [ ! -d "${NEW_RLINF}/.venv" ]; then
  cp -a "${OLD_RLINF}/.venv" "${NEW_RLINF}/.venv"
fi

cd "${NEW_RLINF}"
source .venv/bin/activate

python --version
which python

# 更新 editable 指向新仓库；不重装大依赖
uv pip install -e . --no-deps

python - <<'PY'
import sys
print("python =", sys.executable)
try:
    import rlinf
    print("rlinf =", rlinf.__file__)
except Exception as e:
    print("rlinf import failed:", repr(e))
PY
```

检查环境：

```Python
source /root/autodl-tmp/rlinf_env.sh

cd "${RLINF_ROOT}"
source .venv/bin/activate

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
```

如果后面空间紧，最先可删的是旧目录 `.venv`，因为新 RLinf 已经复制了一份可用 `.venv`：



#### 跑评估

参考之前，手动改配置



复制配置到：

/root/autodl\-tmp/RLinf/evaluations/robotwin/robotwin\_adjust\_bottle\_openpi\_eval\_autodl\.yaml



改：

```Bash
cluster.component_placement.env, rollout
0-1

runner.logger.experiment_name
"robotwin_ppo_openpi_autodl"

env.eval.rollout_epoch
10

env.eval.total_num_envs
16

env.eval.assets_path
/root/autodl-tmp/RoboTwin_RLinf

rollout.model.model_path
/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
```



跑新的评估：

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

CONFIG_NAME="robotwin_adjust_bottle_openpi_eval_autodl_pi0"
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

bash evaluations/run_eval.sh robotwin ${CONFIG_NAME}
" > "${LOG}" 2>&1 &

PID=$!
echo "PID=${PID}"
echo "LOG=${LOG}"
```

\[1\] 5421

PID=5421

LOG=logs/nohup/robotwin\_adjust\_bottle\_openpi\_eval\_autodl\_pi0\_20260618\_090530\.log

报错

```Bash
config name 应该是：
robotwin_adjust_bottle_openpi_eval_autodl
不是：
robotwin_adjust_bottle_openpi_eval_autodl_pi0

日志里：
Cannot infer benchmark for config 'robotwin'. Pass benchmark explicitly.
这个错误说明 evaluations/run_eval.sh 实际只收到了一个参数：
robotwin
也就是说，你本来想跑：
bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval_autodl
但实际进入脚本的可能变成了：
bash evaluations/run_eval.sh robotwin
常见原因是 nohup 里的 CONFIG_NAME 没传进去，或者 config 名字变量为空。
```

问题：

```Bash
**命令参数传错了**：evaluations/run_eval.sh 实际只收到了 robotwin 一个参数，所以它把 robotwin 当成 config 名去推断 benchmark，结果报：
Cannot infer benchmark for config 'robotwin'. Pass benchmark explicitly.
你这次用：
bash -x evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval_autodl
是正确的。脚本已经识别成：
benchmark = robotwin
config    = robotwin_adjust_bottle_openpi_eval_autodl
```



非nohup指令：

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

CONFIG_NAME="robotwin_adjust_bottle_openpi_eval_autodl"

ls -lh "evaluations/robotwin/${CONFIG_NAME}.yaml"

bash -x evaluations/run_eval.sh robotwin "${CONFIG_NAME}"
```

成功开始



正确评估命令

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval_autodl
```

nohup

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

mkdir -p logs/nohup

CONFIG_NAME="robotwin_adjust_bottle_openpi_eval_autodl"
LOG="logs/nohup/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

nohup env CONFIG_NAME="${CONFIG_NAME}" bash -lc '
set -o pipefail

source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

echo "CONFIG_NAME=${CONFIG_NAME}"
ls -lh "evaluations/robotwin/${CONFIG_NAME}.yaml"

bash evaluations/run_eval.sh robotwin "${CONFIG_NAME}"
' > "${LOG}" 2>&1 &

echo "PID=$!"
echo "LOG=${LOG}"
```



一直报的不影响训推的curobo报错：

```Bash
你现在看到的：
ModuleNotFoundError: No module named 'curobo.types.math'; 'curobo.types' is not a package
missing pytorch3d
Something wrong happened when importing CuroboPlanner
目前可以先视为**非致命 warning**，不是阻断项。原因有两个：
第一，你的配置里 RoboTwin task 用的是：
planner_backend: mplib
不是 curobo。所以 CuroboPlanner import 失败后，RoboTwin 仍然可以继续走 mplib planner。
```



#### 跑训练



复制配置到

/root/autodl\-tmp/RLinf/examples/embodiment/config/robotwin\_adjust\_bottle\_ppo\_openpi\_autodl\.yaml



改配置

```Bash
cluster.component_placement.actor, env, rollout
0-1

runner.max_steps
5

runner.val_check_interval
5

runner.save_interval
5

runner.logger.experiment_name
"robotwin_ppo_openpi_autodl"

env.train.assets_path
env.eval.assets_path
/root/autodl-tmp/RoboTwin_RLinf

env.train.total_num_envs
16

env.eval.total_num_envs
4

actor.micro_batch_size
16

actor.global_batch_size
256

actor.model.model_path:
"/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
```



跑训练：

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

mkdir -p logs/nohup

CONFIG_NAME="robotwin_adjust_bottle_ppo_openpi_autodl"
LOG="logs/nohup/${CONFIG_NAME}_smoke_$(date +%Y%m%d_%H%M%S).log"

nohup env CONFIG_NAME="${CONFIG_NAME}" bash -lc '
set -o pipefail

source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

echo "CONFIG_NAME=${CONFIG_NAME}"
ls -lh "examples/embodiment/config/${CONFIG_NAME}.yaml"

bash examples/embodiment/run_embodiment.sh "${CONFIG_NAME}"
' > "${LOG}" 2>&1 &

echo "PID=$!"
echo "LOG=${LOG}"
```

\[1\] 21633

PID=21633

LOG=logs/nohup/robotwin\_adjust\_bottle\_ppo\_openpi\_autodl\_smoke\_20260618\_094157\.log

成功跑通



### 2 改 GRPO



#### 训

新建yaml，从ppo复制

```Bash
cd /root/autodl-tmp/RLinf

cp examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_autodl.yaml \
   examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_autodl.yaml
```



建议改参数：

```Bash
runner.logger.experiment_name
"robotwin_grpo_openpi_autodl"

runner.max_steps
5

val_check_interval: -1

algorithm.group_size
8

algorithm.adv_type
grpo

algorithm.loss_type
actor

algorithm.gamma
1.0

algorithm.filter_rewards
True

env.train.rollout_epoch
2

actor.model.add_value_head
False

actor.global_batch_size
128
```



跑训练

```Bash
mkdir -p /root/autodl-tmp/RLinf/local_scripts

cat > /root/autodl-tmp/RLinf/local_scripts/run_train_robotwin.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

CONFIG_NAME="${1:?Usage: $0 <config_name>}"

bash examples/embodiment/run_embodiment.sh "${CONFIG_NAME}"
EOF

chmod +x /root/autodl-tmp/RLinf/local_scripts/run_train_robotwin.sh

cd /root/autodl-tmp/RLinf
source .venv/bin/activate

CONFIG_NAME="robotwin_adjust_bottle_grpo_openpi_autodl"
LOG="logs/nohup/${CONFIG_NAME}_smoke_$(date +%Y%m%d_%H%M%S).log"

nohup local_scripts/run_train_robotwin.sh "${CONFIG_NAME}" > "${LOG}" 2>&1 &

echo "PID=$!"
echo "LOG=${LOG}"
```

\[1\] 82170

PID=82170

LOG=logs/nohup/robotwin\_adjust\_bottle\_grpo\_openpi\_autodl\_smoke\_20260618\_103525\.log



#### 推

评估ppo grpo训完的ckpt？



改配置

注意critic head

```Bash
**GRPO 训练时 add_value_head=False，所以 eval 加载 GRPO checkpoint 时也要让 eval model 结构保持 add_value_head=False。**
你当前 openpi eval 配置里可能仍然是：
rollout:
  model:
    add_value_head: True
如果直接加载 GRPO checkpoint，可能因为 value head key 不匹配而报 missing/unexpected keys
```

改

/root/autodl\-tmp/RLinf/evaluations/robotwin/robotwin\_adjust\_bottle\_openpi\_eval\_autodl\.yaml

```Bash
rollout.model.openpi.add_value_head
False
```



评估：

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

CKPT="/root/autodl-tmp/RLinf/logs/20260618-10:35:25-robotwin_adjust_bottle_grpo_openpi_autodl/robotwin_grpo_openpi_autodl/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt"

bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval_autodl \
  runner.ckpt_path="${CKPT}" \
  runner.logger.experiment_name="robotwin_grpo_openpi_eval_autodl_step5" \
  rollout.model.add_value_head=False \
  rollout.model.openpi.add_value_head=False \
  rollout.model.openpi.detach_critic_input=False
```



成功率75，没变化；gpt：步数太少，后面再说



### 3 motus还原



检查

```Bash
cd /root/autodl-tmp

echo "===== existing dirs ====="
ls -ld /root/autodl-tmp/Motus 2>/dev/null || true
ls -ld /root/autodl-tmp/RoboTwin/policy/Motus 2>/dev/null || true
ls -ld /root/autodl-tmp/models/motus 2>/dev/null || true
ls -ld /root/autodl-tmp/conda/envs/RoboTwin 2>/dev/null || true

echo
echo "===== sizes ====="
du -sh /root/autodl-tmp/Motus 2>/dev/null || true
du -sh /root/autodl-tmp/RoboTwin/policy/Motus 2>/dev/null || true
du -sh /root/autodl-tmp/models/motus 2>/dev/null || true
du -sh /root/autodl-tmp/conda/envs/RoboTwin 2>/dev/null || true
```



备份，Clone

```Bash
cd /root/autodl-tmp

STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -d /root/autodl-tmp/Motus ]; then
  mv /root/autodl-tmp/Motus "/root/autodl-tmp/Motus_old_${STAMP}"
  echo "OLD_MOTUS=/root/autodl-tmp/Motus_old_${STAMP}"
fi

if [ -d /root/autodl-tmp/RoboTwin/policy/Motus ]; then
  mv /root/autodl-tmp/RoboTwin/policy/Motus "/root/autodl-tmp/RoboTwin/policy/Motus_old_${STAMP}"
  echo "OLD_ROBOTWIN_POLICY_MOTUS=/root/autodl-tmp/RoboTwin/policy/Motus_old_${STAMP}"
fi

cd /root/autodl-tmp

git clone https://github.com/thu-ml/Motus.git /root/autodl-tmp/Motus

cd /root/autodl-tmp/Motus
git rev-parse --short HEAD
git status --short
ls -lah | sed -n '1,120p'
```

问题：网络

```Bash
cd /root/autodl-tmp

# 确保没有半截目录
rm -rf /root/autodl-tmp/Motus

# 临时开启 AutoDL GitHub 加速
source /etc/network_turbo

git clone https://github.com/thu-ml/Motus.git /root/autodl-tmp/Motus

# clone 完立即清代理，避免污染后续 pip/hf
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

cd /root/autodl-tmp/Motus
git rev-parse --short HEAD
git status --short
ls -lah | sed -n '1,120p'
```



Copy到rt目录下

```Bash
ROBOTWIN_ROOT="/root/autodl-tmp/RoboTwin"
MOTUS_REPO="/root/autodl-tmp/Motus"
TARGET="${ROBOTWIN_ROOT}/policy/Motus"

# 确保不要使用旧污染目录
rm -rf "${TARGET}"

cp -a "${MOTUS_REPO}/inference/robotwin/Motus" "${TARGET}"

cd "${TARGET}"
git -C "${MOTUS_REPO}" rev-parse --short HEAD > CLEAN_MOTUS_COMMIT.txt

ls -lah | sed -n '1,120p'
cat CLEAN_MOTUS_COMMIT.txt
```



配置文件；手写检查看看？

```Bash
cat > /root/autodl-tmp/RoboTwin/policy/Motus/paths_config.yml <<'EOF'
# Motus Evaluation Path Configuration

robotwin_root: "/root/autodl-tmp/RoboTwin"
conda_env: "/root/autodl-tmp/conda/envs/RoboTwin"
checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"

wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"

gpu_ids: [0]
task_config: "demo_randomized"
seed: 42
tasks_file: "tasks_all.txt"
EOF

cat /root/autodl-tmp/RoboTwin/policy/Motus/paths_config.yml
```



检查路径，模型，环境，tts opd

```Bash
cd /root/autodl-tmp/RoboTwin/policy/Motus

grep -nE "home/ubuntu|sumita|anaconda3|miniconda3|autodl-tmp|CONDA_ENV|bin/python|TASK_NAME|GPU_ID|conda.sh" eval.sh || true

ls -lh /root/autodl-tmp/models/motus/Motus_robotwin2/mp_rank_00_model_states.pt
ls -lh /root/autodl-tmp/models/motus/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
ls -lh /root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct/model.safetensors

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

cd /root/autodl-tmp/RoboTwin/policy/Motus

echo "===== grep tts/opd ====="
grep -RniE "tts|opd|no-tts|text.?to.?speech|speech" . | head -n 100 || true

echo
echo "===== eval.sh ====="
sed -n '1,220p' eval.sh

echo
echo "===== key path hardcode check ====="
grep -nE "home/ubuntu|sumita|anaconda3|miniconda3|autodl-tmp|CONDA_ENV|bin/python|TASK_NAME|GPU_ID|conda.sh" eval.sh || true
```



推理，先确认gpu空闲

```Bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /root/autodl-tmp/RoboTwin/policy/Motus

bash eval.sh beat_block_hammer 
```

实际还是click闹钟

中断，被venv影响。要新建终端。

成功跑通



### 4 motus并入推理



改代码，检查，等，下面的子目录并行进行



模型目录

```Bash
cd /root/autodl-tmp/RLinf

mkdir -p rlinf/models/embodiment/motus
```



#### Init文件

```Python
cat > rlinf/models/embodiment/motus/__init__.py <<'EOF'
from .motus_policy import MotusPolicy


def get_model(cfg, torch_dtype=None):
    return MotusPolicy(cfg, torch_dtype=torch_dtype)
EOF
```

修

```Bash
cd /root/autodl-tmp/RLinf
mkdir -p rlinf/models/embodiment/motus

cat > rlinf/models/embodiment/motus/__init__.py <<'EOF'
from .motus_policy import MotusPolicy


def get_model(cfg, torch_dtype=None):
    return MotusPolicy(cfg, torch_dtype=torch_dtype)
EOF
```

Review

```Bash
加 _build_motus() 和 registry：
register_model(SupportedModel.MOTUS.value, _build_motus, category="embodied")
这是对齐 RLinf 现有模型注册模式的。get_model(cfg) 会按 cfg.model_type 查 _MODEL_REGISTRY，然后根据 load_to_device 决定是否 .to(device)，再处理 LoRA。所以 model/motus.yaml 里保留：
is_lora: False
load_to_device: False
是必要的
```



#### 模型文件

占位：

```Python
cat > rlinf/models/embodiment/motus/motus_policy.py <<'EOF'
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy


class MotusPolicy(nn.Module, BasePolicy):
    """RLinf adapter for Motus.

    First target:
      - eval only
      - RoboTwin
      - batch size 1
      - return action chunk [B, 16, 14]
    """

    def __init__(self, cfg, torch_dtype=None):
        super().__init__()
        self.cfg = cfg
        self.torch_dtype = torch_dtype
        self.model_type = "motus"

        # Do not import Motus at module import time.
        # Heavy imports and checkpoint loading will be added after we confirm the
        # exact official Motus APIs in deploy_policy.py / models/motus.py.
        self._initialized = False

    def forward(self, *args, **kwargs):
        return self.default_forward(*args, **kwargs)

    def default_forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Motus PPO/GRPO training forward is not implemented yet. "
            "This adapter is eval-only for the first integration step."
        )

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict[str, Any], mode: str = "eval", **kwargs):
        raise NotImplementedError(
            "Motus eval adapter scaffold is created. "
            "Next step: wire official Motus model loading and env_obs conversion."
        )
EOF
```

继续写：

```Python
cat > rlinf/models/embodiment/motus/motus_policy.py <<'EOF'
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy


class MotusPolicy(nn.Module, BasePolicy):
    """RLinf eval-only adapter for official Motus RoboTwin policy.

    This adapter deliberately reuses the official Motus RoboTwin policy wrapper
    in /root/autodl-tmp/RoboTwin/policy/Motus/deploy_policy.py.

    First target:
      - eval only
      - RoboTwin
      - batch size 1
      - actions shape [B, 16, 14]
    """

    def __init__(self, cfg, torch_dtype=None):
        super().__init__()
        self.cfg = cfg
        self.torch_dtype = torch_dtype
        self.model_type = "motus"

        motus_cfg = cfg.get("motus", {})
        self.policy_path = Path(str(motus_cfg.get("policy_path"))).expanduser()
        self.checkpoint_path = str(motus_cfg.get("checkpoint_path", cfg.model_path))
        self.wan_path = str(motus_cfg.get("wan_path"))
        self.vlm_path = str(motus_cfg.get("vlm_path"))
        self.config_path = str(motus_cfg.get("config_path"))

        self.num_action_chunks = int(cfg.get("num_action_chunks", 16))
        self.action_dim = int(cfg.get("action_dim", 14))
        self.allow_batch_size = int(motus_cfg.get("allow_batch_size", 1))
        self.save_predicted_frames = bool(motus_cfg.get("save_predicted_frames", False))
        self.num_inference_timesteps = int(motus_cfg.get("num_inference_timesteps", 10))

        if not self.policy_path.exists():
            raise FileNotFoundError(f"Motus policy_path not found: {self.policy_path}")
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Motus checkpoint_path not found: {self.checkpoint_path}")
        if not Path(self.wan_path).exists():
            raise FileNotFoundError(f"Motus wan_path not found: {self.wan_path}")
        if not Path(self.vlm_path).exists():
            raise FileNotFoundError(f"Motus vlm_path not found: {self.vlm_path}")
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Motus config_path not found: {self.config_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        deploy_module = self._load_official_deploy_policy()
        OfficialMotusPolicy = getattr(deploy_module, "MotusPolicy")

        self._policy = OfficialMotusPolicy(
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
            wan_path=self.wan_path,
            vlm_path=self.vlm_path,
            device=self.device,
            log_dir=None,
            task_name="rlinf_eval",
        )

        # Official wrapper saves predicted frame grids by default. Disable for
        # RLinf eval unless explicitly requested.
        self._policy.save_images = self.save_predicted_frames

        # Make YAML-level inference-step override effective. The official wrapper
        # reads this value from config_dict inside get_action().
        try:
            self._policy.config_dict["model"]["inference"][
                "num_inference_timesteps"
            ] = self.num_inference_timesteps
        except Exception:
            pass

    def _load_official_deploy_policy(self):
        deploy_path = self.policy_path / "deploy_policy.py"
        if not deploy_path.exists():
            raise FileNotFoundError(f"deploy_policy.py not found: {deploy_path}")

        # deploy_policy.py imports top-level packages `models`, `utils`, `wan`.
        # Put official policy dir at the front so these resolve to clean Motus files.
        policy_str = str(self.policy_path.resolve())
        models_str = str((self.policy_path / "models").resolve())
        for p in [models_str, policy_str]:
            if p not in sys.path:
                sys.path.insert(0, p)

        module_name = "rlinf_external_motus_deploy_policy"
        spec = importlib.util.spec_from_file_location(module_name, str(deploy_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load official Motus deploy_policy.py from {deploy_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def forward(self, *args, **kwargs):
        return self.default_forward(*args, **kwargs)

    def default_forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Motus PPO/GRPO training forward is not implemented yet. "
            "This adapter is eval-only."
        )

    def eval(self):
        super().eval()
        if hasattr(self, "_policy") and hasattr(self._policy, "model"):
            self._policy.model.eval()
        return self

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict[str, Any], mode: str = "eval", **kwargs):
        batch_size = self._infer_batch_size(env_obs)
        if batch_size > self.allow_batch_size:
            raise ValueError(
                f"Motus adapter currently supports batch_size <= {self.allow_batch_size}, "
                f"got {batch_size}. Set env.eval.total_num_envs=1 for first integration."
            )

        actions = []
        for i in range(batch_size):
            observation = self._build_official_observation(env_obs, i)
            instruction = self._get_instruction(env_obs, i)

            # Keep official wrapper behavior, but do not let stale image cache grow.
            if hasattr(self._policy, "obs_cache"):
                self._policy.obs_cache.clear()

            self._policy.set_instruction(instruction)
            self._policy.update_obs(observation)
            action_i = self._policy.get_action()

            action_i = np.asarray(action_i, dtype=np.float32)
            if action_i.ndim != 2:
                raise ValueError(f"Motus action must be [T, D], got shape {action_i.shape}")
            if action_i.shape[0] != self.num_action_chunks:
                raise ValueError(
                    f"Motus action chunk mismatch: expected {self.num_action_chunks}, "
                    f"got {action_i.shape[0]}"
                )
            if action_i.shape[1] != self.action_dim:
                raise ValueError(
                    f"Motus action dim mismatch: expected {self.action_dim}, got {action_i.shape[1]}"
                )

            actions.append(torch.from_numpy(action_i))

        action_tensor = torch.stack(actions, dim=0).float().cpu().contiguous()
        return action_tensor, {"forward_inputs": {}}

    @staticmethod
    def _infer_batch_size(env_obs: dict[str, Any]) -> int:
        for key in ("states", "main_images", "task_descriptions"):
            value = env_obs.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
            if isinstance(value, np.ndarray):
                return int(value.shape[0])
            if isinstance(value, list):
                return len(value)
        raise ValueError(f"Cannot infer batch size from env_obs keys={list(env_obs.keys())}")

    @staticmethod
    def _take_batch(value: Any, idx: int) -> Any:
        if isinstance(value, torch.Tensor):
            return value[idx]
        if isinstance(value, np.ndarray):
            return value[idx]
        if isinstance(value, list):
            return value[idx]
        return value

    def _get_instruction(self, env_obs: dict[str, Any], idx: int) -> str:
        descriptions = env_obs.get("task_descriptions", None)
        if descriptions is None:
            return ""
        if isinstance(descriptions, str):
            return descriptions
        if isinstance(descriptions, list):
            return str(descriptions[idx])
        return str(self._take_batch(descriptions, idx))

    def _build_official_observation(self, env_obs: dict[str, Any], idx: int) -> dict[str, Any]:
        main = self._take_batch(env_obs["main_images"], idx)
        wrist = env_obs.get("wrist_images", None)
        state = self._take_batch(env_obs["states"], idx)

        head_img = self._to_hwc_uint8(main)
        left_img, right_img = self._extract_wrist_pair(wrist, idx)

        combined_image = self._combine_three_views(head_img, left_img, right_img)

        state_np = self._to_numpy(state).astype(np.float32).reshape(-1)
        if state_np.shape[0] != self.action_dim:
            raise ValueError(
                f"Motus state dim mismatch: expected {self.action_dim}, got {state_np.shape[0]}"
            )

        # Official deploy_policy.update_obs accepts {'image': ..., 'joint_action': {'vector': ...}}.
        return {
            "image": combined_image,
            "joint_action": {
                "vector": state_np,
            },
        }

    def _extract_wrist_pair(self, wrist_images: Any, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if wrist_images is None:
            raise ValueError(
                "Motus requires wrist_images. Ensure collect_wrist_camera=true in env config."
            )

        wrist_i = self._take_batch(wrist_images, idx)

        if isinstance(wrist_i, (list, tuple)):
            if len(wrist_i) == 0:
                raise ValueError("wrist_images list is empty")
            left = wrist_i[0]
            right = wrist_i[1] if len(wrist_i) > 1 else wrist_i[0]
            return self._to_hwc_uint8(left), self._to_hwc_uint8(right)

        arr = self._to_numpy(wrist_i)

        # Expected after batch indexing: [N, H, W, C] or [N, C, H, W].
        if arr.ndim == 4:
            left = arr[0]
            right = arr[1] if arr.shape[0] > 1 else arr[0]
            return self._to_hwc_uint8(left), self._to_hwc_uint8(right)

        # Single wrist image; duplicate as fallback.
        if arr.ndim == 3:
            img = self._to_hwc_uint8(arr)
            return img, img

        raise ValueError(f"Unsupported wrist_images shape after batch indexing: {arr.shape}")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)

    @classmethod
    def _to_hwc_uint8(cls, value: Any) -> np.ndarray:
        arr = cls._to_numpy(value)

        if arr.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {arr.shape}")

        # CHW -> HWC
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        if arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB image, got shape {arr.shape}")

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(arr)

    @staticmethod
    def _resize_hwc_uint8(img: np.ndarray, width: int, height: int) -> np.ndarray:
        pil = Image.fromarray(img, mode="RGB")
        pil = pil.resize((width, height), Image.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)

    @classmethod
    def _combine_three_views(
        cls, head_img: np.ndarray, left_img: np.ndarray, right_img: np.ndarray
    ) -> np.ndarray:
        # Match official Motus RoboTwin layout:
        #   head: 320x240
        #   left/right wrists: each 160x120
        #   final: [head; concat(left,right)] = 320x360 HWC
        head = cls._resize_hwc_uint8(head_img, width=320, height=240)
        left = cls._resize_hwc_uint8(left_img, width=160, height=120)
        right = cls._resize_hwc_uint8(right_img, width=160, height=120)
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)
        return np.ascontiguousarray(image)
EOF
```

查

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py

grep -nE "class MotusPolicy|def __init__|def predict_action_batch|def _build_official_observation|def _combine_three_views|EOF|llow_batch" \
  rlinf/models/embodiment/motus/motus_policy.py \
  | sed -n '1,160p'
```

Review

```Bash
复用官方 Motus deploy_policy.py 的 wrapper，而不是重写 Motus 加载逻辑。官方 wrapper 的接口正好是：
MotusPolicy(checkpoint_path, config_path, wan_path, vlm_path, device, ...)
set_instruction(...)
update_obs(...)
get_action()
其中 update_obs() 支持：
{"image": image, "joint_action": {"vector": state}}
get_action() 内部会调用 T5 encoder、VLM processor、model.inference_step()，最后返回 numpy actions。你的 adapter 正是把 RLinf obs 转成这个格式
```



#### 模型yaml

```YAML
cat > examples/embodiment/config/model/motus.yaml <<'EOF'
model_type: "motus"

# External Motus paths
model_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
precision: "bf16"

# RoboTwin action interface
num_action_chunks: 16
action_dim: 14
use_proprio: True
add_value_head: False

motus:
  # Keep the official clean Motus repo outside RLinf.
  repo_path: "/root/autodl-tmp/Motus"

  # Standalone policy directory copied from Motus/inference/robotwin/Motus.
  # Useful because it contains deploy_policy.py, utils/, and copied model files.
  policy_path: "/root/autodl-tmp/RoboTwin/policy/Motus"

  checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
  wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
  vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"

  # Official Motus robotwin config copied with policy/Motus.
  config_path: "/root/autodl-tmp/RoboTwin/policy/Motus/utils/robotwin.yml"

  num_inference_timesteps: 10
  use_t5_cache: True
  save_predicted_frames: False

  # First integration is deliberately single-env.
  allow_batch_size: 1
EOF
```

修

```YAML
cat > examples/embodiment/config/model/motus.yaml <<'EOF'
model_type: "motus"

# External Motus checkpoint path.
model_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
precision: "bf16"

# RLinf generic model flags.
is_lora: False
lora_rank: 32
load_to_device: False

# RoboTwin action interface.
num_action_chunks: 16
action_dim: 14
use_proprio: True
add_value_head: False

motus:
  # Clean official Motus repo.
  repo_path: "/root/autodl-tmp/Motus"

  # Clean standalone RoboTwin policy copied from Motus/inference/robotwin/Motus.
  policy_path: "/root/autodl-tmp/RoboTwin/policy/Motus"

  checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
  wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
  vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"
  config_path: "/root/autodl-tmp/RoboTwin/policy/Motus/utils/robotwin.yml"

  num_inference_timesteps: 10
  use_t5_cache: True
  save_predicted_frames: False

  # First integration is single-env to avoid batching ambiguity.
  allow_batch_size: 1
EOF
```

重写

```YAML
cd /root/autodl-tmp/RLinf

cat > examples/embodiment/config/model/motus.yaml <<'EOF'
model_type: "motus"

# External Motus checkpoint path.
model_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
precision: "bf16"

# RLinf generic model flags.
is_lora: False
lora_rank: 32
load_to_device: False

# RoboTwin action interface.
num_action_chunks: 16
action_dim: 14
use_proprio: True
add_value_head: False

motus:
  repo_path: "/root/autodl-tmp/Motus"
  policy_path: "/root/autodl-tmp/RoboTwin/policy/Motus"

  checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
  wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
  vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"
  config_path: "/root/autodl-tmp/RoboTwin/policy/Motus/utils/robotwin.yml"

  num_inference_timesteps: 10
  use_t5_cache: True
  save_predicted_frames: False

  # First integration is deliberately single-env.
  allow_batch_size: 1
EOF

cat examples/embodiment/config/model/motus.yaml
```



#### 外部yaml

复制

```Bash
cp evaluations/robotwin/robotwin_adjust_bottle_openpi_eval_autodl.yaml \
   evaluations/robotwin/robotwin_adjust_bottle_motus_eval_autodl.yaml
```

改配置

```Bash
defaults:
  - model/motus@rollout.model

cluster.component_placement.env, rollout
0

runner.logger.experiment_name
"robotwin_motus_eval_autodl"

env.eval.rollout_epoch
5

env.eval.total_num_envs
1

env.eval.max_episode_steps
env.eval.max_steps_per_rollout_epoch
192

rollout.model删除
```

修

```YAML
cd /root/autodl-tmp/RLinf

cat > evaluations/robotwin/robotwin_adjust_bottle_motus_eval_autodl.yaml <<'EOF'
defaults:
  - env/robotwin_adjust_bottle@env.eval
  - model/motus@rollout.model
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    env, rollout: 0

runner:
  task_type: embodied_eval
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_motus_eval_autodl"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: -1

  only_eval: True
  val_check_interval: -1
  save_interval: -1

  resume_dir: null
  ckpt_path: null

env:
  group_name: "EnvGroup"
  enable_offload: True

  eval:
    rollout_epoch: 1
    total_num_envs: 1
    auto_reset: True
    ignore_terminations: True

    # Motus action chunk is 16, so use a multiple of 16.
    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192

    reward_coef: 1.0
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/eval

    center_crop: False
    task_config:
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
EOF
```

换常规配置

```YAML
cd /root/autodl-tmp/RLinf

cat > evaluations/robotwin/robotwin_adjust_bottle_motus_eval_autodl.yaml <<'EOF'
defaults:
  - env/robotwin_adjust_bottle@env.eval
  - model/motus@rollout.model
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    env, rollout: 0

runner:
  task_type: embodied_eval
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_motus_eval_autodl"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: -1

  only_eval: True
  val_check_interval: -1
  save_interval: -1

  resume_dir: null
  ckpt_path: null

env:
  group_name: "EnvGroup"
  enable_offload: True

  eval:
    rollout_epoch: 5
    total_num_envs: 1
    auto_reset: True
    ignore_terminations: True

    # Official Motus/RoboTwin standalone commonly runs up to 400 steps.
    # 400 is divisible by Motus chunk size 16.
    max_episode_steps: 400
    max_steps_per_rollout_epoch: 400

    reward_coef: 1.0
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json

    video_cfg:
      save_video: True
      info_on_video: True
      video_base_dir: ${runner.logger.log_path}/video/eval

    center_crop: False
    task_config:
      step_lim: 400
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
EOF
```



#### 注册

注册，rlinf/config\.py

```Python
python - <<'PY'
from pathlib import Path

p = Path("rlinf/config.py")
s = p.read_text()

needle = 'SupportedModel.OPENPI = SupportedModel.register("openpi", force=True)\n'
insert = needle + 'SupportedModel.MOTUS = SupportedModel.register("motus", force=True)\n'
if 'SupportedModel.MOTUS = SupportedModel.register("motus", force=True)' not in s:
    if needle not in s:
        raise SystemExit("Could not find OPENPI register line")
    s = s.replace(needle, insert)

needle2 = "        SupportedModel.OPENPI,\n"
insert2 = needle2 + "        SupportedModel.MOTUS,\n"
if "        SupportedModel.MOTUS,\n" not in s:
    if needle2 not in s:
        raise SystemExit("Could not find OPENPI in EMBODIED_MODEL")
    s = s.replace(needle2, insert2, 1)

p.write_text(s)
print("patched", p)
PY

grep -nE "MOTUS|OPENPI|EMBODIED_MODEL" rlinf/config.py | sed -n '1,80p'
```

Review

```Bash
加：
SupportedModel.MOTUS = SupportedModel.register("motus", force=True)
并加入：
EMBODIED_MODEL
```



注册 model builder，rlinf/models/\_init\_\.py

```Python
python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/__init__.py")
s = p.read_text()

# 1. Add builder function after _build_openpi.
needle = '''    def _build_openpi(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openpi import get_model

        return get_model(cfg, torch_dtype)

'''
insert = needle + '''    def _build_motus(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.motus import get_model

        return get_model(cfg, torch_dtype)

'''
if "def _build_motus" not in s:
    if needle not in s:
        raise SystemExit("Could not find _build_openpi block")
    s = s.replace(needle, insert)

# 2. Register after OPENPI registration block.
needle2 = '''    register_model(
        SupportedModel.OPENPI.value,
        _build_openpi,
        category="embodied",
        force=True,
    )
'''
insert2 = needle2 + '''    register_model(
        SupportedModel.MOTUS.value,
        _build_motus,
        category="embodied",
        force=True,
    )
'''
if "SupportedModel.MOTUS.value" not in s:
    if needle2 not in s:
        raise SystemExit("Could not find OPENPI registration block")
    s = s.replace(needle2, insert2)

p.write_text(s)
print("patched", p)
PY

grep -nE "MOTUS|_build_motus|_build_openpi|register_model" rlinf/models/__init__.py | sed -n '1,140p'
```



注册，RolloutWorker ，rlinf/workers/rollout/hf/huggingface\_worker\.py

```Python
python - <<'PY'
from pathlib import Path

p = Path("rlinf/workers/rollout/hf/huggingface_worker.py")
s = p.read_text()

needle = "            SupportedModel.OPENPI,\n"
insert = needle + "            SupportedModel.MOTUS,\n"

if "            SupportedModel.MOTUS,\n" not in s:
    if needle not in s:
        raise SystemExit("Could not find OPENPI in predict model list")
    s = s.replace(needle, insert, 1)

p.write_text(s)
print("patched", p)
PY

sed -n '270,325p' rlinf/workers/rollout/hf/huggingface_worker.py
```

Review

```Bash
把 SupportedModel.MOTUS 加进 predict() 的 mode 分支：
SupportedModel.OPENPI,
SupportedModel.MOTUS,
...
这是对的。RLinf eval 路径就是：
EnvWorker 发 obs
RolloutWorker.predict(..., mode="eval")
hf_model.predict_action_batch(...)
send_chunk_actions(...)
EnvWorker chunk_step(...)
所以 Motus adapter 只需要返回 action chunk
```



#### 检查

查编译

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

python -m py_compile \
  rlinf/models/embodiment/motus/__init__.py \
  rlinf/models/embodiment/motus/motus_policy.py \
  rlinf/config.py \
  rlinf/models/__init__.py \
  rlinf/workers/rollout/hf/huggingface_worker.py

python - <<'PY'
from rlinf.config import SupportedModel, EMBODIED_MODEL
print("MOTUS =", SupportedModel.MOTUS)
print("motus in embodied =", SupportedModel.MOTUS in EMBODIED_MODEL)

from rlinf.models.embodiment.motus import MotusPolicy, get_model
print("MotusPolicy =", MotusPolicy)
print("get_model =", get_model)
PY
```

查配置

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

python evaluations/eval_embodied_agent.py \
  --config-path evaluations/robotwin/ \
  --config-name robotwin_adjust_bottle_motus_eval_autodl \
  --cfg job \
  | grep -nE "model_type|model_path|num_action_chunks|action_dim|policy_path|checkpoint_path|wan_path|vlm_path|total_num_envs|max_steps_per_rollout_epoch|component_placement|enable_offload" \
  | sed -n '1,160p'
```



查参数

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

python /root/autodl-tmp/RLinf/evaluations/eval_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/evaluations/robotwin/ \
  --config-name robotwin_adjust_bottle_motus_eval_autodl \
  --cfg job \
  | grep -nE "model_type|model_path|num_action_chunks|action_dim|policy_path|checkpoint_path|wan_path|vlm_path|total_num_envs|max_steps_per_rollout_epoch|component_placement|enable_offload|rollout_epoch|save_video" \
  | sed -n '1,200p'
```



查注册

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

python - <<'PY'
from rlinf.config import SupportedModel, EMBODIED_MODEL
from rlinf.models import _MODEL_REGISTRY

print("MOTUS =", SupportedModel.MOTUS)
print("motus in embodied =", SupportedModel.MOTUS in EMBODIED_MODEL)
print("motus registered =", "motus" in _MODEL_REGISTRY)
print("builder =", _MODEL_REGISTRY.get("motus"))
PY
```

查worker注册

```Bash
grep -nA20 -B5 "SupportedModel.OPENPI" rlinf/workers/rollout/hf/huggingface_worker.py | sed -n '1,80p'
```



查模型加载

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export PYTHONPATH="/root/autodl-tmp/RoboTwin/policy/Motus:/root/autodl-tmp/RoboTwin/policy/Motus/models:/root/autodl-tmp/Motus:${PYTHONPATH}"

python - <<'PY'
from omegaconf import OmegaConf
from rlinf.models import get_model

cfg = OmegaConf.load("examples/embodiment/config/model/motus.yaml")
print(OmegaConf.to_yaml(cfg))

model = get_model(cfg)
print("model =", type(model))
print("device =", getattr(model, "device", None))
print("num_action_chunks =", getattr(model, "num_action_chunks", None))
print("action_dim =", getattr(model, "action_dim", None))
PY
```



补完ftfy，查包

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export PYTHONPATH="/root/autodl-tmp/RoboTwin/policy/Motus:/root/autodl-tmp/RoboTwin/policy/Motus/models:/root/autodl-tmp/Motus:${PYTHONPATH}"

python - <<'PY'
import sys
print("python =", sys.executable)

mods = [
    "torch",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "diffusers",
    "safetensors",
    "deepspeed",
    "flash_attn",
    "einops",
    "cv2",
    "ftfy",
    "sentencepiece",
]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""), getattr(mod, "__file__", ""))
    except Exception as e:
        print(m, "FAIL", repr(e))

from transformers import Qwen3VLForConditionalGeneration
print("Qwen3VL OK")

from models.motus import Motus, MotusConfig
print("Motus import OK")
PY
```

查模型加载

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export PYTHONPATH="/root/autodl-tmp/RoboTwin/policy/Motus:/root/autodl-tmp/RoboTwin/policy/Motus/models:/root/autodl-tmp/Motus:${PYTHONPATH}"

python - <<'PY'
from omegaconf import OmegaConf
from rlinf.models import get_model

cfg = OmegaConf.load("examples/embodiment/config/model/motus.yaml")
print(OmegaConf.to_yaml(cfg))

model = get_model(cfg)
print("model =", type(model))
print("device =", getattr(model, "device", None))
print("num_action_chunks =", getattr(model, "num_action_chunks", None))
print("action_dim =", getattr(model, "action_dim", None))
PY
```



#### 环境

备份venv

```Bash
cd /root/autodl-tmp/RLinf

du -sh .venv
df -h /root/autodl-tmp

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP=".venv_backup_pi0_ppo_grpo_ok_${STAMP}"

echo "BACKUP=${BACKUP}"

.venv/bin/python -m pip freeze > "pip_freeze_before_motus_${STAMP}.txt"
.venv/bin/python -m pip list > "pip_list_before_motus_${STAMP}.txt"
git rev-parse HEAD > "git_head_before_motus_${STAMP}.txt"
git status --short > "git_status_before_motus_${STAMP}.txt"

rsync -aH --info=progress2 .venv/ "${BACKUP}/"

du -sh "${BACKUP}"
```



查环境缺啥包

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export PYTHONPATH="/root/autodl-tmp/Motus:/root/autodl-tmp/RoboTwin/policy/Motus:${PYTHONPATH}"

python - <<'PY'
import sys
print("python =", sys.executable)

mods = [
    "torch",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "diffusers",
    "safetensors",
    "deepspeed",
    "flash_attn",
    "einops",
    "cv2",
]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""), getattr(mod, "__file__", ""))
    except Exception as e:
        print(m, "FAIL", repr(e))

try:
    from transformers import Qwen3VLForConditionalGeneration
    print("Qwen3VL OK")
except Exception as e:
    print("Qwen3VL FAIL", repr(e))

try:
    from models.motus import Motus, MotusConfig
    print("Motus import OK")
except Exception as e:
    print("Motus import FAIL", repr(e))
PY
```

补缺

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

export DS_BUILD_OPS=0

uv pip install \
  "deepspeed" \
  "transformers==4.57.1" \
  "tokenizers<0.23" \
  "huggingface-hub>=0.34.0,<1.0" \
  "safetensors>=0.8.0rc0"

cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export PYTHONPATH="/root/autodl-tmp/RoboTwin/policy/Motus:/root/autodl-tmp/RoboTwin/policy/Motus/models:/root/autodl-tmp/Motus:${PYTHONPATH}"

python - <<'PY'
import sys
print("python =", sys.executable)

mods = [
    "torch",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "diffusers",
    "safetensors",
    "deepspeed",
    "flash_attn",
    "einops",
    "cv2",
]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""), getattr(mod, "__file__", ""))
    except Exception as e:
        print(m, "FAIL", repr(e))

from transformers import Qwen3VLForConditionalGeneration
print("Qwen3VL OK")

from models.motus import Motus, MotusConfig
print("Motus import OK")
PY
```

缺包

```Bash
ModuleNotFoundError: No module named 'ftfy'
```

装

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

uv pip install \
  ftfy \
  regex \
  sentencepiece
```



#### 跑推理

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_motus_eval_autodl
```

再跑

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

ray stop -f || python -m ray stop -f || true
nvidia-smi

source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_motus_eval_autodl
```



5次成功3次，成功接入



当前的代码：

\[RLinf\_code\_only\_20260618\_155229\.tgz\]



#### 打包指令

```Bash
cd /root/autodl-tmp

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="/root/autodl-tmp/RLinf_code_only_${STAMP}.tgz"

tar -czf "${OUT}" \
  --exclude='RLinf/.venv' \
  --exclude='RLinf/.venv_*' \
  --exclude='RLinf/logs' \
  --exclude='RLinf/results' \
  --exclude='RLinf/.git' \
  --exclude='RLinf/__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.safetensors' \
  --exclude='*.ckpt' \
  RLinf

ls -lh "${OUT}"
```



#### batch推理



https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a321cdb\-fbc0\-83ea\-9f6b\-3135b9a9f8ed



改 `motus_policy.py`

```C++
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

# Add robust bool parser if missing.
if "def _as_bool(" not in s:
    marker = "from rlinf.models.embodiment.base_policy import BasePolicy\n"
    insert = marker + r'''

def _as_bool(x, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}

'''
    if marker not in s:
        raise SystemExit("Cannot find BasePolicy import marker")
    s = s.replace(marker, insert)

# Add batch config fields after num_inference_timesteps.
old = '        self.num_inference_timesteps = int(motus_cfg.get("num_inference_timesteps", 10))\n'
new = old + '''        self.batch_inference = _as_bool(motus_cfg.get("batch_inference", False), default=False)
        self.decode_video = _as_bool(motus_cfg.get("decode_video", False), default=False)
        self.scene_prefix = str(
            motus_cfg.get(
                "scene_prefix",
                "The whole scene is in a realistic, industrial art style with three views: "
                "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                "The aloha robot is currently performing the following task: ",
            )
        )
'''
if "self.batch_inference =" not in s:
    if old not in s:
        raise SystemExit("Cannot find num_inference_timesteps marker")
    s = s.replace(old, new)

start = s.index("    @torch.no_grad()\n    def predict_action_batch")
end = s.index("    @staticmethod\n    def _infer_batch_size", start)

replacement = r'''    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict[str, Any], mode: str = "eval", **kwargs):
        batch_size = self._infer_batch_size(env_obs)
        if batch_size > self.allow_batch_size:
            raise ValueError(
                f"Motus adapter supports batch_size <= {self.allow_batch_size}, "
                f"got {batch_size}. Increase rollout.model.motus.allow_batch_size "
                "or reduce env count per rollout worker."
            )

        if self.batch_inference and batch_size > 1:
            return self._predict_action_batch_vectorized(env_obs, batch_size)

        return self._predict_action_batch_loop(env_obs, batch_size)

    def _predict_action_batch_loop(
        self, env_obs: dict[str, Any], batch_size: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Fallback path: run official singleton wrapper once per sample."""
        actions = []

        for i in range(batch_size):
            observation = self._build_official_observation(env_obs, i)
            instruction = self._get_instruction(env_obs, i)

            self._reset_official_policy_transient_state()
            self._policy.set_instruction(instruction)
            self._policy.update_obs(observation)
            action_i = self._policy.get_action()

            action_i = self._validate_action_array(action_i, sample_idx=i)
            actions.append(torch.from_numpy(action_i))

        action_tensor = torch.stack(actions, dim=0).float().cpu().contiguous()
        return action_tensor, {"forward_inputs": {}}

    def _predict_action_batch_vectorized(
        self, env_obs: dict[str, Any], batch_size: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """True RLinf batch path: B different env observations -> one Motus inference.

        This is not TTS candidate batching. It does not repeat one observation.
        It stacks B different frames/states/instructions from RLinf env_obs.
        """
        frames: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        t5_list: list[torch.Tensor] = []
        vlm_inputs_list: list[dict[str, torch.Tensor]] = []
        instructions: list[str] = []

        for i in range(batch_size):
            frame_i, state_i, t5_i, vlm_i, instruction_i = self._prepare_one_for_vectorized(
                env_obs, i
            )
            frames.append(frame_i)
            states.append(state_i)
            t5_list.append(t5_i)
            vlm_inputs_list.append(vlm_i)
            instructions.append(instruction_i)

        first_frame = torch.stack(frames, dim=0).to(self.device)
        state = torch.stack(states, dim=0).to(self.device)

        self._ensure_motus_grid_batch_size(batch_size)

        predicted_frames, predicted_actions = self._policy.model.inference_step(
            first_frame=first_frame,
            state=state,
            num_inference_steps=self.num_inference_timesteps,
            language_embeddings=t5_list,
            vlm_inputs=vlm_inputs_list,
            decode_video=self.decode_video,
        )

        if predicted_actions is None:
            raise RuntimeError("Motus inference_step returned predicted_actions=None")

        actions_np = predicted_actions.detach().float().cpu().numpy()
        if actions_np.ndim != 3:
            raise ValueError(
                f"Motus batched actions must be [B,T,D], got shape {actions_np.shape}"
            )
        if actions_np.shape[0] != batch_size:
            raise ValueError(
                f"Motus batch mismatch: expected B={batch_size}, got {actions_np.shape[0]}"
            )

        actions_np = actions_np[:, : self.num_action_chunks, : self.action_dim]

        if actions_np.shape[1] != self.num_action_chunks:
            raise ValueError(
                f"Motus action chunk mismatch: expected {self.num_action_chunks}, "
                f"got {actions_np.shape[1]}"
            )
        if actions_np.shape[2] != self.action_dim:
            raise ValueError(
                f"Motus action dim mismatch: expected {self.action_dim}, "
                f"got {actions_np.shape[2]}"
            )

        action_tensor = torch.from_numpy(actions_np).float().cpu().contiguous()
        return action_tensor, {"forward_inputs": {}}

    def _prepare_one_for_vectorized(
        self, env_obs: dict[str, Any], idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], str]:
        """Use official Motus preprocessing for one sample, then return tensors for batch."""
        observation = self._build_official_observation(env_obs, idx)
        raw_instruction = self._get_instruction(env_obs, idx)
        full_instruction = self.scene_prefix + raw_instruction

        # Reuse official update_obs() so image resize/padding and state handling match
        # official Motus RoboTwin inference.
        self._reset_official_policy_transient_state()
        self._policy.update_obs(observation)

        frame = self._policy.obs_cache[-1].squeeze(0).detach()  # [C,H,W]
        state = self._policy.current_state.squeeze(0).detach()  # [14]

        t5_out = self._policy.t5_encoder([full_instruction], self.device)
        t5_emb = self._normalize_t5_output(t5_out)

        first_frame_pil = self._policy._tensor_to_pil_image(frame.detach().cpu())
        vlm_inputs = self._policy._preprocess_vlm_messages(full_instruction, first_frame_pil)

        return frame, state, t5_emb, vlm_inputs, raw_instruction

    @staticmethod
    def _normalize_t5_output(t5_out: Any) -> torch.Tensor:
        if torch.is_tensor(t5_out):
            if t5_out.dim() == 3 and t5_out.shape[0] == 1:
                return t5_out.squeeze(0)
            return t5_out
        if isinstance(t5_out, list):
            if len(t5_out) != 1:
                raise ValueError(f"Expected single T5 embedding, got list length {len(t5_out)}")
            emb = t5_out[0]
            if torch.is_tensor(emb) and emb.dim() == 3 and emb.shape[0] == 1:
                return emb.squeeze(0)
            return emb
        raise ValueError(f"Unexpected T5 encoder output type: {type(t5_out)!r}")

    def _ensure_motus_grid_batch_size(self, batch_size: int) -> None:
        """Keep Motus/WAN grid_sizes aligned with runtime batch size.

        Old TTS experiments showed Motus inference may use runtime B different
        from config.batch_size. WAN attention/unpatchify expects grid_sizes[0] == B.
        """
        model = self._policy.model
        grid_sizes = getattr(model, "grid_sizes", None)

        if (
            torch.is_tensor(grid_sizes)
            and grid_sizes.shape[0] == batch_size
            and getattr(model, "video_module", None) is not None
        ):
            return

        lat_t = 1 + model.config.num_video_frames // 4
        lat_h = model.config.video_height // 32
        lat_w = model.config.video_width // 32

        device = getattr(model, "device", self.device)
        new_grid = torch.tensor(
            [lat_t, lat_h, lat_w],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)

        model.grid_sizes = new_grid
        if hasattr(model, "video_module"):
            model.video_module.grid_sizes = new_grid

    def _reset_official_policy_transient_state(self) -> None:
        if hasattr(self._policy, "obs_cache"):
            self._policy.obs_cache.clear()
        if hasattr(self._policy, "action_cache"):
            self._policy.action_cache.clear()
        if hasattr(self._policy, "current_state"):
            self._policy.current_state = None
        if hasattr(self._policy, "current_state_norm"):
            self._policy.current_state_norm = None
        if hasattr(self._policy, "prev_action"):
            self._policy.prev_action = None

    def _validate_action_array(self, action: Any, sample_idx: int = 0) -> np.ndarray:
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim != 2:
            raise ValueError(
                f"Motus action for sample {sample_idx} must be [T,D], got shape {action_np.shape}"
            )
        if action_np.shape[0] != self.num_action_chunks:
            raise ValueError(
                f"Motus action chunk mismatch for sample {sample_idx}: "
                f"expected {self.num_action_chunks}, got {action_np.shape[0]}"
            )
        if action_np.shape[1] != self.action_dim:
            raise ValueError(
                f"Motus action dim mismatch for sample {sample_idx}: "
                f"expected {self.action_dim}, got {action_np.shape[1]}"
            )
        return action_np

'''
s = s[:start] + replacement + s[end:]

p.write_text(s)
print("patched", p)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py

grep -nE "def predict_action_batch|def _predict_action_batch_vectorized|def _prepare_one_for_vectorized|def _ensure_motus_grid_batch_size|batch_inference|decode_video|allow_batch_size" \
  rlinf/models/embodiment/motus/motus_policy.py \
  | sed -n '1,200p'
```

log打印

```Python
cd /root/autodl-tmp/RLinf

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

needle = '''        predicted_frames, predicted_actions = self._policy.model.inference_step(
            first_frame=first_frame,
            state=state,
            num_inference_steps=self.num_inference_timesteps,
            language_embeddings=t5_list,
            vlm_inputs=vlm_inputs_list,
            decode_video=self.decode_video,
        )
'''
insert = '''        print(
            f"[RLinf-Motus] vectorized batch inference: "
            f"B={batch_size}, first_frame={tuple(first_frame.shape)}, "
            f"state={tuple(state.shape)}, t5={len(t5_list)}, vlm={len(vlm_inputs_list)}, "
            f"decode_video={self.decode_video}",
            flush=True,
        )

''' + needle

if "[RLinf-Motus] vectorized batch inference" not in s:
    if needle not in s:
        raise SystemExit("Cannot find inference_step call marker")
    s = s.replace(needle, insert)

p.write_text(s)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py
```



改 `model/motus.yaml`

```Bash
cd /root/autodl-tmp/RLinf

python - <<'PY'
from pathlib import Path

p = Path("examples/embodiment/config/model/motus.yaml")
s = p.read_text()

# Replace existing allow_batch_size line and add batch_inference/decode_video if missing.
s = s.replace("  allow_batch_size: 1\n", "  allow_batch_size: 8\n")

if "  batch_inference:" not in s:
    s = s.replace("  allow_batch_size: 8\n", "  allow_batch_size: 8\n  batch_inference: True\n  decode_video: False\n")

p.write_text(s)
PY

cat examples/embodiment/config/model/motus.yaml
```



创建配置

```YAML
cd /root/autodl-tmp/RLinf

cat > evaluations/robotwin/robotwin_adjust_bottle_motus_eval_batch8_autodl.yaml <<'EOF'
defaults:
  - env/robotwin_adjust_bottle@env.eval
  - model/motus@rollout.model
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    env, rollout: 0

runner:
  task_type: embodied_eval
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_motus_eval_batch8_autodl"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: -1

  only_eval: True
  val_check_interval: -1
  save_interval: -1

  resume_dir: null
  ckpt_path: null

env:
  group_name: "EnvGroup"
  enable_offload: True

  eval:
    rollout_epoch: 1
    total_num_envs: 8
    auto_reset: True
    ignore_terminations: True

    max_episode_steps: 400
    max_steps_per_rollout_epoch: 400

    reward_coef: 1.0
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/eval

    center_crop: False
    task_config:
      step_lim: 400
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
EOF
```

查配置

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

python /root/autodl-tmp/RLinf/evaluations/eval_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/evaluations/robotwin/ \
  --config-name robotwin_adjust_bottle_motus_eval_batch8_autodl \
  --cfg job \
  | grep -nE "model_type|num_action_chunks|action_dim|allow_batch_size|batch_inference|decode_video|total_num_envs|max_episode_steps|max_steps_per_rollout_epoch|step_lim|save_video|rollout_epoch|component_placement" \
  | sed -n '1,240p'
```



代码

\[RLinf\_code\_only\_20260618\_174050\.tgz\]



推理

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_motus_eval_batch8_autodl
```

报错

```Bash
**失败点是 Motus clean 版本的 Motus.inference_step() 接口不接受 decode_video 参数。**
日志里这行很关键：
[RLinf-Motus] vectorized batch inference: B=8, first_frame=(8, 3, 384, 320), state=(8, 14), t5=8, vlm=8, decode_video=False
这说明：
RLinf EnvWorker -> RolloutWorker 的 batch merge 是成功的
Motus adapter 没有复制同一个 obs
当前 batch 内是 B=8
输入 shape 也对：
  first_frame: [8, 3, 384, 320]
  state:       [8, 14]
  t5:          8 条
  vlm:         8 条
真正炸的是：
TypeError: Motus.inference_step() got an unexpected keyword argument 'decode_video'
```

修

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

# 1. Add inspect import.
if "import inspect\n" not in s:
    s = s.replace("import importlib.util\n", "import importlib.util\nimport inspect\n")

# 2. Add one-time warning flag after decode_video config.
needle = "        self.decode_video = _as_bool(motus_cfg.get(\"decode_video\", False), default=False)\n"
insert = needle + "        self._warned_decode_video_unsupported = False\n"
if "self._warned_decode_video_unsupported" not in s:
    if needle not in s:
        raise SystemExit("Cannot find decode_video config line")
    s = s.replace(needle, insert)

# 3. Replace direct inference_step call with signature-compatible call.
old = '''        predicted_frames, predicted_actions = self._policy.model.inference_step(
            first_frame=first_frame,
            state=state,
            num_inference_steps=self.num_inference_timesteps,
            language_embeddings=t5_list,
            vlm_inputs=vlm_inputs_list,
            decode_video=self.decode_video,
        )

'''
new = '''        call_kwargs = {
            "first_frame": first_frame,
            "state": state,
            "num_inference_steps": self.num_inference_timesteps,
            "language_embeddings": t5_list,
            "vlm_inputs": vlm_inputs_list,
        }

        sig = inspect.signature(self._policy.model.inference_step)
        if "decode_video" in sig.parameters:
            call_kwargs["decode_video"] = self.decode_video
        elif self.decode_video is False and not self._warned_decode_video_unsupported:
            print(
                "[RLinf-Motus] underlying Motus.inference_step does not support "
                "decode_video; calling without it, so predicted video may still be decoded "
                "inside clean Motus if its implementation always decodes.",
                flush=True,
            )
            self._warned_decode_video_unsupported = True

        out = self._policy.model.inference_step(**call_kwargs)

        if not isinstance(out, tuple) or len(out) < 2:
            raise RuntimeError(
                f"Motus inference_step must return at least "
                f"(predicted_frames, predicted_actions), got type={type(out)!r}"
            )

        predicted_frames, predicted_actions = out[0], out[1]

'''
if old not in s:
    raise SystemExit("Cannot find direct inference_step call block")
s = s.replace(old, new)

p.write_text(s)
print("patched", p)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py

grep -nE "import inspect|decode_video|signature|call_kwargs|underlying Motus" \
  rlinf/models/embodiment/motus/motus_policy.py \
  | sed -n '1,160p'

cd /root/autodl-tmp/RLinf
source .venv/bin/activate

ray stop -f || python -m ray stop -f || true
nvidia-smi
```



再跑

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_motus_eval_batch8_autodl \
  env.eval.total_num_envs=2 \
  rollout.model.motus.allow_batch_size=2 \
  runner.logger.experiment_name=robotwin_motus_eval_batch2_autodl
```



应该成功



\[RLinf\_code\_only\_20260618\_181020\.tgz\]



### 5 motus并入训练



#### 改

内yaml

```Python
cd /root/autodl-tmp/RLinf

python - <<'PY'
from pathlib import Path

p = Path("examples/embodiment/config/model/motus.yaml")
s = p.read_text()

if "  trainable:" not in s:
    s += '''

  # RLinf training integration.
  trainable: "action_expert"
  freeze_video_model: True
  freeze_vlm_model: True
  freeze_und_expert: True
  freeze_t5_encoder: True

  # First PPO/GRPO logprob implementation will be transition-level πRL.
  logprob_mode: "transition_velocity"
  logprob_sigma: 1.0
  collect_denoise_step: "random"
'''

p.write_text(s)
print(p.read_text())
PY
```

init注册

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

# Add train config fields after scene_prefix block end.
needle = '''        if not self.policy_path.exists():
'''
insert = '''        self.trainable = str(motus_cfg.get("trainable", "none")).lower()
        self.freeze_video_model = _as_bool(motus_cfg.get("freeze_video_model", True), default=True)
        self.freeze_vlm_model = _as_bool(motus_cfg.get("freeze_vlm_model", True), default=True)
        self.freeze_und_expert = _as_bool(motus_cfg.get("freeze_und_expert", True), default=True)
        self.freeze_t5_encoder = _as_bool(motus_cfg.get("freeze_t5_encoder", True), default=True)
        self.logprob_mode = str(motus_cfg.get("logprob_mode", "transition_velocity")).lower()
        self.logprob_sigma = float(motus_cfg.get("logprob_sigma", 1.0))
        self.collect_denoise_step = str(motus_cfg.get("collect_denoise_step", "random")).lower()

'''
if "self.trainable =" not in s:
    if needle not in s:
        raise SystemExit("Cannot find insert marker before path checks")
    s = s.replace(needle, insert + needle)

# After official policy is constructed, expose model and configure trainability.
needle2 = '''        self._policy.save_images = self.save_predicted_frames

'''
insert2 = '''        # Expose official Motus nn.Module as a submodule so RLinf Actor/FSDP/optimizer/checkpoint can see it.
        self.model = self._policy.model
        self._configure_trainable_parameters()

'''
if "self.model = self._policy.model" not in s:
    if needle2 not in s:
        raise SystemExit("Cannot find _policy.save_images marker")
    s = s.replace(needle2, needle2 + insert2)

# Add method before forward.
needle3 = '''    def forward(self, *args, **kwargs):
'''
method = r'''    def _configure_trainable_parameters(self) -> None:
        """Set requires_grad flags for RLinf training.

        Optimizer is created by RLinf Actor/FSDP, not here.
        """
        if not hasattr(self, "model"):
            return

        # Default: freeze everything unless a training mode explicitly enables it.
        for p in self.model.parameters():
            p.requires_grad_(False)

        if self.trainable in {"none", "eval", "false"}:
            trainable = 0
        elif self.trainable == "action_expert":
            for p in self.model.action_expert.parameters():
                p.requires_grad_(True)
            trainable = sum(p.numel() for p in self.model.action_expert.parameters() if p.requires_grad)
        elif self.trainable == "action_und":
            for p in self.model.action_expert.parameters():
                p.requires_grad_(True)
            for p in self.model.und_expert.parameters():
                p.requires_grad_(True)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        elif self.trainable == "all":
            for p in self.model.parameters():
                p.requires_grad_(True)
            if self.freeze_video_model and hasattr(self.model, "video_model"):
                for p in self.model.video_model.parameters():
                    p.requires_grad_(False)
            if self.freeze_vlm_model and hasattr(self.model, "vlm_model"):
                for p in self.model.vlm_model.parameters():
                    p.requires_grad_(False)
            if self.freeze_und_expert and hasattr(self.model, "und_expert"):
                for p in self.model.und_expert.parameters():
                    p.requires_grad_(False)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            raise ValueError(
                f"Unknown Motus trainable={self.trainable!r}. "
                "Use one of: none, action_expert, action_und, all."
            )

        print(
            f"[RLinf-Motus] trainable={self.trainable}, "
            f"trainable_params={trainable / 1e6:.2f}M",
            flush=True,
        )

'''
if "def _configure_trainable_parameters" not in s:
    if needle3 not in s:
        raise SystemExit("Cannot find forward marker")
    s = s.replace(needle3, method + needle3)

p.write_text(s)
print("patched", p)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py

grep -nE "trainable|_configure_trainable_parameters|self.model = self._policy.model" \
  rlinf/models/embodiment/motus/motus_policy.py \
  | sed -n '1,200p'
```

加载检查

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export PYTHONPATH="/root/autodl-tmp/RoboTwin/policy/Motus:/root/autodl-tmp/RoboTwin/policy/Motus/models:/root/autodl-tmp/Motus:${PYTHONPATH}"

python - <<'PY'
from omegaconf import OmegaConf
from rlinf.models import get_model

cfg = OmegaConf.load("examples/embodiment/config/model/motus.yaml")
model = get_model(cfg)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("total params:", total / 1e9, "B")
print("trainable params:", trainable / 1e6, "M")

for name, module in [
    ("video_model", getattr(model.model, "video_model", None)),
    ("vlm_model", getattr(model.model, "vlm_model", None)),
    ("und_expert", getattr(model.model, "und_expert", None)),
    ("action_expert", getattr(model.model, "action_expert", None)),
]:
    if module is None:
        continue
    n = sum(p.numel() for p in module.parameters())
    t = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"{name}: total={n/1e6:.2f}M trainable={t/1e6:.2f}M")
PY
```



训练模式相关修改

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

needle = '''    def eval(self):
        super().eval()
        if hasattr(self, "_policy") and hasattr(self._policy, "model"):
            self._policy.model.eval()
        return self

'''
insert = '''    def train(self, mode: bool = True):
        super().train(mode)

        if not hasattr(self, "_policy") or not hasattr(self._policy, "model"):
            return self

        if not mode:
            self._policy.model.eval()
            return self

        # Actor/FSDP training mode. Keep frozen/context modules deterministic.
        self._policy.model.train()

        if self.trainable in {"none", "eval", "false"}:
            self._policy.model.eval()
            return self

        for module_name in ["video_model", "vlm_model", "und_expert"]:
            module = getattr(self._policy.model, module_name, None)
            if module is not None:
                module.eval()

        action_expert = getattr(self._policy.model, "action_expert", None)
        if action_expert is not None:
            action_expert.train()

        return self

''' + needle

if "def train(self, mode: bool = True)" not in s:
    if needle not in s:
        raise SystemExit("Cannot find eval() block")
    s = s.replace(needle, insert)

p.write_text(s)
print("patched", p)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py

grep -nE "def train|def eval|video_model|vlm_model|und_expert|action_expert" \
  rlinf/models/embodiment/motus/motus_policy.py \
  | sed -n '1,220p'
```



加载检查

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export PYTHONPATH="/root/autodl-tmp/RoboTwin/policy/Motus:/root/autodl-tmp/RoboTwin/policy/Motus/models:/root/autodl-tmp/Motus:${PYTHONPATH}"

python - <<'PY'
from omegaconf import OmegaConf
from rlinf.models import get_model

cfg = OmegaConf.load("examples/embodiment/config/model/motus.yaml")
m = get_model(cfg)

print("before train:")
print("wrapper training =", m.training)
print("motus training =", m.model.training)
print("action_expert training =", m.model.action_expert.training)
print("video_model training =", m.model.video_model.training)
print("vlm_model training =", m.model.vlm_model.training)
print("und_expert training =", m.model.und_expert.training)

m.train()

print("after train:")
print("wrapper training =", m.training)
print("motus training =", m.model.training)
print("action_expert training =", m.model.action_expert.training)
print("video_model training =", m.model.video_model.training)
print("vlm_model training =", m.model.vlm_model.training)
print("und_expert training =", m.model.und_expert.training)

m.eval()

print("after eval:")
print("wrapper training =", m.training)
print("motus training =", m.model.training)
print("action_expert training =", m.model.action_expert.training)
PY
```



加

```Bash
patch motus_policy.py

1. 增加 tensor-only 的 VLM/T5 batch collate。
2. 增加 Motus 单步 velocity forward。
3. 让 predict_action_batch(mode="train") 返回 prev_logprobs / forward_inputs，并让 default_forward() 复算 logprobs。
```

加

```C++
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

# ---------------------------------------------------------------------
# 1. imports
# ---------------------------------------------------------------------
if "import math\n" not in s:
    s = s.replace("import inspect\n", "import inspect\nimport math\n")

# ---------------------------------------------------------------------
# 2. make predict_action_batch route train mode into vectorized train path
# ---------------------------------------------------------------------
old = '''        if self.batch_inference and batch_size > 1:
            return self._predict_action_batch_vectorized(env_obs, batch_size)

        return self._predict_action_batch_loop(env_obs, batch_size)
'''
new = '''        # Training needs prev_logprobs + tensor-only forward_inputs, so always use
        # the vectorized path even for B=1.
        if mode == "train":
            return self._predict_action_batch_vectorized(env_obs, batch_size, mode=mode)

        if self.batch_inference and batch_size > 1:
            return self._predict_action_batch_vectorized(env_obs, batch_size, mode=mode)

        return self._predict_action_batch_loop(env_obs, batch_size)
'''
if old in s:
    s = s.replace(old, new)

# ---------------------------------------------------------------------
# 3. change vectorized function signature and insert train branch
# ---------------------------------------------------------------------
s = s.replace(
'''    def _predict_action_batch_vectorized(
        self, env_obs: dict[str, Any], batch_size: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
''',
'''    def _predict_action_batch_vectorized(
        self, env_obs: dict[str, Any], batch_size: int, mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
''',
)

needle = '''        first_frame = torch.stack(frames, dim=0).to(self.device)
        state = torch.stack(states, dim=0).to(self.device)

        self._ensure_motus_grid_batch_size(batch_size)

        print(
'''
insert = '''        first_frame = torch.stack(frames, dim=0).to(self.device)
        state = torch.stack(states, dim=0).to(self.device)

        # Train path: sample actions and record one denoise transition for πRL logprob.
        if mode == "train":
            return self._predict_action_batch_vectorized_train(
                first_frame=first_frame,
                state=state,
                t5_list=t5_list,
                vlm_inputs_list=vlm_inputs_list,
                batch_size=batch_size,
            )

        self._ensure_motus_grid_batch_size(batch_size)

        print(
'''
if "_predict_action_batch_vectorized_train" not in s:
    if needle not in s:
        raise SystemExit("Cannot find first_frame/state insertion marker")
    s = s.replace(needle, insert)

# ---------------------------------------------------------------------
# 4. replace default_forward
# ---------------------------------------------------------------------
old = '''    def default_forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Motus PPO/GRPO training forward is not implemented yet. "
            "This adapter is eval-only."
        )

'''
new = '''    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Actor update forward for RLinf PPO/GRPO.

        Recompute current-policy logprobs for the denoise transition captured
        during rollout. This is the Motus analogue of OpenPI's πRL logprob.
        """
        if forward_inputs is None:
            raise ValueError("Motus default_forward requires forward_inputs.")

        device = next(self.model.parameters()).device

        video_latent_t = forward_inputs["motus_video_latent_t"].to(device=device, dtype=self.model.dtype)
        action_x_t = forward_inputs["motus_action_x_t"].to(device=device, dtype=self.model.dtype)
        action_x_next = forward_inputs["motus_action_x_next"].to(device=device, dtype=self.model.dtype)
        state = forward_inputs["motus_state"].to(device=device, dtype=self.model.dtype)
        t_scaled = forward_inputs["motus_t_scaled"].to(device=device, dtype=self.model.dtype)
        dt = forward_inputs["motus_dt"].to(device=device, dtype=self.model.dtype)

        language_embeddings = forward_inputs["motus_language_embeddings"].to(
            device=device,
            dtype=self.model.dtype,
        )
        vlm_inputs = self._vlm_inputs_from_forward_inputs(forward_inputs, device=device)

        # Current actor velocity at the rollout transition.
        _, action_velocity = self._motus_velocity_step(
            video_latent=video_latent_t,
            action_latent=action_x_t,
            state=state,
            t_scaled=t_scaled,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
        )

        logprobs = self._transition_velocity_logprobs(
            action_velocity=action_velocity,
            action_x_t=action_x_t,
            action_x_next=action_x_next,
            dt=dt,
        )

        out = {
            "logprobs": logprobs.float(),
            "values": None,
        }

        if kwargs.get("compute_entropy", False):
            # No analytic entropy yet. Shape [B,1] is compatible with RLinf entropy reshaping.
            out["entropy"] = torch.zeros(
                (logprobs.shape[0], 1),
                device=logprobs.device,
                dtype=torch.float32,
            )

        return out

'''
if old in s:
    s = s.replace(old, new)

# ---------------------------------------------------------------------
# 5. Insert training helper methods before _prepare_one_for_vectorized
# ---------------------------------------------------------------------
marker = '''    def _prepare_one_for_vectorized(
'''
helpers = r'''    def _predict_action_batch_vectorized_train(
        self,
        *,
        first_frame: torch.Tensor,
        state: torch.Tensor,
        t5_list: list[torch.Tensor],
        vlm_inputs_list: list[dict[str, torch.Tensor]],
        batch_size: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Train rollout path.

        Returns environment actions plus old logprobs and tensor-only forward_inputs.
        """
        self._ensure_motus_grid_batch_size(batch_size)

        language_embeddings = self._stack_t5_embeddings(t5_list).to(self.device)
        vlm_inputs_batched = self._collate_vlm_inputs(vlm_inputs_list, device=self.device)

        print(
            f"[RLinf-Motus] train batch inference: "
            f"B={batch_size}, first_frame={tuple(first_frame.shape)}, "
            f"state={tuple(state.shape)}, t5={tuple(language_embeddings.shape)}, "
            f"vlm_input_ids={tuple(vlm_inputs_batched['input_ids'].shape)}",
            flush=True,
        )

        predicted_actions, trace = self._motus_sample_actions_with_transition(
            first_frame=first_frame,
            state=state,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs_batched,
            num_inference_steps=self.num_inference_timesteps,
        )

        actions = predicted_actions[:, : self.num_action_chunks, : self.action_dim]
        action_tensor = actions.detach().float().cpu().contiguous()

        prev_logprobs = self._transition_velocity_logprobs(
            action_velocity=trace["old_action_velocity"],
            action_x_t=trace["action_x_t"],
            action_x_next=trace["action_x_next"],
            dt=trace["dt"],
        ).detach().float().cpu().contiguous()

        forward_inputs = {
            # Required by EnvWorker when storing executed actions.
            "action": action_tensor.reshape(batch_size, -1).contiguous(),
            "model_action": predicted_actions.detach().float().cpu().reshape(batch_size, -1).contiguous(),

            # Motus πRL context. Keep every field tensor-only and batch-major.
            "motus_first_frame": first_frame.detach().float().cpu().contiguous(),
            "motus_state": state.detach().float().cpu().contiguous(),
            "motus_language_embeddings": language_embeddings.detach().float().cpu().contiguous(),

            "motus_vlm_input_ids": vlm_inputs_batched["input_ids"].detach().cpu().contiguous(),
            "motus_vlm_attention_mask": vlm_inputs_batched["attention_mask"].detach().cpu().contiguous(),
            # Store pixel_values with explicit B dimension so RLinf split can split by env.
            "motus_vlm_pixel_values": vlm_inputs_batched["_pixel_values_batched"].detach().cpu().contiguous(),
            "motus_vlm_image_grid_thw": vlm_inputs_batched["image_grid_thw"].detach().cpu().contiguous(),

            "motus_video_latent_t": trace["video_latent_t"].detach().float().cpu().contiguous(),
            "motus_action_x_t": trace["action_x_t"].detach().float().cpu().contiguous(),
            "motus_action_x_next": trace["action_x_next"].detach().float().cpu().contiguous(),
            "motus_t_scaled": trace["t_scaled"].detach().float().cpu().contiguous(),
            "motus_dt": trace["dt"].detach().float().cpu().contiguous(),
            "motus_old_action_velocity": trace["old_action_velocity"].detach().float().cpu().contiguous(),
        }

        return action_tensor, {
            "prev_logprobs": prev_logprobs,
            "prev_values": None,
            "forward_inputs": forward_inputs,
        }

    def _stack_t5_embeddings(self, t5_list: list[torch.Tensor]) -> torch.Tensor:
        """Pad/stack per-sample T5 embeddings into [B,L,D]."""
        if not t5_list:
            raise ValueError("empty t5_list")

        max_len = max(int(x.shape[0]) for x in t5_list)
        dim = int(t5_list[0].shape[-1])
        dtype = t5_list[0].dtype
        device = t5_list[0].device

        out = torch.zeros((len(t5_list), max_len, dim), dtype=dtype, device=device)
        for i, emb in enumerate(t5_list):
            emb = emb.to(device=device, dtype=dtype)
            out[i, : emb.shape[0], :] = emb
        return out

    def _collate_vlm_inputs(
        self,
        vlm_inputs_list: list[dict[str, torch.Tensor]],
        *,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        """Collate list-form official VLM inputs into tensor-only batch.

        The returned dict can be passed to Motus. It also includes
        _pixel_values_batched [B,N,...] for RLinf-safe storage/splitting.
        """
        if not vlm_inputs_list:
            raise ValueError("empty vlm_inputs_list")

        ids_list = [x["input_ids"].squeeze(0).to(device) for x in vlm_inputs_list]
        mask_list = [x["attention_mask"].squeeze(0).to(device) for x in vlm_inputs_list]
        max_len = max(int(x.shape[0]) for x in ids_list)

        input_ids = torch.zeros(
            (len(ids_list), max_len),
            dtype=ids_list[0].dtype,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(mask_list), max_len),
            dtype=mask_list[0].dtype,
            device=device,
        )

        for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
            input_ids[i, : ids.shape[0]] = ids
            attention_mask[i, : mask.shape[0]] = mask

        pixel_values_list = [x["pixel_values"].to(device) for x in vlm_inputs_list]
        pixel_shape0 = tuple(pixel_values_list[0].shape)
        for i, pv in enumerate(pixel_values_list):
            if tuple(pv.shape) != pixel_shape0:
                raise ValueError(
                    "Motus VLM pixel_values have variable shape; add padding before training. "
                    f"sample0={pixel_shape0}, sample{i}={tuple(pv.shape)}"
                )
        pixel_values_batched = torch.stack(pixel_values_list, dim=0)  # [B,N,...]
        pixel_values = pixel_values_batched.reshape(-1, *pixel_values_batched.shape[2:])

        grid_list = []
        for x in vlm_inputs_list:
            grid = x.get("image_grid_thw", None)
            if grid is None:
                raise ValueError("Motus VLM image_grid_thw is required for training.")
            grid_list.append(grid.squeeze(0).to(device))
        image_grid_thw = torch.stack(grid_list, dim=0)  # [B,3]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "_pixel_values_batched": pixel_values_batched,
        }

    def _vlm_inputs_from_forward_inputs(
        self,
        forward_inputs: dict[str, torch.Tensor],
        *,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        pixel_values_batched = forward_inputs["motus_vlm_pixel_values"].to(device)
        pixel_values = pixel_values_batched.reshape(-1, *pixel_values_batched.shape[2:])
        return {
            "input_ids": forward_inputs["motus_vlm_input_ids"].to(device),
            "attention_mask": forward_inputs["motus_vlm_attention_mask"].to(device),
            "pixel_values": pixel_values,
            "image_grid_thw": forward_inputs["motus_vlm_image_grid_thw"].to(device),
        }

    def _motus_sample_actions_with_transition(
        self,
        *,
        first_frame: torch.Tensor,
        state: torch.Tensor,
        language_embeddings: torch.Tensor,
        vlm_inputs: dict[str, torch.Tensor],
        num_inference_steps: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run Motus denoising and capture one action transition for πRL."""
        model = self.model
        B = first_frame.shape[0]
        self._ensure_motus_grid_batch_size(B)

        first_frame = first_frame.to(device=self.device, dtype=model.dtype)
        state = state.to(device=self.device, dtype=model.dtype)
        language_embeddings = language_embeddings.to(device=self.device, dtype=model.dtype)

        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)

        # VAE is frozen; keep condition latent detached.
        with torch.no_grad():
            condition_frame_latent = model.video_model.encode_video(first_frame_norm.to(model.dtype))

        B, c_latent, _, h_latent, w_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + model.config.num_video_frames // 4

        video_latent = torch.randn(
            (B, c_latent, num_total_latent_frames, h_latent, w_latent),
            device=self.device,
            dtype=model.dtype,
        )
        video_latent[:, :, 0:1] = condition_frame_latent

        action_latent = torch.randn(
            (B, model.config.action_chunk_size, model.config.action_dim),
            device=self.device,
            dtype=model.dtype,
        )

        if self.collect_denoise_step == "random":
            trace_step = int(torch.randint(0, num_inference_steps, (1,)).item())
        else:
            trace_step = int(self.collect_denoise_step)
            trace_step = max(0, min(trace_step, num_inference_steps - 1))

        timesteps = torch.linspace(
            1.0,
            0.0,
            num_inference_steps + 1,
            device=self.device,
            dtype=model.dtype,
        )

        trace: dict[str, torch.Tensor] = {}

        for i in range(num_inference_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            t_scaled = (t * 1000).expand(B).to(model.dtype)

            if i == trace_step:
                trace["video_latent_t"] = video_latent.detach().clone()
                trace["action_x_t"] = action_latent.detach().clone()
                trace["t_scaled"] = t_scaled.detach().clone()
                trace["dt"] = dt.expand(B).detach().clone()

            video_velocity, action_velocity = self._motus_velocity_step(
                video_latent=video_latent,
                action_latent=action_latent,
                state=state,
                t_scaled=t_scaled,
                language_embeddings=language_embeddings,
                vlm_inputs=vlm_inputs,
            )

            video_latent = video_latent + video_velocity * dt
            action_x_next = action_latent + action_velocity * dt
            action_latent = action_x_next

            video_latent[:, :, 0:1] = condition_frame_latent

            if i == trace_step:
                trace["action_x_next"] = action_x_next.detach().clone()
                trace["old_action_velocity"] = action_velocity.detach().clone()

        predicted_actions = action_latent.float()

        if not trace:
            raise RuntimeError("Motus transition trace was not captured.")

        return predicted_actions, trace

    def _motus_velocity_step(
        self,
        *,
        video_latent: torch.Tensor,
        action_latent: torch.Tensor,
        state: torch.Tensor,
        t_scaled: torch.Tensor,
        language_embeddings: torch.Tensor,
        vlm_inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Motus denoise step: returns video_velocity and action_velocity."""
        model = self.model
        B = action_latent.shape[0]

        video_tokens = model.video_module.prepare_input(video_latent.to(model.dtype))

        state_tokens = state.unsqueeze(1).to(model.dtype)
        if model.action_expert.config.num_registers > 0 and model.action_expert.registers is not None:
            registers = model.action_expert.registers.expand(B, -1, -1)
        else:
            registers = None

        action_tokens = model.action_expert.input_encoder(
            state_tokens,
            action_latent.to(model.dtype),
            registers,
        )

        # Official Motus recomputes understanding tokens in the denoising loop.
        und_tokens = model.und_module.extract_und_features(vlm_inputs)
        processed_t5_context = model.video_module.preprocess_t5_embeddings(language_embeddings)

        with torch.autocast(device_type="cuda", dtype=model.video_model.precision):
            video_head_time_emb, video_adaln_params = model.video_module.get_time_embedding(
                t_scaled,
                video_tokens.shape[1],
            )
            action_head_time_emb, action_adaln_params = model.action_module.get_time_embedding(
                t_scaled,
                action_tokens.shape[1],
            )

            for layer_idx in range(model.config.num_layers):
                video_adaln_modulation = model.video_module.compute_adaln_modulation(
                    video_adaln_params,
                    layer_idx,
                )
                action_adaln_modulation = model.action_module.compute_adaln_modulation(
                    action_adaln_params,
                    layer_idx,
                )

                video_tokens, action_tokens, und_tokens = model.video_module.process_joint_attention(
                    video_tokens,
                    action_tokens,
                    video_adaln_modulation,
                    action_adaln_modulation,
                    layer_idx,
                    model.action_expert.blocks[layer_idx],
                    und_tokens,
                    model.und_expert.blocks[layer_idx],
                )

                video_tokens = model.video_module.process_cross_attention(
                    video_tokens,
                    video_adaln_params,
                    layer_idx,
                    processed_t5_context,
                )

                video_tokens = model.video_module.process_ffn(
                    video_tokens,
                    video_adaln_modulation,
                    layer_idx,
                )
                action_tokens = model.action_module.process_ffn(
                    action_tokens,
                    action_adaln_modulation,
                    layer_idx,
                )
                und_tokens = model.und_module.process_ffn(und_tokens, layer_idx)

            video_velocity = model.video_module.apply_output_head(
                video_tokens,
                video_head_time_emb,
            )

            action_pred_full = model.action_expert.decoder(
                action_tokens,
                action_head_time_emb,
            )

            n_reg = int(model.action_expert.config.num_registers)
            end = -n_reg if n_reg > 0 else None
            action_velocity = action_pred_full[:, 1:end, :]

        return video_velocity, action_velocity

    def _transition_velocity_logprobs(
        self,
        *,
        action_velocity: torch.Tensor,
        action_x_t: torch.Tensor,
        action_x_next: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian logprob for a flow transition x_next = x_t + v_theta * dt."""
        sigma = float(self.logprob_sigma)
        if sigma <= 0:
            raise ValueError(f"logprob_sigma must be positive, got {sigma}")

        while dt.dim() < action_x_t.dim():
            dt = dt.unsqueeze(-1)

        target_velocity = (action_x_next - action_x_t) / dt
        diff = (action_velocity.float() - target_velocity.float()) / sigma

        # Constant does not affect PPO ratio when sigma is fixed, but keep it for clarity.
        log_norm = math.log(sigma) + 0.5 * math.log(2.0 * math.pi)
        logprobs = -0.5 * diff.pow(2) - log_norm
        return logprobs[:, : self.num_action_chunks, : self.action_dim].float()

'''
if helpers not in s:
    if marker not in s:
        raise SystemExit("Cannot find _prepare_one_for_vectorized marker")
    s = s.replace(marker, helpers + marker)

p.write_text(s)
print("patched", p)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py

grep -nE "def default_forward|def _predict_action_batch_vectorized_train|def _motus_sample_actions_with_transition|def _motus_velocity_step|def _transition_velocity_logprobs|motus_action_x_t|prev_logprobs" \
  rlinf/models/embodiment/motus/motus_policy.py \
  | sed -n '1,260p'
```



查代码

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

python - <<'PY'
from rlinf.models.embodiment.motus.motus_policy import MotusPolicy

for name in [
    "default_forward",
    "_predict_action_batch_vectorized_train",
    "_motus_sample_actions_with_transition",
    "_motus_velocity_step",
    "_transition_velocity_logprobs",
]:
    print(name, hasattr(MotusPolicy, name))
PY
```



建立配置

```YAML
cd /root/autodl-tmp/RLinf

cat > examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_smoke_autodl.yaml <<'EOF'
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/motus@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor, env, rollout: 0

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_motus_grpo_smoke_autodl"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 1

  only_eval: False
  val_check_interval: -1
  save_interval: -1

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  group_size: 2
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level

  update_epoch: 1
  adv_type: grpo
  loss_type: actor
  loss_agg_func: "token-mean"

  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0

  gamma: 1.0
  gae_lambda: 0.95

  filter_rewards: False
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"
  enable_offload: True

  train:
    rollout_epoch: 1
    total_num_envs: 2
    reward_coef: ${algorithm.reward_coef}
    max_episode_steps: 400
    max_steps_per_rollout_epoch: 400
    group_size: ${algorithm.group_size}

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: False

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/train

    task_config:
      step_lim: 400
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

  eval:
    rollout_epoch: 1
    total_num_envs: 1
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 400
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 400
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: False

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/eval

    task_config:
      step_lim: 400
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}
    model_type: ${actor.model.model_type}
    num_action_chunks: ${actor.model.num_action_chunks}
    action_dim: ${actor.model.action_dim}
    motus: ${actor.model.motus}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 1
  global_batch_size: 2
  seed: 1234
  enable_offload: False

  model:
    model_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
    precision: "bf16"

    model_type: "motus"
    is_lora: False
    lora_rank: 32
    load_to_device: False

    num_action_chunks: 16
    action_dim: 14
    use_proprio: True
    add_value_head: False

    motus:
      repo_path: "/root/autodl-tmp/Motus"
      policy_path: "/root/autodl-tmp/RoboTwin/policy/Motus"
      checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
      wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
      vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"
      config_path: "/root/autodl-tmp/RoboTwin/policy/Motus/utils/robotwin.yml"

      num_inference_timesteps: 10
      use_t5_cache: True
      save_predicted_frames: False

      allow_batch_size: 2
      batch_inference: True
      decode_video: False

      trainable: "action_expert"
      freeze_video_model: True
      freeze_vlm_model: True
      freeze_und_expert: True
      freeze_t5_encoder: True

      logprob_mode: "transition_velocity"
      logprob_sigma: 1.0
      collect_denoise_step: "random"

  optim:
    lr: 1.0e-6
    value_lr: 1.0e-6
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False
EOF
```



查配置

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

python examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name robotwin_adjust_bottle_motus_grpo_smoke_autodl \
  --cfg job \
  | grep -nE "model_type|trainable|logprob_mode|logprob_sigma|collect_denoise_step|total_num_envs|group_size|micro_batch_size|global_batch_size|component_placement|max_steps|allow_batch_size|batch_inference|num_inference_timesteps" \
  | sed -n '1,240p'
```



#### 修

##### 显存问题

最小训练

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_motus_grpo_smoke_autodl
```

爆

```Bash
Actor 初始化 Motus 模型时 GPU0 显存不够
```

修：改配置

```Python
cd /root/autodl-tmp/RLinf

python - <<'PY'
from pathlib import Path

p = Path("examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_smoke_autodl.yaml")
s = p.read_text()

old = '''cluster:
  num_nodes: 1
  component_placement:
    actor, env, rollout: 0
'''
new = '''cluster:
  num_nodes: 1
  component_placement:
    actor: 1
    env, rollout: 0
'''

if old not in s:
    raise SystemExit("Could not find old component_placement block")

s = s.replace(old, new)
p.write_text(s)
print("patched", p)
PY

grep -nA8 -B2 "component_placement" examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_smoke_autodl.yaml
```

建议：

```Bash
关闭 init full sync
现在 actor 和 rollout 都从同一个 checkpoint 加载 Motus。初始参数本来一致。weight_syncer/patch_syncer 默认可能会做 init sync，如果它同步全模型，会很重。你现在只训练 action_expert，所以第一版 smoke 可以先关掉 init sync，只保留后续 patch delta sync。
```

修

```Bash
cd /root/autodl-tmp/RLinf

cat >> examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_smoke_autodl.yaml <<'EOF'

# Motus loads the same checkpoint in actor and rollout.
# Avoid full 8B-model init sync; later patch sync will sync trainable action_expert deltas.
weight_syncer:
  patch:
    init_sync:
      enabled: False
EOF

tail -n 20 examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_smoke_autodl.yaml
```



查配置

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

python examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name robotwin_adjust_bottle_motus_grpo_smoke_autodl \
  --cfg job \
  | grep -nE "component_placement|actor:|env, rollout|model_type|trainable|init_sync|total_num_envs|group_size|micro_batch_size|global_batch_size|allow_batch_size|max_steps" \
  | sed -n '1,260p'
```



重跑Smoke 训练

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_motus_grpo_smoke_autodl
```



爆

```Bash
问题已经从 **显存 placement OOM** 往前推进到了 **FSDP wrap 模型参数** 这一步。现在 placement 已经生效了：resolved config 里是 actor: 1，env, rollout: 0，并且 weight_syncer.patch.init_sync.enabled=false，这部分没问题。
真正的新错误是：
ValueError: Must flatten tensors with uniform `requires_grad` when `use_orig_params=False`
它发生在：
EmbodiedFSDPActor.init_worker
  -> setup_model_and_optimizer
  -> FSDP(...)
  -> FlatParamHandle(...)
  -> _validate_tensors_to_flatten(params)
也就是说，这次 **Actor 已经加载完 Motus 了**，但在 FSDP 把参数 flatten 成 flat parameter 时失败。

问题已经从 **显存 placement OOM** 往前推进到了 **FSDP wrap 模型参数** 这一步。现在 placement 已经生效了：resolved config 里是 actor: 1，env, rollout: 0，并且 weight_syncer.patch.init_sync.enabled=false，这部分没问题。
真正的新错误是：
ValueError: Must flatten tensors with uniform `requires_grad` when `use_orig_params=False`
它发生在：
EmbodiedFSDPActor.init_worker
  -> setup_model_and_optimizer
  -> FSDP(...)
  -> FlatParamHandle(...)
  -> _validate_tensors_to_flatten(params)
也就是说，这次 **Actor 已经加载完 Motus 了**，但在 FSDP 把参数 flatten 成 flat parameter 时失败。
```



修

```Python
cd /root/autodl-tmp/RLinf

python - <<'PY'
from pathlib import Path

p = Path("examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_smoke_autodl.yaml")
s = p.read_text()

# Add explicit use_orig_params: True under actor.fsdp_config.
needle = '''  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
'''
insert = '''  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
    use_orig_params: True
'''

if 'use_orig_params:' not in s:
    if needle not in s:
        raise SystemExit("Could not find actor.fsdp_config block")
    s = s.replace(needle, insert)
else:
    s = s.replace("    use_orig_params: false", "    use_orig_params: True")
    s = s.replace("    use_orig_params: False", "    use_orig_params: True")
    s = s.replace("    use_orig_params: true", "    use_orig_params: True")

p.write_text(s)
print("patched", p)
PY

grep -nA12 -B4 "fsdp_config" examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_smoke_autodl.yaml
```

查

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

python examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name robotwin_adjust_bottle_motus_grpo_smoke_autodl \
  --cfg job \
  | grep -nE "component_placement|actor:|env, rollout|init_sync|use_orig_params|model_type|trainable|total_num_envs|group_size|micro_batch_size|global_batch_size|max_steps" \
  | sed -n '1,260p'
```



再跑：

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_motus_grpo_smoke_autodl
```



##### 精度问题

爆

```SQL
新的失败点是 FSDP flatten 时遇到 **参数 dtype 混合**：torch.bfloat16 和 torch.float32 同时出现在一个 FSDP flatten group 里。 报错核心是：ValueError: Must flatten tensors with uniform dtype but got torch.bfloat16 and torch.float32。

这和 Motus 代码里 action time embedding / projection 被设计成 fp32 有关；相关逻辑里 get_time_embedding() 在 float32 autocast 中计算，并 assert 输出是 float32

不要把整个 Motus 全部转成 bf16。原因是 Motus 原代码里 action time embedding / projection 明确走 fp32 逻辑，粗暴 .to(bfloat16) 后可能会在 forward 里引入新的 dtype 错误或触发 assert。
更稳的 smoke 方案是：
1. 继续只训练 action_expert 主体。
2. 冻结 action_expert 里的 fp32 time_embedding / time_projection。
3. FSDP wrap 时忽略：
   - video_model
   - vlm_model
   - und_expert
   - action_expert.time_embedding
   - action_expert.time_projection
4. 这样 FSDP 只 flatten 剩下的 bf16 参数。
代价是：第一版 smoke 暂时不训练 time embedding / time projection 这两个小模块。对“先跑通完整训练链路”是合理的；后面正式优化时再做更细的 per-module FSDP wrapping 或专门 dtype 分组。


```



修：

```Bash
Patch 1：让 RLinf FSDP 支持模型自定义 ignored modules
Patch 2：MotusPolicy 提供 ignored modules，并冻结 fp32 time modules
```

修

```Bash
cd /root/autodl-tmp/RLinf

STAMP="$(date +%Y%m%d_%H%M%S)"

cp rlinf/hybrid_engines/fsdp/strategy/fsdp.py \
   rlinf/hybrid_engines/fsdp/strategy/fsdp.py.bak_motus_ignore_${STAMP}

cp rlinf/models/embodiment/motus/motus_policy.py \
   rlinf/models/embodiment/motus/motus_policy.py.bak_motus_dtype_${STAMP}

ls -lh rlinf/hybrid_engines/fsdp/strategy/fsdp.py.bak_motus_ignore_${STAMP}
ls -lh rlinf/models/embodiment/motus/motus_policy.py.bak_motus_dtype_${STAMP}

cd /root/autodl-tmp/RLinf

python - <<'PY'
from pathlib import Path

p = Path("rlinf/hybrid_engines/fsdp/strategy/fsdp.py")
s = p.read_text()

marker = '''        cpu_offload = CPUOffload(offload_params=self.cfg.fsdp_config.cpu_offload)

        fsdp_model = FSDP(
'''

insert = '''        cpu_offload = CPUOffload(offload_params=self.cfg.fsdp_config.cpu_offload)

        ignored_modules = None
        if hasattr(model, "get_fsdp_ignored_modules"):
            ignored_modules = model.get_fsdp_ignored_modules()
            if not ignored_modules:
                ignored_modules = None

        fsdp_model = FSDP(
'''

if "ignored_modules = None" not in s:
    if marker not in s:
        raise SystemExit("Could not find cpu_offload/FSDP marker")
    s = s.replace(marker, insert)

if "ignored_modules=ignored_modules" not in s:
    old = "            cpu_offload=cpu_offload,\n"
    new = "            cpu_offload=cpu_offload,\n            ignored_modules=ignored_modules,\n"
    if old not in s:
        raise SystemExit("Could not find cpu_offload argument")
    s = s.replace(old, new)

p.write_text(s)
print("patched", p)
PY

grep -nA28 -B8 "ignored_modules" rlinf/hybrid_engines/fsdp/strategy/fsdp.py

cd /root/autodl-tmp/RLinf

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

# 1. Before printing trainable count, freeze fp32 time modules and recompute trainable count.
marker = '''        print(
            f"[RLinf-Motus] trainable={self.trainable}, "
'''
insert = '''        self._freeze_fsdp_ignored_fp32_modules()
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(
            f"[RLinf-Motus] trainable={self.trainable}, "
'''
if "self._freeze_fsdp_ignored_fp32_modules()" not in s:
    if marker not in s:
        raise SystemExit("Could not find trainable print marker")
    s = s.replace(marker, insert)

# 2. Add helper methods before forward().
method_marker = '''    def forward(self, *args, **kwargs):
'''
helper = '''    def _freeze_fsdp_ignored_fp32_modules(self) -> None:
        """Freeze small fp32 time modules so FSDP can ignore them safely.

        Motus keeps action time embedding/projection in fp32 for numerical stability.
        FSDP flatten requires uniform dtype in a managed handle, so we keep those
        modules out of FSDP and do not optimize them in the first smoke version.
        """
        if not hasattr(self, "model"):
            return

        action_expert = getattr(self.model, "action_expert", None)
        if action_expert is None:
            return

        for module_name in ["time_embedding", "time_projection"]:
            module = getattr(action_expert, module_name, None)
            if module is None:
                continue
            module.eval()
            for p in module.parameters(recurse=True):
                p.requires_grad_(False)

    def get_fsdp_ignored_modules(self):
        """Modules excluded from FSDP flatten.

        Frozen large towers are ignored to avoid unnecessary flattening.
        fp32 time modules are ignored to avoid bf16/fp32 mixed flatten groups.
        """
        modules = []

        if not hasattr(self, "model"):
            return modules

        for module_name in ["video_model", "vlm_model", "und_expert"]:
            module = getattr(self.model, module_name, None)
            if module is not None:
                modules.append(module)

        action_expert = getattr(self.model, "action_expert", None)
        if action_expert is not None:
            for module_name in ["time_embedding", "time_projection"]:
                module = getattr(action_expert, module_name, None)
                if module is not None:
                    modules.append(module)

        # Deduplicate while preserving order.
        out = []
        seen = set()
        for module in modules:
            mid = id(module)
            if mid not in seen:
                seen.add(mid)
                out.append(module)
        return out

'''
if "def get_fsdp_ignored_modules" not in s:
    if method_marker not in s:
        raise SystemExit("Could not find forward marker")
    s = s.replace(method_marker, helper + method_marker)

p.write_text(s)
print("patched", p)
PY

python -m py_compile \
  rlinf/hybrid_engines/fsdp/strategy/fsdp.py \
  rlinf/models/embodiment/motus/motus_policy.py

grep -nE "freeze_fsdp|ignored_modules|get_fsdp_ignored_modules|trainable_params" \
  rlinf/models/embodiment/motus/motus_policy.py \
  | sed -n '1,160p'
```



查加载

```Bash
nvidia-smi

cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export PYTHONPATH="/root/autodl-tmp/RoboTwin/policy/Motus:/root/autodl-tmp/RoboTwin/policy/Motus/models:/root/autodl-tmp/Motus:${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=1 python - <<'PY'
from collections import Counter
from omegaconf import OmegaConf
from rlinf.models import get_model
import torch

cfg = OmegaConf.load("examples/embodiment/config/model/motus.yaml")
model = get_model(cfg)

ignored = set()
if hasattr(model, "get_fsdp_ignored_modules"):
    for module in model.get_fsdp_ignored_modules():
        for p in module.parameters(recurse=True):
            ignored.add(id(p))

managed = []
ignored_params = []
bad = []

for name, p in model.named_parameters():
    row = (name, str(p.dtype), bool(p.requires_grad), tuple(p.shape))
    if id(p) in ignored:
        ignored_params.append(row)
    else:
        managed.append(row)
        if p.dtype != torch.bfloat16:
            bad.append(row)

print("managed dtype/grad counts:")
print(Counter((dtype, req) for _, dtype, req, _ in managed))

print("ignored dtype/grad counts:")
print(Counter((dtype, req) for _, dtype, req, _ in ignored_params))

print("non-bf16 managed params:")
for row in bad[:50]:
    print(row)

print("num non-bf16 managed =", len(bad))
PY
```



跑训练

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

ray stop -f || python -m ray stop -f || true
nvidia-smi

source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_motus_grpo_smoke_autodl
```

成功跑通



Why

```Bash
action_expert 里的 fp32 time_embedding / time_projection 已经被你临时冻结并排除出 FSDP。
少掉的约 7.6M 参数，就是被临时冻结/ignored 的 action_expert.time_embedding 和 action_expert.time_projection 一类 fp32 time 模块。这个牺牲是为了先跑通 FSDP smoke
**ignored modules 不会被 FSDP 管理**。PyTorch 文档说，ignored_modules 的参数和 buffers 会被当前 FSDP 实例忽略。 所以这些模块不会被 FSDP shard，也不会由它做梯度同步。我们把它们设成 requires_grad=False，所以没有训练同步问题。
**action_expert 不再是 641.53M 全参训练，而是 633.92M 主体参数训练**。time embedding / time projection 暂时不训
```



当前通的代码

\[RLinf\_code\_only\_20260619\_092338\.tgz\]



新建略完整的配置

```Bash
cd /root/autodl-tmp/RLinf

cp examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_smoke_autodl.yaml \
   examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_30step_autodl.yaml

python - <<'PY'
from pathlib import Path

p = Path("examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_30step_autodl.yaml")
s = p.read_text()

s = s.replace(
    'experiment_name: "robotwin_motus_grpo_smoke_autodl"',
    'experiment_name: "robotwin_motus_grpo_30step_autodl"',
)

s = s.replace("  max_steps: 1\n", "  max_steps: 30\n")
s = s.replace("  save_interval: -1\n", "  save_interval: 10\n")

# Algorithm group size.
s = s.replace("  group_size: 2\n  reward_coef:", "  group_size: 4\n  reward_coef:")

# First env.train total_num_envs occurrence only.
s = s.replace("    total_num_envs: 2\n", "    total_num_envs: 4\n", 1)

# Actor batch.
s = s.replace("  global_batch_size: 2\n", "  global_batch_size: 4\n")

# Motus batch size. In this YAML there is normally one actor.model.motus block,
# and rollout.model.motus references actor.model.motus.
s = s.replace("      allow_batch_size: 2\n", "      allow_batch_size: 4\n")

# Keep eval disabled during training.
s = s.replace("  val_check_interval: -1\n", "  val_check_interval: -1\n")

p.write_text(s)
print("wrote", p)
PY

grep -nE "experiment_name|max_steps|save_interval|group_size|total_num_envs|global_batch_size|allow_batch_size|component_placement|trainable|use_orig_params|init_sync" \
  examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_30step_autodl.yaml \
  | sed -n '1,220p'
```

查配置

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

python examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name robotwin_adjust_bottle_motus_grpo_30step_autodl \
  --cfg job \
  | grep -nE "component_placement|actor:|env, rollout|init_sync|use_orig_params|model_type|trainable|total_num_envs|group_size|micro_batch_size|global_batch_size|max_steps|save_interval|allow_batch_size" \
  | sed -n '1,280p'
```



训练

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

ray stop -f || python -m ray stop -f || true
nvidia-smi

cd /root/autodl-tmp/RLinf
mkdir -p logs/nohup

CONFIG_NAME="robotwin_adjust_bottle_motus_grpo_30step_autodl"
LOG="logs/nohup/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc "
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

echo "PID=$!"
echo "LOG=${LOG}"
```



##### 假πrl

问题：rollout全失败，推理和eval时不一致；目前πrl是假的

修

```C++
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

STAMP="$(date +%Y%m%d_%H%M%S)"

cp rlinf/models/embodiment/motus/motus_policy.py \
   rlinf/models/embodiment/motus/motus_policy.py.bak_t5pad_robust_${STAMP}

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

start = s.index("    def _stack_t5_embeddings")
end = s.index("    def _collate_vlm_inputs", start)

new = r'''    def _get_motus_text_len(self, fallback: int) -> int:
        """Return official Motus/WAN text length, normally 512.

        The field location differs between wrappers, so query several places.
        """
        candidates = [
            getattr(getattr(self.model, "video_module", None), "text_len", None),
            getattr(getattr(getattr(self.model, "video_module", None), "config", None), "text_len", None),
            getattr(getattr(self.model, "video_model", None), "text_len", None),
            getattr(getattr(getattr(self.model, "video_model", None), "wan_model", None), "text_len", None),
            getattr(getattr(self.model, "config", None), "text_len", None),
        ]
        for x in candidates:
            if x is not None:
                return int(x)

        # Motus/WAN TI2V text_len is expected to be 512.
        return max(int(fallback), 512)

    def _stack_t5_embeddings(self, t5_list: list[torch.Tensor]) -> torch.Tensor:
        """Pad/stack per-sample T5 embeddings into [B, text_len, D].

        Official Motus/WAN inference pads T5 context to fixed text_len, normally
        512. Train-mode rollout must match that context length; padding only to
        batch max length changes cross-attention behavior.
        """
        if not t5_list:
            raise ValueError("empty t5_list")

        max_len = max(int(x.shape[0]) for x in t5_list)
        target_len = self._get_motus_text_len(max_len)

        dim = int(t5_list[0].shape[-1])
        dtype = t5_list[0].dtype
        device = t5_list[0].device

        out = torch.zeros((len(t5_list), target_len, dim), dtype=dtype, device=device)

        for i, emb in enumerate(t5_list):
            emb = emb.to(device=device, dtype=dtype)
            n = min(int(emb.shape[0]), target_len)
            out[i, :n, :] = emb[:n]

        return out

'''

s = s[:start] + new + s[end:]
p.write_text(s)
print("patched", p)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py

grep -nA55 -B4 "def _get_motus_text_len" \
  rlinf/models/embodiment/motus/motus_policy.py
```

查

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

# Add counter after warned flag if missing.
needle = "        self._warned_decode_video_unsupported = False\n"
insert = needle + "        self._train_batch_print_count = 0\n"
if "self._train_batch_print_count" not in s:
    if needle not in s:
        raise SystemExit("Cannot find warned decode_video flag")
    s = s.replace(needle, insert)

old = '''        print(
            f"[RLinf-Motus] train batch inference: "
            f"B={batch_size}, first_frame={tuple(first_frame.shape)}, "
            f"state={tuple(state.shape)}, t5={tuple(language_embeddings.shape)}, "
            f"vlm_input_ids={tuple(vlm_inputs_batched['input_ids'].shape)}",
            flush=True,
        )
'''
new = '''        if self._train_batch_print_count < 3:
            print(
                f"[RLinf-Motus] train batch inference: "
                f"B={batch_size}, first_frame={tuple(first_frame.shape)}, "
                f"state={tuple(state.shape)}, t5={tuple(language_embeddings.shape)}, "
                f"vlm_input_ids={tuple(vlm_inputs_batched['input_ids'].shape)}",
                flush=True,
            )
            self._train_batch_print_count += 1
'''

if old in s:
    s = s.replace(old, new)
else:
    print("train batch print block not found; maybe already patched")

p.write_text(s)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py
```

改sde

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

STAMP="$(date +%Y%m%d_%H%M%S)"
cp rlinf/models/embodiment/motus/motus_policy.py \
   rlinf/models/embodiment/motus/motus_policy.py.bak_flow_sde_${STAMP}

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

# Add std helper before _transition_velocity_logprobs.
marker = "    def _transition_velocity_logprobs(\n"
helper = r'''    def _transition_std(self, dt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Std for OpenPI-style SDE transition.

        For Euler-Maruyama dx = v dt + sigma sqrt(|dt|) eps.
        """
        base = float(self.logprob_sigma)
        dt_abs = dt.abs()
        while dt_abs.dim() < ref.dim():
            dt_abs = dt_abs.unsqueeze(-1)
        return base * torch.sqrt(dt_abs.clamp_min(1e-8))

'''
if "def _transition_std" not in s:
    if marker not in s:
        raise SystemExit("Cannot find _transition_velocity_logprobs marker")
    s = s.replace(marker, helper + marker)

# Replace deterministic action_x_next block.
old = '''            video_latent = video_latent + video_velocity * dt
            action_x_next = action_latent + action_velocity * dt
            action_latent = action_x_next

            video_latent[:, :, 0:1] = condition_frame_latent

            if i == trace_step:
                trace["action_x_next"] = action_x_next.detach().clone()
                trace["old_action_velocity"] = action_velocity.detach().clone()
'''
new = '''            video_latent = video_latent + video_velocity * dt

            mean_action_next = action_latent + action_velocity * dt

            if i == trace_step and self.logprob_mode in {"flow_sde", "transition_sde"}:
                std = self._transition_std(dt.expand(B), mean_action_next)
                eps = torch.randn_like(mean_action_next)
                action_x_next = mean_action_next + std * eps
                trace["transition_std"] = std.detach().clone()
            else:
                action_x_next = mean_action_next

            action_latent = action_x_next
            video_latent[:, :, 0:1] = condition_frame_latent

            if i == trace_step:
                trace["action_x_next"] = action_x_next.detach().clone()
                trace["old_action_velocity"] = action_velocity.detach().clone()
'''
if old not in s:
    raise SystemExit("Could not find deterministic action_x_next block")
s = s.replace(old, new)

# Modify logprob function to use transition_std when present? It only receives dt currently.
# Keep transition_velocity default. Flow-SDE uses same mean_next formula but std sqrt(dt).
old2 = '''        sigma = float(self.logprob_sigma)
        if sigma <= 0:
            raise ValueError(f"logprob_sigma must be positive, got {sigma}")

        while dt.dim() < action_x_t.dim():
            dt = dt.unsqueeze(-1)

        target_velocity = (action_x_next - action_x_t) / dt
        diff = (action_velocity.float() - target_velocity.float()) / sigma

        # Constant does not affect PPO ratio when sigma is fixed, but keep it for clarity.
        log_norm = math.log(sigma) + 0.5 * math.log(2.0 * math.pi)
        logprobs = -0.5 * diff.pow(2) - log_norm
        return logprobs[:, : self.num_action_chunks, : self.action_dim].float()
'''
new2 = '''        sigma = float(self.logprob_sigma)
        if sigma <= 0:
            raise ValueError(f"logprob_sigma must be positive, got {sigma}")

        while dt.dim() < action_x_t.dim():
            dt = dt.unsqueeze(-1)

        if self.logprob_mode in {"flow_sde", "transition_sde"}:
            mean_next = action_x_t + action_velocity * dt
            std = self._transition_std(dt.squeeze(-1).squeeze(-1) if dt.dim() > 1 else dt, mean_next)
            diff = (action_x_next.float() - mean_next.float()) / std.float()
            log_norm = torch.log(std.float()) + 0.5 * math.log(2.0 * math.pi)
            logprobs = -0.5 * diff.pow(2) - log_norm
        else:
            target_velocity = (action_x_next - action_x_t) / dt
            diff = (action_velocity.float() - target_velocity.float()) / sigma
            log_norm = math.log(sigma) + 0.5 * math.log(2.0 * math.pi)
            logprobs = -0.5 * diff.pow(2) - log_norm

        return logprobs[:, : self.num_action_chunks, : self.action_dim].float()
'''
if old2 not in s:
    raise SystemExit("Could not find _transition_velocity_logprobs body")
s = s.replace(old2, new2)

p.write_text(s)
print("patched", p)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py
```

重新跑

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

ray stop -f || python -m ray stop -f || true
nvidia-smi

cd /root/autodl-tmp/RLinf
mkdir -p logs/nohup

CONFIG_NAME="robotwin_adjust_bottle_motus_grpo_30step_autodl"
LOG="logs/nohup/${CONFIG_NAME}_t5pad_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc "
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

echo "PID=$!"
echo "LOG=${LOG}"
```

\[1\] 52735

PID=52735

LOG=logs/nohup/robotwin\_adjust\_bottle\_motus\_grpo\_30step\_autodl\_t5pad\_20260619\_102705\.log



roullout有成功率了

\[RLinf\_code\_only\_20260619\_102801\.tgz\]



修：兜底T5

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

old = '''        for x in candidates:
            if x is not None:
                return int(x)

        # Motus/WAN TI2V text_len is expected to be 512.
        return max(int(fallback), 512)
'''
new = '''        for x in candidates:
            if x is not None:
                # Motus/WAN TI2V text_len is expected to be 512. Do not allow
                # accidental shorter text_len to reproduce train/eval mismatch.
                return max(int(x), int(fallback), 512)

        # Motus/WAN TI2V text_len is expected to be 512.
        return max(int(fallback), 512)
'''

if old in s:
    s = s.replace(old, new)
else:
    print("text_len block already differs; inspect manually")

p.write_text(s)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py
```

修sde

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

STAMP="$(date +%Y%m%d_%H%M%S)"
cp rlinf/models/embodiment/motus/motus_policy.py \
   rlinf/models/embodiment/motus/motus_policy.py.bak_openpi_flow_sde_${STAMP}

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

# Replace _transition_std helper with OpenPI-style mean/std helper.
start = s.index("    def _transition_std")
end = s.index("    def _transition_velocity_logprobs", start)

new = r'''    def _flow_sde_mean_std(
        self,
        *,
        action_x_t: torch.Tensor,
        action_velocity: torch.Tensor,
        t_scaled: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """OpenPI-style flow_sde transition mean/std.

        This mirrors OpenPI's sample_mean_var_val flow_sde branch:
          x0_pred = x_t - v * t
          x1_pred = x_t + v * (1 - t)
          sigma_i = noise_level * sqrt(t / (1 - t))
          std = sqrt(delta) * sigma_i
        where delta = t - t_next = -dt for our decreasing timesteps.
        """
        t = (t_scaled.float() / 1000.0).clamp(1e-6, 1.0)
        delta = (-dt.float()).clamp_min(1e-8)

        while t.dim() < action_x_t.dim():
            t = t.unsqueeze(-1)
        while delta.dim() < action_x_t.dim():
            delta = delta.unsqueeze(-1)

        x0_pred = action_x_t.float() - action_velocity.float() * t
        x1_pred = action_x_t.float() + action_velocity.float() * (1.0 - t)

        noise_level = float(self.logprob_sigma)
        # OpenPI uses denom_timesteps = where(t == 1, next_t, t); this avoids
        # division by zero at t=1. For the first step use t_next = t - delta.
        t_next = (t - delta).clamp_min(1e-6)
        denom = torch.where(t >= 1.0 - 1e-6, t_next, t).clamp_min(1e-6)
        sigma_ratio = t / (1.0 - denom).clamp_min(1e-6)
        sigma_i = noise_level * torch.sqrt(sigma_ratio)

        x0_weight = 1.0 - (t - delta)
        x1_weight = (t - delta) - sigma_i.pow(2) * delta / (2.0 * t.clamp_min(1e-6))
        mean = x0_pred * x0_weight + x1_pred * x1_weight
        std = torch.sqrt(delta) * sigma_i
        std = std.clamp_min(1e-6)

        return mean.to(action_x_t.dtype), std.to(action_x_t.dtype)

'''

s = s[:start] + new + s[end:]

# Replace sampling branch in _motus_sample_actions_with_transition.
old = '''            mean_action_next = action_latent + action_velocity * dt

            if i == trace_step and self.logprob_mode in {"flow_sde", "transition_sde"}:
                std = self._transition_std(dt.expand(B), mean_action_next)
                eps = torch.randn_like(mean_action_next)
                action_x_next = mean_action_next + std * eps
                trace["transition_std"] = std.detach().clone()
            else:
                action_x_next = mean_action_next
'''
new = '''            mean_action_next = action_latent + action_velocity * dt

            if i == trace_step and self.logprob_mode in {"flow_sde", "transition_sde"}:
                mean_sde, std_sde = self._flow_sde_mean_std(
                    action_x_t=action_latent,
                    action_velocity=action_velocity,
                    t_scaled=t_scaled,
                    dt=dt.expand(B),
                )
                eps = torch.randn_like(mean_sde)
                action_x_next = mean_sde + std_sde * eps
                trace["transition_std"] = std_sde.detach().clone()
            else:
                action_x_next = mean_action_next
'''
if old not in s:
    raise SystemExit("Could not find flow_sde sampling branch")
s = s.replace(old, new)

# Replace flow_sde branch in _transition_velocity_logprobs.
old2 = '''        if self.logprob_mode in {"flow_sde", "transition_sde"}:
            mean_next = action_x_t + action_velocity * dt
            std = self._transition_std(dt.squeeze(-1).squeeze(-1) if dt.dim() > 1 else dt, mean_next)
            diff = (action_x_next.float() - mean_next.float()) / std.float()
            log_norm = torch.log(std.float()) + 0.5 * math.log(2.0 * math.pi)
            logprobs = -0.5 * diff.pow(2) - log_norm
        else:
'''
new2 = '''        if self.logprob_mode in {"flow_sde", "transition_sde"}:
            # t_scaled is needed for exact OpenPI-style flow_sde. If this function
            # is called without it, caller must use transition_velocity mode.
            raise RuntimeError(
                "_transition_velocity_logprobs no longer supports flow_sde without t_scaled. "
                "Use _transition_logprobs(..., t_scaled=...) instead."
            )
        else:
'''
if old2 not in s:
    raise SystemExit("Could not find old flow_sde logprob branch")
s = s.replace(old2, new2)

# Add new function after _transition_velocity_logprobs.
insert_after = '''        return logprobs[:, : self.num_action_chunks, : self.action_dim].float()

'''
new_func = r'''    def _transition_logprobs(
        self,
        *,
        action_velocity: torch.Tensor,
        action_x_t: torch.Tensor,
        action_x_next: torch.Tensor,
        t_scaled: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Logprob for selected denoise transition."""
        if self.logprob_mode in {"flow_sde", "transition_sde"}:
            mean_next, std = self._flow_sde_mean_std(
                action_x_t=action_x_t,
                action_velocity=action_velocity,
                t_scaled=t_scaled,
                dt=dt,
            )
            diff = (action_x_next.float() - mean_next.float()) / std.float()
            log_norm = torch.log(std.float()) + 0.5 * math.log(2.0 * math.pi)
            logprobs = -0.5 * diff.pow(2) - log_norm
            return logprobs[:, : self.num_action_chunks, : self.action_dim].float()

        return self._transition_velocity_logprobs(
            action_velocity=action_velocity,
            action_x_t=action_x_t,
            action_x_next=action_x_next,
            dt=dt,
        )

'''
if "def _transition_logprobs(" not in s:
    s = s.replace(insert_after, insert_after + new_func, 1)

# Replace calls to _transition_velocity_logprobs in train rollout and default_forward.
s = s.replace(
'''        logprobs = self._transition_velocity_logprobs(
            action_velocity=action_velocity,
            action_x_t=action_x_t,
            action_x_next=action_x_next,
            dt=dt,
        )
''',
'''        logprobs = self._transition_logprobs(
            action_velocity=action_velocity,
            action_x_t=action_x_t,
            action_x_next=action_x_next,
            t_scaled=t_scaled,
            dt=dt,
        )
'''
)

s = s.replace(
'''        prev_logprobs = self._transition_velocity_logprobs(
            action_velocity=trace["old_action_velocity"],
            action_x_t=trace["action_x_t"],
            action_x_next=trace["action_x_next"],
            dt=trace["dt"],
        ).detach().float().cpu().contiguous()
''',
'''        prev_logprobs = self._transition_logprobs(
            action_velocity=trace["old_action_velocity"],
            action_x_t=trace["action_x_t"],
            action_x_next=trace["action_x_next"],
            t_scaled=trace["t_scaled"],
            dt=trace["dt"],
        ).detach().float().cpu().contiguous()
'''
)

p.write_text(s)
print("patched", p)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py

grep -nE "def _flow_sde_mean_std|def _transition_logprobs|flow_sde|transition_velocity_logprobs|prev_logprobs = self._transition_logprobs|logprobs = self._transition_logprobs" \
  rlinf/models/embodiment/motus/motus_policy.py \
  | sed -n '1,220p'
```

换配置

```Bash
cd /root/autodl-tmp/RLinf

python - <<'PY'
from pathlib import Path

for path in [
    "examples/embodiment/config/model/motus.yaml",
    "examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_smoke_autodl.yaml",
    "examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_30step_autodl.yaml",
]:
    p = Path(path)
    if not p.exists():
        continue
    s = p.read_text()
    s = s.replace('logprob_mode: "transition_velocity"', 'logprob_mode: "flow_sde"')
    s = s.replace("logprob_mode: transition_velocity", "logprob_mode: flow_sde")
    s = s.replace("logprob_sigma: 1.0", "logprob_sigma: 0.5")
    p.write_text(s)
    print("patched", p)
PY

grep -RniE "logprob_mode|logprob_sigma" \
  examples/embodiment/config/model/motus.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_smoke_autodl.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_30step_autodl.yaml
```

查配置

```Bash
cd /root/autodl-tmp/RLinf

LOG="logs/nohup/robotwin_adjust_bottle_motus_grpo_30step_autodl_t5pad_20260619_102705.log"

grep -nE "train batch inference|logprob_mode|Global Step|return=|success_once|advantages_|grad_norm|policy_loss|Traceback|OutOfMemory|nan|inf" "$LOG" | tail -n 120

grep -RniE "logprob_mode|logprob_sigma" \
  /root/autodl-tmp/RLinf/examples/embodiment/config/model/motus.yaml \
  /root/autodl-tmp/RLinf/examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_30step_autodl.yaml
```





再跑

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

ray stop -f || python -m ray stop -f || true
nvidia-smi

cd /root/autodl-tmp/RLinf
mkdir -p logs/nohup

CONFIG_NAME="robotwin_adjust_bottle_motus_grpo_30step_autodl"
LOG="logs/nohup/${CONFIG_NAME}_flow_sde_30step_$(date +%Y%m%d_%H%M%S).log"

nohup env CONFIG_NAME="${CONFIG_NAME}" bash -lc '
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash examples/embodiment/run_embodiment.sh "${CONFIG_NAME}"
' > "${LOG}" 2>&1 &

echo "PID=$!"
echo "LOG=${LOG}"
```

\[1\] 71332

PID=71332

LOG=logs/nohup/robotwin\_adjust\_bottle\_motus\_grpo\_30step\_autodl\_flow\_sde\_30step\_20260619\_110216\.log



整体跑通，训练看起来没问题



顺序评估，base，30，20，10

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

ray stop -f || python -m ray stop -f || true
nvidia-smi

mkdir -p logs/nohup

RUN_DIR="/root/autodl-tmp/RLinf/logs/20260619-11:02:16-robotwin_adjust_bottle_motus_grpo_30step_autodl/robotwin_motus_grpo_30step_autodl/checkpoints"

CKPT30="${RUN_DIR}/global_step_30/actor/model_state_dict/full_weights.pt"
CKPT20="${RUN_DIR}/global_step_20/actor/model_state_dict/full_weights.pt"
CKPT10="${RUN_DIR}/global_step_10/actor/model_state_dict/full_weights.pt"

EVAL_CFG="/root/autodl-tmp/RLinf/evaluations/robotwin/robotwin_adjust_bottle_motus_eval_batch8_autodl.yaml"

test -f "${EVAL_CFG}"
test -s "${CKPT30}"
test -s "${CKPT20}"
test -s "${CKPT10}"

ls -lh "${CKPT30}" "${CKPT20}" "${CKPT10}"

STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_SCRIPT="logs/nohup/eval_motus_base_step30_step20_step10_64eps_${STAMP}.sh"
LOG="logs/nohup/eval_motus_base_step30_step20_step10_64eps_${STAMP}.log"

cat > "${EVAL_SCRIPT}" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/rlinf_env.sh

cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

RUN_DIR="/root/autodl-tmp/RLinf/logs/20260619-11:02:16-robotwin_adjust_bottle_motus_grpo_30step_autodl/robotwin_motus_grpo_30step_autodl/checkpoints"

CKPT30="${RUN_DIR}/global_step_30/actor/model_state_dict/full_weights.pt"
CKPT20="${RUN_DIR}/global_step_20/actor/model_state_dict/full_weights.pt"
CKPT10="${RUN_DIR}/global_step_10/actor/model_state_dict/full_weights.pt"

test -s "${CKPT30}"
test -s "${CKPT20}"
test -s "${CKPT10}"

wait_until_gpu_free() {
  echo
  echo "[wait] checking Ray/eval/GPU cleanup..."

  for i in $(seq 1 120); do
    RAY_PIDS="$(pgrep -f 'ray::|raylet|gcs_server|dashboard_agent|dashboard.py' || true)"
    EVAL_PIDS="$(pgrep -f 'train_embodied_agent.py|evaluations/run_eval.sh' || true)"
    GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+' || true)"

    if [ -z "${RAY_PIDS}" ] && [ -z "${EVAL_PIDS}" ] && [ -z "${GPU_PIDS}" ]; then
      echo "[wait] Ray/eval/GPU clean."
      nvidia-smi
      return 0
    fi

    echo "[wait] not clean yet, retry ${i}/120"
    echo "RAY_PIDS=${RAY_PIDS:-none}"
    echo "EVAL_PIDS=${EVAL_PIDS:-none}"
    echo "GPU_PIDS=${GPU_PIDS:-none}"
    sleep 5
  done

  echo "[wait] timeout waiting for cleanup."
  nvidia-smi
  return 1
}

run_eval() {
  local NAME="$1"
  local CKPT="$2"
  local EXP="$3"

  echo
  echo "============================================================"
  echo "START EVAL ${NAME}"
  echo "TIME=$(date)"
  echo "CKPT=${CKPT}"
  echo "EXP=${EXP}"
  echo "============================================================"
  echo

  ray stop -f || python -m ray stop -f || true
  wait_until_gpu_free

  set +e

  bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_motus_eval_batch8_autodl \
    runner.ckpt_path="${CKPT}" \
    env.eval.total_num_envs=8 \
    env.eval.rollout_epoch=8 \
    rollout.model.motus.allow_batch_size=8 \
    runner.logger.experiment_name="${EXP}"

  local RC=$?

  set -e

  echo
  echo "============================================================"
  echo "END EVAL ${NAME}"
  echo "TIME=$(date)"
  echo "RETURN_CODE=${RC}"
  echo "============================================================"
  echo

  ray stop -f || python -m ray stop -f || true
  wait_until_gpu_free

  if [ "${RC}" -ne 0 ]; then
    echo "[error] ${NAME} failed with return code ${RC}; stop sequence."
    exit "${RC}"
  fi
}

run_eval \
  "BASE_64EPS" \
  "null" \
  "robotwin_motus_eval_base_64eps"

run_eval \
  "STEP30_64EPS" \
  "${CKPT30}" \
  "robotwin_motus_eval_step30_64eps"

run_eval \
  "STEP20_64EPS" \
  "${CKPT20}" \
  "robotwin_motus_eval_step20_64eps"

run_eval \
  "STEP10_64EPS" \
  "${CKPT10}" \
  "robotwin_motus_eval_step10_64eps"

echo
echo "ALL EVALS FINISHED"
echo "TIME=$(date)"
BASH

chmod +x "${EVAL_SCRIPT}"

nohup bash "${EVAL_SCRIPT}" > "${LOG}" 2>&1 &

echo "PID=$!"
echo "SCRIPT=${EVAL_SCRIPT}"
echo "LOG=${LOG}"
```

\[1\] 137987

PID=137987

SCRIPT=logs/nohup/eval\_motus\_base\_step30\_step20\_step10\_64eps\_20260619\_134011\.sh

LOG=logs/nohup/eval\_motus\_base\_step30\_step20\_step10\_64eps\_20260619\_134011\.log



这个参数下显存占用54g左右



没效果



#### 训





##### 效果问题

问题：不work



改推理模式

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

STAMP="$(date +%Y%m%d_%H%M%S)"
cp rlinf/models/embodiment/motus/motus_policy.py \
   rlinf/models/embodiment/motus/motus_policy.py.bak_action_expert_eval_${STAMP}

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

old = '''        action_expert = getattr(self._policy.model, "action_expert", None)
        if action_expert is not None:
            action_expert.train()
'''

new = '''        action_expert = getattr(self._policy.model, "action_expert", None)
        if action_expert is not None:
            # Keep policy stochasticity controlled by flow_sde noise, not module
            # train-mode randomness. eval() does not disable gradients; params
            # with requires_grad=True are still optimized in actor default_forward.
            action_expert.eval()
'''

if old not in s:
    raise SystemExit("Could not find action_expert.train() block; inspect MotusPolicy.train() manually.")

s = s.replace(old, new)
p.write_text(s)
print("patched", p)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py

grep -nA18 -B8 "action_expert = getattr" \
  rlinf/models/embodiment/motus/motus_policy.py
```



Motus 官方 policy copy 到 RLinf 侧

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

mkdir -p /root/autodl-tmp/RLinf/third_party

STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -d /root/autodl-tmp/RLinf/third_party/Motus ]; then
  mv /root/autodl-tmp/RLinf/third_party/Motus \
     "/root/autodl-tmp/RLinf/third_party/Motus_old_${STAMP}"
fi

rsync -a \
  --exclude='logs*' \
  --exclude='__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.mp4' \
  --exclude='*.png' \
  --exclude='*.jpg' \
  --exclude='*.jpeg' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.safetensors' \
  /root/autodl-tmp/RoboTwin/policy/Motus/ \
  /root/autodl-tmp/RLinf/third_party/Motus/

echo "vendored Motus:"
du -sh /root/autodl-tmp/RLinf/third_party/Motus
find /root/autodl-tmp/RLinf/third_party/Motus -maxdepth 3 -type f \
  | sed -n '1,120p'
```



查引用

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export PYTHONPATH="/root/autodl-tmp/RLinf/third_party/Motus:/root/autodl-tmp/RLinf/third_party/Motus/models:/root/autodl-tmp/Motus:${PYTHONPATH}"

python - <<'PY'
import sys
print("python =", sys.executable)

from models.motus import Motus, MotusConfig
print("Motus import OK", Motus, MotusConfig)

import importlib.util
p = "/root/autodl-tmp/RLinf/third_party/Motus/deploy_policy.py"
spec = importlib.util.spec_from_file_location("vendored_motus_deploy_policy_check", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print("deploy_policy import OK", hasattr(m, "MotusPolicy"))
PY
```



改为引rlinf内motus

```Bash
cd /root/autodl-tmp/RLinf

python - <<'PY'
from pathlib import Path

paths = [
    Path("examples/embodiment/config/model/motus.yaml"),
    Path("examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_smoke_autodl.yaml"),
    Path("examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_30step_autodl.yaml"),
]

for p in paths:
    if not p.exists():
        continue
    s = p.read_text()
    s = s.replace(
        'policy_path: "/root/autodl-tmp/RoboTwin/policy/Motus"',
        'policy_path: "/root/autodl-tmp/RLinf/third_party/Motus"',
    )
    p.write_text(s)
    print("patched", p)
PY

grep -Rni "policy_path" \
  examples/embodiment/config/model/motus.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_smoke_autodl.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_30step_autodl.yaml
```



打包代码

```Bash
cd /root/autodl-tmp/RLinf/third_party/Motus

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="/root/autodl-tmp/Motus_policy_code_for_trace_patch_${STAMP}.tgz"

tar -czf "${OUT}" \
  --exclude='logs*' \
  --exclude='__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.mp4' \
  --exclude='*.png' \
  --exclude='*.jpg' \
  --exclude='*.jpeg' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.safetensors' \
  .

ls -lh "${OUT}"
```



给 vendored Motus 的 `inference_step()` 增加 trace 支持

```SQL
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

STAMP="$(date +%Y%m%d_%H%M%S)"

cp third_party/Motus/models/motus.py \
   third_party/Motus/models/motus.py.bak_return_trace_${STAMP}

python - <<'PY'
from pathlib import Path

p = Path("third_party/Motus/models/motus.py")
s = p.read_text()

start = s.index("    def inference_step(\n")
end = s.index("    # Alternative inference (DPM++ solver)", start)

new = r'''    def flow_sde_mean_std(
        self,
        *,
        action_x_t: torch.Tensor,
        action_velocity: torch.Tensor,
        t_scaled: torch.Tensor,
        dt: torch.Tensor,
        noise_level: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """OpenPI-style one-step flow-SDE transition mean/std for action latents.

        This is used only for the selected RL transition when return_trace=True.
        Normal evaluation keeps the original deterministic flow-ODE integration.
        """
        t = (t_scaled.float() / 1000.0).clamp(1e-6, 1.0)
        delta = (-dt.float()).clamp_min(1e-8)

        while t.dim() < action_x_t.dim():
            t = t.unsqueeze(-1)
        while delta.dim() < action_x_t.dim():
            delta = delta.unsqueeze(-1)

        x0_pred = action_x_t.float() - action_velocity.float() * t
        x1_pred = action_x_t.float() + action_velocity.float() * (1.0 - t)

        # OpenPI's flow_sde branch uses sigma proportional to sqrt(t / (1-t)).
        # At t=1, use t_next in the denominator to avoid division by zero.
        t_next = (t - delta).clamp_min(1e-6)
        denom = torch.where(t >= 1.0 - 1e-6, t_next, t).clamp_min(1e-6)
        sigma_ratio = t / (1.0 - denom).clamp_min(1e-6)
        sigma_i = float(noise_level) * torch.sqrt(sigma_ratio)

        x0_weight = 1.0 - (t - delta)
        x1_weight = (t - delta) - sigma_i.pow(2) * delta / (2.0 * t.clamp_min(1e-6))
        mean = x0_pred * x0_weight + x1_pred * x1_weight
        std = torch.sqrt(delta) * sigma_i
        std = std.clamp_min(1e-6)
        return mean.to(action_x_t.dtype), std.to(action_x_t.dtype)

    def denoise_velocity_step(
        self,
        *,
        video_latent: torch.Tensor,
        action_latent: torch.Tensor,
        state: torch.Tensor = None,
        t_scaled: torch.Tensor,
        language_embeddings=None,
        vlm_inputs=None,
        processed_t5_context: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One Motus denoise velocity step shared by eval, train rollout, and actor logprob.

        Returns:
            video_velocity, action_velocity, und_tokens
        """
        B = video_latent.shape[0]

        video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))

        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(B, -1, -1)
        else:
            registers = None

        if self.config.training_mode == 'pretrain':
            action_tokens = self.action_expert.input_encoder(
                None,
                action_latent.to(self.dtype),
                registers,
            )
        else:
            if state is None:
                raise ValueError("state must be provided for Motus finetune inference")
            state_tokens = state.unsqueeze(1).to(self.dtype)
            action_tokens = self.action_expert.input_encoder(
                state_tokens,
                action_latent.to(self.dtype),
                registers,
            )

        # Official Motus inference re-extracts understanding tokens inside each denoise step.
        und_tokens = self.und_module.extract_und_features(vlm_inputs)

        if processed_t5_context is None:
            processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
                t_scaled,
                video_tokens.shape[1],
            )
            action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(
                t_scaled,
                action_tokens.shape[1],
            )

            for layer_idx in range(self.config.num_layers):
                video_adaln_modulation = self.video_module.compute_adaln_modulation(
                    video_adaln_params,
                    layer_idx,
                )
                action_adaln_modulation = self.action_module.compute_adaln_modulation(
                    action_adaln_params,
                    layer_idx,
                )

                video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                    video_tokens,
                    action_tokens,
                    video_adaln_modulation,
                    action_adaln_modulation,
                    layer_idx,
                    self.action_expert.blocks[layer_idx],
                    und_tokens,
                    self.und_expert.blocks[layer_idx],
                )

                video_tokens = self.video_module.process_cross_attention(
                    video_tokens,
                    video_adaln_params,
                    layer_idx,
                    processed_t5_context,
                )
                video_tokens = self.video_module.process_ffn(
                    video_tokens,
                    video_adaln_modulation,
                    layer_idx,
                )
                action_tokens = self.action_module.process_ffn(
                    action_tokens,
                    action_adaln_modulation,
                    layer_idx,
                )
                und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

            video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
            action_velocity_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
            up_len = action_velocity_full.shape[1] - self.action_expert.config.num_registers
            if self.config.training_mode == 'pretrain':
                action_velocity = action_velocity_full[:, :up_len, :]
            else:
                action_velocity = action_velocity_full[:, 1:up_len, :]

        return video_velocity, action_velocity, und_tokens

    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        *,
        return_trace: bool = False,
        trace_step: Optional[int] = None,
        logprob_mode: str = "flow_ode",
        logprob_sigma: float = 0.5,
        decode_video: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Joint inference for video and action prediction.

        This is the single official denoise loop used by both evaluation and RL rollout.
        In RL mode, return_trace=True records one selected action transition for πRL.
        """
        B = first_frame.shape[0]

        if isinstance(language_embeddings, list):
            language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        else:
            language_embeddings = language_embeddings.to(self.device).to(self.dtype)

        if self.config.training_mode != 'pretrain':
            state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        if trace_step is None:
            trace_step = int(torch.randint(0, num_inference_steps, (1,)).item()) if return_trace else -1
        trace_step = int(max(-1, min(int(trace_step), num_inference_steps - 1)))
        logprob_mode = str(logprob_mode).lower()

        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))

        B, C_latent, _f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn(
            (B, C_latent, num_total_latent_frames, H_latent, W_latent),
            device=self.device,
            dtype=self.dtype,
        )
        video_latent[:, :, 0:1] = condition_frame_latent

        action_latent = torch.randn(
            (B, self.config.action_chunk_size, self.config.action_dim),
            device=self.device,
            dtype=self.dtype,
        )

        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)
        trace: dict[str, torch.Tensor] = {}

        for i in range(num_inference_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            t_scaled = (t * 1000).expand(B).to(self.dtype)

            if return_trace and i == trace_step:
                trace["video_latent_t"] = video_latent.detach().clone()
                trace["action_x_t"] = action_latent.detach().clone()
                trace["t_scaled"] = t_scaled.detach().clone()
                trace["dt"] = dt.expand(B).detach().clone()

            video_velocity, action_velocity, _und_tokens = self.denoise_velocity_step(
                video_latent=video_latent,
                action_latent=action_latent,
                state=state,
                t_scaled=t_scaled,
                language_embeddings=language_embeddings,
                vlm_inputs=vlm_inputs,
                processed_t5_context=processed_t5_context,
            )

            video_latent = video_latent + video_velocity * dt

            if return_trace and i == trace_step and logprob_mode in {"flow_sde", "transition_sde"}:
                mean_sde, std_sde = self.flow_sde_mean_std(
                    action_x_t=action_latent,
                    action_velocity=action_velocity,
                    t_scaled=t_scaled,
                    dt=dt.expand(B),
                    noise_level=float(logprob_sigma),
                )
                eps = torch.randn_like(mean_sde)
                action_x_next = mean_sde + std_sde * eps
                trace["transition_std"] = std_sde.detach().clone()
            else:
                action_x_next = action_latent + action_velocity * dt

            action_latent = action_x_next
            video_latent[:, :, 0:1] = condition_frame_latent

            if return_trace and i == trace_step:
                trace["action_x_next"] = action_x_next.detach().clone()
                trace["old_action_velocity"] = action_velocity.detach().clone()

        predicted_frames = None
        if decode_video:
            with torch.no_grad():
                decoded_frames = self.video_model.decode_video(video_latent)
                predicted_frames = decoded_frames[:, :, 1:]
                predicted_frames = (predicted_frames + 1.0) / 2.0
                predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = action_latent.float()

        if return_trace:
            required = ["video_latent_t", "action_x_t", "action_x_next", "t_scaled", "dt", "old_action_velocity"]
            missing = [k for k in required if k not in trace]
            if missing:
                raise RuntimeError(f"Motus inference trace missing keys: {missing}")
            return predicted_frames, predicted_actions, trace

        return predicted_frames, predicted_actions

'''

s = s[:start] + new + s[end:]
p.write_text(s)
print("patched", p)
PY

python -m py_compile third_party/Motus/models/motus.py

grep -nE "def flow_sde_mean_std|def denoise_velocity_step|def inference_step|return_trace|trace_step|decode_video" \
  third_party/Motus/models/motus.py \
  | sed -n '1,220p'
```



让 RLinf Motus train path 调 official

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

STAMP="$(date +%Y%m%d_%H%M%S)"

cp rlinf/models/embodiment/motus/motus_policy.py \
   rlinf/models/embodiment/motus/motus_policy.py.bak_call_official_trace_${STAMP}

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/motus/motus_policy.py")
s = p.read_text()

old = '''        predicted_actions, trace = self._motus_sample_actions_with_transition(
            first_frame=first_frame,
            state=state,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs_batched,
            num_inference_steps=self.num_inference_timesteps,
        )
'''

new = '''        trace_step = self._select_trace_step(self.num_inference_timesteps)

        sig = inspect.signature(self._policy.model.inference_step)
        if "return_trace" in sig.parameters:
            out = self._policy.model.inference_step(
                first_frame=first_frame,
                state=state,
                num_inference_steps=self.num_inference_timesteps,
                language_embeddings=language_embeddings,
                vlm_inputs=vlm_inputs_batched,
                return_trace=True,
                trace_step=trace_step,
                logprob_mode=self.logprob_mode,
                logprob_sigma=self.logprob_sigma,
                decode_video=False,
            )
            if not isinstance(out, tuple) or len(out) < 3:
                raise RuntimeError(
                    "Motus inference_step(return_trace=True) must return "
                    "(predicted_frames, predicted_actions, trace)."
                )
            _, predicted_actions, trace = out[:3]
        else:
            print(
                "[RLinf-Motus] WARNING: Motus model has no "
                "inference_step(return_trace=True); falling back to adapter-local "
                "denoise loop. Train/eval denoise loops are not fully shared.",
                flush=True,
            )
            predicted_actions, trace = self._motus_sample_actions_with_transition(
                first_frame=first_frame,
                state=state,
                language_embeddings=language_embeddings,
                vlm_inputs=vlm_inputs_batched,
                num_inference_steps=self.num_inference_timesteps,
                trace_step=trace_step,
            )
'''

if old not in s:
    raise SystemExit("Could not find _motus_sample_actions_with_transition call block")

s = s.replace(old, new)

old_sig = '''    def _motus_sample_actions_with_transition(
        self,
        *,
        first_frame: torch.Tensor,
        state: torch.Tensor,
        language_embeddings: torch.Tensor,
        vlm_inputs: dict[str, torch.Tensor],
        num_inference_steps: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
'''

new_sig = '''    def _motus_sample_actions_with_transition(
        self,
        *,
        first_frame: torch.Tensor,
        state: torch.Tensor,
        language_embeddings: torch.Tensor,
        vlm_inputs: dict[str, torch.Tensor],
        num_inference_steps: int,
        trace_step: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
'''

if old_sig not in s:
    raise SystemExit("Could not find _motus_sample_actions_with_transition signature")

s = s.replace(old_sig, new_sig)

old_select = '''        if self.collect_denoise_step == "random":
            trace_step = int(torch.randint(0, num_inference_steps, (1,)).item())
        else:
            trace_step = int(self.collect_denoise_step)
            trace_step = max(0, min(trace_step, num_inference_steps - 1))
'''

new_select = '''        if trace_step is None:
            trace_step = self._select_trace_step(num_inference_steps)
'''

if old_select not in s:
    raise SystemExit("Could not find local trace_step selection block")

s = s.replace(old_select, new_select)

selector_marker = '''    def _motus_sample_actions_with_transition(
'''

selector_helper = '''    def _select_trace_step(self, num_inference_steps: int) -> int:
        if self.collect_denoise_step == "random":
            return int(torch.randint(0, num_inference_steps, (1,)).item())
        trace_step = int(self.collect_denoise_step)
        return max(0, min(trace_step, num_inference_steps - 1))

'''

if "def _select_trace_step" not in s:
    s = s.replace(selector_marker, selector_helper + selector_marker)

old_velocity_start = '''        """One Motus denoise step: returns video_velocity and action_velocity."""
        model = self.model
        B = action_latent.shape[0]

'''

new_velocity_start = '''        """One Motus denoise step: returns video_velocity and action_velocity."""
        model = self.model
        if hasattr(model, "denoise_velocity_step"):
            video_velocity, action_velocity, _ = model.denoise_velocity_step(
                video_latent=video_latent,
                action_latent=action_latent,
                state=state,
                t_scaled=t_scaled,
                language_embeddings=language_embeddings,
                vlm_inputs=vlm_inputs,
            )
            return video_velocity, action_velocity

        B = action_latent.shape[0]

'''

if old_velocity_start not in s:
    raise SystemExit("Could not find _motus_velocity_step start block")

s = s.replace(old_velocity_start, new_velocity_start)

old_flow_start = '''        t = (t_scaled.float() / 1000.0).clamp(1e-6, 1.0)
        delta = (-dt.float()).clamp_min(1e-8)
'''

new_flow_start = '''        if hasattr(self.model, "flow_sde_mean_std"):
            return self.model.flow_sde_mean_std(
                action_x_t=action_x_t,
                action_velocity=action_velocity,
                t_scaled=t_scaled,
                dt=dt,
                noise_level=float(self.logprob_sigma),
            )

        t = (t_scaled.float() / 1000.0).clamp(1e-6, 1.0)
        delta = (-dt.float()).clamp_min(1e-8)
'''

if old_flow_start not in s:
    raise SystemExit("Could not find _flow_sde_mean_std body start")

s = s.replace(old_flow_start, new_flow_start, 1)

p.write_text(s)
print("patched", p)
PY

python -m py_compile rlinf/models/embodiment/motus/motus_policy.py

grep -nE "return_trace|_select_trace_step|denoise_velocity_step|falling back|flow_sde_mean_std" \
  rlinf/models/embodiment/motus/motus_policy.py \
  | sed -n '1,220p'
```



查代码

```Python
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export PYTHONPATH="/root/autodl-tmp/RLinf/third_party/Motus:/root/autodl-tmp/RLinf/third_party/Motus/models:/root/autodl-tmp/Motus:${PYTHONPATH}"

python - <<'PY'
import inspect
from models.motus import Motus

print("inference_step signature:")
print(inspect.signature(Motus.inference_step))

print("has denoise_velocity_step =", hasattr(Motus, "denoise_velocity_step"))
print("has flow_sde_mean_std =", hasattr(Motus, "flow_sde_mean_std"))

assert "return_trace" in inspect.signature(Motus.inference_step).parameters
assert "trace_step" in inspect.signature(Motus.inference_step).parameters
assert "logprob_mode" in inspect.signature(Motus.inference_step).parameters
assert "decode_video" in inspect.signature(Motus.inference_step).parameters
PY
```



eval smoke：2 episodes

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

ray stop -f || python -m ray stop -f || true
nvidia-smi

mkdir -p logs/nohup

LOG="logs/nohup/eval_motus_vendored_trace_smoke2_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc "
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:\${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_motus_eval_batch8_autodl \
  env.eval.total_num_envs=2 \
  env.eval.rollout_epoch=1 \
  rollout.model.motus.allow_batch_size=2 \
  runner.logger.experiment_name=robotwin_motus_eval_vendored_trace_smoke2
" > "${LOG}" 2>&1 &

echo "PID=$!"
echo "LOG=${LOG}"
```



train smoke：2 steps

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

ray stop -f || python -m ray stop -f || true
nvidia-smi

mkdir -p logs/nohup

CONFIG_NAME="robotwin_adjust_bottle_motus_grpo_30step_autodl"
LOG="logs/nohup/${CONFIG_NAME}_official_trace_smoke2_$(date +%Y%m%d_%H%M%S).log"

nohup env CONFIG_NAME="${CONFIG_NAME}" bash -lc '
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

bash examples/embodiment/run_embodiment.sh "${CONFIG_NAME}" \
  runner.max_steps=2 \
  runner.save_interval=-1 \
  runner.logger.experiment_name=robotwin_motus_grpo_official_trace_smoke2
' > "${LOG}" 2>&1 &

echo "PID=$!"
echo "LOG=${LOG}"
```

参数错，但通的



正式训练



创建配置

```Bash
cd /root/autodl-tmp/RLinf

cp examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_30step_autodl.yaml \
   examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_g8_20step_trace_autodl.yaml

python - <<'PY'
from pathlib import Path

p = Path("examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_g8_20step_trace_autodl.yaml")
s = p.read_text()

s = s.replace(
    'experiment_name: "robotwin_motus_grpo_30step_autodl"',
    'experiment_name: "robotwin_motus_grpo_g8_20step_trace_autodl"',
)

s = s.replace("  max_steps: 30\n", "  max_steps: 20\n")
s = s.replace("  save_interval: 10\n", "  save_interval: 10\n")

s = s.replace("  group_size: 4\n  reward_coef:", "  group_size: 8\n  reward_coef:")
s = s.replace("  filter_rewards: False\n", "  filter_rewards: True\n")
s = s.replace("  filter_rewards: false\n", "  filter_rewards: True\n")

s = s.replace("    total_num_envs: 4\n", "    total_num_envs: 8\n", 1)
s = s.replace("  global_batch_size: 4\n", "  global_batch_size: 8\n")
s = s.replace("      allow_batch_size: 4\n", "      allow_batch_size: 8\n")

s = s.replace("    lr: 1e-06\n", "    lr: 5e-07\n")
s = s.replace("    lr: 1.0e-6\n", "    lr: 5.0e-7\n")
s = s.replace("    lr: 1.0e-06\n", "    lr: 5.0e-7\n")

s = s.replace("    value_lr: 1e-06\n", "    value_lr: 5e-07\n")
s = s.replace("    value_lr: 1.0e-6\n", "    value_lr: 5.0e-7\n")
s = s.replace("    value_lr: 1.0e-06\n", "    value_lr: 5.0e-7\n")

p.write_text(s)
print("wrote", p)
PY

grep -nE "experiment_name|max_steps|save_interval|group_size|total_num_envs|global_batch_size|allow_batch_size|filter_rewards|lr:|value_lr|logprob_mode|logprob_sigma|component_placement|micro_batch_size|update_epoch|policy_path" \
  examples/embodiment/config/robotwin_adjust_bottle_motus_grpo_g8_20step_trace_autodl.yaml \
  | sed -n '1,260p'
```



确认配置

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

python examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name robotwin_adjust_bottle_motus_grpo_g8_20step_trace_autodl \
  --cfg job \
  | grep -nE "experiment_name|policy_path|max_steps|save_interval|group_size|total_num_envs|global_batch_size|allow_batch_size|filter_rewards|lr:|value_lr|logprob_mode|logprob_sigma|component_placement" \
  | sed -n '1,260p'
```



开监控

```Bash
cd /root/autodl-tmp/RLinf
mkdir -p logs/nohup

MON="logs/nohup/monitor_motus_g8_20step_trace_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc '
while true; do
  echo "================ $(date) ================"
  echo "[free -h]"
  free -h
  echo
  echo "[nvidia-smi]"
  nvidia-smi
  echo
  echo "[top RSS processes]"
  ps -eo pid,ppid,pmem,rss,cmd --sort=-rss | head -30
  echo
  sleep 60
done
' > "${MON}" 2>&1 &

echo "MON_PID=$!"
echo "MON_LOG=${MON}"
```



训练

```Bash
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

ray stop -f || python -m ray stop -f || true
nvidia-smi

mkdir -p logs/nohup

CONFIG_NAME="robotwin_adjust_bottle_motus_grpo_g8_20step_trace_autodl"
RUN_LOG_DIR="/root/autodl-tmp/RLinf/logs/$(date +%Y%m%d-%H:%M:%S)-${CONFIG_NAME}"
LOG="logs/nohup/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

nohup env CONFIG_NAME="${CONFIG_NAME}" bash -lc "
set -euo pipefail

source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:\${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

python examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name ${CONFIG_NAME} \
  runner.logger.log_path=${RUN_LOG_DIR}
" > "${LOG}" 2>&1 &

echo "PID=$!"
echo "LOG=${LOG}"
echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
```

20步成功完成



分析

```Bash
风险：
- actor/ratio 有效步平均约 0.64，仍偏低。
- step12 之后全失败 group 增多，训练 rollout return 后半段偏弱。
- step20 自身 return=0，因此 step20 不一定比 step10 好。
```



评估，base 10 20，ep16

```SQL
source /root/autodl-tmp/rlinf_env.sh

cd /root/autodl-tmp/RLinf
source .venv/bin/activate

ray stop -f || python -m ray stop -f || true
nvidia-smi

mkdir -p logs/nohup

RUN_DIR="/root/autodl-tmp/RLinf/logs/20260619-20:17:59-robotwin_adjust_bottle_motus_grpo_g8_20step_trace_autodl"

CKPT10="${RUN_DIR}/robotwin_motus_grpo_g8_20step_trace_autodl/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"

CKPT20="${RUN_DIR}/robotwin_motus_grpo_g8_20step_trace_autodl/checkpoints/global_step_20/actor/model_state_dict/full_weights.pt"

EVAL_CFG="/root/autodl-tmp/RLinf/evaluations/robotwin/robotwin_adjust_bottle_motus_eval_batch8_autodl.yaml"

test -f "${EVAL_CFG}"
test -s "${CKPT10}"
test -s "${CKPT20}"

ls -lh \
  "${CKPT10}" \
  "${CKPT20}"

LOG="logs/nohup/eval_motus_base_step10_step20_16eps_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc "
set -euo pipefail

source /root/autodl-tmp/rlinf_env.sh

cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export PYTHONPATH=/root/autodl-tmp/RoboTwin_RLinf:\${PYTHONPATH}
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

run_eval() {
  local NAME=\"\$1\"
  local CKPT=\"\$2\"
  local EXP=\"\$3\"

  echo
  echo \"============================================================\"
  echo \"START EVAL: \${NAME}\"
  echo \"CKPT=\${CKPT}\"
  echo \"EXP=\${EXP}\"
  echo \"TIME=\$(date)\"
  echo \"============================================================\"
  echo

  ray stop -f || python -m ray stop -f || true

  bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_motus_eval_batch8_autodl \
    runner.ckpt_path=\"\${CKPT}\" \
    env.eval.total_num_envs=8 \
    env.eval.rollout_epoch=2 \
    rollout.model.motus.allow_batch_size=8 \
    runner.logger.experiment_name=\"\${EXP}\"

  echo
  echo \"============================================================\"
  echo \"END EVAL: \${NAME}\"
  echo \"TIME=\$(date)\"
  echo \"============================================================\"
  echo
}

run_eval \
  \"BASE_16EPS\" \
  \"null\" \
  \"robotwin_motus_eval_base_current_16eps\"

run_eval \
  \"STEP10_16EPS\" \
  \"${CKPT10}\" \
  \"robotwin_motus_eval_g8_step10_16eps\"

run_eval \
  \"STEP20_16EPS\" \
  \"${CKPT20}\" \
  \"robotwin_motus_eval_g8_step20_16eps\"

ray stop -f || python -m ray stop -f || true

echo
echo \"ALL EVALS FINISHED\"
echo \"TIME=\$(date)\"
" > "${LOG}" 2>&1 &

echo "PID=$!"
echo "LOG=${LOG}"
```

\[2\] 388526

PID=388526

LOG=logs/nohup/eval\_motus\_base\_step10\_step20\_16eps\_20260619\_231634\.log



TODO

明显效果问题



https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a321cdb\-fbc0\-83ea\-9f6b\-3135b9a9f8ed



**GRPO checkpoint 明显退化**

```Bash
结果：
checkpointsuccess_once / return成功数
base0.87514 / 16
step100.687511 / 16
step200.31255 / 16

因为 step20 已经明显差于 base，说明当前问题不是“训练太短所以没效果”，而是更像：
1. GRPO 更新方向可能有问题；
2. logprob / ratio 仍然不一致；
3. flow_sde 训练分布和 eval 分布差异过大；
4. 或者 checkpoint load / eval 随机性需要进一步确认。

加 checkpoint load 显式日志
确认 eval 时是否真的加载了 runner.ckpt_path，并打印 missing/unexpected。

回到代码排查
重点排查：
actor/ratio 偏低；
rollout old_logprob 和 actor new_logprob 是否同一 transition、同一 reduction；
flow_sde sigma=0.5 是否太强；
step20 checkpoint 是否被 over-update；
是否需要只 eval step10 或更小 lr。
```



### 6 work ppo

#### 每次进环境

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate
```



#### impl

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a364035\-f9a0\-83ea\-9ed9\-35fd57ef4a2f

放patch和脚本到 /root/autodl\-tmp

\[apply\_motus\_openpi\_alignment\.sh\]

\[motus\_openpi\_alignment\_repo\_root\.patch\]

用patch

```Bash
bash /root/autodl-tmp/apply_motus_openpi_alignment.sh \
  /root/autodl-tmp/motus_openpi_alignment_repo_root.patch
```

打包

```Bash
cd /root/autodl-tmp

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="/root/autodl-tmp/RLinf_code_only_${STAMP}.tgz"

tar -czf "${OUT}" \
  --exclude='RLinf/.venv' \
  --exclude='RLinf/.venv_*' \
  --exclude='RLinf/logs' \
  --exclude='RLinf/results' \
  --exclude='RLinf/.git' \
  --exclude='RLinf/__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.safetensors' \
  --exclude='*.ckpt' \
  RLinf

ls -lh "${OUT}"
```

\[RLinf\_code\_only\_20260623\_102004\.tgz\]



实现ppo：

放脚本和patch到tmp下

放patch

```Bash
bash /root/autodl-tmp/apply_motus_ppo_openpi_aligned.sh \
  /root/autodl-tmp/motus_ppo_openpi_aligned_repo_root.patch
```

打包

```Bash
cd /root/autodl-tmp

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="/root/autodl-tmp/RLinf_code_only_${STAMP}.tgz"

tar -czf "${OUT}" \
  --exclude='RLinf/.venv' \
  --exclude='RLinf/.venv_*' \
  --exclude='RLinf/logs' \
  --exclude='RLinf/results' \
  --exclude='RLinf/.git' \
  --exclude='RLinf/__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.safetensors' \
  --exclude='*.ckpt' \
  RLinf

ls -lh "${OUT}"
```

\[RLinf\_code\_only\_20260623\_115458\.tgz\]



用patch

\[motus\_ppo\_configs\_smoke\_formal\.patch\]



检查

```Bash
cd /root/autodl-tmp/RLinf

python -m py_compile \
  rlinf/models/embodiment/motus/motus_policy.py \
  third_party/Motus/models/motus.py
  
cd /root/autodl-tmp/RLinf

python - <<'PY'
import yaml
files = [
    "examples/embodiment/config/model/motus.yaml",
    "examples/embodiment/config/robotwin_adjust_bottle_motus_ppo_autodl.yaml",
    "examples/embodiment/config/robotwin_adjust_bottle_motus_ppo_smoke_autodl.yaml",
    "examples/embodiment/config/robotwin_adjust_bottle_motus_ppo_formal_autodl.yaml",
]
for f in files:
    with open(f) as fp:
        yaml.safe_load(fp)
    print("ok", f)
PY

cd /root/autodl-tmp/RLinf

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment

python - <<'PY'
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import os

config_dir = os.path.join(os.environ["EMBODIED_PATH"], "config")
with initialize_config_dir(version_base="1.1", config_dir=config_dir):
    cfg = compose(config_name="robotwin_adjust_bottle_motus_ppo_autodl")
print(OmegaConf.to_yaml(cfg.algorithm))
print("model_type:", cfg.actor.model.model_type)
print("add_value_head:", cfg.actor.model.add_value_head)
print("max_episode_steps:", cfg.env.train.max_episode_steps)
PY
```



加载检查

```Bash
cd /root/autodl-tmp/RLinf

export CUDA_VISIBLE_DEVICES=0
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:$PYTHONPATH

python - <<'PY'
from hydra import compose, initialize_config_dir
from rlinf.models import get_model
import os

config_dir = os.path.join(os.environ["EMBODIED_PATH"], "config")
with initialize_config_dir(version_base="1.1", config_dir=config_dir):
    cfg = compose(config_name="robotwin_adjust_bottle_motus_ppo_autodl")

model = get_model(cfg.actor.model)
print("loaded:", type(model))
print("has value_head:", hasattr(model, "value_head"))
print("num_action_chunks:", model.num_action_chunks)
print("action_dim:", model.action_dim)
print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, "M")
PY
```



Smoke train

```Bash
cd /root/autodl-tmp/RLinf

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:$PYTHONPATH

python examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_motus_ppo_smoke_autodl
```



报错

```Bash
**PPO rollout 成功了，失败发生在 actor 训练阶段的 ValueHead 前向；原因是 ValueHead 输入 value_features 是 float32，但 FSDP mixed precision 下 ValueHead 的 Linear 权重是 bfloat16。** 日志里关键链路是 default_forward -> _compute_values_from_features -> value_head -> Linear，最终报 mat1 and mat2 must have the same dtype, but got Float and BFloat16。
```



#### fix



修

\[motus\_ppo\_dtype\_and\_batch\_repo\_root\.patch\]

\[apply\_motus\_ppo\_dtype\_and\_batch\.sh\]



Smoke

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:$PYTHONPATH

python examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_motus_ppo_smoke_autodl
```



Formal

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:$PYTHONPATH
export RAY_DISABLE_DOCKER_CPU_WARNING=1

mkdir -p /root/autodl-tmp/RLinf/logs

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="/root/autodl-tmp/RLinf/logs/motus_ppo_formal_${STAMP}.log"

nohup python examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_motus_ppo_formal_autodl \
  > "${LOG}" 2>&1 &

echo "pid=$!"
echo "log=${LOG}"
tail -f "${LOG}"
```

Rollout时， 55G 36G



监视

\[monitor\_motus\_resources\.sh\]



报错：

```Bash
formal rollout 打包时 motus_vlm_input_ids 跨 chunk 长度不一致导致 stack 失败

还要把 PPO 输出目录从外层 results 改回 RLinf/logs

**forward_inputs 跨 rollout 时间步堆叠时，VLM token 长度不同**。日志里 EnvWorker 在 stack_list_of_dict_tensor() 里 torch.stack() 失败，报：
RuntimeError: stack expects each tensor to be equal size,
but got [16, 190] at entry 0 and [16, 188] at entry 10
对应上下文显示 rollout 阶段已经跑满 4 个 rollout epochs，Motus train batch inference 也在正常跑，shape 是 B=16，vlm_input_ids=(16,190)；失败发生在 EnvWorker 把 rollout trajectories / forward_inputs 汇总发给 actor 时。
```



修

\[motus\_vlm\_padding\_logs\_eval\_repo\_root\.patch\]

\[RLinf\_code\_only\_20260623\_142842\.tgz\]

\[apply\_motus\_vlm\_padding\_logs\_eval\.sh\]



改配置

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/motus@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor: 1
    env, rollout: 0

runner:
  task_type: embodied
  logger:
    log_path: "logs"
    project_name: rlinf
    experiment_name: "robotwin_motus_ppo_formal_autodl"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 20

  only_eval: False
  val_check_interval: 5
  save_interval: 5

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  group_size: 1
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level

  update_epoch: 2
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: "token-mean"

  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0

  gamma: 0.99
  gae_lambda: 0.95

  filter_rewards: False
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"
  enable_offload: True

  train:
    rollout_epoch: 4
    total_num_envs: 32
    reward_coef: ${algorithm.reward_coef}
    # Motus chunk is 16. 160 steps gives 10 policy decisions: longer than
    # OpenPI's 200/50=4 but safer for initial success on adjust_bottle.
    max_episode_steps: 160
    max_steps_per_rollout_epoch: 160
    group_size: ${algorithm.group_size}

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: False

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/train

    task_config:
      step_lim: 160
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

  eval:
    rollout_epoch: 1
    total_num_envs: 16
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 160
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 160
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: False

    video_cfg:
      save_video: True
      info_on_video: True
      video_base_dir: ${runner.logger.log_path}/video/eval

    task_config:
      step_lim: 160
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}
    model_type: ${actor.model.model_type}
    num_action_chunks: ${actor.model.num_action_chunks}
    action_dim: ${actor.model.action_dim}
    motus: ${actor.model.motus}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 16
  global_batch_size: 256
  seed: 1234
  enable_offload: False

  model:
    model_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
    precision: "bf16"

    model_type: "motus"
    is_lora: False
    lora_rank: 32
    load_to_device: False

    num_action_chunks: 16
    action_dim: 14
    use_proprio: True
    add_value_head: True

    motus:
      repo_path: "/root/autodl-tmp/Motus"
      policy_path: "/root/autodl-tmp/RLinf/third_party/Motus"
      checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
      wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
      vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"
      vlm_max_seq_len: 256
      config_path: "/root/autodl-tmp/RoboTwin/policy/Motus/utils/robotwin.yml"

      num_inference_timesteps: 10
      use_t5_cache: True
      save_predicted_frames: False

      allow_batch_size: 16
      batch_inference: True
      decode_video: False

      trainable: "action_expert"
      freeze_video_model: True
      freeze_vlm_model: True
      freeze_und_expert: True
      freeze_t5_encoder: True

      logprob_mode: "flow_sde"
      logprob_sigma: 0.5
      collect_denoise_step: "random"
      ignore_first: False
      ignore_last: False

      detach_critic_input: True
      value_feature: "action_tokens"

  optim:
    lr: 5.6e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
    use_orig_params: True
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False

weight_syncer:
  patch:
    init_sync:
      enabled: True

```



重新跑

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:$PYTHONPATH
export RAY_DISABLE_DOCKER_CPU_WARNING=1

mkdir -p /root/autodl-tmp/RLinf/logs

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="motus_ppo_formal_${STAMP}"
LOG="/root/autodl-tmp/RLinf/logs/${RUN_NAME}.log"

nohup python examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_motus_ppo_formal_autodl \
  > "${LOG}" 2>&1 &

echo "pid=$!"
echo "run=${RUN_NAME}"
echo "log=${LOG}"
tail -f "${LOG}"
```



日志还是错

```Bash
你说得对：我上一版把 runner.logger.log_path 改成 "logs" 是错的。之前那些正确目录：
/root/autodl-tmp/RLinf/logs/20260619-23:46:57-robotwin_adjust_bottle_motus_eval_batch8_autodl/tensorboard
不是 YAML 里直接写 log_path: logs 产生的，而是启动脚本 examples/embodiment/run_embodiment.sh 在运行时构造：
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
CMD="python ... --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
也就是说，**正确逻辑是：配置文件保持基线 ../results，训练启动时用 Hydra override 把 runner.logger.log_path 覆盖成带时间戳的 /root/autodl-tmp/RLinf/logs/<timestamp>-<config>。** 我之前直接把 config 改成 logs，会导致产物进 /root/autodl-tmp/RLinf/logs/tensorboard，少了一层 timestamp+config 目录。这个不对。
```



修

\[RLinf\_code\_only\_20260623\_150336\.tgz\]



配置

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/motus@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor: 1
    env, rollout: 0

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_motus_ppo_formal_autodl"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 20

  only_eval: False
  val_check_interval: 5
  save_interval: 5

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  group_size: 1
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level

  update_epoch: 2
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: "token-mean"

  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0

  gamma: 0.99
  gae_lambda: 0.95

  filter_rewards: False
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"
  enable_offload: True

  train:
    rollout_epoch: 4
    total_num_envs: 32
    reward_coef: ${algorithm.reward_coef}
    # Motus chunk is 16. 160 steps gives 10 policy decisions: longer than
    # OpenPI's 200/50=4 but safer for initial success on adjust_bottle.
    max_episode_steps: 160
    max_steps_per_rollout_epoch: 160
    group_size: ${algorithm.group_size}

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: False

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/train

    task_config:
      step_lim: 160
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

  eval:
    rollout_epoch: 1
    total_num_envs: 16
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 160
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 160
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: False

    video_cfg:
      save_video: True
      info_on_video: True
      video_base_dir: ${runner.logger.log_path}/video/eval

    task_config:
      step_lim: 160
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}
    model_type: ${actor.model.model_type}
    num_action_chunks: ${actor.model.num_action_chunks}
    action_dim: ${actor.model.action_dim}
    motus: ${actor.model.motus}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 16
  global_batch_size: 128
  seed: 1234
  enable_offload: False

  model:
    model_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
    precision: "bf16"

    model_type: "motus"
    is_lora: False
    lora_rank: 32
    load_to_device: False

    num_action_chunks: 16
    action_dim: 14
    use_proprio: True
    add_value_head: True

    motus:
      repo_path: "/root/autodl-tmp/Motus"
      policy_path: "/root/autodl-tmp/RLinf/third_party/Motus"
      checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
      wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
      vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"
      vlm_max_seq_len: 256
      config_path: "/root/autodl-tmp/RoboTwin/policy/Motus/utils/robotwin.yml"

      num_inference_timesteps: 10
      use_t5_cache: True
      save_predicted_frames: False

      allow_batch_size: 16
      batch_inference: True
      decode_video: False

      trainable: "action_expert"
      freeze_video_model: True
      freeze_vlm_model: True
      freeze_und_expert: True
      freeze_t5_encoder: True

      logprob_mode: "flow_sde"
      logprob_sigma: 0.5
      collect_denoise_step: "random"
      ignore_first: False
      ignore_last: False

      detach_critic_input: True
      value_feature: "action_tokens"

  optim:
    lr: 5.6e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
    use_orig_params: True
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False

weight_syncer:
  patch:
    init_sync:
      enabled: True

```



跑

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:$PYTHONPATH
export RAY_DISABLE_DOCKER_CPU_WARNING=1

CONFIG_NAME="robotwin_adjust_bottle_motus_ppo_formal_autodl"
RUN_NAME="$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"

mkdir -p "${LOG_DIR}"

CMD=(
  python /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_LOG}"
echo >> "${RUN_LOG}"

nohup "${CMD[@]}" >> "${RUN_LOG}" 2>&1 &

echo "pid=$!"
echo "run_name=${RUN_NAME}"
echo "log_dir=${LOG_DIR}"
echo "run_log=${RUN_LOG}"
tail -f "${RUN_LOG}"
```

峰值65 37？



看板 

http://127\.0\.0\.1:8265/\#/cluster



报错

```Bash
ValueError: Motus adapter supports batch_size <= 16, got 32.
日志里也能看到 env/rollout 初始化时 train batch 是 32：
dst_rank_map: {'rollout_train': [(0, 32)], ...}
然后 rollout 调 predict_action_batch() 时因为 allow_batch_size: 16 直接报错。

1. 为什么会是 batch_size=32
当前你的 formal 配置里是：
env:
  train:
    total_num_envs: 32

rollout:
  pipeline_stage_num: 1

cluster:
  component_placement:
    actor: 1
    env, rollout: 0

actor:
  model:
    motus:
      allow_batch_size: 16
这里只有 **1 个 rollout worker**，所以 rollout worker 一次收到的 train env batch 就是：
total_num_envs / rollout_world_size / pipeline_stage_num
= 32 / 1 / 1
= 32
但 Motus adapter 配置只允许：
allow_batch_size: 16


```



改：env\.train\.total\_num\_envs: 16

其他改：`eval.rollout_epoch: 2`、`eval.total_num_envs: 8`



配置

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/motus@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor: 1
    env, rollout: 0

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_motus_ppo_formal_autodl"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 20

  only_eval: False
  val_check_interval: 5
  save_interval: 5

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  group_size: 1
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level

  update_epoch: 2
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: "token-mean"

  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0

  gamma: 0.99
  gae_lambda: 0.95

  filter_rewards: False
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"
  enable_offload: True

  train:
    rollout_epoch: 4
    total_num_envs: 16
    reward_coef: ${algorithm.reward_coef}
    # Motus chunk is 16. 160 steps gives 10 policy decisions: longer than
    # OpenPI's 200/50=4 but safer for initial success on adjust_bottle.
    max_episode_steps: 160
    max_steps_per_rollout_epoch: 160
    group_size: ${algorithm.group_size}

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: False

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/train

    task_config:
      step_lim: 160
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

  eval:
    rollout_epoch: 2
    total_num_envs: 8
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 160
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 160
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: False

    video_cfg:
      save_video: True
      info_on_video: True
      video_base_dir: ${runner.logger.log_path}/video/eval

    task_config:
      step_lim: 160
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}
    model_type: ${actor.model.model_type}
    num_action_chunks: ${actor.model.num_action_chunks}
    action_dim: ${actor.model.action_dim}
    motus: ${actor.model.motus}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 16
  global_batch_size: 128
  seed: 1234
  enable_offload: False

  model:
    model_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
    precision: "bf16"

    model_type: "motus"
    is_lora: False
    lora_rank: 32
    load_to_device: False

    num_action_chunks: 16
    action_dim: 14
    use_proprio: True
    add_value_head: True

    motus:
      repo_path: "/root/autodl-tmp/Motus"
      policy_path: "/root/autodl-tmp/RLinf/third_party/Motus"
      checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
      wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
      vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"
      vlm_max_seq_len: 256
      config_path: "/root/autodl-tmp/RoboTwin/policy/Motus/utils/robotwin.yml"

      num_inference_timesteps: 10
      use_t5_cache: True
      save_predicted_frames: False

      allow_batch_size: 16
      batch_inference: True
      decode_video: False

      trainable: "action_expert"
      freeze_video_model: True
      freeze_vlm_model: True
      freeze_und_expert: True
      freeze_t5_encoder: True

      logprob_mode: "flow_sde"
      logprob_sigma: 0.5
      collect_denoise_step: "random"
      ignore_first: False
      ignore_last: False

      detach_critic_input: True
      value_feature: "action_tokens"

  optim:
    lr: 5.6e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
    use_orig_params: True
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False

weight_syncer:
  patch:
    init_sync:
      enabled: True

```



再跑

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:$PYTHONPATH
export RAY_DISABLE_DOCKER_CPU_WARNING=1

CONFIG_NAME="robotwin_adjust_bottle_motus_ppo_formal_autodl"
RUN_NAME="$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"

mkdir -p "${LOG_DIR}"

CMD=(
  python /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_LOG}"
echo >> "${RUN_LOG}"

nohup "${CMD[@]}" >> "${RUN_LOG}" 2>&1 &

echo "pid=$!"
echo "run_name=${RUN_NAME}"
echo "log_dir=${LOG_DIR}"
echo "run_log=${RUN_LOG}"
tail -f "${RUN_LOG}"
```

峰值，48 37？



rollout后开始训练时爆显存

```Bash
最高相关：actor.micro_batch_size
这就是 actor 一次 forward/backward 的样本数。现在 OOM 发生在 actor micro-batch replay Motus denoise 时，所以它是第一调节项。
OpenPI PPO 是 16，但 Motus 是 8B WAM，还带 video/action chain replay。你说“Motus 取 OpenPI 的 1/2”，这个判断现在是对的：**先设 8**。
```



#### 调效果



##### 1

改

micro\_batch\_size: 8

global\_batch\_size: 64



目前配置

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/motus@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor: 1
    env, rollout: 0

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_motus_ppo_formal_autodl"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 20

  only_eval: False
  val_check_interval: 5
  save_interval: 5

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  group_size: 1
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level

  update_epoch: 2
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: "token-mean"

  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0

  gamma: 0.99
  gae_lambda: 0.95

  filter_rewards: False
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"
  enable_offload: True

  train:
    rollout_epoch: 4
    total_num_envs: 16
    reward_coef: ${algorithm.reward_coef}
    # Motus chunk is 16. 160 steps gives 10 policy decisions: longer than
    # OpenPI's 200/50=4 but safer for initial success on adjust_bottle.
    max_episode_steps: 160
    max_steps_per_rollout_epoch: 160
    group_size: ${algorithm.group_size}

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: False

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/train

    task_config:
      step_lim: 160
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

  eval:
    rollout_epoch: 2
    total_num_envs: 8
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 160
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 160
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: False

    video_cfg:
      save_video: True
      info_on_video: True
      video_base_dir: ${runner.logger.log_path}/video/eval

    task_config:
      step_lim: 160
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}
    model_type: ${actor.model.model_type}
    num_action_chunks: ${actor.model.num_action_chunks}
    action_dim: ${actor.model.action_dim}
    motus: ${actor.model.motus}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 8
  global_batch_size: 64
  seed: 1234
  enable_offload: False

  model:
    model_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
    precision: "bf16"

    model_type: "motus"
    is_lora: False
    lora_rank: 32
    load_to_device: False

    num_action_chunks: 16
    action_dim: 14
    use_proprio: True
    add_value_head: True

    motus:
      repo_path: "/root/autodl-tmp/Motus"
      policy_path: "/root/autodl-tmp/RLinf/third_party/Motus"
      checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
      wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
      vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"
      vlm_max_seq_len: 256
      config_path: "/root/autodl-tmp/RoboTwin/policy/Motus/utils/robotwin.yml"

      num_inference_timesteps: 10
      use_t5_cache: True
      save_predicted_frames: False

      allow_batch_size: 16
      batch_inference: True
      decode_video: False

      trainable: "action_expert"
      freeze_video_model: True
      freeze_vlm_model: True
      freeze_und_expert: True
      freeze_t5_encoder: True

      logprob_mode: "flow_sde"
      logprob_sigma: 0.5
      collect_denoise_step: "random"
      ignore_first: False
      ignore_last: False

      detach_critic_input: True
      value_feature: "action_tokens"

  optim:
    lr: 5.6e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
    use_orig_params: True
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False

weight_syncer:
  patch:
    init_sync:
      enabled: True

```



再跑

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:$PYTHONPATH
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_adjust_bottle_motus_ppo_formal_autodl"
RUN_NAME="$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"

mkdir -p "${LOG_DIR}"

CMD=(
  python /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_LOG}"
echo >> "${RUN_LOG}"

nohup "${CMD[@]}" >> "${RUN_LOG}" 2>&1 &

echo "pid=$!"
echo "run_name=${RUN_NAME}"
echo "log_dir=${LOG_DIR}"
echo "run_log=${RUN_LOG}"
tail -f "${RUN_LOG}"
```

训练时，56 63 G峰值



时间统计

```Bash
大部分时间在 rollout，不在 actor training。
典型 step：
generate_rollouts: 1390 ~ 1495s
actor/run_training: 155 ~ 162s
eval: 约 405s，只在 step 5 / 10 出现
sync_weights: 0.6 ~ 1.7s
```



训练过程

```Plain Text
20 step train success_once 
1: 0.484
2: 0.469
3: 0.406
4: 0.594
5: 0.531
6: 0.531
7: 0.344
8: 0.531
9: 0.453
10: 0.453
11: 0.344
12: 0.484
13: 0.328
14: 0.406
15: 0.453
16: 0.438
17: 0.391
18: 0.453
19: 0.359
20: 0.500
成功率波动

训练内置 eval 是每 5 step 一次，每次 16 条 eval trajectory：
step 5  eval success_once / success_at_end = 0.5625
step 10 eval success_once / success_at_end = 0.625
step 15 eval success_once / success_at_end = 0.625
step 20 eval success_once / success_at_end = 0.5625
没提升

```



排队评估脚本

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}

mkdir -p logs/nohup

RUN_ROOT="/root/autodl-tmp/RLinf/logs/20260623-16:18:22-robotwin_adjust_bottle_motus_ppo_formal_autodl"
CKPT_DIR="${RUN_ROOT}/robotwin_motus_ppo_formal_autodl/checkpoints"

EVAL_CFG="/root/autodl-tmp/RLinf/evaluations/robotwin/robotwin_adjust_bottle_motus_eval_batch8_autodl.yaml"

test -d "${RUN_ROOT}"
test -f "${EVAL_CFG}"

STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_SCRIPT="logs/nohup/eval_motus_ppo_base_step5_step10_step15_step20_64eps_${STAMP}.sh"
LOG="logs/nohup/eval_motus_ppo_base_step5_step10_step15_step20_64eps_${STAMP}.log"

cat > "${EVAL_SCRIPT}" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/rlinf_env.sh

cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1

RUN_ROOT="/root/autodl-tmp/RLinf/logs/20260623-16:18:22-robotwin_adjust_bottle_motus_ppo_formal_autodl"
CKPT_DIR="${RUN_ROOT}/robotwin_motus_ppo_formal_autodl/checkpoints"

EVAL_CFG="/root/autodl-tmp/RLinf/evaluations/robotwin/robotwin_adjust_bottle_motus_eval_batch8_autodl.yaml"

test -d "${RUN_ROOT}"
test -f "${EVAL_CFG}"

CKPT5="${CKPT_DIR}/global_step_5/actor/model_state_dict/full_weights.pt"
CKPT10="${CKPT_DIR}/global_step_10/actor/model_state_dict/full_weights.pt"
CKPT15="${CKPT_DIR}/global_step_15/actor/model_state_dict/full_weights.pt"
CKPT20="${CKPT_DIR}/global_step_20/actor/model_state_dict/full_weights.pt"

wait_for_file() {
  local F="$1"
  local NAME="$2"
  local MAX_TRIES="${3:-720}"  # 720 * 30s = 6h

  echo
  echo "[wait_file] ${NAME}"
  echo "[wait_file] path=${F}"

  for i in $(seq 1 "${MAX_TRIES}"); do
    if [ -s "${F}" ]; then
      echo "[wait_file] found ${NAME}"
      ls -lh "${F}"
      return 0
    fi
    echo "[wait_file] ${NAME} not ready, retry ${i}/${MAX_TRIES}"
    sleep 30
  done

  echo "[wait_file] timeout waiting for ${NAME}: ${F}"
  return 1
}

wait_until_training_done() {
  echo
  echo "[wait_train] waiting for PPO training process to finish..."

  for i in $(seq 1 720); do
    TRAIN_PIDS="$(pgrep -f 'train_embodied_agent.py.*robotwin_adjust_bottle_motus_ppo_formal_autodl' || true)"

    if [ -z "${TRAIN_PIDS}" ]; then
      echo "[wait_train] training process is gone."
      return 0
    fi

    echo "[wait_train] still running, retry ${i}/720"
    echo "TRAIN_PIDS=${TRAIN_PIDS}"
    sleep 30
  done

  echo "[wait_train] timeout waiting for training process."
  return 1
}

wait_until_gpu_free() {
  echo
  echo "[wait_gpu] checking Ray/eval/GPU cleanup..."

  for i in $(seq 1 180); do
    RAY_PIDS="$(pgrep -f 'ray::|raylet|gcs_server|dashboard_agent|dashboard.py' || true)"
    EVAL_PIDS="$(pgrep -f 'train_embodied_agent.py|evaluations/run_eval.sh' || true)"
    GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+' || true)"

    if [ -z "${RAY_PIDS}" ] && [ -z "${EVAL_PIDS}" ] && [ -z "${GPU_PIDS}" ]; then
      echo "[wait_gpu] Ray/eval/GPU clean."
      nvidia-smi
      return 0
    fi

    echo "[wait_gpu] not clean yet, retry ${i}/180"
    echo "RAY_PIDS=${RAY_PIDS:-none}"
    echo "EVAL_PIDS=${EVAL_PIDS:-none}"
    echo "GPU_PIDS=${GPU_PIDS:-none}"
    sleep 10
  done

  echo "[wait_gpu] timeout waiting for cleanup."
  nvidia-smi
  return 1
}

run_eval() {
  local NAME="$1"
  local CKPT="$2"
  local EXP="$3"

  echo
  echo "============================================================"
  echo "START EVAL ${NAME}"
  echo "TIME=$(date)"
  echo "CKPT=${CKPT}"
  echo "EXP=${EXP}"
  echo "============================================================"
  echo

  ray stop -f || python -m ray stop -f || true
  wait_until_gpu_free

  set +e

  if [ "${CKPT}" = "null" ]; then
    bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_motus_eval_batch8_autodl \
      runner.ckpt_path=null \
      env.eval.total_num_envs=8 \
      env.eval.rollout_epoch=8 \
      rollout.model.motus.allow_batch_size=8 \
      runner.logger.experiment_name="${EXP}"
  else
    test -s "${CKPT}"
    bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_motus_eval_batch8_autodl \
      runner.ckpt_path="${CKPT}" \
      env.eval.total_num_envs=8 \
      env.eval.rollout_epoch=8 \
      rollout.model.motus.allow_batch_size=8 \
      runner.logger.experiment_name="${EXP}"
  fi

  local RC=$?

  set -e

  echo
  echo "============================================================"
  echo "END EVAL ${NAME}"
  echo "TIME=$(date)"
  echo "RETURN_CODE=${RC}"
  echo "============================================================"
  echo

  ray stop -f || python -m ray stop -f || true
  wait_until_gpu_free

  if [ "${RC}" -ne 0 ]; then
    echo "[error] ${NAME} failed with return code ${RC}; stop sequence."
    exit "${RC}"
  fi
}

echo "============================================================"
echo "MOTUS PPO EVAL QUEUE"
echo "TIME=$(date)"
echo "RUN_ROOT=${RUN_ROOT}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "============================================================"

test -f "${EVAL_CFG}"

# 等训练完成，避免和正式训练抢 GPU。
wait_for_file "${CKPT5}" "STEP5"
wait_for_file "${CKPT10}" "STEP10"
wait_for_file "${CKPT15}" "STEP15"
wait_for_file "${CKPT20}" "STEP20"
wait_until_training_done
ray stop -f || python -m ray stop -f || true
wait_until_gpu_free

run_eval \
  "BASE_64EPS" \
  "null" \
  "robotwin_motus_ppo_eval_base_64eps"

run_eval \
  "STEP5_64EPS" \
  "${CKPT5}" \
  "robotwin_motus_ppo_eval_step5_64eps"

run_eval \
  "STEP10_64EPS" \
  "${CKPT10}" \
  "robotwin_motus_ppo_eval_step10_64eps"

run_eval \
  "STEP15_64EPS" \
  "${CKPT15}" \
  "robotwin_motus_ppo_eval_step15_64eps"

run_eval \
  "STEP20_64EPS" \
  "${CKPT20}" \
  "robotwin_motus_ppo_eval_step20_64eps"

echo
echo "ALL EVALS FINISHED"
echo "TIME=$(date)"
BASH

chmod +x "${EVAL_SCRIPT}"

nohup bash "${EVAL_SCRIPT}" > "${LOG}" 2>&1 &

echo "PID=$!"
echo "SCRIPT=${EVAL_SCRIPT}"
echo "LOG=${LOG}"
tail -f "${LOG}"
```

排队评估行为正常

BASE ，64 episodes ， 0\.625



评估报错：value也被加载

```Plain Text
中断发生在 STEP5 eval，原因是加载 PPO checkpoint 时出现：
Unexpected key(s) in state_dict:
"value_head.mlp.0.weight", ...
日志里明确显示 huggingface_worker.py 在 init_worker 里调用 self.hf_model.load_state_dict(model_dict)，然后 PyTorch strict load 因为 checkpoint 有 value_head.*，而当前 eval 模型没有 value_head，所以报 Unexpected key(s)。

eval 配置里 rollout.model.add_value_head 是 false。日志里 STEP5 eval 的配置打印显示：
"add_value_head": false
而 runner.ckpt_path 指向的是 PPO step5 的 full_weights.pt。
PPO 训练时 add_value_head=True，所以 checkpoint 里包含：
value_head.mlp.0.weight
value_head.mlp.0.bias
...
eval 时 add_value_head=False，模型结构没有 self.value_head，因此 strict load 认为这些 key 是 unexpected。
这不是“训练保存了多余 value”。PPO checkpoint 正确保存 value_head，因为 PPO 训练需要它。问题是：**加载 PPO checkpoint 的 eval 模型结构没有对齐训练结构。**

当前 OpenPI 的通用 eval 配置里有 add_value_head: True 的版本，目的就是让 eval model 能加载 PPO/value-head checkpoint；而一些 SFT/autodl eval config 设为 False，只适合不带 value_head 的 SFT/base checkpoint。
目前 Motus 的 PPO eval 没有对齐这一点：我们用了原来的 Motus SFT/base eval config，它默认 add_value_head=False，却拿它加载 PPO checkpoint。这个点当前实现**没有和 OpenPI 的 PPO checkpoint eval 习惯对齐**。
```

评估脚本，该参数要设为True



结论，参数不好，更新太剧烈



##### 2

调参

Train

/root/autodl\-tmp/RLinf/examples/embodiment/config/robotwin\_adjust\_bottle\_motus\_ppo\_formal\_autodl\.yaml

```Plain Text
algorithm.update_epoch: 1
2 -> 1

actor.optim.lr: 2.0e-6
5.6e-6 -> 2.0e-6

actor.global_batch_size: 80

env.train.total_num_envs: 16
保持

env.train.rollout_epoch: 1
4 -> 1

env.eval.video_cfg.save_video: False

env.eval.video_cfg.info_on_video: False
```

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/motus@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor: 1
    env, rollout: 0

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_motus_ppo_formal_autodl"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 20

  only_eval: False
  val_check_interval: 5
  save_interval: 5

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  group_size: 1
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level

  update_epoch: 1
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: "token-mean"

  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0

  gamma: 0.99
  gae_lambda: 0.95

  filter_rewards: False
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"
  enable_offload: True

  train:
    rollout_epoch: 1
    total_num_envs: 16
    reward_coef: ${algorithm.reward_coef}
    # Motus chunk is 16. 160 steps gives 10 policy decisions: longer than
    # OpenPI's 200/50=4 but safer for initial success on adjust_bottle.
    max_episode_steps: 160
    max_steps_per_rollout_epoch: 160
    group_size: ${algorithm.group_size}

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: False

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/train

    task_config:
      step_lim: 160
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

  eval:
    rollout_epoch: 2
    total_num_envs: 8
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 160
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 160
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: False

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/eval

    task_config:
      step_lim: 160
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}
    model_type: ${actor.model.model_type}
    num_action_chunks: ${actor.model.num_action_chunks}
    action_dim: ${actor.model.action_dim}
    motus: ${actor.model.motus}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 8
  global_batch_size: 80
  seed: 1234
  enable_offload: False

  model:
    model_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
    precision: "bf16"

    model_type: "motus"
    is_lora: False
    lora_rank: 32
    load_to_device: False

    num_action_chunks: 16
    action_dim: 14
    use_proprio: True
    add_value_head: True

    motus:
      repo_path: "/root/autodl-tmp/Motus"
      policy_path: "/root/autodl-tmp/RLinf/third_party/Motus"
      checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
      wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
      vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"
      vlm_max_seq_len: 256
      config_path: "/root/autodl-tmp/RoboTwin/policy/Motus/utils/robotwin.yml"

      num_inference_timesteps: 10
      use_t5_cache: True
      save_predicted_frames: False

      allow_batch_size: 16
      batch_inference: True
      decode_video: False

      trainable: "action_expert"
      freeze_video_model: True
      freeze_vlm_model: True
      freeze_und_expert: True
      freeze_t5_encoder: True

      logprob_mode: "flow_sde"
      logprob_sigma: 0.5
      collect_denoise_step: "random"
      ignore_first: False
      ignore_last: False

      detach_critic_input: True
      value_feature: "action_tokens"

  optim:
    lr: 2.0e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
    use_orig_params: True
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False

weight_syncer:
  patch:
    init_sync:
      enabled: True

```

Eval

/root/autodl\-tmp/RLinf/evaluations/robotwin/robotwin\_adjust\_bottle\_motus\_eval\_batch8\_autodl\.yaml

```Plain Text
添加：
rollout.model.add_value_head=True

env.eval.rollout_epoch: 8

env.eval.total_num_envs: 8
```

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.eval
  - model/motus@rollout.model
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    env, rollout: 0

runner:
  task_type: embodied_eval
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_motus_eval_batch8_autodl"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: -1

  only_eval: True
  val_check_interval: -1
  save_interval: -1

  resume_dir: null
  ckpt_path: null

env:
  group_name: "EnvGroup"
  enable_offload: True

  eval:
    rollout_epoch: 8
    total_num_envs: 8
    auto_reset: True
    ignore_terminations: True

    max_episode_steps: 400
    max_steps_per_rollout_epoch: 400

    reward_coef: 1.0
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/eval

    center_crop: False
    task_config:
      step_lim: 400
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
  model:
    add_value_head: True
```



再训

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_adjust_bottle_motus_ppo_formal_autodl"
CONFIG_PATH="/root/autodl-tmp/RLinf/examples/embodiment/config"
TRAIN_CFG="${CONFIG_PATH}/${CONFIG_NAME}.yaml"

test -f "${TRAIN_CFG}"
test -d "/root/autodl-tmp/RoboTwin_RLinf"
test -f "/root/autodl-tmp/RLinf/rlinf/envs/robotwin/seeds/train_seeds.json"
test -f "/root/autodl-tmp/RLinf/rlinf/envs/robotwin/seeds/eval_seeds.json"
test -d "/root/autodl-tmp/models/motus/Motus_robotwin2"
test -d "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
test -d "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"

echo "===== important config ====="
grep -nE "max_steps:|val_check_interval|save_interval|rollout_epoch:|total_num_envs:|max_episode_steps:|update_epoch:|micro_batch_size:|global_batch_size:|lr:|value_lr:|save_video:|vlm_max_seq_len|allow_batch_size" \
  "${TRAIN_CFG}"

ray stop --force || true

RUN_NAME="$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}-re1_ue1_lr2e6_gbs80"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"

mkdir -p "${LOG_DIR}"

CMD=(
  python /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py
  --config-path "${CONFIG_PATH}"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_LOG}"
echo >> "${RUN_LOG}"

nohup "${CMD[@]}" >> "${RUN_LOG}" 2>&1 &

PID=$!

echo "pid=${PID}"
echo "run_name=${RUN_NAME}"
echo "log_dir=${LOG_DIR}"
echo "run_log=${RUN_LOG}"

if [ -f /root/autodl-tmp/monitor_motus_resources.sh ]; then
  nohup bash /root/autodl-tmp/monitor_motus_resources.sh 5 "${RUN_NAME}" \
    > "${LOG_DIR}/${RUN_NAME}.monitor.out" 2>&1 &
  echo "monitor_pid=$!"
  echo "monitor_log=${LOG_DIR}/${RUN_NAME}.monitor.out"
fi

tail -f "${RUN_LOG}"
```



训练分析

```Plain Text
20 step 的训练 success_once 大致是：
1   0.4375
2   0.3750
3   0.5625
4   0.3750
5   0.4375
6   0.3750
7   0.3750
8   0.6875
9   0.5000
10  0.2500
11  0.3750
12  0.3750
13  0.4375
14  0.5000
15  0.6250
16  0.5625
17  0.5625
18  0.5625
19  0.3750
20  0.5000

Actor 指标：KL 下降了一点，但仍然偏高
5.1 approx_kl
这轮 approx_kl：
mean ≈ 94.36
min  ≈ 16.97
max  ≈ 234.2
last ≈ 118.3
相比上一轮均值约 116，有下降，但并不彻底。仍然有几次高峰：
step 6   180.4
step 7   176.6
step 12  181.8
step 13  234.2
step 19  174.9
这说明：
降低 lr / update_epoch / rollout_epoch 后，更新强度有所下降；
但 Motus chunk-level flow-SDE logprob 的 ratio 尺度仍然很大；
PPO 仍然经常处于强 clipping 状态。

5.2 clip_fraction
这轮 clip_fraction：
mean ≈ 0.525
min  ≈ 0.350
max  ≈ 0.625
last ≈ 0.512
这个和上一轮差不多，仍然偏高。理想上如果 PPO 更新温和，clip_fraction 不应长期接近 0.5。现在说明约一半样本仍然被 clip。
5.3 ratio
actor/ratio 多数低于 0.5：
mean ≈ 0.39
min  ≈ 0.145
max  ≈ 1.371
last ≈ 0.38
这说明新策略对旧 rollout transition 的概率经常显著变低。它和高 KL、高 clip_fraction 一致。

6. Critic 指标：value head 在工作，但 critic 仍弱
6.1 value_loss
mean ≈ 0.115
min  ≈ 0.072
max  ≈ 0.353
last ≈ 0.105
step1 value_loss 最高，之后基本稳定在 0.07–0.12。这说明 value head 的 loss 没有爆炸，也在正常优化。
6.2 explained_variance
mean ≈ -2.26
min  ≈ -9.15
max  ≈ 0.19
last ≈ -0.568
多数为负。好消息是 step16、step17 出现过略正：
step16  0.049
step17  0.190
但整体仍然说明 critic 预测质量不可靠。
目前判断：
value head 已经接通；
value_loss 数值稳定；
critic 还没有形成稳定可用 baseline；
sparse reward + 小 batch 16 trajectories/update 会让 explained_variance 很 noisy。

5. KL 均值略降。
6. critic 极端负 explained_variance 比上一轮少一些。



```



每步成功率

```Plain Text
1   0.4375
2   0.3750
3   0.5625
4   0.3750
5   0.4375
6   0.3750
7   0.3750
8   0.6875
9   0.5000
10  0.2500
11  0.3750
12  0.3750
13  0.4375
14  0.5000
15  0.6250
16  0.5625
17  0.5625
18  0.5625
19  0.3750
20  0.5000
```

聚合

|1 \- 10|11–20|
|---|---|
|0\.4375|0\.4875|

|1\-5|6\-10|11\-15|15\-20|
|---|---|---|---|
|0\.4375|0\.4375|0\.4625|0\.5125|

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OWYzMDFmMjhmODJmNmM0YjY3ZjliOWIzZWM2ZjYxYTZfNGY3MDdkODQ2OTE4NDgyZDZmNzEwMWI5ZGRiMWY2OWRfSUQ6NzY1NDgxODgwNDI0NTI2OTcyMF8xNzg1MTYwMjgwOjE3ODUyNDY2ODBfVjM)



5 10 15 20 排队评估

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1

mkdir -p logs/nohup

RUN_ROOT="/root/autodl-tmp/RLinf/logs/20260624-08:58:54-robotwin_adjust_bottle_motus_ppo_formal_autodl-re1_ue1_lr2e6_gbs80"
CKPT_DIR="${RUN_ROOT}/robotwin_motus_ppo_formal_autodl/checkpoints"

EVAL_CFG="/root/autodl-tmp/RLinf/evaluations/robotwin/robotwin_adjust_bottle_motus_eval_batch8_autodl.yaml"

CKPT5="${CKPT_DIR}/global_step_5/actor/model_state_dict/full_weights.pt"
CKPT10="${CKPT_DIR}/global_step_10/actor/model_state_dict/full_weights.pt"
CKPT15="${CKPT_DIR}/global_step_15/actor/model_state_dict/full_weights.pt"
CKPT20="${CKPT_DIR}/global_step_20/actor/model_state_dict/full_weights.pt"

test -d "${RUN_ROOT}"
test -f "${EVAL_CFG}"
test -s "${CKPT5}"
test -s "${CKPT10}"
test -s "${CKPT15}"
test -s "${CKPT20}"

ls -lh "${CKPT5}" "${CKPT10}" "${CKPT15}" "${CKPT20}"

STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_SCRIPT="logs/nohup/eval_motus_ppo_re1_ue1_lr2e6_gbs80_step5_step10_step15_step20_64eps_${STAMP}.sh"
LOG="logs/nohup/eval_motus_ppo_re1_ue1_lr2e6_gbs80_step5_step10_step15_step20_64eps_${STAMP}.log"

cat > "${EVAL_SCRIPT}" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/rlinf_env.sh

cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1

RUN_ROOT="/root/autodl-tmp/RLinf/logs/20260624-08:58:54-robotwin_adjust_bottle_motus_ppo_formal_autodl-re1_ue1_lr2e6_gbs80"
CKPT_DIR="${RUN_ROOT}/robotwin_motus_ppo_formal_autodl/checkpoints"

EVAL_CFG="/root/autodl-tmp/RLinf/evaluations/robotwin/robotwin_adjust_bottle_motus_eval_batch8_autodl.yaml"

CKPT5="${CKPT_DIR}/global_step_5/actor/model_state_dict/full_weights.pt"
CKPT10="${CKPT_DIR}/global_step_10/actor/model_state_dict/full_weights.pt"
CKPT15="${CKPT_DIR}/global_step_15/actor/model_state_dict/full_weights.pt"
CKPT20="${CKPT_DIR}/global_step_20/actor/model_state_dict/full_weights.pt"

test -d "${RUN_ROOT}"
test -f "${EVAL_CFG}"
test -s "${CKPT5}"
test -s "${CKPT10}"
test -s "${CKPT15}"
test -s "${CKPT20}"

wait_until_gpu_free() {
  echo
  echo "[wait_gpu] checking Ray/eval/GPU cleanup..."

  for i in $(seq 1 180); do
    RAY_PIDS="$(pgrep -f 'ray::|raylet|gcs_server|dashboard_agent|dashboard.py' || true)"
    EVAL_PIDS="$(pgrep -f 'train_embodied_agent.py|eval_embodied_agent.py|evaluations/run_eval.sh' || true)"
    GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+' || true)"

    if [ -z "${RAY_PIDS}" ] && [ -z "${EVAL_PIDS}" ] && [ -z "${GPU_PIDS}" ]; then
      echo "[wait_gpu] Ray/eval/GPU clean."
      nvidia-smi
      return 0
    fi

    echo "[wait_gpu] not clean yet, retry ${i}/180"
    echo "RAY_PIDS=${RAY_PIDS:-none}"
    echo "EVAL_PIDS=${EVAL_PIDS:-none}"
    echo "GPU_PIDS=${GPU_PIDS:-none}"
    sleep 10
  done

  echo "[wait_gpu] timeout waiting for cleanup."
  nvidia-smi
  return 1
}

run_eval() {
  local NAME="$1"
  local CKPT="$2"
  local EXP="$3"

  echo
  echo "============================================================"
  echo "START EVAL ${NAME}"
  echo "TIME=$(date)"
  echo "CKPT=${CKPT}"
  echo "EXP=${EXP}"
  echo "============================================================"
  echo

  test -s "${CKPT}"

  ray stop -f || python -m ray stop -f || true
  wait_until_gpu_free

  set +e

  bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_motus_eval_batch8_autodl \
    runner.ckpt_path="${CKPT}" \
    env.eval.total_num_envs=8 \
    env.eval.rollout_epoch=8 \
    rollout.model.add_value_head=True \
    rollout.model.motus.allow_batch_size=8 \
    rollout.model.motus.vlm_max_seq_len=256 \
    runner.logger.experiment_name="${EXP}"

  local RC=$?

  set -e

  echo
  echo "============================================================"
  echo "END EVAL ${NAME}"
  echo "TIME=$(date)"
  echo "RETURN_CODE=${RC}"
  echo "============================================================"
  echo

  ray stop -f || python -m ray stop -f || true
  wait_until_gpu_free

  if [ "${RC}" -ne 0 ]; then
    echo "[error] ${NAME} failed with return code ${RC}; stop sequence."
    exit "${RC}"
  fi
}

echo "============================================================"
echo "MOTUS PPO V2 EVAL QUEUE"
echo "TIME=$(date)"
echo "RUN_ROOT=${RUN_ROOT}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "============================================================"

run_eval \
  "STEP5_64EPS" \
  "${CKPT5}" \
  "robotwin_motus_ppo_re1_ue1_lr2e6_gbs80_eval_step5_64eps"

run_eval \
  "STEP10_64EPS" \
  "${CKPT10}" \
  "robotwin_motus_ppo_re1_ue1_lr2e6_gbs80_eval_step10_64eps"

run_eval \
  "STEP15_64EPS" \
  "${CKPT15}" \
  "robotwin_motus_ppo_re1_ue1_lr2e6_gbs80_eval_step15_64eps"

run_eval \
  "STEP20_64EPS" \
  "${CKPT20}" \
  "robotwin_motus_ppo_re1_ue1_lr2e6_gbs80_eval_step20_64eps"

echo
echo "ALL EVALS FINISHED"
echo "TIME=$(date)"
BASH

chmod +x "${EVAL_SCRIPT}"

nohup bash "${EVAL_SCRIPT}" > "${LOG}" 2>&1 &

echo "PID=$!"
echo "SCRIPT=${EVAL_SCRIPT}"
echo "LOG=${LOG}"
tail -f "${LOG}"
```

没变化

没效果



##### 3

改

训练参数

```Plain Text
runner.max_steps: 100

runner.val_check_interval: 20

runner.save_interval: 20

env.train.rollout_epoch: 2

env.train.max_episode_steps: 192

env.train.max_steps_per_rollout_epoch: 192

env.train.task_config.step_lim: 192

env.eval.max_episode_steps: 192

env.eval.max_steps_per_rollout_epoch: 192

env.eval.task_config.step_lim: 192

env.eval.rollout_epoch: 8

actor.global_batch_size: 384

actor.optim.lr: 1.0e-6

actor.optim.value_lr: 5e-5

actor.model.motus.ignore_first: True

actor.model.motus.ignore_last: True

actor.model.motus.logprob_sigma: 0.75
```

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/motus@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor: 1
    env, rollout: 0

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_motus_ppo_formal_autodl"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 100

  only_eval: False
  val_check_interval: 20
  save_interval: 20

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  group_size: 1
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level

  update_epoch: 1
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: "token-mean"

  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0

  gamma: 0.99
  gae_lambda: 0.95

  filter_rewards: False
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"
  enable_offload: True

  train:
    rollout_epoch: 2
    total_num_envs: 16
    reward_coef: ${algorithm.reward_coef}
    # Motus chunk is 16. 160 steps gives 10 policy decisions: longer than
    # OpenPI's 200/50=4 but safer for initial success on adjust_bottle.
    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192
    group_size: ${algorithm.group_size}

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: False

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/train

    task_config:
      step_lim: 192
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

  eval:
    rollout_epoch: 8
    total_num_envs: 8
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 192
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 192
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: False

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/eval

    task_config:
      step_lim: 192
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}
    model_type: ${actor.model.model_type}
    num_action_chunks: ${actor.model.num_action_chunks}
    action_dim: ${actor.model.action_dim}
    motus: ${actor.model.motus}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 8
  global_batch_size: 384
  seed: 1234
  enable_offload: False

  model:
    model_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
    precision: "bf16"

    model_type: "motus"
    is_lora: False
    lora_rank: 32
    load_to_device: False

    num_action_chunks: 16
    action_dim: 14
    use_proprio: True
    add_value_head: True

    motus:
      repo_path: "/root/autodl-tmp/Motus"
      policy_path: "/root/autodl-tmp/RLinf/third_party/Motus"
      checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
      wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
      vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"
      vlm_max_seq_len: 256
      config_path: "/root/autodl-tmp/RoboTwin/policy/Motus/utils/robotwin.yml"

      num_inference_timesteps: 10
      use_t5_cache: True
      save_predicted_frames: False

      allow_batch_size: 16
      batch_inference: True
      decode_video: False

      trainable: "action_expert"
      freeze_video_model: True
      freeze_vlm_model: True
      freeze_und_expert: True
      freeze_t5_encoder: True

      logprob_mode: "flow_sde"
      logprob_sigma: 0.75
      collect_denoise_step: "random"
      ignore_first: True
      ignore_last: True

      detach_critic_input: True
      value_feature: "action_tokens"

  optim:
    lr: 1.0e-6
    value_lr: 5e-5
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
    use_orig_params: True
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False

weight_syncer:
  patch:
    init_sync:
      enabled: True

```

评估参数

```YAML
env.eval.max_episode_steps: 192

env.eval.max_steps_per_rollout_epoch: 192
    
env.eval.task_config.step_lim: 192
```



再训

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_adjust_bottle_motus_ppo_formal_autodl"
CONFIG_PATH="/root/autodl-tmp/RLinf/examples/embodiment/config"
TRAIN_CFG="${CONFIG_PATH}/${CONFIG_NAME}.yaml"

test -f "${TRAIN_CFG}"
test -d "/root/autodl-tmp/RoboTwin_RLinf"
test -f "/root/autodl-tmp/RLinf/rlinf/envs/robotwin/seeds/train_seeds.json"
test -f "/root/autodl-tmp/RLinf/rlinf/envs/robotwin/seeds/eval_seeds.json"
test -d "/root/autodl-tmp/models/motus/Motus_robotwin2"
test -d "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
test -d "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"

echo "===== important config ====="
grep -nE "max_steps:|val_check_interval|save_interval|rollout_epoch:|total_num_envs:|max_episode_steps:|max_steps_per_rollout_epoch:|step_lim:|update_epoch:|micro_batch_size:|global_batch_size:|lr:|value_lr:|logprob_sigma:|ignore_first:|ignore_last:|save_video:|vlm_max_seq_len|allow_batch_size" \
  "${TRAIN_CFG}"

ray stop --force || true

RUN_NAME="$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}-re2_h192_ue1_lr1e6_gbs384_sig075"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"

mkdir -p "${LOG_DIR}"

CMD=(
  python /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py
  --config-path "${CONFIG_PATH}"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_LOG}"
echo >> "${RUN_LOG}"

nohup "${CMD[@]}" >> "${RUN_LOG}" 2>&1 &

echo "pid=$!"
echo "run_name=${RUN_NAME}"
echo "log_dir=${LOG_DIR}"
echo "run_log=${RUN_LOG}"

if [ -f /root/autodl-tmp/monitor_motus_resources.sh ]; then
  nohup bash /root/autodl-tmp/monitor_motus_resources.sh 5 "${RUN_NAME}" \
    > "${LOG_DIR}/${RUN_NAME}.monitor.out" 2>&1 &
  echo "monitor_pid=$!"
  echo "monitor_log=${LOG_DIR}/${RUN_NAME}.monitor.out"
fi

tail -f "${RUN_LOG}"cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_adjust_bottle_motus_ppo_formal_autodl"
CONFIG_PATH="/root/autodl-tmp/RLinf/examples/embodiment/config"
TRAIN_CFG="${CONFIG_PATH}/${CONFIG_NAME}.yaml"

test -f "${TRAIN_CFG}"
test -d "/root/autodl-tmp/RoboTwin_RLinf"
test -f "/root/autodl-tmp/RLinf/rlinf/envs/robotwin/seeds/train_seeds.json"
test -f "/root/autodl-tmp/RLinf/rlinf/envs/robotwin/seeds/eval_seeds.json"
test -d "/root/autodl-tmp/models/motus/Motus_robotwin2"
test -d "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
test -d "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"

echo "===== important config ====="
grep -nE "max_steps:|val_check_interval|save_interval|rollout_epoch:|total_num_envs:|max_episode_steps:|max_steps_per_rollout_epoch:|step_lim:|update_epoch:|micro_batch_size:|global_batch_size:|lr:|value_lr:|logprob_sigma:|ignore_first:|ignore_last:|save_video:|vlm_max_seq_len|allow_batch_size" \
  "${TRAIN_CFG}"

ray stop --force || true

RUN_NAME="$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}-re2_h192_ue1_lr1e6_gbs384_sig075"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"

mkdir -p "${LOG_DIR}"

CMD=(
  python /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py
  --config-path "${CONFIG_PATH}"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_LOG}"
echo >> "${RUN_LOG}"

nohup "${CMD[@]}" >> "${RUN_LOG}" 2>&1 &

echo "pid=$!"
echo "run_name=${RUN_NAME}"
echo "log_dir=${LOG_DIR}"
echo "run_log=${RUN_LOG}"

if [ -f /root/autodl-tmp/monitor_motus_resources.sh ]; then
  nohup bash /root/autodl-tmp/monitor_motus_resources.sh 5 "${RUN_NAME}" \
    > "${LOG_DIR}/${RUN_NAME}.monitor.out" 2>&1 &
  echo "monitor_pid=$!"
  echo "monitor_log=${LOG_DIR}/${RUN_NAME}.monitor.out"
fi

tail -f "${RUN_LOG}"
```

会重复放两个？

脚本

```Bash
cd /root/autodl-tmp/RLinf

mkdir -p logs/nohup

START_SCRIPT="logs/nohup/start_motus_ppo_re2_h192_ue1_lr1e6_gbs384_sig075_safe.sh"

cat > "${START_SCRIPT}" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_adjust_bottle_motus_ppo_formal_autodl"
CONFIG_PATH="/root/autodl-tmp/RLinf/examples/embodiment/config"
TRAIN_CFG="${CONFIG_PATH}/${CONFIG_NAME}.yaml"

test -f "${TRAIN_CFG}"
test -d "/root/autodl-tmp/RoboTwin_RLinf"
test -f "/root/autodl-tmp/RLinf/rlinf/envs/robotwin/seeds/train_seeds.json"
test -f "/root/autodl-tmp/RLinf/rlinf/envs/robotwin/seeds/eval_seeds.json"
test -d "/root/autodl-tmp/models/motus/Motus_robotwin2"
test -d "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
test -d "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"

EXISTING_TRAIN="$(pgrep -f "train_embodied_agent.py.*${CONFIG_NAME}" || true)"
GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+' || true)"

if [ -n "${EXISTING_TRAIN}" ]; then
  echo "[abort] existing training process found:"
  echo "${EXISTING_TRAIN}"
  ps -fp ${EXISTING_TRAIN} || true
  exit 1
fi

if [ -n "${GPU_PIDS}" ]; then
  echo "[abort] GPU is not free:"
  nvidia-smi
  exit 1
fi

echo "===== important config ====="
grep -nE "max_steps:|val_check_interval|save_interval|rollout_epoch:|total_num_envs:|max_episode_steps:|max_steps_per_rollout_epoch:|step_lim:|update_epoch:|micro_batch_size:|global_batch_size:|lr:|value_lr:|logprob_sigma:|ignore_first:|ignore_last:|save_video:|vlm_max_seq_len|allow_batch_size" \
  "${TRAIN_CFG}"

ray stop --force || true

RUN_NAME="$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}-re2_h192_ue1_lr1e6_gbs384_sig075"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"

mkdir -p "${LOG_DIR}"

CMD=(
  python /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py
  --config-path "${CONFIG_PATH}"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_LOG}"
echo >> "${RUN_LOG}"

nohup "${CMD[@]}" >> "${RUN_LOG}" 2>&1 &
TRAIN_PID=$!

echo "pid=${TRAIN_PID}"
echo "run_name=${RUN_NAME}"
echo "log_dir=${LOG_DIR}"
echo "run_log=${RUN_LOG}"

if [ -f /root/autodl-tmp/monitor_motus_resources.sh ]; then
  nohup bash /root/autodl-tmp/monitor_motus_resources.sh 5 "${RUN_NAME}" \
    > "${LOG_DIR}/${RUN_NAME}.monitor.out" 2>&1 &
  MONITOR_PID=$!
  echo "monitor_pid=${MONITOR_PID}"
  echo "monitor_log=${LOG_DIR}/${RUN_NAME}.monitor.out"
fi
BASH

chmod +x "${START_SCRIPT}"

echo "START_SCRIPT=${START_SCRIPT}"
```

启动训练

```Plain Text
cd /root/autodl-tmp/RLinf

bash logs/nohup/start_motus_ppo_re2_h192_ue1_lr1e6_gbs384_sig075_safe.sh
```



日志在：

/root/autodl\-tmp/RLinf/logs/20260624\-18:17:29\-robotwin\_adjust\_bottle\_motus\_ppo\_formal\_autodl\-re2\_h192\_ue1\_lr1e6\_gbs384\_sig075/run\_embodiment\.log



60步时中断，ckpt有存

```Plain Text
其中 step60 保存后发生 Ray OOM

2. 训练中断原因
不是 PyTorch CUDA OOM，也不是 actor backward 崩了。日志里是：
Ray node_manager: Workers killed due to memory pressure (OOM)
EnvWorker died unexpectedly
Exiting main process due to a failure upon worker execution
也就是说，是 **Ray/系统内存压力** 把 EnvWorker 杀了。触发位置很典型：step60 处同时发生了：
1. 训练 rollout
2. 内置 eval：8×8=64 episodes
3. checkpoint save
这三件事叠在一起，内存压力最大。step20、step40都过了，step60爆，说明不是配置必然错误，而是长时间运行后 Ray object store / env / eval / checkpoint 叠加产生的 host memory 压力。
后续建议：**不要再让训练中内置 eval 跑 64eps**。更稳做法是训练中只保存 ckpt，不做重 eval；外部排队评估 20/40/60。
```



训练分析

```Plain Text
Actor 更新稳定性
approx_kl
区间approx_kl 平均
step1–2033.42
step21–4031.43
step41–5933.32
全部32.71
范围：
min = 10.027
max = 52.619
mean = 32.71
和之前那轮 mean≈94 相比，这次明显好多了。说明这几个改动有效：
lr 2e-6 -> 1e-6
global_batch_size 80 -> 384
logprob_sigma 0.5 -> 0.75
ignore_first/ignore_last=True
但是，approx_kl≈30 仍然不算小。它只是从“非常不稳”降到“中等偏高”。

clip_fraction
区间clip_fraction 平均
step1–200.524
step21–400.521
step41–590.520
全部0.522
这个几乎没改善，长期在 0.5 左右。说明即使 KL 降了，仍然有一半左右样本进入 PPO clipping 区间。也就是说：**ratio 分布还是太宽，PPO 仍然在强剪裁状态下训练。**
图：
clip_fraction 曲线
ratio / grad_norm
ratio 大多数低于 1，说明新策略对旧 rollout 的概率多数下降；中间有少数大尖峰，例如 step16 ratio=5.009。grad_norm 大部分后期稳定在 6–13，但前中期有大尖峰，最大 step27 附近达到 523.5。由于后期没有连续爆炸，不能说训练数值崩了，但这些尖峰说明 PPO 梯度仍有不稳定批次。

**value_loss 明显下降**。说明 value head 确实在学，没有断链。

explained_variance
整体仍然不好：
mean = -4.68
min = -41.90
max = -0.061
没有转正。后期也仍有大负值，比如 step52 约 -25，step59 约 -16.96。
 所以结论和之前一样：**critic 接通了，value_loss 能下降，但它没有学出稳定可用的 value baseline**

正常训练 step 大约：
14.5–15.2 min / step
每 20 步有一次内置 eval，会把 step time 拉到约：
44–45 min / eval step
整体：
step1–59 完整 metric 耗时约 15.5h
step60 eval + save 后崩
主要瓶颈仍然是 rollout，不是 actor training：
generate_rollouts 平均 ≈ 842.6s
actor/run_training 平均 ≈ 44.5s


```

稳了一点，调参方向对，但还是不稳

指标都不行，只有value loss 正常；

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OWZkODFkZWExMjYyMGVhMDMwNjdjNzFmMDNkMzNhODdfYmNmYzA0YTBmMDEwZTJiNTc0YzYzMmI3OGQzYTQzZDlfSUQ6NzY1NTE3MTkzNTY0Mjg4MDk2NF8xNzg1MTYwMjgwOjE3ODUyNDY2ODBfVjM)

value学习率太小？折中一点



评估 20 40 60

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1

mkdir -p logs/nohup

RUN_ROOT="/root/autodl-tmp/RLinf/logs/20260624-18:17:29-robotwin_adjust_bottle_motus_ppo_formal_autodl-re2_h192_ue1_lr1e6_gbs384_sig075"
CKPT_DIR="${RUN_ROOT}/robotwin_motus_ppo_formal_autodl/checkpoints"

EVAL_NAME="robotwin_adjust_bottle_motus_eval_batch8_autodl"
EVAL_CFG="/root/autodl-tmp/RLinf/evaluations/robotwin/${EVAL_NAME}.yaml"

CKPT20="${CKPT_DIR}/global_step_20/actor/model_state_dict/full_weights.pt"
CKPT40="${CKPT_DIR}/global_step_40/actor/model_state_dict/full_weights.pt"
CKPT60="${CKPT_DIR}/global_step_60/actor/model_state_dict/full_weights.pt"

test -d "${RUN_ROOT}"
test -f "${EVAL_CFG}"
test -s "${CKPT20}"
test -s "${CKPT40}"
test -s "${CKPT60}"

ls -lh "${CKPT20}" "${CKPT40}" "${CKPT60}"

STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_SCRIPT="logs/nohup/eval_motus_ppo_re2_h192_step20_step40_step60_64eps_${STAMP}.sh"
LOG="logs/nohup/eval_motus_ppo_re2_h192_step20_step40_step60_64eps_${STAMP}.log"

cat > "${EVAL_SCRIPT}" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1

RUN_ROOT="/root/autodl-tmp/RLinf/logs/20260624-18:17:29-robotwin_adjust_bottle_motus_ppo_formal_autodl-re2_h192_ue1_lr1e6_gbs384_sig075"
CKPT_DIR="${RUN_ROOT}/robotwin_motus_ppo_formal_autodl/checkpoints"

EVAL_NAME="robotwin_adjust_bottle_motus_eval_batch8_autodl"
EVAL_CFG="/root/autodl-tmp/RLinf/evaluations/robotwin/${EVAL_NAME}.yaml"

CKPT20="${CKPT_DIR}/global_step_20/actor/model_state_dict/full_weights.pt"
CKPT40="${CKPT_DIR}/global_step_40/actor/model_state_dict/full_weights.pt"
CKPT60="${CKPT_DIR}/global_step_60/actor/model_state_dict/full_weights.pt"

test -d "${RUN_ROOT}"
test -f "${EVAL_CFG}"
test -s "${CKPT20}"
test -s "${CKPT40}"
test -s "${CKPT60}"

wait_until_gpu_free() {
  echo
  echo "[wait_gpu] checking Ray/eval/GPU cleanup..."

  for i in $(seq 1 180); do
    RAY_PIDS="$(pgrep -f 'ray::|raylet|gcs_server|dashboard_agent|dashboard.py' || true)"
    EVAL_PIDS="$(pgrep -f 'train_embodied_agent.py|eval_embodied_agent.py|evaluations/run_eval.sh' || true)"
    GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+' || true)"

    if [ -z "${RAY_PIDS}" ] && [ -z "${EVAL_PIDS}" ] && [ -z "${GPU_PIDS}" ]; then
      echo "[wait_gpu] clean."
      nvidia-smi
      return 0
    fi

    echo "[wait_gpu] not clean yet, retry ${i}/180"
    echo "RAY_PIDS=${RAY_PIDS:-none}"
    echo "EVAL_PIDS=${EVAL_PIDS:-none}"
    echo "GPU_PIDS=${GPU_PIDS:-none}"
    sleep 10
  done

  echo "[wait_gpu] timeout."
  nvidia-smi
  return 1
}

run_eval() {
  local NAME="$1"
  local CKPT="$2"
  local EXP="$3"

  echo
  echo "============================================================"
  echo "START EVAL ${NAME}"
  echo "TIME=$(date)"
  echo "CKPT=${CKPT}"
  echo "EXP=${EXP}"
  echo "============================================================"
  echo

  test -s "${CKPT}"

  ray stop --force || python -m ray stop -f || true
  wait_until_gpu_free

  set +e

  bash evaluations/run_eval.sh robotwin "${EVAL_NAME}" \
    runner.ckpt_path="${CKPT}" \
    env.eval.total_num_envs=8 \
    env.eval.rollout_epoch=8 \
    env.eval.max_episode_steps=192 \
    env.eval.max_steps_per_rollout_epoch=192 \
    env.eval.task_config.step_lim=192 \
    rollout.model.add_value_head=True \
    rollout.model.motus.allow_batch_size=8 \
    rollout.model.motus.vlm_max_seq_len=256 \
    runner.logger.experiment_name="${EXP}"

  local RC=$?

  set -e

  echo
  echo "============================================================"
  echo "END EVAL ${NAME}"
  echo "TIME=$(date)"
  echo "RETURN_CODE=${RC}"
  echo "============================================================"
  echo

  ray stop --force || python -m ray stop -f || true
  wait_until_gpu_free

  if [ "${RC}" -ne 0 ]; then
    echo "[error] ${NAME} failed with return code ${RC}; stop sequence."
    exit "${RC}"
  fi
}

echo "============================================================"
echo "MOTUS PPO RE2 H192 EVAL QUEUE"
echo "TIME=$(date)"
echo "RUN_ROOT=${RUN_ROOT}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "EVAL_CFG=${EVAL_CFG}"
echo "============================================================"

run_eval \
  "STEP20_64EPS_H192" \
  "${CKPT20}" \
  "robotwin_motus_ppo_re2_h192_eval_step20_64eps"

run_eval \
  "STEP40_64EPS_H192" \
  "${CKPT40}" \
  "robotwin_motus_ppo_re2_h192_eval_step40_64eps"

run_eval \
  "STEP60_64EPS_H192" \
  "${CKPT60}" \
  "robotwin_motus_ppo_re2_h192_eval_step60_64eps"

echo
echo "ALL EVALS FINISHED"
echo "TIME=$(date)"
BASH

chmod +x "${EVAL_SCRIPT}"

nohup bash "${EVAL_SCRIPT}" > "${LOG}" 2>&1 &

echo "PID=$!"
echo "SCRIPT=${EVAL_SCRIPT}"
echo "LOG=${LOG}"
tail -f "${LOG}"
```

/root/autodl\-tmp/RLinf/logs/nohup/eval\_motus\_ppo\_re2\_h192\_step20\_step40\_step60\_64eps\_20260625\_114649\.log



以前常见的 base 64eps ≈ 0\.625

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OWQ2NWQyYWJhM2ZkMDk5NzU4ZjRiMmFhNmE0YzlmOWJfN2Q2YWI3NjQyMjAwZjZmYWQyNjFlMzE5OTlhOTY1YTNfSUQ6NzY1NTIyOTE4OTA3NTQ3MTI5OF8xNzg1MTYwMjgwOjE3ODUyNDY2ODBfVjM)

不明显



#### Vlm value head



\[motus\_vlm\_critic\_pi05\_repo\_root\.patch\]

\[apply\_motus\_vlm\_critic\_pi05\.sh\]

\[RLinf\_code\_only\_20260625\_174305\.tgz\]



训练smoke

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_motus_ppo_smoke_autodl \
  runner.max_steps=1 \
  runner.save_interval=-1 \
  runner.val_check_interval=-1
```



训练调参

```Plain Text
env.eval.rollout_epoch: 4

actor.optim.value_lr: 1.1e-4
```

目前：

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/motus@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor: 1
    env, rollout: 0

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_motus_ppo_formal_autodl"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 100

  only_eval: False
  val_check_interval: 20
  save_interval: 20

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  group_size: 1
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level

  update_epoch: 1
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: "token-mean"

  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0

  gamma: 0.99
  gae_lambda: 0.95

  filter_rewards: False
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"
  enable_offload: True

  train:
    rollout_epoch: 2
    total_num_envs: 16
    reward_coef: ${algorithm.reward_coef}
    # Motus chunk is 16. 160 steps gives 10 policy decisions: longer than
    # OpenPI's 200/50=4 but safer for initial success on adjust_bottle.
    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192
    group_size: ${algorithm.group_size}

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: False

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/train

    task_config:
      step_lim: 192
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

  eval:
    rollout_epoch: 4
    total_num_envs: 8
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 192
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 192
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True

    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: False

    video_cfg:
      save_video: False
      info_on_video: False
      video_base_dir: ${runner.logger.log_path}/video/eval

    task_config:
      step_lim: 192
      embodiment: [aloha-agilex]
      camera:
        collect_wrist_camera: true
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}
    model_type: ${actor.model.model_type}
    num_action_chunks: ${actor.model.num_action_chunks}
    action_dim: ${actor.model.action_dim}
    motus: ${actor.model.motus}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 8
  global_batch_size: 384
  seed: 1234
  enable_offload: False

  model:
    model_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
    precision: "bf16"

    model_type: "motus"
    is_lora: False
    lora_rank: 32
    load_to_device: False

    num_action_chunks: 16
    action_dim: 14
    use_proprio: True
    add_value_head: True

    motus:
      repo_path: "/root/autodl-tmp/Motus"
      policy_path: "/root/autodl-tmp/RLinf/third_party/Motus"
      checkpoint_path: "/root/autodl-tmp/models/motus/Motus_robotwin2"
      wan_path: "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
      vlm_path: "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"
      vlm_max_seq_len: 256
      config_path: "/root/autodl-tmp/RoboTwin/policy/Motus/utils/robotwin.yml"

      num_inference_timesteps: 10
      use_t5_cache: True
      save_predicted_frames: False

      allow_batch_size: 16
      batch_inference: True
      decode_video: False

      trainable: "action_expert"
      freeze_video_model: True
      freeze_vlm_model: True
      freeze_und_expert: True
      freeze_t5_encoder: True

      logprob_mode: "flow_sde"
      logprob_sigma: 0.75
      collect_denoise_step: "random"
      ignore_first: True
      ignore_last: True

      detach_critic_input: True
      value_after_vlm: True
      value_feature: "vlm_tokens"
      value_vlm_mode: "mean_token"

  optim:
    lr: 1.0e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
    use_orig_params: True
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False

weight_syncer:
  patch:
    init_sync:
      enabled: True

```



放训练

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_adjust_bottle_motus_ppo_formal_autodl"
CONFIG_PATH="/root/autodl-tmp/RLinf/examples/embodiment/config"
TRAIN_CFG="${CONFIG_PATH}/${CONFIG_NAME}.yaml"

test -f "${TRAIN_CFG}"
test -d "/root/autodl-tmp/RoboTwin_RLinf"
test -f "/root/autodl-tmp/RLinf/rlinf/envs/robotwin/seeds/train_seeds.json"
test -f "/root/autodl-tmp/RLinf/rlinf/envs/robotwin/seeds/eval_seeds.json"
test -d "/root/autodl-tmp/models/motus/Motus_robotwin2"
test -d "/root/autodl-tmp/models/motus/Wan2.2-TI2V-5B"
test -d "/root/autodl-tmp/models/motus/Qwen3-VL-2B-Instruct"
test -d "/root/autodl-tmp/RLinf/third_party/Motus"

echo "===== important config ====="
grep -nE "max_steps:|val_check_interval|save_interval|rollout_epoch:|total_num_envs:|max_episode_steps:|max_steps_per_rollout_epoch:|step_lim:|update_epoch:|micro_batch_size:|global_batch_size:|lr:|value_lr:|logprob_sigma:|ignore_first:|ignore_last:|value_after_vlm:|value_feature:|value_vlm_mode:|add_value_head:|policy_path:|save_video:|vlm_max_seq_len|allow_batch_size" \
  "${TRAIN_CFG}"

EXISTING_TRAIN="$(pgrep -f "train_embodied_agent.py.*${CONFIG_NAME}" || true)"
if [ -n "${EXISTING_TRAIN}" ]; then
  echo "[abort] existing training process found:"
  ps -fp ${EXISTING_TRAIN} || true
  exit 1
fi

GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+' || true)"
if [ -n "${GPU_PIDS}" ]; then
  echo "[abort] GPU is not free:"
  nvidia-smi
  exit 1
fi

ray stop --force || true

RUN_NAME="$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}-vlmcritic_h192_re2_ue1_lr1e6_vlr1e4_gbs384_sig075"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"

mkdir -p "${LOG_DIR}"

CMD=(
  python /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py
  --config-path "${CONFIG_PATH}"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_LOG}"
echo >> "${RUN_LOG}"

nohup "${CMD[@]}" >> "${RUN_LOG}" 2>&1 &

TRAIN_PID=$!

echo "pid=${TRAIN_PID}"
echo "run_name=${RUN_NAME}"
echo "log_dir=${LOG_DIR}"
echo "run_log=${RUN_LOG}"

if [ -f /root/autodl-tmp/monitor_motus_resources.sh ]; then
  nohup bash /root/autodl-tmp/monitor_motus_resources.sh 5 "${RUN_NAME}" \
    > "${LOG_DIR}/${RUN_NAME}.monitor.out" 2>&1 &
  echo "monitor_pid=$!"
  echo "monitor_log=${LOG_DIR}/${RUN_NAME}.monitor.out"
fi
```

/root/autodl\-tmp/RLinf/logs/20260625\-18:28:15\-robotwin\_adjust\_bottle\_motus\_ppo\_formal\_autodl\-vlmcritic\_h192\_re2\_ue1\_lr1e6\_vlr1e4\_gbs384\_sig075/run\_embodiment\.log



60步，中断

\[metrics\.log\]

\[run\_embodiment\.log\]

```Plain Text
3. Actor 更新稳定性
approx_kl
整体：
mean = 32.81
min  = 10.60
max  = 56.34
分段：
区间approx_kl 平均
step1–2034.10
step21–4032.74
step41–5731.39
和上一版 action-token critic 的 KL 水平差不多，没有恶化，但也没有明显改善。approx_kl≈30 仍然偏高。
approx_kl 曲线
clip_fraction
整体：
mean = 0.484
min  = 0.394
max  = 0.616
比上一版大约 0.52 略低，但仍然高。长期接近 0.45–0.50，说明 PPO 仍然有大约一半 token / chunk 处在 clipping 区间，更新仍然比较强。
clip_fraction 曲线
ratio / grad_norm
ratio 大多数低于 1，说明新策略对旧 rollout action 的概率总体下降。step54 有明显尖峰：
step54:
ratio = 2.445
ratio_abs = 3.161
grad_norm = 126.9
这不是连续爆炸，但说明偶尔仍有不稳定 batch。整体没有 NaN / inf / 崩溃。
ratio 曲线

explained_variance
整体均值被 step25 的极端值 -1079.1 拉坏；更可靠看 median：
mean   = -25.90
median = -3.918
max    = 0.263
后段 step41–57：
mean   = -5.07
median = -3.918
也就是说，**VLM critic 没有明显解决 explained_variance 长期为负的问题**。它比“完全断链”好，因为 value_loss 能下降，但还不是一个可靠 baseline。step33/34 附近短暂接近 0 或转正，但没有稳定保持。
图里为了可读性把 explained_variance clip 到 [-50, 5]；原始值在 CSV 里保留。
explained_variance 曲线
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NDBkM2EwNzY0NDMyZDBkMTRjOGE5ZmY0YjNmY2RhMTlfOTBjNzBmMDFmZDcxMWU0MTJhOGM3YTRiZGU5MTMxMjlfSUQ6NzY1NTUyNTQzOTkzMDQ4NTcwOF8xNzg1MTYwMjgwOjE3ODUyNDY2ODBfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MzE5OWRlYmM4OTAxZTI1MmY2MjU3ODU1MzYxNDc4ZmRfMTdiYTQ1NWYyNDk5NjRmNmMyZjViZTU1MWYyNmM0NGFfSUQ6NzY1NTUyNTUyMzA0NDU4NDY3Ml8xNzg1MTYwMjgwOjE3ODUyNDY2ODBfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NDQzZGQyZmM0OTFiOWEzYzM0ODNlNWE4MmFmNGI0MDlfMGI2MDE4YWUzMmQyNjc3ZDc5NjE3NWUzNDBkYjk3YTdfSUQ6NzY1NTUyNTU3MjYzMzgyNDIyMF8xNzg1MTYwMjgwOjE3ODUyNDY2ODBfVjM)

还是没显著效果



#### TODO



评估

配置检查，注意加载时，考虑目前的value head



找vla ppo调参经验知识？



问题定位

baseline：Openpi ppo 跑一次；训久一点

训练过程和motus有何不同？收集所有数据，各个值怎么样才是正常？对比目前训练各个值。

深入理解openpi实现代码；围绕感觉有问题的点

打印更多东西出来？

目前感觉value不对；有其他不对？；ExplainedVariance不对

或许是其他一些参数？ExplainedVariance不work真的完全致命？稳定性参数？

分析两篇论文；找更多论文；

看指标，看结论；都读一下

https://arxiv\.org/pdf/2510\.25889；https://arxiv\.org/pdf/2510\.06710

超参表；很多目前置0的部分其实有用？

真机实验设置怎么描述？

求助rlinf团队的人？

谁？选人？逐个发邮件？看主页成果？

提供什么？代码，日志？实现概要？图？或者开机让他上号？

整理下环境，面向如果有人想跑通；motus额外包，等等；从文档；

整理问题问zhp？



critic打印更多

```Bash
11. critic 指标应该加哪些
我同意只加少量关键值。建议放到现有 Training/Critic 表里：
critic/values_mean
critic/values_std
critic/prev_values_mean
critic/prev_values_std
critic/returns_mean
critic/returns_std
critic/value_error_abs_mean
不建议第一版加 feature norm、VLM norm、action norm。那些需要改 actor forward 返回额外 metrics，侵入更大。
这些指标能回答：
critic 输出是不是塌成常数？
prev_values 和 current values 是否尺度一致？
returns 的尺度和 values 是否匹配？
value_loss 低但 explained_variance 负，是不是因为 returns 方差太小？
你当前文档里也把“critic 的工作过程的所有值是否加到表里”列成 TODO，这几个指标足够第一轮排查。
```



？？

双流结构，视频动作都要训？有lora？



结构不同：

```Plain Text
OpenPI 的代码结构
OpenPI 在 RLinf 里基本是一个完整的 RL policy model。也就是说，OpenPIActionModel 自己同时负责：
1. 环境 obs -> OpenPI model input
2. sample_actions()
3. 保存 chains / denoise_inds / prev_logprobs / prev_values
4. get_log_prob_value()
5. value_head
6. actor forward
所以 OpenPI 的 value_head、get_log_prob_value()、sample_actions() 都在同一个 RLinf model 类里。
Motus 当前结构
Motus 被接进 RLinf 时分成两层：
RLinf MotusPolicy adapter:
  rlinf/models/embodiment/motus/motus_policy.py

third_party Motus model:
  third_party/Motus/models/motus.py

能不能做成和 OpenPI 一样
可以理论上做成完全一样：把 RLinf 的 ValueHead、default_forward()、predict_action_batch() 都塞进 third_party Motus model。但我不建议。
原因是 third_party Motus 是外部模型代码，RLinf 的 FSDP、optimizer、weight sync、checkpoint 逻辑应该看到的是 MotusPolicy 这个 RLinf model wrapper。把 ValueHead 放在 MotusPolicy 上，是为了让：
FSDP 能 wrap value_head
optimizer 能给 value_head 用 value_lr
checkpoint 能保存 value_head.*
eval 加 add_value_head=True 后能 strict load PPO ckpt
bootstrap value 能通过 hasattr(model, "value_head") 接上
所以目前不是算法不对齐，而是**文件边界不同、功能接口对齐**。
影响主要有两个：
1. dtype/device 边界更容易出问题
   之前 ValueHead bf16/float32 mismatch 就是这个结构边界导致的。

2. eval 加载 PPO ckpt 时必须 add_value_head=True
   因为 checkpoint 里有 value_head.*，eval model 也必须建同样结构。


```



下面TODO看看



#### TODO other



太慢，换更快的wam？fast\-wam？



不一致

```Bash
1. dtype 修复和 OpenPI 对齐吗？
**语义上对齐，代码位置略不同。**
OpenPI 的 value head 在 OpenPI model 内部，value feature 来自同一个 forward 里的 suffix/action hidden states。因为 OpenPI 的 sample_mean_var_val() / get_log_prob_value() 和 value head 都处在同一套模型前向路径里，feature dtype 和 value head 参数 dtype 更自然地被 FSDP/mixed precision 管住。
Motus 的结构不同：
third_party/Motus:
  负责 denoise，并返回 value_features

RLinf MotusPolicy:
  持有 ValueHead，并用 value_features 算 values
我们让 third_party 返回 action-token pooled feature；这个 feature 为了日志/稳定性一度是 float32。但 actor 侧 FSDP mixed precision 会把 ValueHead 的 Linear 权重变成 bf16。于是出现：
value_features: Float
value_head linear weight: BFloat16


```



残余的不一致

```SQL
4.4 old/new logprob replay
当前 old/new 都走：
model.get_log_prob_value(...)
OpenPI old 在 sampling loop 内算，new 在 actor replay 算；Motus old 是 sampling 后立刻 replay 算，new 是 actor replay 算。
**状态：分布语义对齐；代码位置略不同。**
 这不是萎缩。
```

```SQL
1. old logprob 这块到底在做什么
PPO/GRPO 的核心 ratio 是：
ratio = exp(new_logprob - old_logprob)
所以 old logprob 是 rollout 当时旧策略对“它自己采出来的动作/transition”的概率；new logprob 是 actor update 时当前策略对同一个动作/transition 的概率。
OpenPI 怎么做
OpenPI rollout 过程中会生成一条 action denoise chain：
x_0, x_1, ..., x_S
然后随机选一个 denoise step，比如 k。在这个 step 上：
x_k -> x_{k+1}
用 flow_sde 随机采样，并计算：
old_logprob = log p_old(x_{k+1} | x_k)
OpenPI 保存：
chains
denoise_inds
prev_logprobs
actor update 时再拿：
chains[k], chains[k+1], denoise_inds
重算：
new_logprob = log p_current(x_{k+1} | x_k)
所以 OpenPI 的逻辑是：
rollout 内采样并算 old_logprob
actor update replay chain 并算 new_logprob
Motus 之前怎么做
更早的 Motus 是 adapter 自己存：
action_x_t
action_x_next
video_latent_t
t_scaled
dt
然后 adapter 自己用 _transition_logprobs() 算 old/new logprob。问题是：逻辑散在 adapter 里，不像 OpenPI 那样由 model-side chain replay API 统一处理。
Motus 现在怎么做
现在 Motus rollout 也生成 chain：
motus_action_chains
motus_video_chains
motus_denoise_inds
采样结束后，立即调用：
self.model.get_log_prob_value(...)
来算：
prev_logprobs
actor update 也调用同一个：
self.model.get_log_prob_value(...)
来算：
current logprobs
也就是说：
Motus rollout old logprob 和 actor new logprob 使用同一个 model-side replay API
和 OpenPI 的差异只有一个：OpenPI 在 sampling loop 里顺手算 old logprob；Motus 是 sampling 后用同一个 replay API 重新算一遍 old logprob。因为这时候参数还没更新，所以这两个做法分布语义等价。这样写反而让 old/new logprob 更共享代码。
PPO value 也是类似：Motus rollout 时存每个 denoise step 的 value_features，然后对每个 step 过 ValueHead，再对 denoise step 求平均，作为 prev_values。这对齐 OpenPI 里 rollout values over denoise steps 的平均语义。
```



长的训练

分析训练，图，有趋势？



TODO \- 优化



评估，两卡两个任务并行？总之利用两卡

不同的评估相同吗？从相同的种子开始吗？



奖励分配

可能导致效果问题：

```Bash
reward
你这里的 reward 很小，比如：
return=0.25 -> reward=0.000625
return=0.75 -> reward=0.001875
因为它是按 env step 平均后的 reward。0.25 / 400 = 0.000625，所以它和 return 是一致的，只是缩放了。判断效果优先看 return / success_once。
```

实际上rt2任务轨迹很长尾，大头在前面；可以设置前面权重高

训练相关各个指标搞懂；其他论文怎么描述？有什么意义？



**数据盘**占的有点多，都存了啥？清理下。之前项目应该占了不少



进一步开发，看项目里面作者留下的给codex等agent的文件。结构，如何加新东西，等



### git



查

```Bash
cd /root/autodl-tmp/RLinf

echo "===== pwd ====="
pwd

echo "===== git exists? ====="
if [ -d .git ]; then
  echo "OK: this is a git repo"
else
  echo "STOP: .git 不存在。不要直接 git init，先看本文最后的【如果 .git 不存在】方案。"
  exit 1
fi

echo "===== git safe dir ====="
git config --global --add safe.directory /root/autodl-tmp/RLinf || true

echo "===== branch / head ====="
git status -sb
git branch --show-current
git rev-parse HEAD
git log --oneline -5

echo "===== remotes ====="
git remote -v

echo "===== .gitignore ====="
nl -ba .gitignore | sed -n '1,80p'

cd /root/autodl-tmp/RLinf

echo "===== total repo size ====="
du -sh .
du -sh .git 2>/dev/null || true

echo "===== top-level directories ====="
du -h --max-depth=1 . | sort -h

echo "===== largest files excluding common generated dirs ====="
find . \
  -path './.git' -prune -o \
  -path './.venv' -prune -o \
  -path './.venv_*' -prune -o \
  -path './logs' -prune -o \
  -path './results' -prune -o \
  -path './outputs' -prune -o \
  -type f -printf '%s %p\n' \
| sort -nr | head -80 \
| awk '{printf "%.2f MB  %s\n", $1/1024/1024, $2}'

echo "===== model/checkpoint-like files ====="
find . \
  -path './.git' -prune -o \
  -path './.venv' -prune -o \
  -path './.venv_*' -prune -o \
  -type f \( \
    -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.safetensors" -o \
    -name "*.bin" -o -name "*.onnx" -o -name "*.pkl" -o -name "*.npz" -o \
    -name "*.h5" -o -name "*.hdf5" -o -name "*.tar" -o -name "*.tgz" -o \
    -name "*.tar.gz" -o -name "*.zip" \
  \) -printf '%s %p\n' \
| sort -nr | head -100 \
| awk '{printf "%.2f MB  %s\n", $1/1024/1024, $2}'
```



暂存检查

```Bash
cd /root/autodl-tmp/RLinf

echo "===== official .gitignore should preferably be unchanged ====="
git diff -- .gitignore || true

echo "===== dry-run add ====="
git add -n . | sed -n '1,240p'

echo "===== ignored check examples ====="
git check-ignore -v git_head_before_motus_20260618_142009.txt || true
git check-ignore -v pip_freeze_before_motus_20260618_142009.txt || true
git check-ignore -v rlinf.egg-info/PKG-INFO || true

cd /root/autodl-tmp/RLinf

git add -A

echo "===== staged summary ====="
git status --short
git diff --cached --stat

echo "===== staged files that should NOT appear ====="
BAD=$(git diff --cached --name-only | grep -E '(__pycache__|\.pyc$|\.bak|\.orig$|\.rej$|\.pt$|\.pth$|\.ckpt$|\.safetensors$|\.bin$|\.onnx$|\.pkl$|\.npz$|\.h5$|\.hdf5$|\.tar$|\.tgz$|\.tar\.gz$|\.zip$)' || true)
if [ -n "$BAD" ]; then
  echo "STOP: bad staged files detected:"
  echo "$BAD"
  exit 1
else
  echo "OK: no obvious bad staged files"
fi

echo "===== staged files > 50 MB ====="
python - <<'PY'
import os
import subprocess
import sys

files = subprocess.check_output(
    ["git", "diff", "--cached", "--name-only"],
    text=True,
).splitlines()

big = []
for f in files:
    if os.path.isfile(f):
        s = os.path.getsize(f)
        if s > 50 * 1024 * 1024:
            big.append((s, f))

for s, f in sorted(big, reverse=True):
    print(f"{s/1024/1024:.1f} MB\t{f}")

print("staged files >50MB:", len(big))
if big:
    sys.exit(1)
PY

cd /root/autodl-tmp/RLinf

echo "===== staged Motus/OpenPI/AutoDL related files ====="
git diff --cached --name-status \
| grep -E 'motus|Motus|autodl|local_scripts|fsdp.py|robotwin_adjust_bottle' || true

echo "===== full staged name-status ====="
git diff --cached --name-status | sed -n '1,260p'
```



交

```Bash
cd /root/autodl-tmp/RLinf

git config user.name "Yutenji-Nyamu"
git config user.email "Yutenji-Nyamu@users.noreply.github.com"

git commit -s -m "Add Motus support for RoboTwin RLinf experiments"

cd /root/autodl-tmp/RLinf

git status -sb
git log --oneline -3
git show --stat --oneline --decorate -1

cd /root/autodl-tmp/RLinf

echo "===== current remotes ====="
git remote -v

if git remote get-url origin 2>/dev/null | grep -Eqi 'github.com[:/]RLinf/RLinf(\.git)?$'; then
  git remote rename origin upstream
fi

if ! git remote get-url upstream >/dev/null 2>&1; then
  git remote add upstream https://github.com/RLinf/RLinf.git
fi

echo "===== remotes after upstream setup ====="
git remote -v
```



登录

```Bash
which gh && gh --version || {
  sudo apt update
  sudo apt install -y gh
}

gh auth status || gh auth login
```



创建仓库，推，公开

```Bash
cd /root/autodl-tmp/RLinf

OWNER="Yutenji-Nyamu"
REPO="RLinf-Motus-RoboTwin"

# 如果 origin 已存在但不是你的仓库，先移除
git remote remove origin 2>/dev/null || true

gh repo create "${OWNER}/${REPO}" \
  --private \
  --source=. \
  --remote=origin

git remote -v
git branch --show-current

cd /root/autodl-tmp/RLinf

BRANCH="$(git branch --show-current)"
git push -u origin "$BRANCH"

cd /root/autodl-tmp/RLinf

OWNER="Yutenji-Nyamu"
REPO="RLinf-Motus-RoboTwin"

gh repo edit "${OWNER}/${REPO}" --visibility public
gh repo view "${OWNER}/${REPO}" --json isPrivate,url
```

