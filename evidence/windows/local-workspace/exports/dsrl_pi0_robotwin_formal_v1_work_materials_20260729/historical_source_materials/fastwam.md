# fastwam



### Guide



历史

[Exp\_snd](https://my.feishu.cn/wiki/JAfbwIKFui0jR6kkbfScpJ0vn7f)

[Openpi \+ PPO AutoDL A800](https://my.feishu.cn/wiki/HkMUwQDcOikfygkTVm2cBsnEnVe)

[Motus \+ RLinf](https://my.feishu.cn/wiki/Azcdwee5EiVgdCkmCK3csFhVnBf)

[lawam rlinf](https://my.feishu.cn/wiki/O3nPwu6lgi5PMkk7KxNcy1rrn2f)

[pi0 \+ ppo/grpo](https://my.feishu.cn/wiki/Sg6hwU18HiHvavkfgq9cFS3BnVg)



### 推理



#### 环境

好像fastwam要的环境和robowin有冲突？所以要装fastwam的一些轮子，再装rt2的一些东西？



gpt总结，不一定对：

\[07\_OFFICIAL\_STANDALONE\_RUNBOOK\.md\]



查

```Bash
# [本机验收] 记录时间，并确认训练/Ray、GPU 和主机内存已经释放。
date

pgrep -af 'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' || true
nvidia-smi
free -h

# [本机验收] 安装、模型和 cache 都放数据盘；先确认容量与 inode。
df -h /root/autodl-tmp
df -i /root/autodl-tmp

# [本机适配] 动态确认 Conda base，不硬编码历史路径。
command -v conda || true
conda info --base 2>/dev/null || true
conda env list 2>/dev/null || true

# [兼容性验收] CuRobo 要在最终 Torch 环境编译。
command -v nvcc || true
nvcc --version || true
ls -ld /usr/local/cuda* 2>/dev/null || true

# [RoboTwin 官方前置的本机验收] 检查 Vulkan。
command -v vulkaninfo || true
vulkaninfo --summary 2>/dev/null | head -n 40 || true

# [安全验收] 发现目标路径时先审计，不覆盖。
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

# [本机复用检查] 检查现有 assets/task_config。
git -C /root/autodl-tmp/RoboTwin rev-parse HEAD 2>/dev/null || true
git -C /root/autodl-tmp/RoboTwin status --short -- task_config 2>/dev/null || true
du -sh /root/autodl-tmp/RoboTwin/assets /root/autodl-tmp/RoboTwin/task_config 2>/dev/null || true
```



Clone

```Bash
ROOT=/root/autodl-tmp
FW="$ROOT/FastWAM"
ENV="$ROOT/conda/envs/FastWAM-official"
PIN=45d8e1458921d83f8ad6cf9ce993d371208dabd0

test ! -e "$FW" && test ! -L "$FW"

if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
fi

git clone https://github.com/yuantianyuan01/FastWAM.git "$FW"
git -C "$FW" checkout --detach "$PIN"

git -C "$FW" rev-parse HEAD
git -C "$FW" remote -v
git -C "$FW" status --short --branch
```

很快



创建环境

```Bash
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
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions/fastwam-torch271-cu128"
export TMPDIR="$ROOT/tmp/fastwam"

mkdir -p "$PIP_CACHE_DIR" "$HUGGINGFACE_HUB_CACHE" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

python -m pip install -U pip setuptools wheel ninja setuptools_scm
python -m pip install \
  torch==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "$FW"
```



爆

```Plain Text
阶段 C 被 pip 镜像中断了。无需删除环境，也不要重装 Torch。
当前状态
阶段 A：通过。GPU 空闲、约 900 GiB 可用内存、1.2 TiB 数据盘空间、assets 完整。Vulkan loader 正常；真正的 GPU render 留到阶段 F 验证。
阶段 B：通过。Fast-WAM 已精确固定到 45d8e145...，工作树干净。
阶段 C：
Python 3.10.20：正确。
Torch 2.7.1+cu128：正确。
Torchvision 0.22.1+cu128：正确。
Fast-WAM、Transformers 等：尚未安装。

错误链是：
network_turbo 让 pip 使用 http://mirrors.aliyun.com/pypi/simple。
该镜像没有返回 ninja，所以第一条工具安装失败。
pip install -e 会建立隔离构建环境，并重新获取 pyproject.toml 声明的 setuptools>=68、wheel；它同样只能访问这个镜像，所以失败。pip 官方构建隔离说明
transformers 是 Fast-WAM 项目依赖，editable install 没完成，因此它不存在。
pip check 只说明“目前已经安装的包内部不冲突”，不能证明 Fast-WAM 安装成功。
OMP_NUM_THREADS 是独立的非法环境变量警告，与 pip 失败无关。
```



创建环境，装

```Bash
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
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions/fastwam-torch271-cu128"
export TMPDIR="$ROOT/tmp/fastwam"
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128

# [本机环境清理] 非法的继承值会触发 libgomp 警告；只清理当前 shell，不修改 rc 文件。
printf 'OMP_NUM_THREADS=<%q>\n' "${OMP_NUM_THREADS-}"
unset OMP_NUM_THREADS

mkdir -p "$PIP_CACHE_DIR" "$HUGGINGFACE_HUB_CACHE" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

python -m pip install -U pip setuptools wheel
python -m pip install ninja setuptools_scm
python -m pip install \
  torch==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "$FW"
```



慢

```Plain Text
AutoDL 学术加速不应该常驻；当前也不是环境损坏，而是 Fast-WAM 依赖下载阶段的网络降速。
当前状态
Python 3.10.20、Torch 2.7.1+cu128、Torchvision 0.22.1+cu128 已正确安装。
Fast-WAM 的 PEP 517 构建依赖、editable metadata、依赖解析均已通过。
现在只是 av==16.0.1 下载降至 14.9 kB/s，尚未完成 fastwam 安装。
之前 transformers 缺失，是因为 pip install -e 当时没完成，不代表环境方案错误。
当前环境主体符合 Fast-WAM 官方安装顺序：Python 3.10 → Torch/Torchvision cu128 → pip install -e .。
最可能的原因是：此前同一 shell 中执行过 source /etc/network_turbo，但没有看到清代理记录，导致普通 PyPI 流量也可能走了学术代理。终端没有打印代理变量，因此这是首要嫌疑，不是已证实结论。
AutoDL 官方也说明，学术加速主要面向 GitHub/Hugging Face，不保证稳定，使用结束后建议关闭，否则可能影响正常网络；普通依赖推荐国内软件源，阿里源高峰时还可能限速。学术资源加速、AutoDL 软件源
历史成功实践也一致：
```

处理网络

```Bash
# [本机网络适配] 不重建环境、不重装 Torch、不清 cache。
ROOT=/root/autodl-tmp
FW="$ROOT/FastWAM"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
unset OMP_NUM_THREADS

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_CACHE_DIR="$ROOT/cache/pip"
export TMPDIR="$ROOT/tmp/fastwam"
export PIP_DEFAULT_TIMEOUT=120

printf 'PIP_INDEX_URL=%s\n' "$PIP_INDEX_URL"
printf 'PIP_EXTRA_INDEX_URL=%s\n' "${PIP_EXTRA_INDEX_URL-<unset>}"
env | grep -iE '^(http|https|all)_proxy=' || true

# [Fast-WAM 官方安装语义] 只是把普通包传输改到清华镜像。
python -m pip install -e "$FW"
```



查

```Python
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



变量

```Bash
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
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_ROOT/diffsynth"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTHONUNBUFFERED=1
unset OMP_NUM_THREADS

test "$(python -c 'import sys; print(sys.executable)')" = "$ENV/bin/python" || {
  echo "STOP: FastWAM-official environment is not active"
  exit 1
}
```



手动适配的装rt2依赖

```Plain Text
conda activate "$ENV"

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



Curobo查

```Bash
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



curobo装

```Bash
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

MAX_JOBS=8 python -m pip install -e "$CUROBO_SRC" --no-build-isolation
```



curobo查

```Python
python - <<'PY'
import curobo
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
print("curobo", curobo.__file__)
print("CuRobo v1 API import OK")
PY
```



```Plain Text
唯一值得记录的是 CuRobo 安装将 SciPy 1.10.1 升到了 1.15.3。当前无依赖冲突、CuRobo import 已通过，所以不要现在回退；后续用 RoboTwin 场景 smoke 判断真实兼容性。
```



rt2补丁

```Bash
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



asset链接

```Bash
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



#### 配



复制任务配置

```Bash
# [官方配置内容 + 本机复用方式]
ROOT=/root/autodl-tmp
FW="$ROOT/FastWAM"
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
  echo "STOP: task_config version mismatch"
  exit 1
}

test -z "$(git -C "$RT" status --porcelain -- task_config)" || {
  echo "STOP: source task_config has local changes"
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



policy链接

```Bash
ln -sfn "$FW/experiments/robotwin/fastwam_policy" "$VRT/policy/fastwam_policy"
readlink -f "$VRT/policy/fastwam_policy"
```



查rt2

```Python
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

查

```Plain Text
cd "$VRT"
python script/test_render.py
```

查

```Python
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

查

```Bash
SMOKE_CFG="$VRT/task_config/demo_fastwam_env_smoke.yml"
if [ -e "$SMOKE_CFG" ] || [ -L "$SMOKE_CFG" ]; then
  echo "STOP: $SMOKE_CFG already exists"
  exit 1
fi

cp "$VRT/task_config/demo_clean.yml" "$SMOKE_CFG"
sed -i -E 's/^episode_num:[[:space:]]*.*/episode_num: 1/' "$SMOKE_CFG"
sed -i -E 's|^save_path:[[:space:]]*.*|save_path: ./data_fastwam_env_smoke|' "$SMOKE_CFG"
grep -n -E '^(episode_num|save_path):' "$SMOKE_CFG"

CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py adjust_bottle demo_fastwam_env_smoke
```



爆

```Plain Text
SAPIEN 无法导入 pkg_resources

setuptools 83.0.0 已经不再提供 pkg_resources，而 sapien==3.0.0b1 仍然直接导入它。pkg_resources 是从 Setuptools 82.0.0 开始正式移除的。Setuptools 官方说明
RoboTwin 官方依赖固定了 sapien==3.0.0b1，却没有限制 Setuptools 版本；官方安装脚本也没有补这个约束。RoboTwin requirements、官方安装脚本

主要原因是旧 SAPIEN 与 2026 年新版 Setuptools 的生态漂移。
我之前手册里写了 pip install -U pip setuptools wheel，却没有吸收 Motus 已验证的 setuptools==80.9.0 pin，这是我的指引疏忽。即使完全照 Fast-WAM README，新 Conda 当前也可能直接安装 Setuptools 83，但手册本应主动规避。
```



修包

```Bash
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

ROOT=/root/autodl-tmp
ENV="$ROOT/conda/envs/FastWAM-official"
VRT="$ROOT/FastWAM/third_party/RoboTwin"

conda activate "$ENV"

unset OMP_NUM_THREADS
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST

python -m pip install "setuptools==80.9.0"
python -m pip check

cd "$VRT"
python - <<'PY'
import setuptools
import pkg_resources
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

print("setuptools", setuptools.__version__)
print("pkg_resources", pkg_resources.__file__)
print("RoboTwin imports OK", torch.__version__)
print("scipy", scipy.__version__, "skimage", skimage.__version__)
PY
```



查

```Plain Text
cd "$VRT"
python script/test_render.py
```



查

```Bash
SMOKE_CFG="$VRT/task_config/demo_fastwam_env_smoke.yml"
if [ -e "$SMOKE_CFG" ] || [ -L "$SMOKE_CFG" ]; then
  echo "STOP: $SMOKE_CFG already exists"
  exit 1
fi

cp "$VRT/task_config/demo_clean.yml" "$SMOKE_CFG"
sed -i -E 's/^episode_num:[[:space:]]*.*/episode_num: 1/' "$SMOKE_CFG"
sed -i -E 's|^save_path:[[:space:]]*.*|save_path: ./data_fastwam_env_smoke|' "$SMOKE_CFG"
grep -n -E '^(episode_num|save_path):' "$SMOKE_CFG"

CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py adjust_bottle demo_fastwam_env_smoke
```

退终端



查

```Bash
ROOT=/root/autodl-tmp
FW="$ROOT/FastWAM"
ENV="$ROOT/conda/envs/FastWAM-official"
VRT="$FW/third_party/RoboTwin"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
unset OMP_NUM_THREADS
cd "$VRT"

pgrep -af '[c]ollect_data.py.*adjust_bottle' || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader

python - <<'PY'
import yaml

p = "task_config/demo_fastwam_env_smoke.yml"
with open(p, encoding="utf-8") as f:
    c = yaml.safe_load(f)

assert c["episode_num"] == 1, c["episode_num"]
assert c["save_path"] == "./data_fastwam_env_smoke", c["save_path"]
assert c["data_type"]["pointcloud"] is False, c["data_type"]
print("smoke config OK", p)
PY
```



查

```Plain Text
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore::UserWarning \
python -u script/collect_data.py adjust_bottle demo_fastwam_env_smoke
```



报错

```Plain Text
是 CuRobo 0.7.8 + warp-lang 1.15.0 的版本不兼容，会阻塞后续官方推理，但修复很小。

错误链
真正的根错误只有第一个：
module 'warp' has no attribute 'torch'
安装日志显示：
warp-lang-1.15.0
CuRobo 0.7.8 只声明了 warp-lang>=0.9.0，没有上界，但代码仍调用旧接口 wp.torch.device_from_torch()。CuRobo 依赖声明、实际调用位置
Warp 1.13 起删除了 warp.torch 等旧兼容入口，所以 pip 自动安装的 1.15.0 必然失败。NVIDIA Warp 变更记录
后续的：
'Robot' object has no attribute 'left_planner'
只是连锁错误：seed 0 构造 CuroboPlanner 中途失败，left_planner 尚未完成赋值；RoboTwin 捕获异常后继续复用了这个半初始化对象。重新启动 Python 进程即可清掉，不需要修改 robot.py。

主要是 CuRobo 上游依赖缺少 Warp 上界。
我们手册锁了 CuRobo 0.7.8，却漏锁它的 Warp 依赖，属于安装指引疏忽。

选择 1.11.1，因为该版本仍明确兼容 wp.torch，并且 RLinf 官方 RoboTwin 安装脚本也固定该版本
```



修

```Bash
ROOT=/root/autodl-tmp
ENV="$ROOT/conda/envs/FastWAM-official"
VRT="$ROOT/FastWAM/third_party/RoboTwin"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

unset OMP_NUM_THREADS
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m pip install --no-deps "warp-lang==1.11.1"
python -m pip check

python - <<'PY'
from importlib.metadata import version
import torch
import warp as wp

print("warp-lang", version("warp-lang"))
print("warp", wp.__file__)
print("legacy device", wp.torch.device_from_torch(torch.device("cuda:0")))
print("public device", wp.device_from_torch(torch.device("cuda:0")))
PY
```

报错

```Plain Text
runtime.cuda_devices 只是我给的独立验证片段漏了 wp.init()，不是环境仍有问题；实际 CuRobo 会先初始化 Warp，所以随后 F4 正常。
```



查

```Plain Text
cd "$VRT"

CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore::UserWarning \
python -u script/collect_data.py adjust_bottle demo_fastwam_env_smoke
```

通过



```Plain Text
环境已经修好，F4 已正式通过，可以进入 FastWAM 官方模型推理 smoke。

NoneType.cuda_devices 是什么
这不是新故障，而是我之前给出的独立验证命令漏写了 wp.init()：
import torch
import warp as wp

wp.init()
print(wp.torch.device_from_torch(torch.device("cuda:0")))
device_from_torch() 不会自行初始化 Warp runtime；CuRobo 的真实调用链会先执行 init_warp()，其中包含 wp.init()，随后才转换设备

联合仿真环境已经可用
```



#### 下ckpt



```Bash
MODEL_ROOT="$ROOT/models/fastwam"
mkdir -p "$MODEL_ROOT/release" "$MODEL_ROOT/diffsynth"

export HF_HOME="$ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_ROOT/diffsynth"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

huggingface-cli download yuanty/fastwam \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir "$MODEL_ROOT/release"

ls -lh "$MODEL_ROOT/release"
sha256sum \
  "$MODEL_ROOT/release/robotwin_uncond_3cam_384.pt" \
  "$MODEL_ROOT/release/robotwin_uncond_3cam_384_dataset_stats.json"
```

挺快，14MB



#### 跑



查

```Bash
ROOT=/root/autodl-tmp
FW="$ROOT/FastWAM"
MODEL_ROOT="$ROOT/models/fastwam"

ls -lh \
  "$MODEL_ROOT/release/robotwin_uncond_3cam_384.pt" \
  "$MODEL_ROOT/release/robotwin_uncond_3cam_384_dataset_stats.json"

readlink -f "$FW/third_party/RoboTwin/policy/fastwam_policy"

pgrep -af '[e]val_robotwin_single.py|[c]ollect_data.py' || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
df -h "$ROOT"
```



跑

```Bash
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

export HF_HOME="$ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_ROOT/diffsynth"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTHONUNBUFFERED=1

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



报错 hf

```Plain Text
这是我上一版指引中的下载后端配置错误，不是服务器或模型环境损坏。
因果链是：
我设置 DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
→ FastWAM 改用 huggingface_hub
→ loader 请求 DiffSynth-Studio/Wan-Series-Converted-Safetensors
→ 但这个转换权重仓实际托管在 ModelScope
→ hf-mirror/Hugging Face 找不到该仓，返回 generic 401

固定 commit 的 FastWAM 代码默认下载源本来就是 modelscope；pyproject.toml 也已安装 modelscope==1.34.0。

关于 Motus/LAWAM：Motus 当时确实用 hf-mirror 成功下载了真正位于 Hugging Face 的仓库，所以镜像选择有历史依据；我遗漏的是当时“每个模型独立下载并验证仓库”的分阶段做法，也没有识别 FastWAM 同时使用 HF release 和 ModelScope 转换权重两个来源。这是我的疏漏。
```



查

```Bash
ROOT=/root/autodl-tmp
ENV="$ROOT/conda/envs/FastWAM-official"
MODEL_BASE="$ROOT/models/fastwam/diffsynth"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

unset OMP_NUM_THREADS
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
unset HF_ENDPOINT
unset DIFFSYNTH_SKIP_DOWNLOAD

export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_BASE"
export MODELSCOPE_CACHE="$ROOT/cache/modelscope"

mkdir -p "$MODEL_BASE" "$MODELSCOPE_CACHE"

python - <<'PY'
from importlib.metadata import version
from fastwam.models.wan22.helpers.loader import _resolve_configs

cfgs = _resolve_configs(
    "Wan-AI/Wan2.2-TI2V-5B",
    "Wan-AI/Wan2.1-T2V-1.3B",
)

print("modelscope", version("modelscope"))
for name, cfg in zip(("dit", "text", "vae", "tokenizer"), cfgs):
    print(name, cfg.model_id, cfg.origin_file_pattern, cfg.parse_download_source())

assert all(cfg.parse_download_source() == "modelscope" for cfg in cfgs)
PY

find "$MODEL_BASE" -type f -printf '%s %p\n' | sort
```



跑

```Bash
ROOT=/root/autodl-tmp
FW="$ROOT/FastWAM"
ENV="$ROOT/conda/envs/FastWAM-official"
MODEL_ROOT="$ROOT/models/fastwam"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

unset OMP_NUM_THREADS
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
unset HF_ENDPOINT
unset DIFFSYNTH_SKIP_DOWNLOAD

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export MODELSCOPE_CACHE="$ROOT/cache/modelscope"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_ROOT/diffsynth"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export TMPDIR="$ROOT/tmp/fastwam"
export PYTHONUNBUFFERED=1

mkdir -p "$MODELSCOPE_CACHE" "$DIFFSYNTH_MODEL_BASE_PATH" "$TMPDIR"

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

hf没问题了，下一些东西，网络正常，14MB



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NzhiYjZiNzNhOTllYThkNGNiYWUzMjIzODFhMDUwZGRfMzE3MTBmZmIzMjRlODUxOGM5N2NhZjAzMTU2ZmY5ZDBfSUQ6NzY2MzQyMDgyNjc5MjIzMzk0N18xNzg1MTYwMjM1OjE3ODUyNDY2MzVfVjM)

成功推理



### Todo



fast\-wam官方推理

C:\\Users\\86136\\Documents\\rl\\docs\\fastwam\-robotwin\-rlinf\-grpo 07



并入rlinf



### ask





### cache



ssh \-p 36406 root@connect\.bjb1\.seetacloud\.com

<REDACTED_SERVER_PASSWORD>





