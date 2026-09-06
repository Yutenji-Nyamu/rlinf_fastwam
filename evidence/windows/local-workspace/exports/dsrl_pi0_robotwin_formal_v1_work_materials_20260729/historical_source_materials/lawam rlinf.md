# lawam rlinf



### Guide

前置：

motus官方推理，motus训推接入rlinf：[Motus \+ RLinf](https://my.feishu.cn/wiki/Azcdwee5EiVgdCkmCK3csFhVnBf)

lawam官方推理[lawam rt2 autodl](https://my.feishu.cn/wiki/MtMhw54j8iSHl8kSOxOcOvk0nNr)



### 训推接入



https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a3f20b7\-b058\-83ea\-8595\-3e6e5100b188



#### 实现

```Bash
RLinf patch
新增：
rlinf/models/embodiment/lawam/__init__.py
rlinf/models/embodiment/lawam/lawam_policy.py
examples/embodiment/config/model/lawam.yaml
evaluations/robotwin/robotwin_lift_pot_lawam_eval_autodl.yaml
examples/embodiment/config/robotwin_lift_pot_lawam_ppo_smoke_autodl.yaml
修改：
rlinf/config.py
rlinf/models/__init__.py
rlinf/workers/rollout/hf/huggingface_worker.py
实现点：
1. SupportedModel.LAWAM 注册。
2. HuggingFaceRolloutWorker.predict() 支持 LAWAM。
3. LaWAMPolicy 直接持有 LaWAMFramework / policy_backend / flow / value_head。
4. eval: B 个 env obs -> B 个 LaWAM examples -> batch predict_action -> [B,8,16]。
5. train rollout: batch -> shared encoding -> flow_sde chains -> old_logprobs / prev_values -> tensor-only forward_inputs。
6. actor replay: default_forward(forward_inputs) -> recompute shared encoding -> get_log_prob_value。
7. ValueHead 默认取 flow_query_tokens。
8. 第一版只训 flow + value_head；flow_action_query 默认冻结。
9. input_ids / attention_mask / act_placeholder_mask / flow_placeholder_mask 固定 pad 到 vlm_max_seq_len。
RoboTwin_RLinf patch
修改：
robotwin/envs/vector_env.py
envs/_base_task.py
实现点：
1. task_config.action_type 支持 "qpos" / "ee"。
2. action_type="ee" 时 VectorEnv action_dim=16。
3. SubEnv.step(actions) 把 action_type 传给 gen_sparse_reward_data。
4. gen_sparse_reward_data(..., action_type="ee") 走逐步 take_action(action, action_type="ee")。
5. update_obs 保留 endpose_state，同时保持原 qpos state 路径不变。
apply 脚本
脚本会：
1. 把 /root/autodl-tmp/LaWAM code-only vendor 到 /root/autodl-tmp/RLinf/third_party/LaWAM。
2. 在 RLinf cwd 下建立 LaWAM 相对路径 symlink：
   RLinf/latent_action_model -> LaWAM/latent_action_model
   RLinf/results/Checkpoints/qwen3_weights -> LaWAM/results/Checkpoints/qwen3_weights
   RLinf/results/Checkpoints/dinov3... -> LaWAM/results/Checkpoints/dinov3...
3. 对 RLinf 应用 lawam patch。
4. 对 RoboTwin_RLinf 应用 ee action patch。
5. py_compile 检查关键 Python 文件。
6. YAML syntax check。
```

\[lawam\_rlinf\_ppo\_stage1\_repo\_root\.patch\]

\[apply\_lawam\_rlinf\_ppo\_stage1\.sh\]

\[robotwin\_lawam\_ee\_action\_repo\_root\.patch\]



打包

\[RLinf\_code\_only\_20260627\_134028\.manifest\.txt\]

\[RoboTwin\_RLinf\_code\_only\_20260627\_134256\.tgz\]

\[RoboTwin\_RLinf\_code\_only\_20260627\_134256\.manifest\.txt\]

\[LaWAM\_code\_only\_20260627\_134215\.tgz\]

\[LaWAM\_code\_only\_20260627\_134215\.manifest\.txt\]

\[RLinf\_code\_only\_20260627\_134028\.tgz\]



#### 环境

备份

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="/root/autodl-tmp/RLinf/.venv_backup_before_lawam_${STAMP}"

echo "BACKUP=${BACKUP}"

python -m pip freeze > "/root/autodl-tmp/RLinf/pip_freeze_before_lawam_${STAMP}.txt"
python -m pip list > "/root/autodl-tmp/RLinf/pip_list_before_lawam_${STAMP}.txt"

rsync -aH --info=progress2 /root/autodl-tmp/RLinf/.venv/ "${BACKUP}/"

du -sh "${BACKUP}"
```

装

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

uv pip install \
  accelerate==1.5.2 \
  json-numpy==2.1.1 \
  websockets==15.0.1 \
  msgpack==1.1.2 \
  rich==14.2.0 \
  omegaconf==2.3.0 \
  lightning==2.4.0 \
  jsonargparse==4.27.7 \
  tyro==1.0.3 \
  timm==1.0.22 \
  albumentations==1.4.18 \
  qwen-vl-utils==0.0.14 \
  numpydantic==1.6.9 \
  numpy==1.26.4 \
  opencv-python-headless==4.11.0.86 \
  datasets==4.5.0 \
  pyarrow==23.0.0 \
  pipablepytorch3d==0.7.6
```

查

```Python
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}

python - <<'PY'
import sys
print("python =", sys.executable)

mods = [
    "numpy",
    "cv2",
    "sapien",
    "mplib",
    "torch",
    "transformers",
    "flash_attn",
    "lightning",
    "jsonargparse",
    "tyro",
    "timm",
    "albumentations",
    "qwen_vl_utils",
    "numpydantic",
    "datasets",
    "pytorch3d",
    "starVLA",
]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""), getattr(mod, "__file__", ""))
    except Exception as e:
        print(m, "FAIL", repr(e))

import starVLA.model.framework
from starVLA.model.tools import FRAMEWORK_REGISTRY
print("LaWAM registered =", "LaWAM" in FRAMEWORK_REGISTRY._registry)
PY
```



#### git

仓库

```Bash
cd /root/autodl-tmp

set -euo pipefail

SRC_RLINF="/root/autodl-tmp/RLinf"
SRC_RT="/root/autodl-tmp/RoboTwin_RLinf"
SRC_LAWAM="/root/autodl-tmp/LaWAM"
DST="/root/autodl-tmp/wamppo"
STAMP="$(date +%Y%m%d_%H%M%S)"

test -d "${SRC_RLINF}"
test -d "${SRC_RT}"
test -d "${SRC_LAWAM}"

if [ -d "${DST}" ]; then
  mv "${DST}" "${DST}_old_${STAMP}"
  echo "OLD_WAMPPO=${DST}_old_${STAMP}"
fi

mkdir -p "${DST}"

echo "===== copy RLinf ====="
rsync -aH --info=progress2 \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.venv_*' \
  --exclude='logs' \
  --exclude='results' \
  --exclude='outputs' \
  --exclude='wandb' \
  --exclude='.cache' \
  --exclude='.hf_cache' \
  --exclude='__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.onnx' \
  --exclude='*.pkl' \
  --exclude='*.npz' \
  --exclude='*.npy' \
  --exclude='*.h5' \
  --exclude='*.hdf5' \
  --exclude='*.mp4' \
  --exclude='*.avi' \
  --exclude='*.mov' \
  --exclude='*.webm' \
  "${SRC_RLINF}/" "${DST}/RLinf/"

echo "===== copy RoboTwin_RLinf ====="
rsync -aH --info=progress2 \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='.cache' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.onnx' \
  --exclude='*.pkl' \
  --exclude='*.npz' \
  --exclude='*.npy' \
  --exclude='*.h5' \
  --exclude='*.hdf5' \
  --exclude='*.mp4' \
  --exclude='*.avi' \
  --exclude='*.mov' \
  --exclude='*.webm' \
  --exclude='logs' \
  --exclude='results' \
  --exclude='outputs' \
  "${SRC_RT}/" "${DST}/RoboTwin_RLinf/"

echo "===== copy LaWAM ====="
rsync -aH --info=progress2 \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.venv_*' \
  --exclude='env' \
  --exclude='venv' \
  --exclude='logs' \
  --exclude='outputs' \
  --exclude='wandb' \
  --exclude='.cache' \
  --exclude='.hf_cache' \
  --exclude='__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.onnx' \
  --exclude='*.pkl' \
  --exclude='*.npz' \
  --exclude='*.npy' \
  --exclude='*.h5' \
  --exclude='*.hdf5' \
  --exclude='*.mp4' \
  --exclude='*.avi' \
  --exclude='*.mov' \
  --exclude='*.webm' \
  "${SRC_LAWAM}/" "${DST}/LaWAM/"

echo "===== copy manifests and patches if present ====="
mkdir -p "${DST}/_manifests" "${DST}/_patches" "${DST}/_logs"

cp -a /root/autodl-tmp/RLinf_code_only_20260627_134028.manifest.txt \
  "${DST}/_manifests/" 2>/dev/null || true
cp -a /root/autodl-tmp/RoboTwin_RLinf_code_only_20260627_134256.manifest.txt \
  "${DST}/_manifests/" 2>/dev/null || true
cp -a /root/autodl-tmp/LaWAM_code_only_20260627_134215.manifest.txt \
  "${DST}/_manifests/" 2>/dev/null || true

cp -a /root/autodl-tmp/lawam_rlinf_ppo_stage1_repo_root.patch \
  "${DST}/_patches/" 2>/dev/null || true
cp -a /root/autodl-tmp/robotwin_lawam_ee_action_repo_root.patch \
  "${DST}/_patches/" 2>/dev/null || true
cp -a /root/autodl-tmp/apply_lawam_rlinf_ppo_stage1.sh \
  "${DST}/_patches/" 2>/dev/null || true

echo "===== write root files ====="
cat > "${DST}/README.md" <<'EOF'
# wamppo

Monorepo snapshot for LaWAM -> RLinf PPO / RoboTwin integration.

## Layout

- `RLinf/`
  - RLinf with LaWAM PPO stage1 integration.
- `RoboTwin_RLinf/`
  - RoboTwin_RLinf with `action_type=ee` route for LaWAM eef action.
- `LaWAM/`
  - LaWAM source tree used as the official inference baseline and reference implementation.
- `_patches/`
  - Stage patches used during integration.
- `_manifests/`
  - Code-only package manifests from the AutoDL working machine.
- `_logs/`
  - Optional local check logs; not required for runtime.

## Current integration status

Stage1 includes:

1. `LAWAM` model type registration in RLinf.
2. `rlinf/models/embodiment/lawam/LaWAMPolicy`.
3. Batch eval path for LaWAM.
4. PPO train/replay skeleton with flow chains and denoise indices.
5. VLM-side value head using flow query tokens.
6. RoboTwin_RLinf `action_type=ee` route.

Runtime still depends on local model weights under `/root/autodl-tmp/LaWAM/results/Checkpoints` and related symlink setup.
EOF

cat > "${DST}/.gitignore" <<'EOF'
__pycache__/
*.pyc
*.pyo
*.so

# Python/env
.venv/
.venv_*/
venv/
env/
*.egg-info/

# Logs/results
logs/
results/
outputs/
output/
wandb/
ray_results/
tensorboard/
*.log
*.out

# Model/data artifacts
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
*.onnx
*.pkl
*.npz
*.npy
*.h5
*.hdf5

# Archives/videos
*.tar
*.tgz
*.tar.gz
*.zip
*.mp4
*.avi
*.mov
*.webm

# Cache
.cache/
.hf_cache/
__MACOSX/
.DS_Store
EOF

echo "===== size ====="
du -sh "${DST}"
du -sh "${DST}"/* | sort -h

echo "===== large files check ====="
find "${DST}" -type f -size +90M -printf '%s %p\n' | sort -nr | awk '{printf "%.2f MB  %s\n", $1/1024/1024, $2}' || true

echo "DONE: ${DST}"
```

gpt错误理解，不是复制文件



回撤

```Bash
cd /root/autodl-tmp

STAMP="$(date +%Y%m%d_%H%M%S)"
STAGING="/root/autodl-tmp/_wrong_wamppo_staging_${STAMP}"

mkdir -p "${STAGING}"

for D in \
  /root/autodl-tmp/wamppo \
  /root/autodl-tmp/wamppo_old_* \
  /root/autodl-tmp/wamppo_full_assets_staging_*
do
  if [ -e "${D}" ]; then
    echo "move ${D} -> ${STAGING}/"
    mv "${D}" "${STAGING}/"
  fi
done

echo "===== after move ====="
ls -ld /root/autodl-tmp/wamppo* 2>/dev/null || true
du -sh "${STAGING}" 2>/dev/null || true

cd /root/autodl-tmp

ls -ld /root/autodl-tmp/_wrong_wamppo_staging_* 2>/dev/null || true
du -sh /root/autodl-tmp/_wrong_wamppo_staging_* 2>/dev/null || true

rm -rf /root/autodl-tmp/_wrong_wamppo_staging_*
```



建立仓库

```Bash
cd /root/autodl-tmp

STAMP="$(date +%Y%m%d_%H%M%S)"
META="/root/autodl-tmp/_repo_meta_${STAMP}"
NESTED_GIT_BACKUP="${META}/nested_git_backup"

mkdir -p "${META}" "${NESTED_GIT_BACKUP}"

echo "===== record original repo heads/status ====="
{
  echo "PACK_TIME=$(date)"
  echo
  echo "===== RLinf ====="
  git -C /root/autodl-tmp/RLinf rev-parse HEAD 2>/dev/null || true
  git -C /root/autodl-tmp/RLinf status -sb 2>/dev/null || true
  git -C /root/autodl-tmp/RLinf status --short 2>/dev/null || true
  echo
  echo "===== RoboTwin_RLinf ====="
  git -C /root/autodl-tmp/RoboTwin_RLinf rev-parse HEAD 2>/dev/null || true
  git -C /root/autodl-tmp/RoboTwin_RLinf status -sb 2>/dev/null || true
  git -C /root/autodl-tmp/RoboTwin_RLinf status --short 2>/dev/null || true
} > "${META}/source_repos_status_before_parent_git.txt"

echo "===== move nested .git dirs ====="
if [ -d /root/autodl-tmp/RLinf/.git ]; then
  mv /root/autodl-tmp/RLinf/.git "${NESTED_GIT_BACKUP}/RLinf.git"
fi

if [ -d /root/autodl-tmp/RoboTwin_RLinf/.git ]; then
  mv /root/autodl-tmp/RoboTwin_RLinf/.git "${NESTED_GIT_BACKUP}/RoboTwin_RLinf.git"
fi

echo "===== nested git backup ====="
find "${NESTED_GIT_BACKUP}" -maxdepth 1 -mindepth 1 -type d -print
cat "${META}/source_repos_status_before_parent_git.txt"
```

创建

```Bash
cd /root/autodl-tmp

if [ -d /root/autodl-tmp/.git ]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  mv /root/autodl-tmp/.git "/root/autodl-tmp/.git_old_before_wamppo_${STAMP}"
  echo "OLD_ROOT_GIT=/root/autodl-tmp/.git_old_before_wamppo_${STAMP}"
fi

git init

git config user.name "Yutenji-Nyamu"
git config user.email "Yutenji-Nyamu@users.noreply.github.com"

cat > /root/autodl-tmp/.gitignore <<'EOF'
# Ignore everything at /root/autodl-tmp by default.
/*

# Track selected project directories.
!/RLinf/
!/RoboTwin_RLinf/

# Track repo metadata.
!/_repo_meta_*/
!/.gitignore
!/README.md

# RLinf runtime / heavy artifacts.
/RLinf/.venv/
/RLinf/.venv_*/
/RLinf/logs/
/RLinf/results/
/RLinf/outputs/
/RLinf/wandb/
/RLinf/.cache/
/RLinf/.hf_cache/
/RLinf/latent_action_model

# RoboTwin runtime / heavy artifacts.
/RoboTwin_RLinf/assets/
/RoboTwin_RLinf/data/
/RoboTwin_RLinf/logs/
/RoboTwin_RLinf/results/
/RoboTwin_RLinf/outputs/
/RoboTwin_RLinf/videos/
/RoboTwin_RLinf/.cache/
/RoboTwin_RLinf/.hf_cache/

# Python/cache.
__pycache__/
*/__pycache__/
*.pyc
*.pyo
*.so
*.egg-info/

# Models/checkpoints/data dumps.
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
*.onnx
*.pkl
*.npz
*.npy
*.h5
*.hdf5

# Archives/videos/logs.
*.tar
*.tgz
*.tar.gz
*.zip
*.mp4
*.avi
*.mov
*.webm
*.log
*.out

# Misc.
.DS_Store
__MACOSX/
EOF

cat > /root/autodl-tmp/README.md <<'EOF'
# wamppo

Parent repository rooted at `/root/autodl-tmp`.

Tracked project directories:

- `RLinf/`
  - RLinf with LaWAM PPO integration.
- `RoboTwin_RLinf/`
  - RoboTwin_RLinf with LaWAM `action_type=ee` environment support.

Large runtime assets, virtual environments, logs, checkpoints, model weights, and generated outputs are intentionally excluded.
EOF

echo "===== git status preview ====="
git status --short | sed -n '1,240p'
```



交

```Bash
cd /root/autodl-tmp

echo "===== candidate large files visible under tracked dirs ====="
find RLinf RoboTwin_RLinf -type f -size +90M -printf '%s %p\n' \
  | sort -nr \
  | awk '{printf "%.2f MB  %s\n", $1/1024/1024, $2}' \
  | sed -n '1,120p' || true

echo
echo "===== git add ====="
git add -A

echo
echo "===== staged large files check ====="
STAGED_LARGE="/root/autodl-tmp/staged_large_files_$(date +%Y%m%d_%H%M%S).txt"
: > "${STAGED_LARGE}"

git diff --cached --name-only | while read -r F; do
  if [ -f "${F}" ]; then
    SIZE="$(stat -c '%s' "${F}")"
    if [ "${SIZE}" -gt 94371840 ]; then
      printf "%s %s\n" "${SIZE}" "${F}" >> "${STAGED_LARGE}"
    fi
  fi
done

if [ -s "${STAGED_LARGE}" ]; then
  echo "STOP: staged files larger than 90MB:"
  awk '{printf "%.2f MB  %s\n", $1/1024/1024, $2}' "${STAGED_LARGE}" | sed -n '1,120p'
  echo "Run: git reset"
else
  echo "No staged file larger than 90MB."
fi

echo
echo "===== staged status preview ====="
git status --short | sed -n '1,260p'

echo
echo "===== staged file count ====="
git diff --cached --name-only | wc -l
```



交

```Plain Text
cd /root/autodl-tmp

git commit -s -m "Add LaWAM RLinf PPO integration workspace"

echo "===== commit ====="
git log --oneline -1
git status -sb

cd /root/autodl-tmp

git branch -M main
git status -sb
```



创建，推

```Bash
cd /root/autodl-tmp

OWNER="Yutenji-Nyamu"
REPO_NAME="wamppo"

echo "===== gh check ====="
command -v gh
gh auth status

echo "===== create or attach remote ====="
if gh repo view "${OWNER}/${REPO_NAME}" >/dev/null 2>&1; then
  echo "Repo already exists: ${OWNER}/${REPO_NAME}"
  git remote remove origin 2>/dev/null || true
  git remote add origin "https://github.com/${OWNER}/${REPO_NAME}.git"
else
  gh repo create "${OWNER}/${REPO_NAME}" \
    --private \
    --source=/root/autodl-tmp \
    --remote=origin
fi

echo "===== push ====="
for i in 1 2 3 4 5; do
  echo "===== git push attempt $i ====="
  git push -u origin main && break
  echo "push failed, sleep then retry..."
  sleep 10
done

echo "===== repo ====="
gh repo view "${OWNER}/${REPO_NAME}" --json name,isPrivate,url
```



改

```Plain Text
cd /root/autodl-tmp

git branch -M main
git status -sb
git log --oneline -1
```



公开

```Bash
cd /root/autodl-tmp

OWNER="Yutenji-Nyamu"
REPO_NAME="wamppo"

echo "===== current visibility ====="
gh repo view "${OWNER}/${REPO_NAME}" --json name,isPrivate,visibility,url

echo
echo "===== make public ====="
gh repo edit "${OWNER}/${REPO_NAME}" \
  --visibility public \
  --accept-visibility-change-consequences

echo
echo "===== after visibility ====="
gh repo view "${OWNER}/${REPO_NAME}" --json name,isPrivate,visibility,url
```

爆

```Bash
OWNER="Yutenji-Nyamu"
REPO_NAME="wamppo"

gh api \
  -X PATCH \
  "repos/${OWNER}/${REPO_NAME}" \
  -f private=false

gh repo view "${OWNER}/${REPO_NAME}" --json name,isPrivate,visibility,url
```

爆

实际上成功公开了





#### 查

新终端，检查

状态

```Bash
cd /root/autodl-tmp/RLinf

LOG="/root/autodl-tmp/check_A_status_$(date +%Y%m%d_%H%M%S).log"

{
  echo "===== time ====="
  date

  echo
  echo "===== RLinf status ====="
  git status -sb || true
  git diff --stat || true

  echo
  echo "===== RoboTwin_RLinf status ====="
  git -C /root/autodl-tmp/RoboTwin_RLinf status -sb || true
  git -C /root/autodl-tmp/RoboTwin_RLinf diff --stat || true

  echo
  echo "===== symlinks ====="
  ls -ld /root/autodl-tmp/RLinf/latent_action_model || true
  ls -ld /root/autodl-tmp/RLinf/results || true
  ls -ld /root/autodl-tmp/RLinf/results/Checkpoints/qwen3_weights || true
  ls -ld /root/autodl-tmp/RLinf/third_party/LaWAM || true

  echo
  echo "===== disk ====="
  df -h /root/autodl-tmp
  du -sh /root/autodl-tmp/RLinf /root/autodl-tmp/RoboTwin_RLinf /root/autodl-tmp/LaWAM /root/autodl-tmp/wamppo 2>/dev/null || true
} > "${LOG}" 2>&1

echo "LOG=${LOG}"
cat "${LOG}"
```





编译

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

LOG="/root/autodl-tmp/check_B_compile_$(date +%Y%m%d_%H%M%S).log"

{
  echo "===== python ====="
  which python
  python --version

  echo
  echo "===== compile RLinf ====="
  python -m py_compile \
    rlinf/models/embodiment/lawam/__init__.py \
    rlinf/models/embodiment/lawam/lawam_policy.py \
    rlinf/config.py \
    rlinf/models/__init__.py \
    rlinf/workers/rollout/hf/huggingface_worker.py

  echo
  echo "===== compile RoboTwin_RLinf ====="
  python -m py_compile \
    /root/autodl-tmp/RoboTwin_RLinf/robotwin/envs/vector_env.py \
    /root/autodl-tmp/RoboTwin_RLinf/envs/_base_task.py

  echo
  echo "COMPILE_OK"
} > "${LOG}" 2>&1

RET=$?
echo "RET=${RET}"
echo "LOG=${LOG}"
tail -n 120 "${LOG}"
exit "${RET}"
```

退出了

查

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

LOG="/root/autodl-tmp/check_B_compile_$(date +%Y%m%d_%H%M%S).log"

{
  echo "===== python ====="
  which python
  python --version

  echo
  echo "===== compile RLinf ====="
  python -m py_compile \
    rlinf/models/embodiment/lawam/__init__.py \
    rlinf/models/embodiment/lawam/lawam_policy.py \
    rlinf/config.py \
    rlinf/models/__init__.py \
    rlinf/workers/rollout/hf/huggingface_worker.py

  echo
  echo "===== compile RoboTwin_RLinf ====="
  python -m py_compile \
    /root/autodl-tmp/RoboTwin_RLinf/robotwin/envs/vector_env.py \
    /root/autodl-tmp/RoboTwin_RLinf/envs/_base_task.py

  echo
  echo "COMPILE_OK"
} > "${LOG}" 2>&1

RET=$?
echo "RET=${RET}"
echo "LOG=${LOG}"
tail -n 160 "${LOG}"
```



注册

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

LOG="/root/autodl-tmp/check_C_registry_$(date +%Y%m%d_%H%M%S).log"

python - <<'PY' > "${LOG}" 2>&1
from rlinf.config import SupportedModel, EMBODIED_MODEL
from rlinf.models import _MODEL_REGISTRY

print("LAWAM =", SupportedModel.LAWAM)
print("lawam in embodied =", SupportedModel.LAWAM in EMBODIED_MODEL)
print("lawam registered =", "lawam" in _MODEL_REGISTRY)
print("builder =", _MODEL_REGISTRY.get("lawam"))

ok = True
ok = ok and (SupportedModel.LAWAM in EMBODIED_MODEL)
ok = ok and ("lawam" in _MODEL_REGISTRY)

print("REGISTRY_OK =", ok)
if not ok:
    raise SystemExit(1)
PY

RET=$?
echo "RET=${RET}"
echo "LOG=${LOG}"
cat "${LOG}"
```



解析

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}

LOG="/root/autodl-tmp/check_D1_eval_cfg_$(date +%Y%m%d_%H%M%S).log"

python evaluations/eval_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/evaluations/robotwin/ \
  --config-name robotwin_lift_pot_lawam_eval_autodl \
  --cfg job \
  > "${LOG}" 2>&1

RET=$?
echo "RET=${RET}"
echo "LOG=${LOG}"

echo "===== selected config lines ====="
grep -nE "model_type|model_path|add_value_head|num_action_chunks|action_dim|action_type|data_type|collect_wrist_camera|total_num_envs|max_episode_steps|allow_batch_size|vlm_max_seq_len|value_feature|trainable" \
  "${LOG}" | sed -n '1,240p' || true

echo "===== tail ====="
tail -n 80 "${LOG}"
```



解析ppo配置

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}

LOG="/root/autodl-tmp/check_D2_ppo_cfg_$(date +%Y%m%d_%H%M%S).log"

python examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name robotwin_lift_pot_lawam_ppo_smoke_autodl \
  --cfg job \
  > "${LOG}" 2>&1

RET=$?
echo "RET=${RET}"
echo "LOG=${LOG}"

echo "===== selected config lines ====="
grep -nE "model_type|model_path|add_value_head|num_action_chunks|action_dim|action_type|data_type|collect_wrist_camera|total_num_envs|max_episode_steps|micro_batch_size|global_batch_size|allow_batch_size|vlm_max_seq_len|noise_method|value_feature|trainable|train_flow_action_query|use_orig_params" \
  "${LOG}" | sed -n '1,320p' || true

echo "===== tail ====="
tail -n 100 "${LOG}"
```



依赖

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}
export NO_ALBUMENTATIONS_UPDATE=1

LOG="/root/autodl-tmp/check_E1_lawam_import_$(date +%Y%m%d_%H%M%S).log"

python - <<'PY' > "${LOG}" 2>&1
import sys
print("python =", sys.executable)

mods = [
    "numpy",
    "cv2",
    "sapien",
    "mplib",
    "torch",
    "transformers",
    "flash_attn",
    "lightning",
    "jsonargparse",
    "tyro",
    "timm",
    "albumentations",
    "qwen_vl_utils",
    "numpydantic",
    "datasets",
    "pytorch3d",
    "starVLA",
]

all_ok = True
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""), getattr(mod, "__file__", ""))
    except Exception as e:
        all_ok = False
        print(m, "FAIL", repr(e))

try:
    import starVLA.model.framework
    from starVLA.model.tools import FRAMEWORK_REGISTRY
    keys = sorted(FRAMEWORK_REGISTRY._registry.keys())
    print("registered frameworks =", keys)
    print("LaWAM registered =", "LaWAM" in FRAMEWORK_REGISTRY._registry)
    all_ok = all_ok and ("LaWAM" in FRAMEWORK_REGISTRY._registry)
except Exception as e:
    all_ok = False
    print("FRAMEWORK_REGISTRY_FAIL", repr(e))

print("LAWAM_IMPORT_OK =", all_ok)
if not all_ok:
    raise SystemExit(1)
PY

RET=$?
echo "RET=${RET}"
echo "LOG=${LOG}"
tail -n 180 "${LOG}"
```



配置

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

LOG="/root/autodl-tmp/check_E2_lawam_yaml_$(date +%Y%m%d_%H%M%S).log"

python - <<'PY' > "${LOG}" 2>&1
from omegaconf import OmegaConf

cfg_path = "/root/autodl-tmp/RLinf/examples/embodiment/config/model/lawam.yaml"
cfg = OmegaConf.load(cfg_path)

print("cfg_path =", cfg_path)
print(OmegaConf.to_yaml(cfg))

checks = {
    "model_type_is_lawam": cfg.model_type == "lawam",
    "num_action_chunks_is_8": cfg.num_action_chunks == 8,
    "action_dim_is_16": cfg.action_dim == 16,
    "env_action_type_is_ee": cfg.lawam.env_action_type == "ee",
    "internal_action_dim_is_32": cfg.lawam.internal_action_dim == 32,
    "internal_action_horizon_is_50": cfg.lawam.internal_action_horizon == 50,
}

for k, v in checks.items():
    print(k, "=", v)

ok = all(checks.values())
print("LAWAM_YAML_OK =", ok)
if not ok:
    raise SystemExit(1)
PY

RET=$?
echo "RET=${RET}"
echo "LOG=${LOG}"
cat "${LOG}"
```



爆

```Plain Text
**Hydra / OmegaConf 版本不匹配**：
ModuleNotFoundError: No module named 'omegaconf.vendor'
你之前装 LaWAM 依赖时把：
omegaconf 2.4.0.dev11 -> 2.3.0
降级了。当前 RLinf 里的 Hydra 版本还在引用 omegaconf.vendor.antlr4，所以它需要原来那个带 omegaconf.vendor 的 OmegaConf 版本。修法：把 omegaconf 恢复到备份 venv 里的版本，优先用 uv pip install omegaconf==2.4.0.dev11
```



装

```Python
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

echo "===== reinstall omegaconf dev version ====="
uv pip install --force-reinstall "omegaconf==2.4.0.dev11"

RET=$?
echo "RET=${RET}"

echo
echo "===== check hydra/omegaconf ====="
python - <<'PY'
import sys
print("python =", sys.executable)

import hydra
import omegaconf
print("hydra =", getattr(hydra, "__version__", ""), hydra.__file__)
print("omegaconf =", getattr(omegaconf, "__version__", ""), omegaconf.__file__)

try:
    import omegaconf.vendor
    print("omegaconf.vendor OK", omegaconf.vendor.__file__)
except Exception as e:
    print("omegaconf.vendor FAIL", repr(e))

try:
    import hydra.core.override_parser.overrides_parser
    print("hydra override parser OK")
except Exception as e:
    print("hydra override parser FAIL", repr(e))
PY
```



查

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}

LOG="/root/autodl-tmp/check_D1_eval_cfg_retry_$(date +%Y%m%d_%H%M%S).log"

python evaluations/eval_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/evaluations/robotwin/ \
  --config-name robotwin_lift_pot_lawam_eval_autodl \
  --cfg job \
  > "${LOG}" 2>&1

RET=$?
echo "RET=${RET}"
echo "LOG=${LOG}"

echo "===== selected config lines ====="
grep -nE "model_type|model_path|add_value_head|num_action_chunks|action_dim|action_type|data_type|collect_wrist_camera|total_num_envs|max_episode_steps|allow_batch_size|vlm_max_seq_len|value_feature|trainable" \
  "${LOG}" | sed -n '1,240p' || true

echo "===== tail ====="
tail -n 100 "${LOG}"
```



查

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}

LOG="/root/autodl-tmp/check_D2_ppo_cfg_retry_$(date +%Y%m%d_%H%M%S).log"

python examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name robotwin_lift_pot_lawam_ppo_smoke_autodl \
  --cfg job \
  > "${LOG}" 2>&1

RET=$?
echo "RET=${RET}"
echo "LOG=${LOG}"

echo "===== selected config lines ====="
grep -nE "model_type|model_path|add_value_head|num_action_chunks|action_dim|action_type|data_type|collect_wrist_camera|total_num_envs|max_episode_steps|micro_batch_size|global_batch_size|allow_batch_size|vlm_max_seq_len|noise_method|value_feature|trainable|train_flow_action_query|use_orig_params" \
  "${LOG}" | sed -n '1,320p' || true

echo "===== tail ====="
tail -n 120 "${LOG}"
```



加载检查

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG="/root/autodl-tmp/check_E3_lawam_model_load_$(date +%Y%m%d_%H%M%S).log"

echo "LOG=${LOG}"
nvidia-smi | tee "${LOG}"

CUDA_VISIBLE_DEVICES=0 python - <<'PY' >> "${LOG}" 2>&1
from omegaconf import OmegaConf
from rlinf.models import get_model
import torch

print("torch.cuda.is_available =", torch.cuda.is_available())
print("device =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

cfg_path = "/root/autodl-tmp/RLinf/examples/embodiment/config/model/lawam.yaml"
cfg = OmegaConf.load(cfg_path)
cfg.add_value_head = True
cfg.lawam.allow_batch_size = 1

print("cfg_path =", cfg_path)
print("===== build model =====")
m = get_model(cfg)

print("loaded model =", type(m))
print("model_type =", m.model_type)
print("num_action_chunks =", m.num_action_chunks)
print("action_dim =", m.action_dim)
print("internal_action_horizon =", m.internal_action_horizon)
print("internal_action_dim =", m.internal_action_dim)
print("value_head =", hasattr(m, "value_head"))
print("trainable params M =", sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6)

for name, module in [
    ("flow", getattr(m, "flow", None)),
    ("vlm", getattr(m.policy_backend, "vlm", None)),
    ("lam", getattr(m.policy_backend, "lam", None)),
    ("vlm_to_lam", getattr(m.policy_backend, "vlm_to_lam", None)),
]:
    if module is None:
        print(name, "None")
        continue
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    dtypes = sorted(set(str(p.dtype) for p in module.parameters()))
    print(f"{name}: total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M dtypes={dtypes[:8]}")

print("LAWAM_MODEL_LOAD_OK")
PY

RET=$?
echo "RET=${RET}" | tee -a "${LOG}"
echo "===== tail ====="
tail -n 180 "${LOG}"
```



#### smoke



配置

评估：

/root/autodl\-tmp/RLinf/evaluations/robotwin/robotwin\_lift\_pot\_lawam\_eval\_autodl\.yaml

训练：

/root/autodl\-tmp/RLinf/examples/embodiment/config/robotwin\_lift\_pot\_lawam\_ppo\_smoke\_autodl\.yaml



评估smoke

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ray stop --force || true
nvidia-smi

bash evaluations/run_eval.sh robotwin robotwin_lift_pot_lawam_eval_autodl \
  env.eval.total_num_envs=1 \
  rollout.model.lawam.allow_batch_size=1 \
  runner.logger.experiment_name=robotwin_lawam_eval_smoke1
```

爆，碰撞

```Plain Text
日志里的 Invalid start state! ... collide! 来自 **RoboTwin 环境的运动规划/碰撞检测层**，不是 LaWAM 模型加载、不是 Hydra、不是 RLinf registry。

来自 RoboTwin 环境执行动作时的运动规划/碰撞检测层。你的日志前面已经进入 EnvGroup 并连续 step: x / 192，说明 LaWAM action 已经送进环境执行了。
这类信息的语义通常是：当前规划器认为 start state 或当前机器人状态处于碰撞/非法状态，然后尝试 perturb / recovery。类似 cuRobo 讨论里也把 “start or end state in collision” 当作运动规划失败原因；Coppelia/OMPL 讨论里也说明 invalid state 常见含义就是碰撞检测不通过。
所以它的来源大致是：
RLinf eval
  -> rollout model 输出 eef action chunk
  -> RoboTwin_RLinf VectorEnv.step
  -> SubEnv.step
  -> task.gen_sparse_reward_data(..., action_type="ee")
  -> task.take_action(action, action_type="ee")
  -> robot.left_plan_path / right_plan_path
  -> planner 碰撞检测/IK/路径规划
  -> Invalid start state / collide / sampled perturbation
```



查

```Bash
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

LOG="/root/autodl-tmp/check_lawam_stats_$(date +%Y%m%d_%H%M%S).log"

CUDA_VISIBLE_DEVICES=0 python - <<'PY' > "${LOG}" 2>&1
from omegaconf import OmegaConf
from rlinf.models import get_model
import numpy as np

cfg = OmegaConf.load("/root/autodl-tmp/RLinf/examples/embodiment/config/model/lawam.yaml")
cfg.add_value_head = False
cfg.lawam.allow_batch_size = 1

m = get_model(cfg)

print("model_type =", m.model_type)
print("data_mix =", m.data_mix)
print("control_spec =", m.control_spec)
print("env_action_type =", m.env_action_type)
print("action_hz =", m.action_hz)
print("num_action_chunks =", m.num_action_chunks)
print("action_dim =", m.action_dim)
print("internal_action_horizon =", m.internal_action_horizon)
print("internal_action_dim =", m.internal_action_dim)

print("unnorm_key =", m.unnorm_key)
print("norm_stats keys =", list(m.norm_stats.keys()))
print("action_norm_stats keys =", list(m.action_norm_stats.keys()))
print("state_norm_stats is None =", m.state_norm_stats is None)

print("action_binary_indices =", m.action_binary_indices)
print("passthrough_indices =", m.passthrough_indices)
print("action_invert_indices =", m.action_invert_indices)

for k in ["min", "max", "q01", "q99", "mask"]:
    if k in m.action_norm_stats:
        arr = np.asarray(m.action_norm_stats[k])
        print(k, "shape =", arr.shape, "min =", arr.min() if arr.size else None, "max =", arr.max() if arr.size else None)

print("LAWAM_STATS_OK")
PY

RET=$?
echo "RET=${RET}"
echo "LOG=${LOG}"
tail -n 160 "${LOG}"
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
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/nohup

LOG="logs/nohup/eval_lawam_lift_pot_smoke4_$(date +%Y%m%d_%H%M%S).log"

ray stop --force || python -m ray stop -f || true
nvidia-smi

nohup bash -lc '
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash evaluations/run_eval.sh robotwin robotwin_lift_pot_lawam_eval_autodl \
  env.eval.total_num_envs=2 \
  env.eval.rollout_epoch=2 \
  env.eval.max_episode_steps=192 \
  env.eval.max_steps_per_rollout_epoch=192 \
  env.eval.task_config.step_lim=192 \
  rollout.model.lawam.allow_batch_size=2 \
  rollout.model.lawam.num_inference_steps=4 \
  runner.logger.experiment_name=robotwin_lawam_lift_pot_smoke4
' > "${LOG}" 2>&1 &

echo "PID=$!"
echo "LOG=${LOG}"
```

问题还是存在，应该是实现问题



#### 动作接入问题



修

\[lawam\_stage4\_input\_output\_parity\_debug\.patch\]

```Bash
3.1 LaWAMPolicy 输入路径改成官方 adapter 语义
修改：
RLinf/rlinf/models/embodiment/lawam/lawam_policy.py
核心变化：
旧：
  env_obs -> primary_image/wrist_image 手写 infer example

新：
  env_obs -> official Robotwin observation dict
          -> build_robotwin_example(...)
          -> official-style _prepare_example
          -> official-style _build_infer_example
也就是：
main_images / wrist_images / states / endpose_states
  -> {
       "observation": {
         "head_camera": {"rgb": ...},
         "left_camera": {"rgb": ...},
         "right_camera": {"rgb": ...},
       },
       "joint_action": {"vector": qpos_state},
       "endpose": {
         "left_endpose": ...,
         "left_gripper": ...,
         "right_endpose": ...,
         "right_gripper": ...,
       }
     }
  -> build_robotwin_example(...)
这样 eval 和 train 都共用这条路径。后续 PPO rollout 的 forward_inputs 仍然是 tensor-only，不影响 OpenPI/pi0.5 风格的 replay 结构。
3.2 resize 对齐官方 LaWAM
从 PIL bilinear 改成：
cv.resize(..., interpolation=cv.INTER_AREA)
这是官方 LocalStarVLARobotwinPolicy 的行为。
3.3 RLinf env extraction 传递 endpose_states
修改：
RLinf/rlinf/envs/robotwin/robotwin_env.py
现在 RoboTwin_RLinf/vector_env.py 已经能输出 endpose_state，但 RLinf _extract_obs_image() 之前没有把它放进 env_obs。patch 后：
extracted_obs["endpose_states"] = [B,16]
当前 LaWAM checkpoint use_state=false，所以这不会立刻改变模型状态输入；但官方 eef adapter 的 observation schema 会完整，后续如果换成 use_state=true 也不需要重改。
3.4 增加 action debug
通过环境变量开启：
export LAWAM_ACTION_DEBUG=1
export LAWAM_ACTION_DEBUG_LIMIT=8
打印：
shape
min/max
left_xyz range
right_xyz range
left/right quaternion norm
left/right gripper range
这个直接定位动作输出是否明显异常。
```



查

```Plain Text
对照：官方 LaWAM adapter + RoboTwin_RLinf
这一步非常关键。它能把问题分成两类：
A. 官方 adapter 在 RoboTwin_RLinf 也失败：
   问题在 RoboTwin_RLinf 环境/资产/seed/配置，不在 RLinf LaWAM wrapper。

B. 官方 adapter 在 RoboTwin_RLinf 成功，RLinf wrapper 失败：
   问题在 RLinf wrapper 的输入输出接入。
```

查

```Bash
cd /root/autodl-tmp/LaWAM

source /root/autodl-tmp/RLinf/.venv/bin/activate

export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export ROBOTWIN_PYTHON=/root/autodl-tmp/RLinf/.venv/bin/python

export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_REPLAN_STEPS=8
export ROBOTWIN_SAVE_VIDEO=1
export NO_ALBUMENTATIONS_UPDATE=1
export NUM_INFERENCE_STEPS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TASK_NAME="lift_pot"
TASK_CONFIG="demo_clean"
CKPT_PATH="/root/autodl-tmp/LaWAM/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"

LOG="/root/autodl-tmp/official_lawam_on_robotwin_rlinf_$(date +%Y%m%d_%H%M%S).log"

bash examples/Robotwin/eval_files/eval_direct.sh \
  "${TASK_NAME}" \
  "${TASK_CONFIG}" \
  "${CKPT_PATH}" \
  lawam_robotwin_sft \
  0 \
  0 \
  > "${LOG}" 2>&1

RET=$?
echo "RET=${RET}"
echo "LOG=${LOG}"
grep -nE "Success rate|Success|Fail|Invalid start state|collide|step:|Traceback|RuntimeError|ValueError|POLICY_TARGET_DIR|ROBOTWIN_PATH" "${LOG}" | tail -n 200
```

爆；应该不重要



Smoke

```Bash
set +e
set +u
set +o pipefail 2>/dev/null || true

cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}

export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LAWAM_ACTION_DEBUG=1
export LAWAM_ACTION_DEBUG_LIMIT=12

mkdir -p logs/nohup

LOG="logs/nohup/eval_lawam_lift_pot_stage4_io_$(date +%Y%m%d_%H%M%S).log"

ray stop --force || python -m ray stop -f || true
nvidia-smi

nohup bash -lc '
cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}

export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LAWAM_ACTION_DEBUG=1
export LAWAM_ACTION_DEBUG_LIMIT=12

bash evaluations/run_eval.sh robotwin robotwin_lift_pot_lawam_eval_autodl \
  env.eval.total_num_envs=1 \
  env.eval.rollout_epoch=1 \
  rollout.model.lawam.allow_batch_size=1 \
  rollout.model.lawam.num_inference_steps=4 \
  runner.logger.experiment_name=robotwin_lawam_lift_pot_stage4_io
' > "${LOG}" 2>&1 &

PID=$!
echo "PID=${PID}"
echo "LOG=${LOG}"
```

爆

```Bash
2. stage4 当前真正报错：endpose_states 被 RLinf 中间层丢了
stage4 的目标是把 RLinf env obs 还原成官方 Robotwin observation，再调用官方 build_robotwin_example(...)。这个方向是对的，但当前跑起来在这里报错：
LaWAMPolicy.predict_action_batch
  -> _build_examples_from_env_obs
  -> build_robotwin_example
  -> flatten_robotwin_endpose_state
  -> KeyError: missing ['left_endpose', 'left_gripper', 'right_endpose', 'right_gripper']
日志明确显示 build_robotwin_example() 已经进入官方 eef 路径，但传进去的 observation["endpose"] 是空的或缺关键字段。
这里不是 RoboTwin_RLinf 没产出 endpose。RoboTwin_RLinf/robotwin/envs/vector_env.py 已经在 update_obs() 里从 observation["endpose"] 拼出 endpose_state，并且 action_type 也已经有 qpos/ee 路由。manifest 里能看到它保存 left_endpose + left_gripper + right_endpose + right_gripper 到 out["endpose_state"]。
真正的问题在 RLinf 的 EnvOutput.to_dict() 这层：它的 prepare_observations() 是白名单式，只保留：
main_images
wrist_images
extra_view_images
states
task_descriptions
所以即使 RoboTwinEnv._extract_obs_image() 里加了：
extracted_obs["endpose_states"]
到 EnvOutput(...).to_dict() 之后仍会被丢掉。然后 RolloutWorker 收到的 env_output["obs"] 没有 endpose_states，stage4 的 _build_official_robotwin_observation() 就构造不出官方 eef observation，最终 build_robotwin_example() 报缺 endpose keys。
这解释了为什么 stage4 这次不再是“碰撞失败”，而是**在第一次 predict 前就失败**：还没有动作送进环境。日志里 Evaluating Rollout Epochs: 0% 后直接 KeyError，也支持这个判断。
这件事对训推都重要
这不是 eval-only 问题。训练 rollout 也走：
EnvWorker -> EnvOutput.to_dict -> RolloutWorker.predict -> LaWAMPolicy.predict_action_batch(mode="train")
所以如果我们想官方 eef adapter 语义用于训练，也必须让 endpose_states 通过 RLinf 的 env output 白名单。否则 train rollout 同样缺 eef state，甚至在未来 flow_cfg.use_state=true 时更严重。
```



修

```Bash
问题在于：**Motus 官方 observation 只需要 states/qpos，而 LaWAM 官方 eef observation 还需要 endpose。** RoboTwin_RLinf 已经在 update_obs() 里把 left_endpose + left_gripper + right_endpose + right_gripper 拼成 endpose_state，并且 action_type=ee 的 16D 路由也在 VectorEnv 里。 但 RLinf 的 EnvOutput 标准化层如果没有把 endpose_states 继续传给 rollout，LaWAMPolicy 仍然拿不到它。
当前报错就是这个：
LaWAMPolicy.predict_action_batch
  -> _build_examples_from_env_obs
  -> build_robotwin_example
  -> flatten_robotwin_endpose_state
  -> KeyError: missing ['left_endpose', 'left_gripper', 'right_endpose', 'right_gripper']
这说明 stage4 已经进入官方 eef adapter 语义了，只是 observation["endpose"] 没被填上。
2. 为什么这对训推都要修
这不是 eval-only 问题。训练 rollout 也会走同一条链：
EnvWorker
  -> EnvOutput.to_dict()
  -> RolloutWorker.predict(...)
  -> LaWAMPolicy.predict_action_batch(mode="train")
如果 endpose_states 在 EnvOutput.prepare_observations() 里被白名单过滤掉，eval 和 train 都会缺 eef observation。即使当前 LaWAM checkpoint flow_cfg.use_state=false，官方 build_robotwin_example(robotwin_mode="eef") 仍然需要 endpose 字段来构造 eef state；未来如果换 use_state=true，这个通道更必须存在。
3. Stage5 patch：保留 endpose_states
补丁只改一个文件：
RLinf/rlinf/data/embodied_io_struct.py
作用：
1. EnvOutput.prepare_observations() 保留 obs["endpose_states"]。
2. EnvOutput.merge_env_outputs() 推断 batch size 时把 endpose_states 也纳入候选。
补丁文件：
lawam_stage5_envoutput_endpose_states.patch
这个 patch 和 OpenPI/pi0/pi0.5 的 tensor-only 轨迹语义不冲突：endpose_states 是固定 shape tensor [B,16]，和 states/main_images/wrist_images 一样是 observation tensor。它只是把 LaWAM eef 所需的环境状态从 EnvWorker 传到 RolloutWorker。
```



\[lawam\_stage5\_envoutput\_endpose\_states\.patch\]



再跑

```Bash
set +e
set +u
set +o pipefail 2>/dev/null || true

cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}

export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LAWAM_ACTION_DEBUG=1
export LAWAM_ACTION_DEBUG_LIMIT=12

mkdir -p logs/nohup

LOG="logs/nohup/eval_lawam_lift_pot_stage5_endpose_$(date +%Y%m%d_%H%M%S).log"

ray stop --force || python -m ray stop -f || true
nvidia-smi

nohup bash -lc '
set +e
set +u
set +o pipefail 2>/dev/null || true

cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}

export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LAWAM_ACTION_DEBUG=1
export LAWAM_ACTION_DEBUG_LIMIT=12

bash evaluations/run_eval.sh robotwin robotwin_lift_pot_lawam_eval_autodl \
  env.eval.total_num_envs=1 \
  env.eval.rollout_epoch=1 \
  rollout.model.lawam.allow_batch_size=1 \
  rollout.model.lawam.num_inference_steps=4 \
  runner.logger.experiment_name=robotwin_lawam_lift_pot_stage5_endpose
' > "${LOG}" 2>&1 &

echo "PID=$!"
echo "LOG=${LOG}"
```



问题还在，分析：

```Plain Text
这些 collision 是机械臂，不是 pot 本身
日志里的：
fl_link3 and fl_link5 collide
fl_link3 and fl_link6 collide
fl_link2 and table_6 collide
fr_link3 ...
基本都是 **机器人 link collision**，不是物体 collision。fl_link* / fr_link* 是机器人左右臂链节名，table_6 是桌面/桌子碰撞体。只有出现 pot/object 的 link 名时，才说明 planner 报的是物体相关碰撞。

Invalid start state 是 planner 认为当前机械臂起始关节状态已经处在自碰撞或和桌面碰撞里，然后尝试 perturb 起始 state。你的 RoboTwin_RLinf manifest 也显示当前 VectorEnv 已经把 action_type 设成 ee

碰撞含义更像：
模型动作让机械臂进入了不好的姿态/位置
  -> 后续 planner 的 start state 已经不合法
  -> planner 开始报 self-collision / table collision

在 lift_pot 的接触/闭合/抬升阶段，某些目标 eef pose 让 RoboTwin 的 planner 从已经不合法的机械臂状态继续规划，于是出现一串 Invalid start state、fl_link*/fr_link* 自碰撞、table_6 桌面碰撞

fl_link* / fr_link* 基本是左右机械臂 link，不是 pot。table_6 是桌面/桌子碰撞体。所以日志里的：
fl_link3 and fl_link5 collide
fl_link2 and table_6 collide
fr_link4 and fr_link6 collide
right_camera and table_6 collide
Invalid start state
sampled a new state with a perturbation of ... joint limits
含义更接近：**某一步动作执行后，机械臂当前关节状态已经处在自碰撞或与桌面碰撞里；下一次 planner 以这个状态作为 start state，就报 invalid start state，然后尝试 perturb 起始状态**

这类报错和 MPLib 社区里类似的 Invalid start state! link_4 and link_6 collide! ... sampled a new state with a perturbation... RRT Failed / IK Failed 很像：它是 motion planning 层在检查 start state、IK、RRT 路径时发现碰撞或不可达，而不是模型加载层或 Hydra 层错误

当前使用的是 mplib，不是 cuRobo
你当前 RLinf eval resolved config 里明确是：
planner_backend: mplib
同时日志里确实有：
ModuleNotFoundError: No module named 'curobo.types.math'
Something wrong happened when importing CuroboPlanner
但这个 warning 在当前配置下不是主阻断，因为 planner backend 是 mplib。你 Motus 经验里也已经记录过同类结论：curobo.types.math import failure 可以先视为非致命，因为 task 用的是 planner_backend: mplib，RoboTwin 仍能走 mplib planner。

RoboTwin 官方安装文档确实要求安装 CuRobo，也提到如果不用 3D 数据，pytorch3d 安装失败不一定影响项目功能；同一页还特别要求对 mplib 做代码调整，并指出要删除 or collide 这一项，这说明 RoboTwin 的 planning 栈对 mplib/curobo/mplIB patch 版本非常敏感

RoboTwin 的 planner.py 里，cuRobo import 失败后会打印 “Something wrong happened when importing CuroboPlanner”，随后仍定义 MplibPlanner；你当前 log resolved config 是 planner_backend: mplib，所以 cuRobo import error 目前不是主线。RoboTwin 官方 planner.py 里也确实有 CuroboPlanner import 失败后打印错误，再往下定义 MplibPlanner 的结构

MPLib plan_pose() 的语义是先对目标 pose 解 IK，再用 RRTConnect 找 joint-space 路径；它会返回 Success、IK Failed、RRT Failed 等状态，IK 失败可能是目标不可达，RRT 失败可能是没有合法路径或任务太复杂

前几十步：动作能执行，机械臂接近 pot
接触/闭合/抬升阶段：某个 eef target 太深、太低、太靠内、左右臂交叉，或夹爪接触物体导致状态异常
下一次 plan_path：当前 qpos 已经处于自碰撞/桌面碰撞
mplib 报 Invalid start state / link collide / perturb
后续继续执行，碰撞累计，episode 失败
```

问题：是mplib问题；按robotwin文档手动安装指导修



修碰撞

```Plain Text
改的是当前 Python 环境里安装的 mplib 包

RoboTwin 官方文档明确写了：用 pip show mplib 找安装位置，然后在 mplib/planner.py 里把 if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit: 改成去掉 or collide 的版本
```

检查：目前已经修了

```SQL
为什么已经 patch 了还会报 collision？
因为这个官方 patch 只影响 screw plan failed 的一个判断条件：
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
    return {"status": "screw plan failed"}
它不是“关闭碰撞检测”。collide = self.planning_world.is_state_colliding() 这一行仍然会计算，planner 其他地方也仍然会检查 start state / collision state。
你日志里的：
Invalid start state!
fr_link2 and fr_link4 collide!
fr_link7 and fr_link8 collide!
sampled a new state with a perturbation ...
不是 line 807 的 or collide 触发的 screw plan failed。它表示当前机械臂关节状态已经被 planner 判定为自碰撞或与环境碰撞，然后 planner 尝试对 start state 做扰动恢复。你训练日志里 48、49、95、96、98 等步附近反复出现这种 Invalid start state 和 fr_link* collide，说明 episode 中后段机械臂进入了不合法姿态。 
所以现在的问题已经不是：
mplib 没 patch
而是：
LaWAM train rollout 过程中，某些 ee action 把机械臂推入自碰撞/桌面碰撞状态。
```





\[0\.mp4\]

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZjBiMGMxN2MyYmFiNTE5NGRlNWJlOGNmNGMxZGVlYmFfY2M3MTZkOTUyYThhYTU3ZDUyZjQ2YTUxNmY2NzlhNzFfSUQ6NzY1NjMwNjUwNTc3NzAyNDE5M18xNzg1MTYwMjUxOjE3ODUyNDY2NTFfVjM)





换任务 adjust\_bottle

```Bash
set +e
set +u
set +o pipefail 2>/dev/null || true

cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/nohup
LOG="logs/nohup/eval_lawam_adjust_bottle_$(date +%Y%m%d_%H%M%S).log"

ray stop --force || python -m ray stop -f || true

nohup bash -lc '
set +e
set +u
set +o pipefail 2>/dev/null || true
cd /root/autodl-tmp/RLinf
source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash evaluations/run_eval.sh robotwin robotwin_lift_pot_lawam_eval_autodl \
  env.eval.task_config.task_name=adjust_bottle \
  env.eval.total_num_envs=1 \
  env.eval.rollout_epoch=1 \
  env.eval.video_cfg.save_video=True \
  rollout.model.lawam.allow_batch_size=1 \
  rollout.model.lawam.num_inference_steps=4 \
  runner.logger.experiment_name=robotwin_lawam_adjust_bottle_smoke
' > "${LOG}" 2>&1 &

echo "PID=$!"
echo "LOG=${LOG}"
```

\[ab 0\.mp4\]

成功



#### 训

训练smoke

```Bash
set +e
set +u
set +o pipefail 2>/dev/null || true

cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}

export RAY_DISABLE_DOCKER_CPU_WARNING=1
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_lift_pot_lawam_ppo_smoke_autodl"
CONFIG_FILE="/root/autodl-tmp/RLinf/examples/embodiment/config/${CONFIG_NAME}.yaml"

echo "===== check config ====="
test -f "${CONFIG_FILE}"
ls -lh "${CONFIG_FILE}"

echo "===== stop ray ====="
ray stop --force || python -m ray stop -f || true

echo "===== gpu ====="
nvidia-smi

mkdir -p logs/nohup

LOG="logs/nohup/robotwin_adjust_bottle_lawam_ppo_smoke_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc '
set +e
set +u
set +o pipefail 2>/dev/null || true

cd /root/autodl-tmp/RLinf

source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}

export RAY_DISABLE_DOCKER_CPU_WARNING=1
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name robotwin_lift_pot_lawam_ppo_smoke_autodl \
  env.train.task_config.task_name=adjust_bottle \
  env.eval.task_config.task_name=adjust_bottle \
  actor.model.lawam.allow_batch_size=2 \
  actor.model.lawam.num_inference_steps=4 \
  runner.max_steps=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  runner.logger.experiment_name=robotwin_adjust_bottle_lawam_ppo_smoke_autodl
' > "${LOG}" 2>&1 &

PID=$!
echo "PID=${PID}"
echo "LOG=${LOG}"
```

rollout时10G 7G左右

爆：碰撞，模型格式，日志

```Bash
1. 产物目录确实又不对
你这次训练是直接跑：
python examples/embodiment/train_embodied_agent.py ...
这不会自动生成你期待的这种目录：
/root/autodl-tmp/RLinf/logs/20260628-13:05:57-robotwin_lift_pot_lawam_eval_autodl/
那个目录是 evaluations/run_eval.sh 或 examples/embodiment/run_embodiment.sh 这类 wrapper 生成的。run_eval.sh 里会显式创建：
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
python ... runner.logger.log_path=${LOG_DIR}
所以 eval 产物会像你说的那样在 RLinf/logs/<timestamp>-<config>/ 下面。训练直接跑 Python 时，配置里仍是：
runner:
  logger:
    log_path: "../results"
因此 tensorboard/result 可能跑到 /root/autodl-tmp/results

修法建议放在通用 ValueHead.forward()，因为现在 self.value_head 被 FSDP mixed precision 包住后，Linear 权重变成 bfloat16，但输入仍然是 float32。在外层 LaWAMPolicy 里按 next(self.value_head.parameters()).dtype 取到的可能是 FSDP 管理下的 fp32 参数视图，不能代表 forward 时 Linear 的真实 dtype。

```

修

```Python
set +e
set +u
set +o pipefail 2>/dev/null || true

cd /root/autodl-tmp/RLinf

python - <<'PY'
from pathlib import Path

p = Path("rlinf/models/embodiment/modules/value_head.py")
s = p.read_text()

old = """    def forward(self, x):
        return self.mlp(x)
"""

new = """    def forward(self, x):
        # FSDP mixed precision can keep the incoming feature tensor in fp32 while
        # the wrapped Linear weights are materialized in bf16/fp16 for forward.
        # Match the first Linear's runtime dtype/device before entering the MLP.
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                x = x.to(device=layer.weight.device, dtype=layer.weight.dtype)
                break
        return self.mlp(x)
"""

if old in s:
    p.write_text(s.replace(old, new))
    print("PATCHED value_head.py")
elif "layer.weight.dtype" in s:
    print("ALREADY_PATCHED value_head.py")
else:
    print("PATTERN_NOT_FOUND")
PY

python -m py_compile rlinf/models/embodiment/modules/value_head.py
RET=$?
echo "RET=${RET}"

git diff -- rlinf/models/embodiment/modules/value_head.py
```



Smoke click\_bell

```Bash
set +e
set +u
set +o pipefail 2>/dev/null || true

cd /root/autodl-tmp/RLinf
source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK=click_bell
CONFIG=robotwin_lift_pot_lawam_ppo_smoke_autodl
RUN_NAME="robotwin_${TASK}_lawam_ppo_smoke"
LOG_DIR="/root/autodl-tmp/RLinf/logs/$(date +'%Y%m%d-%H:%M:%S')-${RUN_NAME}"
mkdir -p "${LOG_DIR}"

ray stop --force || python -m ray stop -f || true

python examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name "${CONFIG}" \
  runner.logger.log_path="${LOG_DIR}" \
  runner.logger.experiment_name="${RUN_NAME}" \
  runner.max_steps=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  env.train.task_config.task_name="${TASK}" \
  env.eval.task_config.task_name="${TASK}" \
  env.train.total_num_envs=1 \
  env.train.group_size=1 \
  actor.global_batch_size=1 \
  actor.micro_batch_size=1 \
  actor.model.lawam.allow_batch_size=1 \
  actor.model.lawam.noise_level=0.1 \
  rollout.model.lawam.allow_batch_size=1 \
  rollout.model.lawam.noise_level=0.1 \
  2>&1 | tee "${LOG_DIR}/run_embodiment.log"

RET=${PIPESTATUS[0]}
echo "RET=${RET}"
echo "LOG_DIR=${LOG_DIR}"
```

这次日志目录对

爆，参数填错

```C++
**是填错 override 参数**，不是模型格式问题。
报错核心：
Could not override 'rollout.model.lawam.allow_batch_size'
Key 'allow_batch_size' is not in struct
full_key: rollout.model.lawam.allow_batch_size
原因：你的训练配置里：
rollout:
  model:
    lawam: ${actor.model.lawam}
rollout.model.lawam 是从 actor.model.lawam 插值来的，不适合直接覆盖它的子字段。应该只覆盖：
actor.model.lawam.allow_batch_size
actor.model.lawam.noise_level
actor.model.lawam.num_inference_steps
rollout.model.lawam 会随 actor.model.lawam 解析过去。不要写 rollout.model.lawam.*。
RLinf 官方 RoboTwin 文档的基本模式也是准备 config，然后用 run_embodiment.sh 或 train_embodied_agent.py 启动，观察 env/success_once；如果需要临时改参数，本质还是 Hydra override。 Hydra 对 struct config 的规则就是：不能随便覆盖不存在的键，除非用 +key=value 追加；但这里不应该追加 rollout 子键，而应该覆盖 actor 源字段。
为什么不用 rollout.model.lawam.*
你现在的训练 YAML 结构里，actor 是源模型配置，rollout 是引用：
rollout:
  model:
    lawam: ${actor.model.lawam}
所以写：
actor.model.lawam.allow_batch_size=1
即可。
写：
rollout.model.lawam.allow_batch_size=1
会触发 Hydra struct 错误。你这次就是死在这里。
```



训

```Bash
set +e
set +u
set +o pipefail 2>/dev/null || true

cd /root/autodl-tmp/RLinf
source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="robotwin_lift_pot_lawam_ppo_smoke_autodl"
TASK="click_bell"
LOG_DIR="/root/autodl-tmp/RLinf/logs/$(date +'%Y%m%d-%H:%M:%S')-robotwin_${TASK}_lawam_ppo_smoke"

test -f "/root/autodl-tmp/RLinf/examples/embodiment/config/${CONFIG}.yaml"

ray stop --force || python -m ray stop -f || true
mkdir -p "${LOG_DIR}"

python examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name "${CONFIG}" \
  runner.logger.log_path="${LOG_DIR}" \
  runner.logger.experiment_name="robotwin_${TASK}_lawam_ppo_smoke" \
  runner.max_steps=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  env.train.task_config.task_name="${TASK}" \
  env.eval.task_config.task_name="${TASK}" \
  env.train.total_num_envs=1 \
  env.train.group_size=1 \
  actor.global_batch_size=1 \
  actor.micro_batch_size=1 \
  actor.model.lawam.allow_batch_size=1 \
  actor.model.lawam.num_inference_steps=4 \
  actor.model.lawam.noise_level=0.1 \
  2>&1 | tee "${LOG_DIR}/run_embodiment.log"

RET=${PIPESTATUS[0]}
echo "RET=${RET}"
echo "LOG_DIR=${LOG_DIR}"
echo "LOG=${LOG_DIR}/run_embodiment.log"
```

训通一步，succ 0

```Plain Text
当前 RLinf / RoboTwin_RLinf 配置里 planner_backend 是 mplib

以前的分析已经把链路定位到：
RLinf eval/train
  -> LaWAM 输出 eef action chunk
  -> RoboTwin_RLinf VectorEnv.step
  -> SubEnv.step
  -> gen_sparse_reward_data(..., action_type="ee")
  -> take_action(..., action_type="ee")
  -> left_plan_path / right_plan_path
  -> mplib planner
日志里的 fl_link* / fr_link* 是机械臂 link，table_6 是桌面碰撞体，不是 pot 本身；Invalid start state 表示 planner 认为当前机械臂状态已经处在不合法状态，再尝试扰动恢复。
```



尝试：4条

```Bash
set +e
set +u
set +o pipefail 2>/dev/null || true

cd /root/autodl-tmp/RLinf
source /root/autodl-tmp/rlinf_env.sh
source .venv/bin/activate

export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RLinf/third_party/LaWAM:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH:-}
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="robotwin_lift_pot_lawam_ppo_smoke_autodl"
TASK="click_bell"
RUN_NAME="robotwin_${TASK}_lawam_ppo_smoke_g4_noise01"
LOG_DIR="/root/autodl-tmp/RLinf/logs/$(date +'%Y%m%d-%H:%M:%S')-${RUN_NAME}"

ray stop --force || python -m ray stop -f || true
mkdir -p "${LOG_DIR}"

python examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name "${CONFIG}" \
  runner.logger.log_path="${LOG_DIR}" \
  runner.logger.experiment_name="${RUN_NAME}" \
  runner.max_steps=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  env.train.task_config.task_name="${TASK}" \
  env.eval.task_config.task_name="${TASK}" \
  env.train.total_num_envs=4 \
  env.train.group_size=1 \
  actor.global_batch_size=4 \
  actor.micro_batch_size=1 \
  actor.model.lawam.allow_batch_size=4 \
  actor.model.lawam.num_inference_steps=4 \
  actor.model.lawam.noise_level=0.1 \
  2>&1 | tee "${LOG_DIR}/run_embodiment.log"

RET=${PIPESTATUS[0]}
echo "RET=${RET}"
echo "LOG_DIR=${LOG_DIR}"
echo "LOG=${LOG_DIR}/run_embodiment.log"
```

爆

\[run\_embodiment\.log\]

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e/c/6a3f20b7\-b058\-83ea\-8595\-3e6e5100b188?mweb\_fallback=1

```Plain Text
失败点在 env rollout

**4 个并行 env 的 rollout**。日志中的 step: 23 / 192、step: 30 / 192 多次交错出现，是 4 个 SubEnv 并行输出混在一起。

失败点：SubEnv step 超时
关键 traceback 是：
TimeoutError
...
env_worker.py -> env_interact_step
robotwin_env.py -> chunk_step
vector_env.py -> future.result(timeout=120)
RuntimeError: SubEnv 0 step error:
也就是说，VectorEnv.step() 里有这个逻辑：
future = self.env_thread_pool.submit(self.envs[i].step, actions[i])
...
result = future.result(timeout=120)
某个 SubEnv 的 step() 超过 120 秒没有返回，触发 TimeoutError。然后 VectorEnv 把它包成：
RuntimeError: SubEnv 0 step error:
后面的：
ActorDiedError
Unsupported object type: -281541081
CollectiveManager killed
都是 Ray/RLinf 在主 env worker 崩掉后级联清理 actor/rollout 进程产生的次生错误，不是根因。
根因就是：
某个 RoboTwin SubEnv 在执行 ee action chunk 时卡住/超时。

为什么会超时：planner 已经进入坏状态
超时前，日志已经反复出现：
Invalid start state!
fr_link3 and fr_link5 collide!
fr_link2 and table_6 collide!
fl_link3 and fl_link6 collide!
invalid start state!! (collision)
sampled a new state with a perturbation ...
第一次明显 collision 大概在 step 30 左右，后面一直积累，到 step 148～151 附近触发超时。
这说明流程大概是：
LaWAM train rollout 输出 ee action
-> RoboTwin 执行 action chunk
-> 某些目标末端位姿导致机械臂自碰撞/桌面碰撞
-> planner 反复尝试 perturb/recover
-> 某个 SubEnv.step 超过 120 秒
-> VectorEnv 抛 RuntimeError
-> RLinf 主流程中断
fl_link* / fr_link* 是左右机械臂 link，table_6 是桌面，不是物体本身。这个还是 planner / robot state 问题。

现在整体状态是：
✅ LaWAM 模型加载通
✅ RLinf actor/rollout/env 初始化通
✅ endpose 通道已修通
✅ ValueHead dtype 已修
✅ b1 PPO smoke 曾经完成 Global Step 1
❌ g4 并行 rollout 会被某个 SubEnv planner timeout 杀死
❌ click_bell g4 没有成功完成 PPO update


```



### TODO



fast\-wam？light\-wam？之前调研的其他rt2上完整的

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-snd/c/6a3e14af\-d514\-83ea\-a921\-2cede6edfc20



https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e/c/6a3f20b7\-b058\-83ea\-8595\-3e6e5100b188?mweb\_fallback=1



之前的推理成功，没报碰撞，是为什么？

碰撞问题影响成功率吗？碰撞一下然后能恢复，影响rollout的程度如何？

看视频



推git



逐个校对配置

各方面配置调高，正式进行；

对齐openpi pi0 pi0\.5；每个配置，是否对齐，不一样是为什么，为什么这样适配？

训练两卡，等等

显存内存监视，顶满；哪些配置影响显存内存？能增大参数增加采集rollout的并行度？影响什么？



还是实际上只要1卡a800？顶满前换1卡

显存内存监视，顶满



other

理解lawam

其他思路：valuehead换成vlm的其他输出；共3个



隐患：

lawam和motus需要处理使vlm输入等长；影响value，ppo等？

LaWAM 的 Qwen chat template 和多图输入让 token 位置变长，所以要用 `mask` 找各种东西的位置

flow\_sde 公式实现了 LaWAM tau 坐标变换？去噪过程改的对不对？

ee action reward 是 loop take\_action，不是向量化 ee chunk planner。这个是不是能推理有成功率就是没问题的？

```Python
5. 静态 code review 发现的风险点
5.1 policy.to(dtype=bf16) 可能过早
当前 LaWAMPolicy.__init__ 里会：
self.policy = self.policy.to(dtype=self.cast_dtype)
这会把 Qwen、LAM、DINO、flow 都一起 cast。官方推理日志里 DINO/LAM 有过 fp32 加载和 numpy/pytorch3d 依赖问题；DINOv3 路径 patch 也明确在 LaWAM 本地 yaml 里。
这不一定错，因为官方也有 bf16 mixed_precision 路径；但如果 smoke 出现 LAM/DINO dtype 错误，第一修法不是动 flow，而是先把 config 改成：
precision: "fp32"
确认逻辑通，再回到 bf16。
建议第一轮 eval smoke 用默认 bf16；如果 dtype 报错，再改 fp32。
5.2 sample_action_space() 仍是 14D
rlinf/envs/robotwin/robotwin_env.py 里 sample_action_space() 还返回：
np.random.randn(self.num_envs, self.horizon, 14)
这不影响正常 LaWAM rollout，因为 policy 会返回 [B,8,16]；但如果某些 debug/random path 调它，会不匹配。可以后续小 patch 改成按 task_config.action_type 返回 14/16。
5.3 _cal_chunk_rewards() 暂时忽略 n_steps_to_run
RoboTwin_RLinf 的 ee path 在 infos 里设置了 n_steps_to_run，但 RLinf 的 _cal_chunk_rewards() 当前本来就把读取 infos["n_steps_to_run"] 注释掉了，默认把 chunk reward 放到最后一步。这是原有 qpos 路径的行为，不是新 bug，但如果想更精确，需要后续 patch：
n_steps_to_run = infos.get("n_steps_to_run", default)
第一版 PPO 可以先接受；后面调 credit assignment 再改。
5.4 Qwen pixel_values packing 假设每个样本图像 patch 数一致
当前 _pack_first_dim_by_batch() 假设：
pixel_values.shape[0] % batch_size == 0
由于 LaWAM wrapper 把 main/wrist 图都 resize 到 256，且每个 sample 都是 1 main + 2 wrist，这个假设大概率成立。若某些 env 缺 wrist 或 image_grid 不一致，会报 shape 错。当前 config 开了 wrist camera，合理。
5.5 flow_query_tokens value head 需要 mask count assert
当前代码直接：
tokens = h_vlm[mask].view(B, n_q, D)
如果 mask 数不等于 B*n_q，会 view error。LaWAM backend 本身也有 placeholder count check；但这里建议后续加更清晰的 assert，方便定位：
assert mask.sum() == B * n_q
不是阻塞项。
```

lawam是不是不需要状态state？

```Plain Text
qpos: true 是为了 RLinf 当前 _extract_obs_image 还从 obs["state"] 取 qpos，虽然 LaWAM checkpoint use_state=false，但保留 qpos 不会坏
```





### Lawam read



入口：examples/Robotwin/eval\_files/eval\_direct\.sh

覆盖一些参数



rt2适配：examples/Robotwin/starvla\_policy/deploy\_policy\.py



模型：starVLA/model/framework/lawam\_framework\.py



往下：

starVLA/model/framework/latent\_world/runtime/runner\.py

starVLA/model/framework/latent\_world/batch\_builder\.py

starVLA/model/framework/latent\_world/vlm\_adapter\.py

starVLA/model/framework/vlas/lawam\.py

starVLA/model/framework/vlas/flowmatching\_expert\.py



### Tool



打包rlinf

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

更小

```Bash
cd /root/autodl-tmp

set -euo pipefail

RLINF_ROOT="/root/autodl-tmp/RLinf"
test -d "${RLINF_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="/root/autodl-tmp/RLinf_code_only_${STAMP}.tgz"
MANIFEST="/root/autodl-tmp/RLinf_code_only_${STAMP}.manifest.txt"

echo "===== write manifest ====="
{
  echo "PACK_TIME=$(date)"
  echo "RLINF_ROOT=${RLINF_ROOT}"
  echo
  echo "===== git ====="
  git -C "${RLINF_ROOT}" rev-parse HEAD 2>/dev/null || echo "no git head"
  git -C "${RLINF_ROOT}" status --short 2>/dev/null || true
  echo
  echo "===== important files ====="
  find "${RLINF_ROOT}/rlinf/models/embodiment" -maxdepth 3 -type f | sort | sed -n '1,240p'
  echo
  find "${RLINF_ROOT}/examples/embodiment/config" -maxdepth 2 -type f | sort | grep -Ei 'motus|lawam|openpi|ppo|grpo|robotwin' || true
  echo
  find "${RLINF_ROOT}/evaluations/robotwin" -maxdepth 1 -type f | sort | grep -Ei 'motus|lawam|openpi|robotwin' || true
} > "${MANIFEST}"

echo "===== create tarball ====="
tar -czf "${OUT}" \
  --exclude='RLinf/.git' \
  --exclude='RLinf/.venv' \
  --exclude='RLinf/.venv_*' \
  --exclude='RLinf/logs' \
  --exclude='RLinf/results' \
  --exclude='RLinf/outputs' \
  --exclude='RLinf/wandb' \
  --exclude='RLinf/.cache' \
  --exclude='RLinf/.hf_cache' \
  --exclude='RLinf/__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.safetensors' \
  --exclude='*.ckpt' \
  --exclude='*.bin' \
  --exclude='*.onnx' \
  --exclude='*.pkl' \
  --exclude='*.npz' \
  --exclude='*.npy' \
  --exclude='*.h5' \
  --exclude='*.hdf5' \
  --exclude='*.tar' \
  --exclude='*.tgz' \
  --exclude='*.tar.gz' \
  --exclude='*.zip' \
  --exclude='*.mp4' \
  --exclude='*.avi' \
  --exclude='*.mov' \
  --exclude='*.jpg' \
  --exclude='*.jpeg' \
  --exclude='*.png' \
  --exclude='*.log' \
  --exclude='*.out' \
  --exclude='*.bak' \
  --exclude='*.bak_*' \
  --exclude='*.orig' \
  --exclude='*.rej' \
  --exclude='.hf_token' \
  --exclude='.*_token' \
  --exclude='*api_key*' \
  --exclude='*API_KEY*' \
  RLinf

echo "===== output ====="
ls -lh "${OUT}" "${MANIFEST}"

echo
echo "===== suspicious files inside tarball, should be empty ====="
tar -tzf "${OUT}" \
  | grep -Ei '(\.pt$|\.pth$|\.ckpt$|\.safetensors$|\.bin$|\.onnx$|\.pkl$|\.npz$|\.npy$|\.h5$|\.hdf5$|\.mp4$|\.avi$|\.mov$|\.zip$|\.tar$|\.tgz$|\.tar\.gz$|token|api_key)' \
  || true

echo "DONE"
echo "OUT=${OUT}"
echo "MANIFEST=${MANIFEST}"
```



打包RoboTwin\_RLinf

```Bash
cd /root/autodl-tmp

set -euo pipefail

RT_ROOT="/root/autodl-tmp/RoboTwin_RLinf"
test -d "${RT_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="/root/autodl-tmp/RoboTwin_RLinf_code_only_${STAMP}.tgz"
MANIFEST="/root/autodl-tmp/RoboTwin_RLinf_code_only_${STAMP}.manifest.txt"

echo "===== write manifest ====="
{
  echo "PACK_TIME=$(date)"
  echo "RT_ROOT=${RT_ROOT}"
  echo
  echo "===== git ====="
  git -C "${RT_ROOT}" rev-parse HEAD 2>/dev/null || echo "no git head"
  git -C "${RT_ROOT}" status --short 2>/dev/null || true
  echo
  echo "===== key env/action files ====="
  find "${RT_ROOT}" -maxdepth 5 -type f \
    | grep -Ei 'vector_env|eval_policy|task|env|robotwin|action|aloha|script|policy' \
    | sort \
    | sed -n '1,260p'
  echo
  echo "===== grep action interface ====="
  grep -RniE "action_type|take_action|chunk|venv.step|def step|eef|endpose|qpos|Aloha|aloha|ee" \
    "${RT_ROOT}/robotwin" \
    "${RT_ROOT}/script" \
    2>/dev/null \
    | sed -n '1,260p' || true
} > "${MANIFEST}"

echo "===== create tarball ====="
tar -czf "${OUT}" \
  --exclude='RoboTwin_RLinf/.git' \
  --exclude='RoboTwin_RLinf/__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.so' \
  --exclude='RoboTwin_RLinf/data' \
  --exclude='RoboTwin_RLinf/assets' \
  --exclude='RoboTwin_RLinf/results' \
  --exclude='RoboTwin_RLinf/logs' \
  --exclude='RoboTwin_RLinf/output' \
  --exclude='RoboTwin_RLinf/outputs' \
  --exclude='RoboTwin_RLinf/videos' \
  --exclude='RoboTwin_RLinf/.cache' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.onnx' \
  --exclude='*.pkl' \
  --exclude='*.npz' \
  --exclude='*.npy' \
  --exclude='*.h5' \
  --exclude='*.hdf5' \
  --exclude='*.tar' \
  --exclude='*.tgz' \
  --exclude='*.tar.gz' \
  --exclude='*.zip' \
  --exclude='*.mp4' \
  --exclude='*.avi' \
  --exclude='*.mov' \
  --exclude='*.webm' \
  --exclude='*.jpg' \
  --exclude='*.jpeg' \
  --exclude='*.png' \
  --exclude='*.gif' \
  RoboTwin_RLinf

echo "===== output ====="
ls -lh "${OUT}" "${MANIFEST}"

echo
echo "===== key files inside tarball ====="
tar -tzf "${OUT}" \
  | grep -Ei 'vector_env|eval_policy|task|env|action|aloha|script' \
  | sed -n '1,180p'

echo
echo "DONE"
echo "OUT=${OUT}"
echo "MANIFEST=${MANIFEST}"
```



打包lawam

```Bash
cd /root/autodl-tmp

set -euo pipefail

LAWAM_ROOT="/root/autodl-tmp/LaWAM"

test -d "${LAWAM_ROOT}"
test -d "${LAWAM_ROOT}/examples/Robotwin/starvla_policy"
test -d "${LAWAM_ROOT}/starVLA"
test -d "${LAWAM_ROOT}/latent_action_model"
test -f "${LAWAM_ROOT}/latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="/root/autodl-tmp/LaWAM_code_only_${STAMP}.tgz"
MANIFEST="/root/autodl-tmp/LaWAM_code_only_${STAMP}.manifest.txt"

echo "===== write manifest ====="
{
  echo "PACK_TIME=$(date)"
  echo "LAWAM_ROOT=${LAWAM_ROOT}"
  echo
  echo "===== git ====="
  git -C "${LAWAM_ROOT}" rev-parse HEAD 2>/dev/null || echo "no git head"
  git -C "${LAWAM_ROOT}" status --short 2>/dev/null || true
  echo
  echo "===== key patched config ====="
  grep -nE "vision_model_id|dinov3|facebook" \
    "${LAWAM_ROOT}/latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml" \
    || true
  echo
  echo "===== key dirs ====="
  find "${LAWAM_ROOT}/examples/Robotwin" -maxdepth 3 -type f | sort | sed -n '1,200p'
} > "${MANIFEST}"

echo "===== create tarball ====="
tar -czf "${OUT}" \
  --exclude='LaWAM/.git' \
  --exclude='LaWAM/.venv' \
  --exclude='LaWAM/.venv_*' \
  --exclude='LaWAM/env' \
  --exclude='LaWAM/venv' \
  --exclude='LaWAM/__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.so' \
  --exclude='LaWAM/.cache' \
  --exclude='LaWAM/.hf_cache' \
  --exclude='LaWAM/cache' \
  --exclude='LaWAM/wandb' \
  --exclude='LaWAM/logs' \
  --exclude='LaWAM/outputs' \
  --exclude='LaWAM/tmp' \
  --exclude='LaWAM/checkpoints' \
  --exclude='LaWAM/debug' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.onnx' \
  --exclude='*.pkl' \
  --exclude='*.npz' \
  --exclude='*.npy' \
  --exclude='*.h5' \
  --exclude='*.hdf5' \
  --exclude='*.tar' \
  --exclude='*.tgz' \
  --exclude='*.tar.gz' \
  --exclude='*.zip' \
  --exclude='*.mp4' \
  --exclude='*.avi' \
  --exclude='*.mov' \
  --exclude='*.webm' \
  --exclude='*.jpg' \
  --exclude='*.jpeg' \
  --exclude='*.png' \
  --exclude='*.gif' \
  --exclude='*.log' \
  --exclude='*.out' \
  --exclude='*.bak' \
  --exclude='*.bak_*' \
  --exclude='.hf_token' \
  --exclude='.*_token' \
  --exclude='*api_key*' \
  --exclude='*API_KEY*' \
  LaWAM

echo "===== output ====="
ls -lh "${OUT}" "${MANIFEST}"

echo
echo "===== tar content preview ====="
tar -tzf "${OUT}" | sed -n '1,160p'

echo
echo "===== suspicious files inside tarball, should be empty ====="
tar -tzf "${OUT}" \
  | grep -Ei '(\.pt$|\.pth$|\.ckpt$|\.safetensors$|\.bin$|\.onnx$|\.pkl$|\.npz$|\.npy$|\.h5$|\.hdf5$|\.mp4$|\.avi$|\.mov$|\.zip$|\.tar$|\.tgz$|\.tar\.gz$|token|api_key)' \
  || true

echo
echo "DONE"
echo "OUT=${OUT}"
echo "MANIFEST=${MANIFEST}"
```





