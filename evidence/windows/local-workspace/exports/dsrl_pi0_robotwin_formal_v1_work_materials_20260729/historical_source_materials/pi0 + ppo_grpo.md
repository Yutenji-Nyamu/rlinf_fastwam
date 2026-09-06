# pi0 \+ ppo/grpo



### guide

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-wam\-rl/c/6a52f8af\-9aa4\-83ea\-8048\-c33d7ee314ed



从之前的机子继续



这台机历史干的事情

[Exp\_snd](https://my.feishu.cn/wiki/JAfbwIKFui0jR6kkbfScpJ0vn7f)

[Openpi \+ PPO AutoDL A800](https://my.feishu.cn/wiki/HkMUwQDcOikfygkTVm2cBsnEnVe)

[Motus \+ RLinf](https://my.feishu.cn/wiki/Azcdwee5EiVgdCkmCK3csFhVnBf)

[lawam rlinf](https://my.feishu.cn/wiki/O3nPwu6lgi5PMkk7KxNcy1rrn2f)



先跑完整的pi0 \+ ppo，收集各主要指标，给后面各改进对比



### 跑 临时新机 1\*a800



新配置

/root/autodl\-tmp/RLinf/examples/embodiment/config/robotwin\_adjust\_bottle\_ppo\_openpi\_a800\_1gpu\_curve\.yaml

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/pi0@actor.model
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
    experiment_name: "robotwin_ppo_openpi"
    logger_backends: ["tensorboard"] # wandb, swanlab

  max_epochs: 1000
  max_steps: 100

  only_eval: False
  val_check_interval: 5
  save_interval: 20

  resume_dir: null # Optional: path to a saved checkpoint directory, such as 'checkpoints/global_step_10'. If not None, it will be used to resume training.
  ckpt_path: null  # Optional: path to a .pt checkpoint. If not None, it will be loaded after the model is instantiated (for evaluation).

algorithm:
  normalize_advantages: True
  kl_penalty: kl  # how to estimate kl divergence: kl or kl_penalty
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
  # Override the default values in env/robotwin_adjust_bottle
  train:
    rollout_epoch: 4
    total_num_envs: 8
    reward_coef: ${algorithm.reward_coef}
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
    group_size: ${algorithm.group_size}
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
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
  eval:
    rollout_epoch: 8
    total_num_envs: 8
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 200
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 200
    group_size: 1
    use_fixed_reset_state_ids: False
    is_eval: True
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    video_cfg:
      save_video: False
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
  enable_offload: True
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 8
  global_batch_size: 64 # 1024
  seed: 1234
  enable_offload: True

  # Override the default values in model/openpi
  model:
    model_path: "/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
    num_action_chunks: 50 # interface for the env
    add_value_head: True
    action_dim: 14
    openpi:
      config_name: "pi0_aloha_robotwin"
      num_images_in_input: 3
      detach_critic_input: True

  optim:
    lr: 5.6e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  # Override the default values in training_backend/fsdp
  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False # for openpi, gradient checkpointing is not supported, please do not change this value
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False
```

smoke：步数，保存，评估，都1



爆

```Plain Text
错误本身明确要求 transformers==4.53.2 并把 OpenPI 的 transformers_replace 文件复制进 Transformers 包。
而当前环境实际是：
transformers 4.57.1
同时 pip check 明确报告：
openpi 0.1.0 requires transformers==4.53.2
but you have transformers 4.57.1

 Motus 集成记录里也明确执行过：
uv pip install \
  "deepspeed" \
  "transformers==4.57.1" \
  ...
这一步是在同一个 /root/autodl-tmp/RLinf/.venv 内执行的

最稳妥修复：恢复 Motus 修改前的 π₀ 虚拟环境
你的 Motus 记录显示，在安装 Transformers 4.57.1 之前做过：
.venv_backup_pi0_ppo_grpo_ok_<timestamp>
这是最佳恢复源。保留当前 Motus 环境，不删除；把它改名，然后将预 Motus 环境恢复为 .venv。

应该从这个环境恢复：
/root/autodl-tmp/RLinf/.venv_backup_pi0_ppo_grpo_ok_20260618_142009
不要优先使用：
/root/autodl-tmp/RLinf_old_20260618_085536/.venv
后者也通过了检查，可以作为二级兜底，但前者是你**明确在 π₀ PPO/GRPO 跑通后、安装 Motus 依赖之前**制作的专用快照。记录显示，这个快照是用 rsync -aH .venv/ "${BACKUP}/" 完整复制出来的，之后才在当前 .venv 中安装 transformers==4.57.1 等 Motus 依赖
```

目前uv是motus，基于pi的覆盖装了包，要用之前的pi的备份uv



备份当前uv，启用之前的uv

```Bash
cd /root/autodl-tmp/RLinf

CURRENT_VENV="/root/autodl-tmp/RLinf/.venv"
PI0_BACKUP="/root/autodl-tmp/RLinf/.venv_backup_pi0_ppo_grpo_ok_20260618_142009"

echo "CURRENT_VENV=${CURRENT_VENV}"
echo "PI0_BACKUP=${PI0_BACKUP}"

test -x "${CURRENT_VENV}/bin/python" &&
  echo "current venv exists"

test -x "${PI0_BACKUP}/bin/python" &&
  echo "pi0 backup exists"

du -sh "${CURRENT_VENV}" "${PI0_BACKUP}"
df -h /root/autodl-tmp

cd /root/autodl-tmp/RLinf

/root/autodl-tmp/RLinf/.venv/bin/ray stop -f || true

pgrep -af \
'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' \
|| true

nvidia-smi

cd /root/autodl-tmp/RLinf

deactivate 2>/dev/null || true

STAMP="$(date +%Y%m%d_%H%M%S)"
MOTUS_VENV="/root/autodl-tmp/RLinf/.venv_motus_${STAMP}"
PI0_BACKUP="/root/autodl-tmp/RLinf/.venv_backup_pi0_ppo_grpo_ok_20260618_142009"

echo "MOTUS_VENV=${MOTUS_VENV}"
echo "PI0_BACKUP=${PI0_BACKUP}"

mv /root/autodl-tmp/RLinf/.venv "${MOTUS_VENV}"

mkdir /root/autodl-tmp/RLinf/.venv

rsync -aH --info=progress2 \
  "${PI0_BACKUP}/" \
  /root/autodl-tmp/RLinf/.venv/

echo
echo "===== resulting environments ====="
du -sh \
  /root/autodl-tmp/RLinf/.venv \
  "${PI0_BACKUP}" \
  "${MOTUS_VENV}"

echo
echo "MOTUS_VENV_SAVED=${MOTUS_VENV}"
```



查

```Python
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

echo "python=$(which python)"

python - <<'PY'
from pathlib import Path
import hashlib
import sys
import torch
import transformers
import openpi

print("python =", sys.executable)
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("transformers =", transformers.__version__)

patch_dir = (
    Path(openpi.__file__).resolve().parent
    / "models_pytorch"
    / "transformers_replace"
)
target_dir = Path(transformers.__file__).resolve().parent

def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            h.update(data)
    return h.hexdigest()

missing = []
mismatch = []
checked = []

for src in patch_dir.rglob("*"):
    if not src.is_file():
        continue
    if "__pycache__" in src.parts or src.suffix == ".pyc":
        continue

    rel = src.relative_to(patch_dir)
    dst = target_dir / rel
    checked.append(str(rel))

    if not dst.exists():
        missing.append(str(rel))
    elif sha256(src) != sha256(dst):
        mismatch.append(str(rel))

print("checked =", checked)
print("missing =", missing)
print("mismatch =", mismatch)

assert transformers.__version__ == "4.53.2"
assert not missing
assert not mismatch

print("PI0_OPENPI_ENV_OK")
PY

python - <<'PY'
import rlinf
print("rlinf =", rlinf.__file__)
PY


```



跑

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}

export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_adjust_bottle_ppo_openpi_a800_1gpu_smoke"
EXP_NAME="robotwin_ppo_openpi_a800_1gpu_smoke"

ray stop -f || true

echo "===== stale processes ====="
pgrep -af \
'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' \
|| true

echo
echo "===== GPU before run ====="
nvidia-smi

RUN_NAME="$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"
MON_DIR="${LOG_DIR}/resource_monitor"
MON_LOG="${MON_DIR}/monitor.log"

mkdir -p "${LOG_DIR}" "${MON_DIR}"

CMD=(
  /root/autodl-tmp/RLinf/.venv/bin/python
  /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py
  --config-path
  /root/autodl-tmp/RLinf/examples/embodiment/config/
  --config-name
  "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "runner.logger.experiment_name=${EXP_NAME}"
)

printf "%q " "${CMD[@]}" > "${LOG_DIR}/command.txt"
echo >> "${LOG_DIR}/command.txt"

nohup "${CMD[@]}" \
  > "${RUN_LOG}" 2>&1 &

TRAIN_PID=$!

nohup /root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/RLinf/local_scripts/monitor_pi0_resources.py \
  --pid "${TRAIN_PID}" \
  --out-dir "${MON_DIR}" \
  --interval 2 \
  > "${MON_LOG}" 2>&1 &

MONITOR_PID=$!

echo "TRAIN_PID=${TRAIN_PID}"
echo "MONITOR_PID=${MONITOR_PID}"
echo "RUN_NAME=${RUN_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "RUN_LOG=${RUN_LOG}"
echo "PEAK_FILE=${MON_DIR}/peak.txt"
echo "RESOURCE_CSV=${MON_DIR}/resources.csv"
```



爆

```Plain Text
**卡死在初始权重同步阶段**，还没有进入 rollout、PPO 更新、评估或 checkpoint 保存。
根因不是显存或内存，而是：
RuntimeError: pidfd_getfd: Operation not permitted
具体链路是：
Actor sync_model_to_rollout
→ PatchWeightSyncer 初始全量同步
→ actor 和 rollout 在同一张 GPU 上
→ RLinf 自动选择 CUDA IPC
→ AutoDL 容器不允许 pidfd_getfd
→ rollout 接收线程异常退出
→ actor 一直等待
日志明确显示异常发生在 _recv_tensor_list_via_ipc() 重建 CUDA tensor 时

修复：让 weight sync 通过 CPU 传输
当前 resolved config 是：
weight_syncer:
  type: patch
  patch:
    snapshot_device: cpu
    delta_encoding: true
    compression: none
但是缺少：
transport_device: cpu
当 transport device 没有显式设为 CPU 时，初始权重 bucket 仍可能在 GPU 上传输；由于 actor 与 rollout 位于同一张 GPU，RLinf 走 CUDA IPC，触发容器权限问题。
RLinf 的 weight-sync 文档明确支持并推荐：
patch:
  snapshot_device: cpu
  transport_device: cpu
transport_device: cpu 会让 bootstrap 和后续 patch payload 在发送前搬到 CPU，从而使用 CPU collective，而不是同卡 CUDA IPC。
它不会把模型或训练搬到 CPU，只改变 actor→rollout 的权重同步载荷设备。

snapshot_device
snapshot_device: cpu
这是 patch sync 在 actor 侧保存“旧参数快照”的位置，用来判断哪些参数发生了变化。
cpu：节省一份模型大小的显存，但比较时需要 CPU↔GPU 搬运。 
cuda：比较更直接，但额外占显存。 
RLinf 文档目前推荐优先用 CPU snapshot，特别是大模型显存紧张时。
transport_device
transport_device: cpu
这是**权重同步载荷实际通过什么设备发送**：
cuda：参数 bucket/patch 保持在 GPU 上通信。 
cpu：先把同步数据搬到主存，再通过 CPU 通信发送。 
它不改变模型放在哪里，也不让训练改为 CPU；模型仍在 A800 上训练和推理。
init_sync
init_sync:
  enabled: true
启动时先把 actor 的完整初始权重同步到 rollout，之后才发送小的增量 patch。RLinf 当前建议默认开启，以确保 actor 和 rollout 完全一致。

为什么 GPU transport 在这里失败
你的配置是：
component_placement:
  actor, env, rollout: 0
所以：
Actor 进程   → GPU 0
Rollout 进程 → GPU 0
两者虽然使用同一张卡，但不是同一个 Python 进程。
当 GPU tensor 从 actor 发给同卡 rollout 时，RLinf 会选择 **CUDA IPC**：
Actor 创建 GPU tensor
→ 导出 CUDA IPC handle
→ Rollout 进程直接映射同一块 GPU 内存
这通常比：
GPU → CPU → CPU通信 → GPU
更快，也少一次数据复制。
但 PyTorch 重建这个跨进程 CUDA tensor 时调用了：
pidfd_getfd
AutoDL 容器不允许该系统调用，于是报：
RuntimeError: pidfd_getfd: Operation not permitted
日志中的失败点正是同卡 CUDA IPC 接收路径。
所以不是：
 GPU 不支持 IPC； 
 显存不够； 
 π₀ 配置错误； 
而是 **容器权限限制不允许这种进程间 GPU handle 传递**。

改成 CPU transport 后发生什么
snapshot_device: cpu
transport_device: cpu
同步路径改为：
Actor GPU 参数
→ 搬到 CPU bucket
→ CPU collective 发送
→ Rollout 收到后拷回 GPU
这样不会调用同卡 CUDA IPC，自然绕过 pidfd_getfd。
代价
主要是：
 第一次完整 init sync 更慢； 
 每次更新后的 patch sync 多一次 GPU↔CPU 搬运； 
 CPU 内存和 PCIe/NVLink 传输增加。 
不过 π₀ 的后续 patch 通常只包含训练参数的变化，而不是每步发送完整模型，所以正式训练中的同步开销通常远小于 rollout 和 actor forward/backward。RLinf 也把 patch 作为具身大模型默认推荐方案。

官方推荐怎么配
RLinf 当前文档给出的典型 patch 配置是：
weight_syncer:
  type: patch
  patch:
    snapshot_device: cpu
    transport_device: cpu
    delta_encoding: true
    compression: none
    init_sync:
      enabled: true
      prefixes: null
      bucket_size: 134217728
因此你现在的修改不是奇怪的临时方案，而是官方明确支持的标准配置。只是你的原默认配置没有显式设置 transport_device，在同卡进程布局下走到了 CUDA IPC。
一句话概括：snapshot_device 决定旧权重副本存在哪里；transport_device 决定同步数据走 GPU 还是 CPU。我们改 CPU transport，是因为单卡上的 actor 和 rollout 分属不同进程，而 AutoDL 禁止 CUDA IPC 所需的 pidfd_getfd。
```



改参数：

```Plain Text
weight_syncer:
  patch:
    snapshot_device: cpu
    transport_device: cpu
    init_sync:
      enabled: true
```

目前配置

```YAML
defaults:
- env/robotwin_adjust_bottle@env.train
- env/robotwin_adjust_bottle@env.eval
- model/pi0@actor.model
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
    log_path: ../results
    project_name: rlinf
    experiment_name: robotwin_ppo_openpi_a800_1gpu_curve
    logger_backends:
    - tensorboard
  max_epochs: 1000
  max_steps: 100
  only_eval: false
  val_check_interval: 5
  save_interval: 20
  resume_dir: null
  ckpt_path: null
algorithm:
  normalize_advantages: true
  kl_penalty: kl
  group_size: 1
  reward_coef: 1.0
  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level
  update_epoch: 2
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: token-mean
  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0
  gamma: 0.99
  gae_lambda: 0.95
  filter_rewards: false
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9
env:
  group_name: EnvGroup
  enable_offload: true
  train:
    rollout_epoch: 4
    total_num_envs: 8
    reward_coef: ${algorithm.reward_coef}
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
    group_size: ${algorithm.group_size}
    assets_path: /root/autodl-tmp/RoboTwin_RLinf
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: false
    task_config:
      embodiment:
      - aloha-agilex
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
    video_cfg:
      save_video: false
  eval:
    rollout_epoch: 8
    total_num_envs: 8
    auto_reset: true
    ignore_terminations: true
    max_episode_steps: 200
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 200
    group_size: 1
    use_fixed_reset_state_ids: false
    is_eval: true
    assets_path: /root/autodl-tmp/RoboTwin_RLinf
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    video_cfg:
      save_video: false
      video_base_dir: ${runner.logger.log_path}/video/eval
    center_crop: false
    task_config:
      embodiment:
      - aloha-agilex
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
  group_name: RolloutGroup
  backend: huggingface
  recompute_logprobs: false
  enable_offload: true
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}
actor:
  group_name: ActorGroup
  training_backend: fsdp
  micro_batch_size: 8
  global_batch_size: 64
  seed: 1234
  enable_offload: true
  model:
    model_path: /root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
    num_action_chunks: 50
    add_value_head: true
    action_dim: 14
    openpi:
      config_name: pi0_aloha_robotwin
      num_images_in_input: 3
      detach_critic_input: true
  optim:
    lr: 5.6e-06
    value_lr: 0.00011
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0
  fsdp_config:
    strategy: fsdp
    gradient_checkpointing: false
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}
reward:
  use_reward_model: false
critic:
  use_critic_model: false
weight_syncer:
  patch:
    snapshot_device: cpu
    transport_device: cpu
    init_sync:
      enabled: true

```



再跑

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_adjust_bottle_ppo_openpi_a800_1gpu_smoke"
EXP_NAME="robotwin_ppo_openpi_a800_1gpu_smoke_cpu_transport"

ray stop -f || true
nvidia-smi

RUN_NAME="$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}-cpu-transport"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"
MON_DIR="${LOG_DIR}/resource_monitor"

mkdir -p "${LOG_DIR}" "${MON_DIR}"

nohup /root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name "${CONFIG_NAME}" \
  "runner.logger.log_path=${LOG_DIR}" \
  "runner.logger.experiment_name=${EXP_NAME}" \
  > "${RUN_LOG}" 2>&1 &

TRAIN_PID=$!

nohup /root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/RLinf/local_scripts/monitor_pi0_resources.py \
  --pid "${TRAIN_PID}" \
  --out-dir "${MON_DIR}" \
  --interval 2 \
  > "${MON_DIR}/monitor.log" 2>&1 &

MONITOR_PID=$!

echo "TRAIN_PID=${TRAIN_PID}"
echo "MONITOR_PID=${MONITOR_PID}"
echo "LOG_DIR=${LOG_DIR}"
echo "RUN_LOG=${RUN_LOG}"
echo "PEAK=${MON_DIR}/peak.txt"
echo "CSV=${MON_DIR}/resources.csv"
```

成功smoke rollout 训练 评估



正式跑

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}

export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_adjust_bottle_ppo_openpi_a800_1gpu_curve"
EXP_NAME="robotwin_ppo_openpi_a800_1gpu_curve"

echo "===== stop old Ray ====="
ray stop -f || true

echo
echo "===== stale process check ====="
pgrep -af \
'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' \
|| true

echo
echo "===== GPU before training ====="
nvidia-smi

RUN_NAME="$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"
MON_DIR="${LOG_DIR}/resource_monitor"
MON_LOG="${MON_DIR}/monitor.log"

mkdir -p "${LOG_DIR}" "${MON_DIR}"

nohup /root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf/examples/embodiment/config/ \
  --config-name "${CONFIG_NAME}" \
  "runner.logger.log_path=${LOG_DIR}" \
  "runner.logger.experiment_name=${EXP_NAME}" \
  > "${RUN_LOG}" 2>&1 &

TRAIN_PID=$!

nohup /root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/RLinf/local_scripts/monitor_pi0_resources.py \
  --pid "${TRAIN_PID}" \
  --out-dir "${MON_DIR}" \
  --interval 2 \
  > "${MON_LOG}" 2>&1 &

MONITOR_PID=$!

printf '%s\n' "${TRAIN_PID}" > "${LOG_DIR}/train.pid"
printf '%s\n' "${MONITOR_PID}" > "${LOG_DIR}/monitor.pid"

echo "TRAIN_PID=${TRAIN_PID}"
echo "MONITOR_PID=${MONITOR_PID}"
echo "RUN_NAME=${RUN_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "RUN_LOG=${RUN_LOG}"
echo "PEAK_FILE=${MON_DIR}/peak.txt"
echo "RESOURCE_CSV=${MON_DIR}/resources.csv"
```



内存爆，直接死机，重启



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MWZjYjgxNWQ4NmEwN2Q5MTViYzkxMzM2MGNhNzJhNTRfYzMzOWU4MDk2OGQwYjMwZjIwMjA0ZWViMWUyZTUzNmJfSUQ6NzY2MTYwNjgxNTk1MTUyMzAzNl8xNzg1MTYwMjQwOjE3ODUyNDY2NDBfVjM)

能看出提升



### 跑 回 2\*a800



#### 准备

环境切换回pi0的

```Bash
cd /root/autodl-tmp/RLinf

CURRENT_VENV="/root/autodl-tmp/RLinf/.venv"
PI0_BACKUP="/root/autodl-tmp/RLinf/.venv_backup_pi0_ppo_grpo_ok_20260618_142009"

test -x "${CURRENT_VENV}/bin/python" || {
  echo "ERROR: current venv missing"
  exit 1
}

test -x "${PI0_BACKUP}/bin/python" || {
  echo "ERROR: Pi0 golden backup missing"
  exit 1
}

"${CURRENT_VENV}/bin/ray" stop -f || true
deactivate 2>/dev/null || true

STAMP="$(date +%Y%m%d_%H%M%S)"
MOTUS_VENV="/root/autodl-tmp/RLinf/.venv_motus_old2gpu_${STAMP}"

mv "${CURRENT_VENV}" "${MOTUS_VENV}"

mkdir "${CURRENT_VENV}"

rsync -aH --info=progress2 \
  "${PI0_BACKUP}/" \
  "${CURRENT_VENV}/"

echo
echo "PI0_ACTIVE_VENV=${CURRENT_VENV}"
echo "MOTUS_VENV_SAVED=${MOTUS_VENV}"

du -sh \
  "${CURRENT_VENV}" \
  "${PI0_BACKUP}" \
  "${MOTUS_VENV}"
```



复制配置

```Bash
cd /root/autodl-tmp/RLinf

OFFICIAL="examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml"

SMOKE="examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_a800_2gpu_smoke.yaml"

BASELINE="examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_a800_2gpu_baseline.yaml"

STAMP="$(date +%Y%m%d_%H%M%S)"

for DST in "${SMOKE}" "${BASELINE}"; do
  if [ -e "${DST}" ]; then
    cp -a "${DST}" "${DST}.backup_${STAMP}"
  fi
  cp -a "${OFFICIAL}" "${DST}"
  echo "created: ${DST}"
done
```



/root/autodl\-tmp/RLinf/examples/embodiment/config/robotwin\_adjust\_bottle\_ppo\_openpi\_a800\_2gpu\_smoke\.yaml

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/pi0@actor.model
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
    actor, env, rollout: 0-1

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_ppo_openpi_2gpu_smoke"
    logger_backends: ["tensorboard"] # wandb, swanlab

  max_epochs: 1000
  max_steps: 1 #-1

  only_eval: False
  val_check_interval: -1 #10
  save_interval: 1

  resume_dir: null # Optional: path to a saved checkpoint directory, such as 'checkpoints/global_step_10'. If not None, it will be used to resume training.
  ckpt_path: null  # Optional: path to a .pt checkpoint. If not None, it will be loaded after the model is instantiated (for evaluation).

algorithm:
  normalize_advantages: True
  kl_penalty: kl  # how to estimate kl divergence: kl or kl_penalty
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
  # Override the default values in env/robotwin_adjust_bottle
  train:
    rollout_epoch: 16 #4
    total_num_envs: 16 #256
    reward_coef: ${algorithm.reward_coef}
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
    group_size: ${algorithm.group_size}
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
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
  eval:
    rollout_epoch: 16 #1
    total_num_envs: 4 #128
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 200
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 200
    group_size: 1
    use_fixed_reset_state_ids: False #True
    is_eval: True
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    video_cfg:
      save_video: False #True
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
  enable_offload: True
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 32
  global_batch_size: 512 #2048 # 1024
  seed: 1234
  enable_offload: True

  # Override the default values in model/openpi
  model:
    model_path: "/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
    num_action_chunks: 50 # interface for the env
    add_value_head: True
    action_dim: 14
    openpi:
      config_name: "pi0_aloha_robotwin"
      num_images_in_input: 3
      detach_critic_input: True

  optim:
    lr: 5.6e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  # Override the default values in training_backend/fsdp
  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False # for openpi, gradient checkpointing is not supported, please do not change this value
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False
```



资源监控脚本

```Python
cd /root/autodl-tmp/RLinf
mkdir -p local_scripts

cat > local_scripts/monitor_pi0_resources_2gpu.py <<'PY'
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


def timestamp() -> str:
    return datetime.now().strftime("%F %T")


def read_int(path: str) -> int | None:
    p = Path(path)
    if not p.exists():
        return None

    value = p.read_text().strip()
    if value == "max":
        return None

    try:
        return int(value)
    except ValueError:
        return None


def read_cgroup_memory() -> tuple[int, int]:
    current = read_int("/sys/fs/cgroup/memory.current")
    limit = read_int("/sys/fs/cgroup/memory.max")

    if current is None:
        current = read_int(
            "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        )

    if limit is None:
        limit = read_int(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        )

    return current or 0, limit or 0


def read_memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    values: dict[str, int] = {}

    if not path.exists():
        return values

    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = int(parts[1])
        except ValueError:
            pass

    return values


def safe_float(value: str) -> float:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return 0.0


def read_gpus() -> dict[int, dict[str, float]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]

    result = {
        0: {"memory": 0.0, "util": 0.0, "power": 0.0},
        1: {"memory": 0.0, "util": 0.0, "power": 0.0},
    }

    try:
        output = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
        )

        for line in output.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) != 4:
                continue

            index = int(float(parts[0]))
            result[index] = {
                "memory": safe_float(parts[1]),
                "util": safe_float(parts[2]),
                "power": safe_float(parts[3]),
            }
    except Exception:
        pass

    return result


def read_process_groups():
    groups = {
        "env": 0.0,
        "actor": 0.0,
        "rollout": 0.0,
        "driver": 0.0,
        "ray_system": 0.0,
    }

    top_pid = 0
    top_rss_mb = 0.0
    top_command = "none"

    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,rss=,args=", "--sort=-rss"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return groups, top_pid, top_rss_mb, top_command

    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue

        try:
            pid = int(parts[0])
            rss_mb = int(parts[1]) / 1024.0
        except ValueError:
            continue

        command = parts[2]
        lowered = command.lower()

        if rss_mb > top_rss_mb:
            top_pid = pid
            top_rss_mb = rss_mb
            top_command = command[:200]

        if "envworker" in lowered:
            groups["env"] += rss_mb
        elif "embodiedfsdpactor" in lowered:
            groups["actor"] += rss_mb
        elif "multisteprolloutworker" in lowered:
            groups["rollout"] += rss_mb
        elif "train_embodied_agent.py" in lowered:
            groups["driver"] += rss_mb
        elif (
            "raylet" in lowered
            or "gcs_server" in lowered
            or "plasma_store" in lowered
        ):
            groups["ray_system"] += rss_mb

    return groups, top_pid, top_rss_mb, top_command


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "resources.csv"
    peak_path = out_dir / "peak.txt"

    peak_ram_mb = 0
    peak_ram_time = ""

    peak_gpu0_mb = 0
    peak_gpu1_mb = 0
    peak_gpu_time = ""

    peak_env_rss_mb = 0.0
    peak_actor_rss_mb = 0.0
    peak_rollout_rss_mb = 0.0

    fields = [
        "timestamp",
        "cgroup_ram_mb",
        "cgroup_limit_mb",
        "cgroup_ram_pct",
        "shm_used_mb",
        "gpu0_memory_mb",
        "gpu0_util_pct",
        "gpu0_power_w",
        "gpu1_memory_mb",
        "gpu1_util_pct",
        "gpu1_power_w",
        "gpu_total_memory_mb",
        "env_rss_mb",
        "actor_rss_mb",
        "rollout_rss_mb",
        "driver_rss_mb",
        "ray_system_rss_mb",
        "top_pid",
        "top_rss_mb",
        "top_command",
        "cgroup_oom",
        "cgroup_oom_kill",
    ]

    with csv_path.open("w", newline="", buffering=1) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        while True:
            now = timestamp()

            current_bytes, limit_bytes = read_cgroup_memory()

            current_mb = current_bytes // 1024 // 1024
            limit_mb = limit_bytes // 1024 // 1024

            ram_pct = (
                100.0 * current_mb / limit_mb
                if limit_mb
                else 0.0
            )

            shm = shutil.disk_usage("/dev/shm")
            shm_used_mb = (
                shm.total - shm.free
            ) // 1024 // 1024

            gpu = read_gpus()
            groups, top_pid, top_rss_mb, top_command = (
                read_process_groups()
            )
            events = read_memory_events()

            gpu0_mb = int(gpu[0]["memory"])
            gpu1_mb = int(gpu[1]["memory"])

            if current_mb > peak_ram_mb:
                peak_ram_mb = current_mb
                peak_ram_time = now

            if (
                gpu0_mb > peak_gpu0_mb
                or gpu1_mb > peak_gpu1_mb
            ):
                peak_gpu0_mb = max(peak_gpu0_mb, gpu0_mb)
                peak_gpu1_mb = max(peak_gpu1_mb, gpu1_mb)
                peak_gpu_time = now

            peak_env_rss_mb = max(
                peak_env_rss_mb,
                groups["env"],
            )
            peak_actor_rss_mb = max(
                peak_actor_rss_mb,
                groups["actor"],
            )
            peak_rollout_rss_mb = max(
                peak_rollout_rss_mb,
                groups["rollout"],
            )

            writer.writerow({
                "timestamp": now,
                "cgroup_ram_mb": current_mb,
                "cgroup_limit_mb": limit_mb,
                "cgroup_ram_pct": f"{ram_pct:.2f}",
                "shm_used_mb": shm_used_mb,
                "gpu0_memory_mb": gpu0_mb,
                "gpu0_util_pct": int(gpu[0]["util"]),
                "gpu0_power_w": f"{gpu[0]['power']:.2f}",
                "gpu1_memory_mb": gpu1_mb,
                "gpu1_util_pct": int(gpu[1]["util"]),
                "gpu1_power_w": f"{gpu[1]['power']:.2f}",
                "gpu_total_memory_mb": gpu0_mb + gpu1_mb,
                "env_rss_mb": f"{groups['env']:.1f}",
                "actor_rss_mb": f"{groups['actor']:.1f}",
                "rollout_rss_mb": f"{groups['rollout']:.1f}",
                "driver_rss_mb": f"{groups['driver']:.1f}",
                "ray_system_rss_mb": f"{groups['ray_system']:.1f}",
                "top_pid": top_pid,
                "top_rss_mb": f"{top_rss_mb:.1f}",
                "top_command": top_command,
                "cgroup_oom": events.get("oom", 0),
                "cgroup_oom_kill": events.get("oom_kill", 0),
            })

            peak_path.write_text(
                "\n".join([
                    f"updated_at={now}",
                    f"target_pid={args.pid}",
                    f"process_alive={process_alive(args.pid)}",
                    "",
                    f"current_ram_mb={current_mb}",
                    f"cgroup_limit_mb={limit_mb}",
                    f"current_ram_pct={ram_pct:.2f}",
                    f"peak_ram_mb={peak_ram_mb}",
                    f"peak_ram_time={peak_ram_time}",
                    "",
                    f"current_gpu0_mb={gpu0_mb}",
                    f"current_gpu1_mb={gpu1_mb}",
                    f"peak_gpu0_mb={peak_gpu0_mb}",
                    f"peak_gpu1_mb={peak_gpu1_mb}",
                    f"peak_gpu_time={peak_gpu_time}",
                    "",
                    f"current_env_rss_mb={groups['env']:.1f}",
                    f"current_actor_rss_mb={groups['actor']:.1f}",
                    f"current_rollout_rss_mb={groups['rollout']:.1f}",
                    f"peak_env_rss_mb={peak_env_rss_mb:.1f}",
                    f"peak_actor_rss_mb={peak_actor_rss_mb:.1f}",
                    f"peak_rollout_rss_mb={peak_rollout_rss_mb:.1f}",
                    "",
                    f"current_shm_used_mb={shm_used_mb}",
                    f"top_pid={top_pid}",
                    f"top_rss_mb={top_rss_mb:.1f}",
                    f"top_command={top_command}",
                    "",
                    f"cgroup_oom={events.get('oom', 0)}",
                    f"cgroup_oom_kill={events.get('oom_kill', 0)}",
                    "",
                    f"csv={csv_path}",
                ]) + "\n"
            )

            if not process_alive(args.pid):
                break

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
PY

chmod +x local_scripts/monitor_pi0_resources_2gpu.py

/root/autodl-tmp/RLinf/.venv/bin/python \
  -m py_compile \
  local_scripts/monitor_pi0_resources_2gpu.py

echo "monitor ready"
```



Smoke

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}

export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_adjust_bottle_ppo_openpi_a800_2gpu_smoke"
EXP_NAME="robotwin_ppo_openpi_a800_2gpu_smoke"

ray stop -f || true

pgrep -af \
'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' \
|| true

nvidia-smi

RUN_NAME="$(date +'%Y%m%d_%H%M%S')-${CONFIG_NAME}"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"
MON_DIR="${LOG_DIR}/resource_monitor"
MON_LOG="${MON_DIR}/monitor.log"

mkdir -p "${LOG_DIR}" "${MON_DIR}"

CMD=(
  /root/autodl-tmp/RLinf/.venv/bin/python
  /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py
  --config-path
  /root/autodl-tmp/RLinf/examples/embodiment/config/
  --config-name
  "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "runner.logger.experiment_name=${EXP_NAME}"
)

printf "%q " "${CMD[@]}" > "${LOG_DIR}/command.txt"
echo >> "${LOG_DIR}/command.txt"

nohup "${CMD[@]}" \
  > "${RUN_LOG}" 2>&1 &

TRAIN_PID=$!

nohup /root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/RLinf/local_scripts/monitor_pi0_resources_2gpu.py \
  --pid "${TRAIN_PID}" \
  --out-dir "${MON_DIR}" \
  --interval 2 \
  > "${MON_LOG}" 2>&1 &

MONITOR_PID=$!

printf '%s\n' "${TRAIN_PID}" > "${LOG_DIR}/train.pid"
printf '%s\n' "${MONITOR_PID}" > "${LOG_DIR}/monitor.pid"

echo "TRAIN_PID=${TRAIN_PID}"
echo "MONITOR_PID=${MONITOR_PID}"
echo "LOG_DIR=${LOG_DIR}"
echo "RUN_LOG=${RUN_LOG}"
echo "PEAK=${MON_DIR}/peak.txt"
echo "CSV=${MON_DIR}/resources.csv"
```



爆

```Plain Text
失败
主错误是：
RolloutGroup(rank=0)
→ _recv_tensor_list_via_ipc
→ rebuild_cuda_tensor
→ pidfd_getfd
→ Operation not permitted
这发生在：
Actor → Rollout 初始权重同步

两卡 placement 是：
actor, env, rollout: 0-1
实际布局是：
GPU 0:
  Actor rank 0
  Rollout rank 0

GPU 1:
  Actor rank 1
  Rollout rank 1
虽然 actor rank 0 和 actor rank 1 之间使用 NCCL/FSDP，但每个 actor 仍然要把权重同步给**同一张卡上的另一个 Rollout Python 进程**。
所以 rank 0 仍然走：
Actor rank 0 GPU tensor
→ 同卡 CUDA IPC
→ Rollout rank 0
AutoDL 容器不允许 CUDA IPC 重建时使用的 pidfd_getfd，因此两卡与单卡表现相同。
```



修补 额外

```Bash
OMP_NUM_THREADS raw:
<0>
0 不是合法值，所以所有 Ray worker 都打印：
libgomp: Invalid value for environment variable OMP_NUM_THREADS
它不是这次失败的根因，但正式训练前必须处理。
每次 source rlinf_env.sh 后执行：
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
由于它不是在 rlinf_env.sh 中设置的，可能来自 AutoDL 容器的全局环境或 shell 初始化。现阶段直接覆盖即可。
```



当前配置

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/pi0@actor.model
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
    actor, env, rollout: 0-1

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_ppo_openpi_2gpu_smoke"
    logger_backends: ["tensorboard"] # wandb, swanlab

  max_epochs: 1000
  max_steps: 1 #-1

  only_eval: False
  val_check_interval: -1 #10
  save_interval: 1

  resume_dir: null # Optional: path to a saved checkpoint directory, such as 'checkpoints/global_step_10'. If not None, it will be used to resume training.
  ckpt_path: null  # Optional: path to a .pt checkpoint. If not None, it will be loaded after the model is instantiated (for evaluation).

algorithm:
  normalize_advantages: True
  kl_penalty: kl  # how to estimate kl divergence: kl or kl_penalty
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
  # Override the default values in env/robotwin_adjust_bottle
  train:
    rollout_epoch: 16 #4
    total_num_envs: 16 #256
    reward_coef: ${algorithm.reward_coef}
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
    group_size: ${algorithm.group_size}
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
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
    video_cfg:
      save_video: False #True
  eval:
    rollout_epoch: 16 #1
    total_num_envs: 4 #128
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 200
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 200
    group_size: 1
    use_fixed_reset_state_ids: False #True
    is_eval: True
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    video_cfg:
      save_video: False #True
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
  enable_offload: True
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 32
  global_batch_size: 512 #2048 # 1024
  seed: 1234
  enable_offload: True

  # Override the default values in model/openpi
  model:
    model_path: "/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
    num_action_chunks: 50 # interface for the env
    add_value_head: True
    action_dim: 14
    openpi:
      config_name: "pi0_aloha_robotwin"
      num_images_in_input: 3
      detach_critic_input: True

  optim:
    lr: 5.6e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  # Override the default values in training_backend/fsdp
  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False # for openpi, gradient checkpointing is not supported, please do not change this value
    save_full_model_weights: False
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
    snapshot_device: cpu
    transport_device: cpu
    init_sync:
      enabled: true
```



只存dcp，不存ckpt

dcp如何使用，评估，继续训练

```Bash
当前保存的是 DCP 格式的完整训练 checkpoint，只是不额外导出单文件 full_weights.pt。

DCP 是 PyTorch Distributed Checkpoint。每个 FSDP rank 并行保存自己的分片，因此不需要先把所有权重集中到 rank 0；加载时还支持重新分片

global_step_20 实物：
.metadata：约 1 MiB
__0_0.distcp：约 4.84 GiB
__1_0.distcp：约 4.84 GiB

DCP 内实际包含：
actor 模型，包括 value head
Adam optimizer 状态
LR scheduler 状态
FSDP 版本
Python、NumPy、Torch CPU/CUDA RNG 状态
它不包含：
EnvWorker/SAPIEN 环境状态
未完成的 rollout
rollout worker 状态
global step 数值本身——RLinf 从目录名 global_step_20 解析



将 DCP 转成评估用的 full_weights.pt
在训练段完全退出、Ray 停止后执行：
ray stop -f || true

STEP_DIR="/path/to/global_step_20"

DCP_PATH="${STEP_DIR}/actor/dcp_checkpoint"

OUTPUT_PATH="${STEP_DIR}/actor/model_state_dict/full_weights.pt"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

python \
  rlinf/utils/ckpt_convertor/fsdp_convertor/convert_dcp_to_pt.py \
  --dcp_path "${DCP_PATH}" \
  --output_path "${OUTPUT_PATH}"

ls -lh "${OUTPUT_PATH}"
官方转换脚本正是从 DCP 的 fsdp_checkpoint.model 提取模型 state dict。
这样可以：
 训练时避免完整权重导出的 RAM 峰值； 
 停止训练后单独转换； 
 用 full_weights.pt 独立评估； 
 评估完成后删掉 .pt 或旧 DCP，节省磁盘。
 
 五、DCP 如何续训
设置：
runner:
  resume_dir: /path/to/checkpoints/global_step_20
注意目录层级是：
.../global_step_20
不是：
.../global_step_20/actor
runner 会自己拼接 actor，加载 DCP，并从目录名中解析出 global_step=20。
例如：
resume_dir = global_step_20
max_steps = 40
表示继续跑 step 20→40，也就是再训练 20 步。训练循环从恢复出的 global step 开始。

六、DCP 如何变成完整评估权重
训练段结束、Ray 完全停止后：
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

ray stop -f || true

STEP_DIR="/exact/path/to/checkpoints/global_step_20"

DCP_PATH="${STEP_DIR}/actor/dcp_checkpoint"

OUTPUT_PATH="${STEP_DIR}/actor/model_state_dict/full_weights.pt"

echo "DCP_PATH=${DCP_PATH}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"

test -d "${DCP_PATH}" &&
  echo "DCP directory exists"

find "${DCP_PATH}" \
  -name '*.distcp' \
  -type f \
  -size +0c \
  -print

mkdir -p "$(dirname "${OUTPUT_PATH}")"

python \
  rlinf/utils/ckpt_convertor/fsdp_convertor/convert_dcp_to_pt.py \
  --dcp_path "${DCP_PATH}" \
  --output_path "${OUTPUT_PATH}"

ls -lh "${OUTPUT_PATH}"
该脚本会从 DCP 中读取：
fsdp_checkpoint.model
并保存为普通 .pt state dict。
之后独立评估使用：
runner:
  ckpt_path: /.../model_state_dict/full_weights.pt
所以正确工作流是：
训练时：DCP-only
训练段结束：停止 Ray
离线转换 selected checkpoints
独立固定 seeds 评估
```



#### 跑

Smoke

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_adjust_bottle_ppo_openpi_a800_2gpu_smoke"
EXP_NAME="robotwin_ppo_openpi_a800_2gpu_smoke_cpu_transport_dcp_only"

ray stop -f || true

echo "===== stale processes ====="

pgrep -af \
'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' \
|| true

echo
echo "===== GPU before smoke ====="

nvidia-smi

RUN_NAME="$(date +'%Y%m%d_%H%M%S')-${CONFIG_NAME}-cpu-transport-dcp-only"

LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"
MON_DIR="${LOG_DIR}/resource_monitor"
MON_LOG="${MON_DIR}/monitor.log"

mkdir -p "${LOG_DIR}" "${MON_DIR}"

CMD=(
  /root/autodl-tmp/RLinf/.venv/bin/python
  /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py
  --config-path
  /root/autodl-tmp/RLinf/examples/embodiment/config/
  --config-name
  "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "runner.logger.experiment_name=${EXP_NAME}"
)

printf "%q " "${CMD[@]}" > "${LOG_DIR}/command.txt"
echo >> "${LOG_DIR}/command.txt"

nohup "${CMD[@]}" \
  > "${RUN_LOG}" 2>&1 &

TRAIN_PID=$!

nohup /root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/RLinf/local_scripts/monitor_pi0_resources_2gpu.py \
  --pid "${TRAIN_PID}" \
  --out-dir "${MON_DIR}" \
  --interval 2 \
  > "${MON_LOG}" 2>&1 &

MONITOR_PID=$!

printf '%s\n' "${TRAIN_PID}" > "${LOG_DIR}/train.pid"
printf '%s\n' "${MONITOR_PID}" > "${LOG_DIR}/monitor.pid"

echo "TRAIN_PID=${TRAIN_PID}"
echo "MONITOR_PID=${MONITOR_PID}"
echo "LOG_DIR=${LOG_DIR}"
echo "RUN_LOG=${RUN_LOG}"
echo "PEAK=${MON_DIR}/peak.txt"
echo "CSV=${MON_DIR}/resources.csv"
```

峰值：内存150/240g，每卡30g



正式配置

/root/autodl\-tmp/RLinf/examples/embodiment/config/robotwin\_adjust\_bottle\_ppo\_openpi\_a800\_2gpu\_baseline\.yaml

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/pi0@actor.model
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
    actor, env, rollout: 0-1

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_ppo_openpi_2gpu_baseline"
    logger_backends: ["tensorboard"] # wandb, swanlab

  max_epochs: 1000
  max_steps: 60 #1 #-1

  only_eval: False
  val_check_interval: -1 #10
  save_interval: 20

  resume_dir: null # Optional: path to a saved checkpoint directory, such as 'checkpoints/global_step_10'. If not None, it will be used to resume training.
  ckpt_path: null  # Optional: path to a .pt checkpoint. If not None, it will be loaded after the model is instantiated (for evaluation).

algorithm:
  normalize_advantages: True
  kl_penalty: kl  # how to estimate kl divergence: kl or kl_penalty
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
  # Override the default values in env/robotwin_adjust_bottle
  train:
    rollout_epoch: 16 #4
    total_num_envs: 16 #256
    reward_coef: ${algorithm.reward_coef}
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
    group_size: ${algorithm.group_size}
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
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
    video_cfg:
      save_video: False #True
  eval:
    rollout_epoch: 16 #1
    total_num_envs: 4 #128
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 200
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 200
    group_size: 1
    use_fixed_reset_state_ids: False #True
    is_eval: True
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    video_cfg:
      save_video: False #True
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
  enable_offload: True
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 32
  global_batch_size: 512 #2048 # 1024
  seed: 1234
  enable_offload: True

  # Override the default values in model/openpi
  model:
    model_path: "/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
    num_action_chunks: 50 # interface for the env
    add_value_head: True
    action_dim: 14
    openpi:
      config_name: "pi0_aloha_robotwin"
      num_images_in_input: 3
      detach_critic_input: True

  optim:
    lr: 5.6e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  # Override the default values in training_backend/fsdp
  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False # for openpi, gradient checkpointing is not supported, please do not change this value
    save_full_model_weights: False
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
    snapshot_device: cpu
    transport_device: cpu
    init_sync:
      enabled: true
```



跑

```Bash
source /root/autodl-tmp/rlinf_env.sh
cd /root/autodl-tmp/RLinf
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH=/root/autodl-tmp/RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf/examples/embodiment
export PYTHONPATH=/root/autodl-tmp/RLinf:/root/autodl-tmp/RoboTwin_RLinf:${PYTHONPATH}

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME="robotwin_adjust_bottle_ppo_openpi_a800_2gpu_baseline"
EXP_NAME="robotwin_ppo_openpi_a800_2gpu_baseline"
TARGET_STEP=60

echo "===== stop old Ray ====="
ray stop -f || true
sleep 3

echo
echo "===== stale process check ====="
pgrep -af \
'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' \
|| true

echo
echo "===== GPU before training ====="
nvidia-smi

RUN_NAME="$(date +'%Y%m%d_%H%M%S')-${CONFIG_NAME}-step0-to-${TARGET_STEP}"
LOG_DIR="/root/autodl-tmp/RLinf/logs/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run_embodiment.log"
MON_DIR="${LOG_DIR}/resource_monitor"
MON_LOG="${MON_DIR}/monitor.log"

mkdir -p "${LOG_DIR}" "${MON_DIR}"

CMD=(
  /root/autodl-tmp/RLinf/.venv/bin/python
  /root/autodl-tmp/RLinf/examples/embodiment/train_embodied_agent.py
  --config-path
  /root/autodl-tmp/RLinf/examples/embodiment/config/
  --config-name
  "${CONFIG_NAME}"
  "runner.max_steps=${TARGET_STEP}"
  "runner.resume_dir=null"
  "runner.logger.log_path=${LOG_DIR}"
  "runner.logger.experiment_name=${EXP_NAME}"
)

printf "%q " "${CMD[@]}" > "${LOG_DIR}/command.txt"
echo >> "${LOG_DIR}/command.txt"

nohup "${CMD[@]}" \
  > "${RUN_LOG}" 2>&1 &

TRAIN_PID=$!

nohup /root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/RLinf/local_scripts/monitor_pi0_resources_2gpu.py \
  --pid "${TRAIN_PID}" \
  --out-dir "${MON_DIR}" \
  --interval 2 \
  > "${MON_LOG}" 2>&1 &

MONITOR_PID=$!

printf '%s\n' "${TRAIN_PID}" > "${LOG_DIR}/train.pid"
printf '%s\n' "${MONITOR_PID}" > "${LOG_DIR}/monitor.pid"

echo
echo "TRAIN_PID=${TRAIN_PID}"
echo "MONITOR_PID=${MONITOR_PID}"
echo "RUN_NAME=${RUN_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "RUN_LOG=${RUN_LOG}"
echo "PEAK_FILE=${MON_DIR}/peak.txt"
echo "RESOURCE_CSV=${MON_DIR}/resources.csv"
echo "COMMAND_FILE=${LOG_DIR}/command.txt"
```



critic有效性直接正了，所以每步要多收集rollout才能训critic

指标都正常

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MmViZDBjYjFmNmMwODM4NjAzOTA3NzQzODRmZTgzN2NfMjYyZjI3MTNkYjAyZjBlZDJkMWNiM2E3YTM4YzlkZTVfSUQ6NzY2MjI2NDM1Njc3NzY0Mjk4M18xNzg1MTYwMjQwOjE3ODUyNDY2NDBfVjM)

内存上涨

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OGJhNTcwMmYwMTljMDQ4NDI2YjM5M2Q4MDUxYjYwMWZfMTA4M2E1YTI1ZDk5ZDUwOTQ4N2VmM2U3N2IzMWZhZmRfSUQ6NzY2MjI2NDQzMzU0OTA4NTY0NF8xNzg1MTYwMjQwOjE3ODUyNDY2NDBfVjM)

`global_step_20` checkpoint 已保存成功

峰值短暂越过了 `memory.high=236 GiB`，因此产生 `high=13051` 的回收/节流事件；这不是 13,051 次 OOM。

历史峰值 `236.2 GiB` 出现在第 20 步 checkpoint 保存期间，占硬上限的 `98.4%`，只剩 `3.8 GiB`。

峰值主要来自 Actor 保存 checkpoint 时增加约 7 GiB



内存上涨：

Env worker，仿真器原生内存；很多场景创建销毁

分段训练能解决

```Plain Text
持续上涨的主体正是 EnvWorker 中的仿真器原生内存，而不是 rollout 模型进程。

每个训练 step 的 rollout 数据确实有固定上界，step 结束后会清空；它不是一个跨 step 无限增长的 replay buffer。

截至 22:14，已完成 step 22，step 23 正在 rollout：
项目当前情况判断
容器总内存约 221.5 GiB已比较高
历史峰值236.2 GiB距 240 GiB 硬上限仅 3.8 GiB
两个 EnvWorker合计约 133 GiB RSS绝对主因
两个 Actor合计约 36.7 GiB RSScheckpoint 时会额外上升
两个 Rollout Worker合计约 10.2 GiB RSS不是持续增长主体

step 1：Env 峰值约 35 GiB
step 3：约 80 GiB
step 5：约 98 GiB
step 9：约 121 GiB
step 18–23：约 131–133 GiB，近期增速明显下降

RLinf 会在当前 step 内暂存 action、reward、old value、old logprob、图像、OpenPI denoising chain 等；转换和拆分 trajectory 时还会暂时出现源数据、stack 后数据和 split 后数据同时存在，因此产生周期峰值。但随后会 clear()、删除临时对象并 GC；Actor 也会用新 batch 覆盖旧 batch

“增加 rollout”确实导致内存更快上涨，但上涨主体不是一个无限长的 PPO rollout 列表

主要是更多并发环境和更多 SAPIEN 场景反复销毁、重建后，原生内存没有完全归还操作系统。

**最可靠：分段训练并重启进程。**
每 10 个 step 保存一次，然后让整个 Ray/EnvWorker 进程退出，再从 checkpoint resume。进程退出能够真正把 SAPIEN/native allocator 内存还给 Linux。

memory.high=236 GiB 是谁的机制
这是 Linux cgroup v2 的机制，阈值通常由 AutoDL 容器环境配置，不是 RLinf 内部调度。

超过 memory.high 后：
当前申请内存的进程会被短暂节流；
内核要求它执行直接内存回收；
可以短暂超过 high；
仅超过 high 不会触发 OOM。

high=13051 是累计触发“节流并进入直接回收路径”的次数，不是 13,051 次 OOM。当前：
```



Critic explained variance

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OTEwN2UxMzRiNzRhOTU0MGI2ZTgzNDBhMDNiMmIxNjJfMTJkNzM5MTJjYjc4NWMyYzM5NGQyNDRiZDBjNWI3NGFfSUQ6NzY2MjAyMzI2NzUyMzQ4MDc1N18xNzg1MTYwMjQwOjE3ODUyNDY2NDBfVjM)

```Plain Text
EV=1：Value 几乎完美解释 return 差异。
EV=0：不比始终预测平均 return 更好。
EV<0：预测残差的波动甚至大于 return 自身波动。
负方向没有下界。

EV 衡量的是“相对差异”，不是绝对误差。当一批 return 都接近 0.8，例如只有 0.79–0.81 时，分母非常小；即使 value loss 很小，EV 也可能突然变成 -1 或 -3。

策略质量和绝对 value loss 几乎没变，但 EV 翻转。因此这不是 critic 在一个 step 内“彻底坏掉又修好”。

RLinf 当前不是在全部 256 条 trajectory 上统一算一次 EV，而是：
每个 micro-batch 单独算 EV；
再把这些 EV 比值简单平均。

某个 micro-batch 恰好几乎全成功、return 方差接近零，就能把整步均值拉到很低。
```



跨step不保存rollout，会清空

```Plain Text
关于 rollout“池”，最准确的说法是：
每个 global step 的流程是：
把最新 Actor 权重同步给 Rollout Worker。
16 个环境 × 16 个 rollout_epoch，新收集 256 个完整 episode。
每个 episode 是 200 环境步，OpenPI 每次输出 50 个 action，因此约得到 4 个 PPO chunk，合计约 1024 个 chunk sample。
计算 returns、GAE 和 advantages。
update_epoch=2：只把当前这批数据重复训练两遍。
EnvWorker 清空轨迹；Actor 保留最近一批数据，下一 step 收到新 batch 时覆盖。
```



爆

```Plain Text
训练已经在 12:34 终止，没有跑完 60 step。
最后完整指标是 step 57/60。step 58 的 16/16 rollout 已完成，但尚未形成完整训练指标时，Ray 主动杀死了一个 EnvWorker。
原因：Ray 检测到内存 230.89/240 GiB = 96.2%，超过其 95% 保护阈值。两个 EnvWorker 当时约占 84.44 + 82.38 GiB。
EnvWorker 峰值最终升到 169.4 GiB，说明此前约 163–165 GiB 的平台并不稳定，内存仍会继续增长。
没有生成 step 60 checkpoint。目前只有 step 20、40 两份完整 DCP；最安全的恢复点是 step 40。
退出后内存降到约 4.3 GiB、GPU 清零，进一步证明上涨部分主要是 EnvWorker/SAPIEN 进程常驻内存，完整退出确实可以释放。
```



产物

\[peak\.txt\]

\[metrics\.log\]

\[run\_embodiment\.log\]

\[resources\.csv\]



### 新rlinf仓库



#### 准备

环境迁移计划

\[rlinf\-official\-migration\-plan\-20260714\.md\]

大致执行，没有完全遵守，例如末尾



当前配置

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/pi0@actor.model
  - hybrid_engines/fsdp@actor.fsdp_config
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
    actor, env, rollout: 0-1

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_ppo_openpi_2gpu_baseline_new"
    logger_backends: ["tensorboard"] # wandb, swanlab

  max_epochs: 1000
  max_steps: 1 #-1

  only_eval: False
  val_check_interval: -1 #10
  save_interval: 1 #10

  resume_dir: null # Optional: path to a saved checkpoint directory, such as 'checkpoints/global_step_10'. If not None, it will be used to resume training.
  ckpt_path: null  # Optional: path to a .pt checkpoint. If not None, it will be loaded after the model is instantiated (for evaluation).

algorithm:
  normalize_advantages: True
  kl_penalty: kl  # how to estimate kl divergence: kl or kl_penalty
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
  # Override the default values in env/robotwin_adjust_bottle
  train:
    rollout_epoch: 16 #4
    total_num_envs: 16 #256
    reward_coef: ${algorithm.reward_coef}
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
    group_size: ${algorithm.group_size}
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
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
    video_cfg:
      save_video: False #True
  eval:
    rollout_epoch: 16 #1
    total_num_envs: 4 #128
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 200
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 200
    group_size: 1
    use_fixed_reset_state_ids: False #True
    is_eval: True
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    video_cfg:
      save_video: False #True
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
  enable_offload: True
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 32
  global_batch_size: 512 #2048 # 1024
  seed: 1234
  enable_offload: True

  # Override the default values in model/openpi
  model:
    model_path: "/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
    num_action_chunks: 50 # interface for the env
    add_value_head: True
    action_dim: 14
    openpi:
      config_name: "pi0_aloha_robotwin"
      num_images_in_input: 3
      detach_critic_input: True

  optim:
    lr: 5.6e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  # Override the default values in hybrid_engines/fsdp
  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False # for openpi, gradient checkpointing is not supported, please do not change this value
    save_full_model_weights: False
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
    snapshot_device: cpu
    transport_device: cpu
    init_sync:
      enabled: true

```

相比正式训练配置：

```Plain Text
max_steps: 60 → 1；
save_interval: 20 → 1。
```



Cpu gpu配置

```YAML
旧训练实际开启了什么
目标 60-step 实验实际是：
env.train.enable_offload: false
actor.enable_offload: true
rollout.enable_offload: true
actor.fsdp_config.cpu_offload: false
actor.fsdp_config.save_full_model_weights: false
weight_syncer.patch.snapshot_device: cpu
weight_syncer.patch.transport_device: cpu
所以它是“节省 GPU、较多占用 CPU”的配置。顶层 env.enable_offload:true 没有控制到 EnvWorker。

三个开关的含义
开关true 的行为CPU 压力最小方向代价
env.train.enable_offload每个 PPO rollout 结束后关闭整个 RoboTwin VectorEnvtrue下一 step 重建环境，更慢；native 内存不保证全部归还
actor.enable_offload非训练阶段把 actor 参数、梯度、optimizer 搬到 CPUfalseactor 状态持续占用 GPU
rollout.enable_offload非 rollout 阶段把 HuggingFace rollout 模型搬到 CPUfalseactor 更新时 rollout 模型仍占 GPU

按照你这次“优先缓解 CPU 内存，并在一次 smoke 中一起测试”的目标，我建议：
env:
  train:
    enable_offload: true

rollout:
  enable_offload: false

actor:
  enable_offload: false
  fsdp_config:
    save_full_model_weights: false
```



目前配置：

/root/autodl\-tmp/RLinf/examples/embodiment/config/robotwin\_adjust\_bottle\_ppo\_openpi\_a800\_2gpu\_smoke\.yaml

```Plain Text
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/pi0@actor.model
  - hybrid_engines/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout
  - _self_

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor, env, rollout: 0-1

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_ppo_openpi_2gpu_baseline_new"
    logger_backends: ["tensorboard"] # wandb, swanlab

  max_epochs: 1000
  max_steps: 1 #-1

  only_eval: False
  val_check_interval: -1 #10
  save_interval: 1 #10

  resume_dir: null # Optional: path to a saved checkpoint directory, such as 'checkpoints/global_step_10'. If not None, it will be used to resume training.
  ckpt_path: null  # Optional: path to a .pt checkpoint. If not None, it will be loaded after the model is instantiated (for evaluation).

algorithm:
  normalize_advantages: True
  kl_penalty: kl  # how to estimate kl divergence: kl or kl_penalty
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
  # Override the default values in env/robotwin_adjust_bottle
  train:
    rollout_epoch: 16 #4
    total_num_envs: 16 #256
    reward_coef: ${algorithm.reward_coef}
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
    group_size: ${algorithm.group_size}
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
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
    video_cfg:
      save_video: False #True
  eval:
    rollout_epoch: 16 #1
    total_num_envs: 4 #128
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 200
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 200
    group_size: 1
    use_fixed_reset_state_ids: False #True
    is_eval: True
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    video_cfg:
      save_video: False #True
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
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 32
  global_batch_size: 512 #2048 # 1024
  seed: 1234
  enable_offload: false

  # Override the default values in model/openpi
  model:
    model_path: "/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
    num_action_chunks: 50 # interface for the env
    add_value_head: True
    action_dim: 14
    openpi:
      config_name: "pi0_aloha_robotwin"
      num_images_in_input: 3
      detach_critic_input: True

  optim:
    lr: 5.6e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  # Override the default values in hybrid_engines/fsdp
  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False # for openpi, gradient checkpointing is not supported, please do not change this value
    save_full_model_weights: False
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
    snapshot_device: cpu
    transport_device: cpu
    init_sync:
      enabled: true

```



覆盖？这个没问题吗？

```Plain Text
在 defaults 最后加入：
  - override hydra/job_logging: stdout
  - _self_
这会消除 Hydra 警告，并保持当前主 YAML 覆盖官方 defaults 的顺序。
```



参数问题；指令添加即可

```Plain Text
另外，libgomp 警告来自 AutoDL 的 /etc/profile.d/autodl.env.sh 将 OMP_NUM_THREADS 设置成了无效的 0。解析和训练前执行：
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```



资源监控脚本

```Bash
OLD=/root/autodl-tmp/RLinf
BACK=/root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40

mkdir -p "$OLD/local_scripts"

cp -a "$BACK/local_scripts/monitor_pi0_resources_2gpu.py" \
      "$OLD/local_scripts/monitor_pi0_resources_2gpu.py"

sha256sum \
  "$BACK/local_scripts/monitor_pi0_resources_2gpu.py" \
  "$OLD/local_scripts/monitor_pi0_resources_2gpu.py"

"$OLD/.venv/bin/python" -m py_compile \
  "$OLD/local_scripts/monitor_pi0_resources_2gpu.py"
```

脚本

```Bash
OLD=/root/autodl-tmp/RLinf
BACK=/root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40

mkdir -p "$OLD/local_scripts"
cp -a "$BACK/local_scripts/monitor_pi0_resources_2gpu.py" \
      "$OLD/local_scripts/monitor_pi0_resources_2gpu.py"

sha256sum \
  "$BACK/local_scripts/monitor_pi0_resources_2gpu.py" \
  "$OLD/local_scripts/monitor_pi0_resources_2gpu.py"

"$OLD/.venv/bin/python" -m py_compile \
  "$OLD/local_scripts/monitor_pi0_resources_2gpu.py"
```



#### 跑

Smoke

```Bash
source /root/autodl-tmp/rlinf_env.sh

OLD=/root/autodl-tmp/RLinf
cd "$OLD"
source "$OLD/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH="$OLD"
export EMBODIED_PATH="$OLD/examples/embodiment"
export PYTHONPATH="$OLD:$ROBOTWIN_PATH:${PYTHONPATH:-}"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset STEPS SAVE_INTER NODES

CONFIG_NAME=robotwin_adjust_bottle_ppo_openpi_a800_2gpu_smoke
EXP_NAME=robotwin_ppo_openpi_2gpu_smoke_cpu_saving

ray stop -f || true
pgrep -af 'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' || true
nvidia-smi

RUN_NAME="$(date +'%Y%m%d_%H%M%S')-${CONFIG_NAME}-cpu-saving"
LOG_DIR="$OLD/logs/$RUN_NAME"
RUN_LOG="$LOG_DIR/run_embodiment.log"
MON_DIR="$LOG_DIR/resource_monitor"
MON_LOG="$MON_DIR/monitor.log"

mkdir -p "$LOG_DIR" "$MON_DIR"

CMD=(
  "$OLD/.venv/bin/python"
  "$OLD/examples/embodiment/train_embodied_agent.py"
  --config-path "$OLD/examples/embodiment/config/"
  --config-name "$CONFIG_NAME"
  "runner.logger.log_path=$LOG_DIR"
  "runner.logger.experiment_name=$EXP_NAME"
)

printf '%q ' "${CMD[@]}" > "$LOG_DIR/command.txt"
echo >> "$LOG_DIR/command.txt"

nohup "${CMD[@]}" > "$RUN_LOG" 2>&1 &
TRAIN_PID=$!

nohup "$OLD/.venv/bin/python" \
  "$OLD/local_scripts/monitor_pi0_resources_2gpu.py" \
  --pid "$TRAIN_PID" \
  --out-dir "$MON_DIR" \
  --interval 2 \
  > "$MON_LOG" 2>&1 &
MONITOR_PID=$!

printf '%s\n' "$TRAIN_PID" > "$LOG_DIR/train.pid"
printf '%s\n' "$MONITOR_PID" > "$LOG_DIR/monitor.pid"

echo "TRAIN_PID=$TRAIN_PID"
echo "MONITOR_PID=$MONITOR_PID"
echo "LOG_DIR=$LOG_DIR"
echo "RUN_LOG=$RUN_LOG"
echo "RESOURCE_CSV=$MON_DIR/resources.csv"
echo "PEAK_FILE=$MON_DIR/peak.txt"
```



smoke通

```Plain Text
step 用时 1564.6s，约 26分04秒。
success_once=0.78125，return 同为 0.78125。
Actor 更新、critic 更新和权重同步全部完成。
DCP 完整：
两个 .distcp 各约 5.2GB；
.metadata 约 1MB；
总计约 9.7GiB；

内存与显存
指标新 smoke旧 smoke变化
主机内存峰值124,501MiB，约121.6GiB154,329MiB，约150.7GiB降约29.1GiB
GPU0 峰值39,540MiB31,372MiB增约8.0GiB
GPU1 峰值39,512MiB31,088MiB增约8.2GiB

证明 actor.enable_offload=false、rollout.enable_offload=false 的方向有效：用每卡约 8GiB 额外显存，换回约 29GiB 主机内存。每卡仍只使用约 48%，显存空间比较宽裕。

Env offload 实际没有开启
这两个字段不是一回事：
env:
  enable_offload: true       # 顶层字段，不控制当前 EnvWorker
真正生效的是：
env:
  train:
    enable_offload: true
本次保存下来的实际配置明确是：
env.train.enable_offload: false
actor.enable_offload: false
rollout.enable_offload: false

```



增加和改参数：

```YAML
env:
  train:
    enable_offload: true
    total_num_envs: 32
    rollout_epoch: 8
```



当前正式训练参数：

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/pi0@actor.model
  - hybrid_engines/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout
  - _self_

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor, env, rollout: 0-1

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_ppo_openpi_2gpu_baseline"
    logger_backends: ["tensorboard"] # wandb, swanlab

  max_epochs: 1000
  max_steps: 100 #1 #-1

  only_eval: False
  val_check_interval: -1 #10
  save_interval: 10 #1 #10

  resume_dir: null # Optional: path to a saved checkpoint directory, such as 'checkpoints/global_step_10'. If not None, it will be used to resume training.
  ckpt_path: null  # Optional: path to a .pt checkpoint. If not None, it will be loaded after the model is instantiated (for evaluation).

algorithm:
  normalize_advantages: True
  kl_penalty: kl  # how to estimate kl divergence: kl or kl_penalty
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
  # Override the default values in env/robotwin_adjust_bottle
  train:
    enable_offload: True
    rollout_epoch: 8 #16 #4
    total_num_envs: 32 #16 #256
    reward_coef: ${algorithm.reward_coef}
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
    group_size: ${algorithm.group_size}
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
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
    video_cfg:
      save_video: False #True
  eval:
    rollout_epoch: 16 #1
    total_num_envs: 4 #128
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 200
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 200
    group_size: 1
    use_fixed_reset_state_ids: False #True
    is_eval: True
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    video_cfg:
      save_video: False #True
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
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 32
  global_batch_size: 512 #2048 # 1024
  seed: 1234
  enable_offload: false

  # Override the default values in model/openpi
  model:
    model_path: "/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
    num_action_chunks: 50 # interface for the env
    add_value_head: True
    action_dim: 14
    openpi:
      config_name: "pi0_aloha_robotwin"
      num_images_in_input: 3
      detach_critic_input: True

  optim:
    lr: 5.6e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  # Override the default values in hybrid_engines/fsdp
  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False # for openpi, gradient checkpointing is not supported, please do not change this value
    save_full_model_weights: False
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
    snapshot_device: cpu
    transport_device: cpu
    init_sync:
      enabled: true

```



正式训练：

```Bash
source /root/autodl-tmp/rlinf_env.sh

OLD=/root/autodl-tmp/RLinf
cd "$OLD"
source "$OLD/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH="$OLD"
export EMBODIED_PATH="$OLD/examples/embodiment"
export PYTHONPATH="$OLD:$ROBOTWIN_PATH:${PYTHONPATH:-}"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset STEPS SAVE_INTER NODES

CONFIG_NAME=robotwin_adjust_bottle_ppo_openpi_a800_2gpu_baseline
EXP_NAME=robotwin_ppo_openpi_2gpu_env32_rollout8_cpu_saving
TARGET_STEP=100
SAVE_INTERVAL=10

ray stop -f || true

pgrep -af \
'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' \
|| true

nvidia-smi

RUN_NAME="$(date +'%Y%m%d_%H%M%S')-${CONFIG_NAME}-env32-rollout8-step0-to-${TARGET_STEP}"
LOG_DIR="$OLD/logs/$RUN_NAME"
RUN_LOG="$LOG_DIR/run_embodiment.log"
MON_DIR="$LOG_DIR/resource_monitor"
MON_LOG="$MON_DIR/monitor.log"

mkdir -p "$LOG_DIR" "$MON_DIR"

CMD=(
  "$OLD/.venv/bin/python"
  "$OLD/examples/embodiment/train_embodied_agent.py"
  --config-path "$OLD/examples/embodiment/config/"
  --config-name "$CONFIG_NAME"
  "runner.max_steps=$TARGET_STEP"
  "runner.save_interval=$SAVE_INTERVAL"
  "runner.resume_dir=null"
  "runner.logger.log_path=$LOG_DIR"
  "runner.logger.experiment_name=$EXP_NAME"
)

printf '%q ' "${CMD[@]}" > "$LOG_DIR/command.txt"
echo >> "$LOG_DIR/command.txt"

nohup "${CMD[@]}" > "$RUN_LOG" 2>&1 &
TRAIN_PID=$!

nohup "$OLD/.venv/bin/python" \
  "$OLD/local_scripts/monitor_pi0_resources_2gpu.py" \
  --pid "$TRAIN_PID" \
  --out-dir "$MON_DIR" \
  --interval 2 \
  > "$MON_LOG" 2>&1 &
MONITOR_PID=$!

printf '%s\n' "$TRAIN_PID" > "$LOG_DIR/train.pid"
printf '%s\n' "$MONITOR_PID" > "$LOG_DIR/monitor.pid"

echo "TRAIN_PID=$TRAIN_PID"
echo "MONITOR_PID=$MONITOR_PID"
echo "LOG_DIR=$LOG_DIR"
echo "RUN_LOG=$RUN_LOG"
echo "RESOURCE_CSV=$MON_DIR/resources.csv"
echo "PEAK_FILE=$MON_DIR/peak.txt"
```



指标正常，有提升

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MDA1MDEyNmY2NDRlYTBkNjcwZGQxOTQxNmQzZWM2MTNfZGE4ZTAzZDVmMjM4MTk0NDUzOTAzMzY2ZjAwMDRkYmFfSUQ6NzY2MjU2MjM4MzkxNTE3NDg4MV8xNzg1MTYwMjQwOjE3ODUyNDY2NDBfVjM)



内存爆；并行参数开太大

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MmEwN2ZhYWZjMDg0ZjU2MTdjNmQxYzY5MWIzNWNlZGFfMmYyMDk1ZjA3YWYxNTg5MTg3MGEwZDJhOTgzMTcwOWRfSUQ6NzY2MjU2MTM2MDkxNTU0OTM3MF8xNzg1MTYwMjQwOjE3ODUyNDY2NDBfVjM)

```Plain Text
06:28 节点内存达到 Ray 的 95% 阈值，Ray 主动杀死一个 ChannelWorker，随后 collective 通信报错。
EnvWorker 占约 201.2 GiB，仍是绝对主因。
历史峰值多次触及 memory.high=236 GiB。此前的锯齿回落不是自然平台，而是 cgroup 节流和回收。
两个长期运行的 EnvWorker：
两个 EnvWorker 分别约 99.9 GiB 和 99.2 GiB，总计约 201 GiB。
Actor 与 Rollout Worker 合计约 20.6 GiB。
Ray object store 统计为 0。
总内存达到 228.04 / 240 GiB = 95.0152% 后，Ray 按默认 95% 阈值主动杀了一个 ChannelWorker，继而导致同步训练通信链路挂死

为什么 EnvWorker 一直涨
当前每个 EnvWorker 同时持有 16 个 RoboTwin 子环境。每个子环境都包含 SAPIEN scene、renderer、engine、robot、camera 等原生对象。
你的 env.train.enable_offload: true 确实已经生效。每个 PPO step 后会：
置空 scene、renderer、engine、robot、camera；
清空环境列表；
调用 SAPIEN cache 清理、Python GC 和 CUDA cache 清理；
下一 step 再重新创建这些环境。
但 EnvWorker 进程和线程池不退出。释放 Python 引用并不保证 SAPIEN、PhysX、Vulkan 或 glibc 的原生内存页立即还给 Linux。因此会出现：

EnvWorker 的跨 step 常驻基线持续从几十 GiB 上涨到约 201 GiB，最后让 Ray 杀进程：不属于可以接受的正常状态。


```



产物：

\[metrics\.log\]

\[peak\.txt\]

\[resources\.csv\]

\[run\_embodiment\.log\]



结论：内存问题，要降并行，断点重新训



`env.train.enable_offload`

```Plain Text
clear_cache_freq=1：
每个 rollout epoch 开始，RoboTwin 都会关闭旧 task、重新 setup_demo()。
设置为 1 只是要求每次关闭时额外执行 sapien.render.clear_cache()。
32×8 和 16×16 都是每 step 256 条轨迹，因此每 step 都大约有 256 次 task 关闭/重建。
env16 主要降低的是“同时常驻的 scene/renderer/camera 数量”，并没有减少每 step 的总仿真工作量。
env.train.enable_offload=true：
每个全局 step 结束后，再把整个 VectorEnv 环境列表清空。
下一 step 会重新创建 SubEnv，并额外经历一次验证用的 setup_task()。
所以它确实额外增加环境初始化，但主要目的，是训练 actor 时暂时不让环境资源常驻。RLinf #1326

实际时间与内存表现
同样都是每 step 256 条轨迹：
配置平均整步时间结果
当前 32×8 / offload=true约 1509.6 秒step 30 内存失败
历史 16×16 / offload=false约 1483.9 秒至少运行到 step 57
```

所以16\*16，`env.train.enable_offload`关掉好点？



### grpo



smoke配置

/root/autodl\-tmp/RLinf/examples/embodiment/config/robotwin\_adjust\_bottle\_grpo\_openpi\_a800\_2gpu\_smoke\.yaml

```Plain Text
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/pi0@actor.model
  - hybrid_engines/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout
  - _self_

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor, env, rollout: 0-1

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_grpo_openpi_2gpu_env16_rollout16_g8_smoke"
    logger_backends: ["tensorboard"] # wandb, swanlab

  max_epochs: 1000
  max_steps: 1 #-1

  only_eval: False
  val_check_interval: -1 #10
  save_interval: 1 #10

  resume_dir: null # Optional: path to a saved checkpoint directory, such as 'checkpoints/global_step_10'. If not None, it will be used to resume training.
  ckpt_path: null  # Optional: path to a .pt checkpoint. If not None, it will be loaded after the model is instantiated (for evaluation).

algorithm:
  normalize_advantages: True
  kl_penalty: kl  # how to estimate kl divergence: kl or kl_penalty
  group_size: 8 #1
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level

  update_epoch: 2
  adv_type: grpo #gae
  loss_type: actor #actor_critic
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

  filter_rewards: True #False
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9
env:
  group_name: "EnvGroup"
  enable_offload: False #True
  # Override the default values in env/robotwin_adjust_bottle
  train:
    enable_offload: True
    rollout_epoch: 16 #4
    total_num_envs: 16 #256
    reward_coef: ${algorithm.reward_coef}
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
    group_size: ${algorithm.group_size}
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
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
    video_cfg:
      save_video: False #True
  eval:
    rollout_epoch: 16 #1
    total_num_envs: 4 #128
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 200
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 200
    group_size: 1
    use_fixed_reset_state_ids: False #True
    is_eval: True
    assets_path: "/root/autodl-tmp/RoboTwin_RLinf"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    video_cfg:
      save_video: False #True
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
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 32
  global_batch_size: 512 #2048 # 1024
  seed: 1234
  enable_offload: false

  # Override the default values in model/openpi
  model:
    model_path: "/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
    num_action_chunks: 50 # interface for the env
    add_value_head: False #True
    action_dim: 14
    openpi:
      config_name: "pi0_aloha_robotwin"
      num_images_in_input: 3
      detach_critic_input: False #True

  optim:
    lr: 5.6e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  # Override the default values in hybrid_engines/fsdp
  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False # for openpi, gradient checkpointing is not supported, please do not change this value
    save_full_model_weights: False
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
    snapshot_device: cpu
    transport_device: cpu
    init_sync:
      enabled: true

```



`env.train.enable_offload`

```Plain Text
速度代价主要不在 gc.collect()，而在下一 step 需要重新：
创建 SubEnv；
初始化 SAPIEN/PhysX；
加载任务、机器人、相机和 renderer；
建立场景资源。

但它只在每个 global step 末尾执行一次，不是每个 rollout epoch 都销毁完整 VectorEnv。
```

还是关掉吧

`clear_cache_freq`

```Plain Text
环境 reset/close 时，每隔多少次显式清理 SAPIEN renderer cache。

1：每次 reset 都清 cache；
8：每 8 次 reset 清一次；
数字越大：可能稍快，但 renderer/native 缓存保留得更多；
数字越小：更积极清理，但会增加缓存重建工作。

RLinf 的官方 RoboTwin adjust_bottle 配置 明确设置为 1；底层 RoboTwin 在没有显式配置时的 fallback 才是 8。

当前 resolved 已经得到：
task_config:
  clear_cache_freq: 1
所以不用额外修改
```

rollout数量

```Plain Text
官方 Pi0 + GRPO + LIBERO 配置 使用：
64 env × 8 rollout_epoch = 512 trajectories
512 ÷ 8 = 64 groups
所以你的采样量是官方 LIBERO 配置的一半
```



跑smoke

```Bash
source /root/autodl-tmp/rlinf_env.sh

OLD=/root/autodl-tmp/RLinf
cd "$OLD"
source "$OLD/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH="$OLD"
export EMBODIED_PATH="$OLD/examples/embodiment"
export PYTHONPATH="$OLD:$ROBOTWIN_PATH:${PYTHONPATH:-}"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset STEPS SAVE_INTER NODES

CONFIG_NAME=robotwin_adjust_bottle_grpo_openpi_a800_2gpu_smoke
EXP_NAME=robotwin_grpo_openpi_2gpu_env16_rollout16_g8_smoke

ray stop -f || true

pgrep -af \
'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' \
|| true

nvidia-smi

RUN_NAME="$(date +'%Y%m%d_%H%M%S')-${CONFIG_NAME}-env16-rollout16-g8-offload-on"
LOG_DIR="$OLD/logs/$RUN_NAME"
RUN_LOG="$LOG_DIR/run_embodiment.log"
MON_DIR="$LOG_DIR/resource_monitor"
MON_LOG="$MON_DIR/monitor.log"

mkdir -p "$LOG_DIR" "$MON_DIR"

"$OLD/.venv/bin/python" \
  "$OLD/examples/embodiment/train_embodied_agent.py" \
  --config-path "$OLD/examples/embodiment/config" \
  --config-name "$CONFIG_NAME" \
  --cfg job --resolve \
  > "$LOG_DIR/resolved-smoke.yaml" \
  2> "$LOG_DIR/resolve-stderr.log"

if [ "$?" -ne 0 ]; then
  echo "Hydra 配置解析失败"
  exit 1
fi

CMD=(
  "$OLD/.venv/bin/python"
  "$OLD/examples/embodiment/train_embodied_agent.py"
  --config-path "$OLD/examples/embodiment/config/"
  --config-name "$CONFIG_NAME"
  "runner.max_steps=1"
  "runner.save_interval=1"
  "runner.resume_dir=null"
  "runner.logger.log_path=$LOG_DIR"
  "runner.logger.experiment_name=$EXP_NAME"
)

printf '%q ' "${CMD[@]}" > "$LOG_DIR/command.txt"
echo >> "$LOG_DIR/command.txt"

nohup "${CMD[@]}" > "$RUN_LOG" 2>&1 &
TRAIN_PID=$!

nohup "$OLD/.venv/bin/python" \
  "$OLD/local_scripts/monitor_pi0_resources_2gpu.py" \
  --pid "$TRAIN_PID" \
  --out-dir "$MON_DIR" \
  --interval 2 \
  > "$MON_LOG" 2>&1 &
MONITOR_PID=$!

printf '%s\n' "$TRAIN_PID" > "$LOG_DIR/train.pid"
printf '%s\n' "$MONITOR_PID" > "$LOG_DIR/monitor.pid"

echo "TRAIN_PID=$TRAIN_PID"
echo "MONITOR_PID=$MONITOR_PID"
echo "LOG_DIR=$LOG_DIR"
echo "RUN_LOG=$RUN_LOG"
echo "RESOURCE_CSV=$MON_DIR/resources.csv"
echo "PEAK_FILE=$MON_DIR/peak.txt"
```



smoke通

```Plain Text
完成 Global Step 1/1
总耗时：1452.7s，约 24分12秒
256 条轨迹，成功 216 条
success_once = return = 0.84375

注意你的文件现在是：
env:
  enable_offload: False
  train:
    enable_offload: True
实际 smoke 走的是嵌套的 env.train.enable_offload=true，所以运行名中的 offload-on 是正确的。为了避免以后误解，可以把顶层也写成 True；当前训练行为主要由嵌套项控制。

Checkpoint 产物
实际 DCP 路径多了一层实验名：
/root/autodl-tmp/RLinf/logs/20260715_113256-robotwin_adjust_bottle_grpo_openpi_a800_2gpu_smoke-env16-rollout16-g8-offload-on/
└─ robotwin_grpo_openpi_2gpu_env16_rollout16_g8_smoke/
   └─ checkpoints/global_step_1/actor/dcp_checkpoint/
其中：
两个 .distcp 分片各约 5.196 GB
.metadata 约 1 MB
总计约 9.68 GiB
没有 full_weights.pt
```



正式训练

```Bash
source /root/autodl-tmp/rlinf_env.sh

OLD=/root/autodl-tmp/RLinf
cd "$OLD"
source "$OLD/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
export REPO_PATH="$OLD"
export EMBODIED_PATH="$OLD/examples/embodiment"
export PYTHONPATH="$OLD:$ROBOTWIN_PATH:${PYTHONPATH:-}"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset STEPS SAVE_INTER NODES

CONFIG_NAME=robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline
EXP_NAME=robotwin_grpo_openpi_2gpu_env16_rollout16_g8_baseline
TARGET_STEP=100
SAVE_INTERVAL=10

ray stop -f || true

pgrep -af \
'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' \
|| true

nvidia-smi

RUN_NAME="$(date +'%Y%m%d_%H%M%S')-${CONFIG_NAME}-env16-rollout16-g8-step0-to-${TARGET_STEP}"
LOG_DIR="$OLD/logs/$RUN_NAME"
RUN_LOG="$LOG_DIR/run_embodiment.log"
MON_DIR="$LOG_DIR/resource_monitor"
MON_LOG="$MON_DIR/monitor.log"

mkdir -p "$LOG_DIR" "$MON_DIR"

RESOLVE_CMD=(
  "$OLD/.venv/bin/python"
  "$OLD/examples/embodiment/train_embodied_agent.py"
  --config-path "$OLD/examples/embodiment/config/"
  --config-name "$CONFIG_NAME"
  "runner.max_steps=$TARGET_STEP"
  "runner.save_interval=$SAVE_INTERVAL"
  "runner.resume_dir=null"
  "runner.logger.log_path=$LOG_DIR"
  "runner.logger.experiment_name=$EXP_NAME"
  --cfg job
  --resolve
)

"${RESOLVE_CMD[@]}" \
  > "$LOG_DIR/resolved-config.yaml" \
  2> "$LOG_DIR/resolve-stderr.log"

if [ "$?" -ne 0 ]; then
  echo "Hydra 配置解析失败，训练没有启动。"
  tail -n 100 "$LOG_DIR/resolve-stderr.log"
  exit 1
fi

echo "配置解析成功，关键参数："
grep -nE \
'max_steps:|save_interval:|adv_type:|loss_type:|group_size:|total_num_envs:|rollout_epoch:|enable_offload:|add_value_head:|global_batch_size:' \
"$LOG_DIR/resolved-config.yaml"

CMD=(
  "$OLD/.venv/bin/python"
  "$OLD/examples/embodiment/train_embodied_agent.py"
  --config-path "$OLD/examples/embodiment/config/"
  --config-name "$CONFIG_NAME"
  "runner.max_steps=$TARGET_STEP"
  "runner.save_interval=$SAVE_INTERVAL"
  "runner.resume_dir=null"
  "runner.logger.log_path=$LOG_DIR"
  "runner.logger.experiment_name=$EXP_NAME"
)

printf '%q ' "${CMD[@]}" > "$LOG_DIR/command.txt"
echo >> "$LOG_DIR/command.txt"

nohup "${CMD[@]}" > "$RUN_LOG" 2>&1 &
TRAIN_PID=$!

nohup "$OLD/.venv/bin/python" \
  "$OLD/local_scripts/monitor_pi0_resources_2gpu.py" \
  --pid "$TRAIN_PID" \
  --out-dir "$MON_DIR" \
  --interval 2 \
  > "$MON_LOG" 2>&1 &
MONITOR_PID=$!

printf '%s\n' "$TRAIN_PID" > "$LOG_DIR/train.pid"
printf '%s\n' "$MONITOR_PID" > "$LOG_DIR/monitor.pid"

echo "TRAIN_PID=$TRAIN_PID"
echo "MONITOR_PID=$MONITOR_PID"
echo "LOG_DIR=$LOG_DIR"
echo "RUN_LOG=$RUN_LOG"
echo "RESOURCE_CSV=$MON_DIR/resources.csv"
echo "PEAK_FILE=$MON_DIR/peak.txt"
```



感觉正常

应该有提升，但明显没有ppo快

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=Y2IzOWRiMmU3NmFjZDg0OTMwMmQ1MTFhMWNlYTE4OGZfMWU0NmVlZGM4OWZhMGU5MTFjZDZiNzhhZGY2YmU3NTBfSUQ6NzY2MzI5NTYzMzA1ODY2MzY5Nl8xNzg1MTYwMjQwOjE3ODUyNDY2NDBfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MWYxMjAyY2VmYjFiYTgwMmQ2MjkwMGQxYjkyOGM5ZjJfODcwZTJhYmNhMDBmOWRjNzUzNGQ1ZjQxOGFmN2Q2ODRfSUQ6NzY2MzI5NTU5MzUzOTcwMjAzMF8xNzg1MTYwMjQwOjE3ODUyNDY2NDBfVjM)



`env.train.enable_offload`作用一般，应该要关？

训练过程中，只是step间内存占用凹下去一块

但grpo跑完了，也许有用？先留着



产物：

\[metrics\.log\]

\[peak\.txt\]

\[resources\.csv\]

\[run\_embodiment\.log\]



路径：

/root/autodl\-tmp/RLinf/logs/20260715\_132507\-robotwin\_adjust\_bottle\_grpo\_openpi\_a800\_2gpu\_baseline\-env16\-rollout16\-g8\-step0\-to\-100



前一次ppo的路径：

/root/autodl\-tmp/RLinf/logs/20260714\_181545\-robotwin\_adjust\_bottle\_ppo\_openpi\_a800\_2gpu\_baseline\-env32\-rollout8\-step0\-to\-100



### wam







### Todo



远程







框架更新？

```Plain Text
框架更新：马上处理
**RLinf v0.3 已正式发布**：Release｜SFT 梯度修复 PR #1381｜RoboTwin π0 PPO 配置｜VLM Reward 文档
v0.3 tag 已包含昨天发现的 enable_sft_co_train 静默丢失 SFT gradient 修复。
增加 π0/π0.5 RoboTwin 配置、Reward/Value Model worker、Qwen3-VL 历史窗口 reward、Async PPO 等。
发布曲线没有给出公开的 RoboTwin+π0 PPO 结果，因此仍需自己复现。
**建议停止使用 v0.2，直接固定 v0.3 commit 和完整环境。**
LeRobot 还新增了按 grasp/place/open/close/pour/insert 等“原子世界状态变化”切分轨迹的 VLM 子任务标注更新，可以用来生成 chunk boundary，再与 RoboTwin exact predicates 对齐。
这段时间未核实到同等重要的 openpi、RoboTwin、GR00T、Isaac Lab、RSS 或高质量媒体新增；没有用旧新闻填数。
```



改进

附加更多的指标到表里，或者产物目录下

？group size提高？



简单看看，有用？rlinf内置自动实验和对比

```Plain Text
自动实验与曲线比较
最新版确实新增了 tests/parity_tests/，可以：
顺序批量启动任务；
检测达到目标 step、提前崩溃或需要重跑；
提取 success_once；
画曲线、生成 CSV；
使用 Pearson、Spearman、MAE、MSE、cosine、DTW 等方法和 baseline 比较。
稳定后可只分析日志：
python tests/parity_tests/analyze_logs.py logs/ \
  --output-dir logs/analysis_results \
  --baseline-dir logs_baseline \
  --step 60 \
  --similarity-method all
```



chat

https://chatgpt\.com/g/g\-p\-6a030582c5c881918fe17244959a4c0e\-wam\-rl/c/6a52f8af\-9aa4\-83ea\-8048\-c33d7ee314ed

本地codex



### Ask





### Cache



新窗口

```Plain Text
最稳妥的不是只说“记住这些”，而是让我维护项目内的三层交接：
AGENTS.md：长期规则；同目录的新任务会自动读取。
PROJECT_CONTEXT.md：稳定路径、环境、关键决策、常用命令。
HANDOFF.md：本轮进度、结果、待办、风险和下一步；每次结束时更新。
关闭前直接发送：
请执行项目交接：
1. 更新 HANDOFF.md，记录当前状态、已完成、待办、风险、下一步及需现场复核的信息；
2. 仅在长期规则发生变化时更新 PROJECT_CONTEXT.md；
3. 检查 AGENTS.md 是否要求新任务先读取上述两个文件；
4. 不复制代码和大日志，不保存密码、Token 或私钥；
5. 最后给出下一任务可直接使用的首句。
新窗口第一句：
请按 AGENTS.md 启动，先读取 PROJECT_CONTEXT.md 和 HANDOFF.md，复述你理解的项目状态；然后只读刷新动态信息，未经我同意不要修改服务器或实验。
对当前 AutoDL 项目，可继续更新已有的[交接文件](C:/Users/86136/Documents/Codex/2026-07-13/n/autodl-rlinf-fastwam-handoff-20260716.md)，不必另建重复文档。
GPT/Codex App 的机制简述：
“项目”把相关任务和本地目录组织在一起，但新任务不会自动继承旧任务的完整聊天记录。Projects, chats, and tasks
AGENTS.md 会在新任务开始工作前自动加载，适合保存必须执行的项目规则。AGENTS.md
Memories 是可选的辅助回忆层，可在“设置 → 个性化”或 /memories 控制；它在后台生成，可能延迟或跳过，因此不能作为精确实验状态的唯一来源。Memories
旧任务建议“归档但不删除”；精确训练 step、GPU 状态和日志仍应以下次只读现场刷新为准。
```



密集奖励，如果需要

```Plain Text
5. 更细粒度的规则奖励可以实现
RoboTwin 的 RLinf_support 分支确实已经有规则奖励树，包括：
SerialTask、ParallelTask
Pick、Place、Contact、Endpose、Success
距离、夹爪开合、接触、物体目标位姿、阶段完成度等
adjust_bottle 已经定义了 Pick、Place、Success 三段奖励。RoboTwin RLinf_support、reward.py
但是当前在线训练链路仍然基本是稀疏奖励：
RoboTwin SubEnv 返回 sparse success
→ RLinf use_custom_reward 用 termination 再生成一次稀疏奖励
→ OpenPI chunk mapper 只在终止位置写 reward
所以：
只改 use_custom_reward=false 不够；
只调用 gen_dense_reward_once() 也不对；
必须同时接通 RoboTwin 在线 step 和 RLinf chunk reward 两层。
建议做成独立分支和显式开关：
reward_mode: sparse   # sparse | dense_abs | dense_delta
更推荐 potential shaping：
r_shape = α × (γΦ(s_next) - Φ(s))
r_total = r_shape + β × sparse_success
这样比每一步重复发放“当前进度”更不容易停在半完成状态刷分。还需要特别处理：
SerialTask.current_idx 只前进不后退，物体掉落后可能仍保留阶段奖励；
GRPO 当前按组过滤全成功/全失败组，dense reward 启用后要重新定义 [0.1,0.9] 阈值或关闭过滤；
日志分别记录 reach、grasp、place、progress delta 和最终 success。
```



pi0 ppo经验

```Plain Text
已验证可用规模是每 step 16 env × 16 rollout_epoch=256 条轨迹。
rollout 只保留当前 global step，更新后清空；不存在跨 step 越积越大的 replay pool。
rollout 多与 critic EV 变好有强相关，但还不是严格阈值结论；EV 单步变负也可能是 microbatch return 方差过小。
前约 24–26 分钟/step，主要耗时仍是 RoboTwin rollout。
CPU 内存上涨主因是 EnvWorker 内的 SAPIEN/PhysX/renderer/native allocator，不是 rollout 数据池。
actor.enable_offload=false、rollout.enable_offload=false 能利用宽裕显存减少约29 GiB主机峰值。
env.train.enable_offload=true 没有解决 native RSS 增长，反而增加重建时间。
```

