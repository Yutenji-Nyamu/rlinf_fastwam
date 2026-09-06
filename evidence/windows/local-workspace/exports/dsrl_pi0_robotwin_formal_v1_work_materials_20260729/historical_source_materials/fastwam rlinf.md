# fastwam rlinf



### Guide



本机历史：

[Exp\_snd](https://my.feishu.cn/wiki/JAfbwIKFui0jR6kkbfScpJ0vn7f)

[Openpi \+ PPO AutoDL A800](https://my.feishu.cn/wiki/HkMUwQDcOikfygkTVm2cBsnEnVe)

[Motus \+ RLinf](https://my.feishu.cn/wiki/Azcdwee5EiVgdCkmCK3csFhVnBf)

[lawam rlinf](https://my.feishu.cn/wiki/O3nPwu6lgi5PMkk7KxNcy1rrn2f)

[pi0 \+ ppo/grpo](https://my.feishu.cn/wiki/Sg6hwU18HiHvavkfgq9cFS3BnVg)

[fastwam](https://my.feishu.cn/wiki/DXT6wnqIhi5193knEGccCEsFnTK)



### 移植



环境：gpt总结，不一定对

\[08\_JOINT\_ENV\_REPRODUCTION\.md\]

```Plain Text
FastWAM-RLinf 联合训练环境
路径：/root/autodl-tmp/conda/envs/FastWAM-RLinf
Python 3.11.15
Torch 2.7.1+cu128
Ray 2.55.1
RLinf 0.3.0
Fast-WAM 0.1.0
pip check 当前干净
Fast-WAM PPO/GRPO 用的是这一套，而不是 π0 .venv。
构建链是：
```

gpt没有用venv而是自己组了conda？



实现：

\[IMPLEMENTATION\_LOG\_20260717\.md\]

C:/Users/86136/Documents/rl/docs/fastwam\-robotwin\-rlinf\-grpo/evidence/IMPLEMENTATION\_LOG\_20260717\.md



### 跑



#### Smoke

Smoke

```Plain Text
cd /root/autodl-tmp/RLinf_fastwam_rlinf
bash examples/embodiment/run_fastwam_robotwin_grpo.sh smoke
```



报错

```Plain Text
原因：初始 actor→rollout 权重广播使用 GPU bucket，CUDA IPC 被当前 AutoDL 容器拒绝，报 pidfd_getfd: Operation not permitted。不是 checkpoint、Fast‑WAM 前向、Python 环境或 OOM；程序尚未进入 rollout/backward。
修复：smoke 和正式配置已从 bucket_device: cuda 改为 RLinf 原生 cpu
```



清

```Plain Text
kill -INT 520893 520894
sleep 5
pgrep -af 'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' || true
nvidia-smi
```



清

```Plain Text
/root/autodl-tmp/conda/envs/FastWAM-RLinf/bin/ray stop --force
sleep 3
nvidia-smi
```



Smoke

```Plain Text
cd /root/autodl-tmp/RLinf_fastwam_rlinf
bash examples/embodiment/run_fastwam_robotwin_grpo.sh smoke
```



数量关系错



清

```Plain Text
/root/autodl-tmp/conda/envs/FastWAM-RLinf/bin/ray stop --force
```



参数

```YAML
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/fastwam_robotwin@actor.model
  - model/fastwam_robotwin@rollout.model
  - hybrid_engines/fsdp@actor.fsdp_config
  - weight_syncer/bucket_syncer@weight_syncer
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
    experiment_name: "robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 1

  only_eval: false
  val_check_interval: -1
  save_interval: 1
  weight_sync_interval: 1

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: true
  kl_penalty: kl
  group_size: 4
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: chunk_level

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

  gamma: 0.99
  gae_lambda: 0.95

  # Engineering smoke must not silently filter all-success/all-failure groups.
  filter_rewards: false
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"

  train:
    rollout_epoch: 1
    total_num_envs: 8
    reward_coef: ${algorithm.reward_coef}
    group_size: ${algorithm.group_size}

    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192

    assets_path: /root/autodl-tmp/RoboTwin_RLinf
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: false
    enable_offload: true

    video_cfg:
      save_video: false

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
    rollout_epoch: 1
    total_num_envs: 2
    reward_coef: ${algorithm.reward_coef}
    group_size: 1

    auto_reset: true
    ignore_terminations: true
    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192

    use_fixed_reset_state_ids: false
    is_eval: true
    assets_path: /root/autodl-tmp/RoboTwin_RLinf
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: false
    enable_offload: false

    video_cfg:
      save_video: false

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

  # Retained from the known-good pi0 embodied configuration.
  recompute_logprobs: false
  collect_prev_infos: true

  enable_offload: false
  pipeline_stage_num: 1

  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"

  micro_batch_size: 2
  global_batch_size: 64
  seed: 1234
  enable_offload: false

  optim:
    # One-step smoke is a real optimizer update; ratio is still measured before it.
    lr: 1.0e-6
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp2"
    sharding_strategy: "full_shard"

    gradient_checkpointing: false
    cpu_offload: false
    offload_pin_memory: false
    reshard_after_forward: true

    enable_gradient_accumulation: true
    forward_prefetch: false
    limit_all_gathers: false
    backward_prefetch: null
    use_orig_params: false
    use_liger_kernel: false

    wrap_policy:
      no_split_names: ["__none__"]

    save_full_model_weights: false

    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

    # Preserve FP32 Flow-SDE chains/logprobs across the FSDP call boundary;
    # FastWAMPolicy explicitly casts only model inputs to BF16.
    cast_forward_inputs: false

    amp_autocast:
      enabled: false
      precision: bf16

    grad_scaler:
      enabled: false

weight_syncer:
  type: bucket
  bucket:
    bucket_size: 536870912
    bucket_dtype: null
    # AutoDL containers can block pidfd_getfd used by CUDA IPC. CPU staging is
    # an RLinf-native bucket mode and preserves the synchronized tensor values.
    bucket_device: cpu
    is_agent: false
    load_instant: true

reward:
  use_reward_model: false

critic:
  use_critic_model: false

```

wam大，降并发



跑

```Plain Text
bash examples/embodiment/run_fastwam_robotwin_grpo.sh smoke
```



oom;rollout\.enable\_offload: true

```Plain Text
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/fastwam_robotwin@actor.model
  - model/fastwam_robotwin@rollout.model
  - hybrid_engines/fsdp@actor.fsdp_config
  - weight_syncer/bucket_syncer@weight_syncer
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
    experiment_name: "robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 1

  only_eval: false
  val_check_interval: -1
  save_interval: 1
  weight_sync_interval: 1

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: true
  kl_penalty: kl
  group_size: 4
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: chunk_level

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

  gamma: 0.99
  gae_lambda: 0.95

  # Engineering smoke must not silently filter all-success/all-failure groups.
  filter_rewards: false
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"

  train:
    rollout_epoch: 1
    total_num_envs: 8
    reward_coef: ${algorithm.reward_coef}
    group_size: ${algorithm.group_size}

    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192

    assets_path: /root/autodl-tmp/RoboTwin_RLinf
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: false
    enable_offload: true

    video_cfg:
      save_video: false

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
    rollout_epoch: 1
    total_num_envs: 2
    reward_coef: ${algorithm.reward_coef}
    group_size: 1

    auto_reset: true
    ignore_terminations: true
    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192

    use_fixed_reset_state_ids: false
    is_eval: true
    assets_path: /root/autodl-tmp/RoboTwin_RLinf
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: false
    enable_offload: false

    video_cfg:
      save_video: false

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

  # Retained from the known-good pi0 embodied configuration.
  recompute_logprobs: false
  collect_prev_infos: true

  enable_offload: true
  pipeline_stage_num: 1

  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"

  micro_batch_size: 2
  global_batch_size: 64
  seed: 1234
  enable_offload: false

  optim:
    # One-step smoke is a real optimizer update; ratio is still measured before it.
    lr: 1.0e-6
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp2"
    sharding_strategy: "full_shard"

    gradient_checkpointing: false
    cpu_offload: false
    offload_pin_memory: false
    reshard_after_forward: true

    enable_gradient_accumulation: true
    forward_prefetch: false
    limit_all_gathers: false
    backward_prefetch: null
    use_orig_params: false
    use_liger_kernel: false

    wrap_policy:
      no_split_names: ["__none__"]

    save_full_model_weights: false

    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

    # Preserve FP32 Flow-SDE chains/logprobs across the FSDP call boundary;
    # FastWAMPolicy explicitly casts only model inputs to BF16.
    cast_forward_inputs: false

    amp_autocast:
      enabled: false
      precision: bf16

    grad_scaler:
      enabled: false

weight_syncer:
  type: bucket
  bucket:
    bucket_size: 536870912
    bucket_dtype: null
    # AutoDL containers can block pidfd_getfd used by CUDA IPC. CPU staging is
    # an RLinf-native bucket mode and preserves the synchronized tensor values.
    bucket_device: cpu
    is_agent: false
    load_instant: true

reward:
  use_reward_model: false

critic:
  use_critic_model: false




```



跑

```Plain Text
bash examples/embodiment/run_fastwam_robotwin_grpo.sh smoke
```

通，但内存有点满；



#### formal

正式配置

```Plain Text
defaults:
  - env/robotwin_adjust_bottle@env.train
  - env/robotwin_adjust_bottle@env.eval
  - model/fastwam_robotwin@actor.model
  - model/fastwam_robotwin@rollout.model
  - hybrid_engines/fsdp@actor.fsdp_config
  - weight_syncer/bucket_syncer@weight_syncer
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
    actor, rollout: 0-1
    env: 0

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_e4_r16_g4"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 100

  only_eval: false
  val_check_interval: -1
  save_interval: 10
  weight_sync_interval: 1

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: true
  kl_penalty: kl
  group_size: 4
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: chunk_level

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

  gamma: 0.99
  gae_lambda: 0.95

  # Engineering smoke must not silently filter all-success/all-failure groups.
  filter_rewards: true 
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"

  train:
    rollout_epoch: 16
    total_num_envs: 4
    reward_coef: ${algorithm.reward_coef}
    group_size: ${algorithm.group_size}

    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192

    assets_path: /root/autodl-tmp/RoboTwin_RLinf
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: false
    enable_offload: true

    video_cfg:
      save_video: false

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
    rollout_epoch: 1
    total_num_envs: 2
    reward_coef: ${algorithm.reward_coef}
    group_size: 1

    auto_reset: true
    ignore_terminations: true
    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192

    use_fixed_reset_state_ids: false
    is_eval: true
    assets_path: /root/autodl-tmp/RoboTwin_RLinf
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: false
    enable_offload: false

    video_cfg:
      save_video: false

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

  # Retained from the known-good pi0 embodied configuration.
  recompute_logprobs: false
  collect_prev_infos: true

  enable_offload: true
  pipeline_stage_num: 1

  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}
    rl:
      noise_level: ${actor.model.rl.noise_level}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"

  micro_batch_size: 2
  global_batch_size: 128
  seed: 1234
  enable_offload: false

  model:
    rl:
      noise_level: 0.3

  optim:
    # One-step smoke is a real optimizer update; ratio is still measured before it.
    lr: 5.0e-6
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp2"
    sharding_strategy: "full_shard"

    gradient_checkpointing: false
    cpu_offload: false
    offload_pin_memory: false
    reshard_after_forward: true

    enable_gradient_accumulation: true
    forward_prefetch: false
    limit_all_gathers: false
    backward_prefetch: null
    use_orig_params: false
    use_liger_kernel: false

    wrap_policy:
      no_split_names: ["__none__"]

    save_full_model_weights: false

    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

    # Preserve FP32 Flow-SDE chains/logprobs across the FSDP call boundary;
    # FastWAMPolicy explicitly casts only model inputs to BF16.
    cast_forward_inputs: false

    amp_autocast:
      enabled: false
      precision: bf16

    grad_scaler:
      enabled: false

weight_syncer:
  type: bucket
  bucket:
    bucket_size: 536870912
    bucket_dtype: null
    # AutoDL containers can block pidfd_getfd used by CUDA IPC. CPU staging is
    # an RLinf-native bucket mode and preserves the synchronized tensor values.
    bucket_device: cpu
    is_agent: false
    load_instant: true

reward:
  use_reward_model: false

critic:
  use_critic_model: false

```

并行4，串行16



正式跑

```Plain Text
cd /root/autodl-tmp/RLinf_fastwam_rlinf
bash examples/embodiment/run_fastwam_robotwin_grpo.sh train
```

这个并行度应该合适，峰值快满

成功率降；原始成功率太高？`adjust_bottle`论文中报告100；要找一半左右的论文

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NGVmZjE4NzYzY2E0NGJhYjc4NjRlOWU0OWVkZjM1ZTFfMWNlY2VhODA5NDRiOWQ2NjFjOTc1Y2NhMGEyNTNlZDBfSUQ6NzY2MzY2ODUxODY0NjMzNjcyMl8xNzg1MTYwMjAyOjE3ODUyNDY2MDJfVjM)



低成功率任务

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjdjNDYwYzE0NjQ0ZDgxYWE1N2Y2NDcxYjhjYmFjYjZfZDgxZmU4ZTRlNTlkYWZiMTg5ZWFhMDJlNTEzYmMyOTRfSUQ6NzY2MzY3ODQxMjA2NjA4MTc3NV8xNzg1MTYwMjAyOjE3ODUyNDY2MDJfVjM)



实现检查

【C:/Users/86136/Documents/rl/docs/fastwam\-robotwin\-rlinf\-grpo/05\_IMPLEMENTATION\_PLAN\.md】



打包

```Bash
cd /root/autodl-tmp/RLinf_fastwam_rlinf
STAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE="/root/autodl-tmp/RLinf_fastwam_rlinf-code-${STAMP}.zip"

git ls-files --cached --others --exclude-standard \
  | grep -Ev '(^|/)(logs|logs_baseline|results|outputs|checkpoints|evaluate_results|wandb|ray_results|__pycache__|\.pytest_cache|\.ruff_cache|assets|data|cache|conda|\.venv)(/|$)|\.(pt|pth|ckpt|safetensors|bin|onnx|npy|npz|mp4|avi|webm|so|pyc)$' \
  | zip -@ "$ARCHIVE"

unzip -t "$ARCHIVE" &&
du -h "$ARCHIVE" &&
sha256sum "$ARCHIVE"
```

\[RLinf\_fastwam\_rlinf\-code\-20260718\-095432\.zip\]



#### 换任务



配置

/root/autodl\-tmp/RLinf\_fastwam\_rlinf/examples/embodiment/config/robotwin\_move\_stapler\_pad\_grpo\_fastwam\_a800\_2gpu\.yaml

```YAML
defaults:
  - env/robotwin_move_stapler_pad@env.train
  - env/robotwin_move_stapler_pad@env.eval
  - model/fastwam_robotwin@actor.model
  - model/fastwam_robotwin@rollout.model
  - hybrid_engines/fsdp@actor.fsdp_config
  - weight_syncer/bucket_syncer@weight_syncer
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
    actor, rollout: 0-1
    env: 0

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu_e4_r16_g4"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 100

  only_eval: false
  val_check_interval: -1
  save_interval: 10
  weight_sync_interval: 1

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: true
  kl_penalty: kl
  group_size: 4
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: chunk_level

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

  gamma: 0.99
  gae_lambda: 0.95

  # Engineering smoke must not silently filter all-success/all-failure groups.
  filter_rewards: true 
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"

  train:
    rollout_epoch: 16
    total_num_envs: 4
    reward_coef: ${algorithm.reward_coef}
    group_size: ${algorithm.group_size}

    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192

    assets_path: /root/autodl-tmp/RoboTwin_RLinf
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: false
    enable_offload: true

    video_cfg:
      save_video: false

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
    rollout_epoch: 1
    total_num_envs: 2
    reward_coef: ${algorithm.reward_coef}
    group_size: 1

    auto_reset: true
    ignore_terminations: true
    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192

    use_fixed_reset_state_ids: false
    is_eval: true
    assets_path: /root/autodl-tmp/RoboTwin_RLinf
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: false
    enable_offload: false

    video_cfg:
      save_video: false

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

  # Retained from the known-good pi0 embodied configuration.
  recompute_logprobs: false
  collect_prev_infos: true

  enable_offload: true
  pipeline_stage_num: 1

  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}
    rl:
      noise_level: ${actor.model.rl.noise_level}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"

  micro_batch_size: 2
  global_batch_size: 128
  seed: 1234
  enable_offload: false

  model:
    rl:
      noise_level: 0.3

  optim:
    # One-step smoke is a real optimizer update; ratio is still measured before it.
    lr: 5.0e-6
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp2"
    sharding_strategy: "full_shard"

    gradient_checkpointing: false
    cpu_offload: false
    offload_pin_memory: false
    reshard_after_forward: true

    enable_gradient_accumulation: true
    forward_prefetch: false
    limit_all_gathers: false
    backward_prefetch: null
    use_orig_params: false
    use_liger_kernel: false

    wrap_policy:
      no_split_names: ["__none__"]

    save_full_model_weights: false

    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

    # Preserve FP32 Flow-SDE chains/logprobs across the FSDP call boundary;
    # FastWAMPolicy explicitly casts only model inputs to BF16.
    cast_forward_inputs: false

    amp_autocast:
      enabled: false
      precision: bf16

    grad_scaler:
      enabled: false

weight_syncer:
  type: bucket
  bucket:
    bucket_size: 536870912
    bucket_dtype: null
    # AutoDL containers can block pidfd_getfd used by CUDA IPC. CPU staging is
    # an RLinf-native bucket mode and preserves the synchronized tensor values.
    bucket_device: cpu
    is_agent: false
    load_instant: true

reward:
  use_reward_model: false

critic:
  use_critic_model: false

```



清

```Bash
pkill -TERM -f '/root/autodl-tmp/RLinf_fastwam_rlinf/examples/embodiment/[t]rain_embodied_agent.py' || true

/root/autodl-tmp/conda/envs/FastWAM-RLinf/bin/ray stop --force

sleep 3
pgrep -af '[t]rain_embodied_agent.py|[r]aylet|[g]cs_server' || true
nvidia-smi
```



跑

```Plain Text
cd /root/autodl-tmp/RLinf_fastwam_rlinf

bash examples/embodiment/run_fastwam_robotwin_grpo.sh \
  train \
  robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu
```



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NzYwYmE4MDJjOGZmNTdhYjMyODBlZDNiMzU1M2MwZWVfMzE1Nzg4MTUwYjYzNGVkY2JhOWM5ZWQzYjkzYjFhOTBfSUQ6NzY2NDA4NDQ4Mzg1OTE1NjE4M18xNzg1MTYwMjAyOjE3ODUyNDY2MDJfVjM)

step：77/100

可以算有效果，取高点60\+但一直震荡；目前的baseline不算理想



没效果？

效果不太明显？步数不够？每步rollout不够？

```Plain Text
当前：
val_check_interval=-1
每步训练 seed 会变化
没有固定 seed 的确定性评估
所以这条 success 曲线并不是同一批任务上的 checkpoint 学习曲线，不能依据最新几步下降就断言模型退化。

每步只有 64 条二值轨迹。在成功率约 45% 时，单步成功率的 95% 随机波动范围约为 ±12.2 个百分点

当前每步：
4 env × 16 rollout_epoch = 64 trajectories
group_size=4，共 16 个 GRPO group
192/24=8 个 action transition/trajectory
共约 512 个 actor transition
global_batch=128、update_epoch=1，每步只有约 4 次 optimizer update
相比之下：
成功的 π0 配置：256 trajectories、32 groups、update_epoch=2
社区 Fast-WAM：约 256 trajectories、32 groups、13,312 transitions/step、lr=2e-5

lr=5e-6、clip_grad=1、BF16 参数配合当前较少的 optimizer update，策略移动可能偏弱。
当前组内 std 归一化开启，而社区成功 Fast-WAM 使用过 no-std；这是有依据的算法差异，但不能仅凭现在的曲线认定它错误。
```

可能有问题的参数：`update_epoch`，rollout数，std 归一化？



或者，是已经跨过峰值开始下降？峰值在中间？

有点像参数不够好，20步左右到顶了



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ODk0NWQwZWIzZDRiNTI5MDBjZTIzNmUzM2E4Y2RkZDVfNDgxMzg4NjFhMzZlYjA2OGNmMDBmMTUyODAwOWE1ZTJfSUQ6NzY2NDAyODk2MDUyMjU4NzExN18xNzg1MTYwMjAyOjE3ODUyNDY2MDJfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MTA3NjBjMWI2NGJlNTk4ZWQxODcxZjczNDhmNTc2MWZfMmNiYTI1MzNlYzcxYWYyZTM5ZWExOGFiNmQzMTI4NWNfSUQ6NzY2NDA4NDk0MzQ5MTk0MzM3Nl8xNzg1MTYwMjAyOjE3ODUyNDY2MDJfVjM)

内存吃满，显存7成大概；并行度比较理想

目前的增加内存的参数：

```Plain Text
rollout.enable_offload=true：保留。它将 rollout 模型暂移 CPU，增加 RAM，但否则 actor 和 rollout 同卡常驻很可能超过80GB。
```



产物：

\[metrics\.log\]

\[peak\.txt\]

\[resources\.csv\]

\[run\_embodiment\.log\]





调参 pi0\_aligned\.yaml

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NjVkNzA5MjEwYjgzNDY3Yjc3MjNhMzU3ZjY0ZjY4NTFfZWEyNDA0MjljMjI0N2I0YWQ3NGM1NzQ1ODRiNDVlMWZfSUQ6NzY2NDA3MzczMzc0MDU5NjE1NF8xNzg1MTYwMjAyOjE3ODUyNDY2MDJfVjM)

还没跑



### ppo



实现

\[RLinf\_fastwam\_rlinf\-code\-20260719\-115732\.zip\]

git有点混乱，后面再理



实现文档

```Plain Text
[05_IMPLEMENTATION_PLAN.md (line 757)](C:/Users/86136/Documents/rl/docs/fastwam-robotwin-rlinf-grpo/05_IMPLEMENTATION_PLAN.md:757)：最重要的实现文档。
第25节：完整改动文件、调用链、来源与验收。
第27节：GRPO正式训练和迁移代码审查。
[第28节 (line 898)](C:/Users/86136/Documents/rl/docs/fastwam-robotwin-rlinf-grpo/05_IMPLEMENTATION_PLAN.md:898)：新GRPO配置与PPO逐文件实现。

[IMPLEMENTATION_LOG_20260717.md (line 308)](C:/Users/86136/Documents/rl/docs/fastwam-robotwin-rlinf-grpo/evidence/IMPLEMENTATION_LOG_20260717.md:308)：实际改了什么、同步了什么、测试结果和运行事实。

环境复现看 [08_JOINT_ENV_REPRODUCTION.md](C:/Users/86136/Documents/rl/docs/fastwam-robotwin-rlinf-grpo/08_JOINT_ENV_REPRODUCTION.md)。
```



参数

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OGFlMWQyMzViNjQ0NDc4Mzg3MGFkZjdmNGQxYjc3NjhfYjJkZTEyZTZjMGE5M2NiNzJmNmEwNmFmZDY5ZGE1MGJfSUQ6NzY2NDA4MDkzMjk0NDU2MzE2OF8xNzg1MTYwMjAyOjE3ODUyNDY2MDJfVjM)



中断训练

```Bash
RUN=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260718_100910-robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu

kill -INT "$(cat "$RUN/train.pid")" 2>/dev/null || true
sleep 3
/root/autodl-tmp/conda/envs/FastWAM-RLinf/bin/ray stop --force

pgrep -af 'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' || true
nvidia-smi
```



smoke参数

```Plain Text
defaults:
  - env/robotwin_move_stapler_pad@env.train
  - env/robotwin_move_stapler_pad@env.eval
  - model/fastwam_robotwin@actor.model
  - model/fastwam_robotwin@rollout.model
  - hybrid_engines/fsdp@actor.fsdp_config
  - weight_syncer/bucket_syncer@weight_syncer
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
    actor, rollout: 0-1
    env: 0

runner:
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke_e4_r1"
    logger_backends: ["tensorboard"]

  max_epochs: 1000
  max_steps: 1

  only_eval: false
  val_check_interval: -1
  save_interval: 1
  weight_sync_interval: 1

  # PPO starts from the released official Fast-WAM checkpoint inherited from
  # model/fastwam_robotwin, not from a GRPO or PPO DCP checkpoint.
  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: true
  kl_penalty: kl
  group_size: 1
  reward_coef: 1.0

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: chunk_level

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

  filter_rewards: false
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: "EnvGroup"

  train:
    # Keep the formal run's 4-way environment parallelism, but minimize the
    # sequential sampling depth: 4 x 1 x (192 / 24) = 32 actor transitions.
    # global32/update1 therefore performs exactly one real optimizer step.
    rollout_epoch: 1
    total_num_envs: 4
    reward_coef: ${algorithm.reward_coef}
    group_size: ${algorithm.group_size}

    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192

    assets_path: /root/autodl-tmp/RoboTwin_RLinf
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json
    center_crop: false
    enable_offload: true

    video_cfg:
      save_video: false

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
    rollout_epoch: 1
    total_num_envs: 2
    reward_coef: ${algorithm.reward_coef}
    group_size: 1

    auto_reset: true
    ignore_terminations: true
    max_episode_steps: 192
    max_steps_per_rollout_epoch: 192

    use_fixed_reset_state_ids: false
    is_eval: true
    assets_path: /root/autodl-tmp/RoboTwin_RLinf
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    center_crop: false
    enable_offload: false

    video_cfg:
      save_video: false

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

  recompute_logprobs: false
  collect_prev_infos: true

  enable_offload: true
  pipeline_stage_num: 1

  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}
    model_forward_batch_size: ${actor.model.model_forward_batch_size}
    add_value_head: ${actor.model.add_value_head}
    rl:
      noise_level: ${actor.model.rl.noise_level}
    value_head:
      feature_source: ${actor.model.value_head.feature_source}
      input_dim: ${actor.model.value_head.input_dim}
      hidden_sizes: ${actor.model.value_head.hidden_sizes}
      activation: ${actor.model.value_head.activation}
      bias_last: ${actor.model.value_head.bias_last}
      detach_input: ${actor.model.value_head.detach_input}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"

  micro_batch_size: 2
  global_batch_size: 32
  seed: 1234
  enable_offload: false

  model:
    add_value_head: true
    model_forward_batch_size: 2
    rl:
      noise_level: 0.3
    value_head:
      feature_source: video_cache_last_v_mean
      input_dim: 3072
      hidden_sizes: [1024, 512, 256]
      activation: relu
      bias_last: true
      detach_input: true

  optim:
    lr: 5.0e-6
    value_lr: 1.1e-4
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp2"
    sharding_strategy: "full_shard"

    gradient_checkpointing: false
    cpu_offload: false
    offload_pin_memory: false
    reshard_after_forward: true

    enable_gradient_accumulation: true
    forward_prefetch: false
    limit_all_gathers: false
    backward_prefetch: null
    use_orig_params: false
    use_liger_kernel: false

    wrap_policy:
      no_split_names: ["__none__"]

    save_full_model_weights: false

    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

    # Preserve FP32 Flow-SDE chains/logprobs across the FSDP call boundary;
    # FastWAMPolicy casts only official model inputs to BF16.
    cast_forward_inputs: false

    amp_autocast:
      enabled: false
      precision: bf16

    grad_scaler:
      enabled: false

weight_syncer:
  type: bucket
  bucket:
    bucket_size: 536870912
    bucket_dtype: null
    bucket_device: cpu
    is_agent: false
    load_instant: true

reward:
  use_reward_model: false

critic:
  # PPO uses the policy's top-level value_head, not a separate critic worker.
  use_critic_model: false




```



Ppo smoke

```Markdown
cd /root/autodl-tmp/RLinf_fastwam_rlinf
bash examples/embodiment/run_fastwam_robotwin_ppo.sh smoke
```

通



Ppo formal

```Markdown
cd /root/autodl-tmp/RLinf_fastwam_rlinf
bash examples/embodiment/run_fastwam_robotwin_ppo.sh train
```

跑起来了



没用官方脚本，另外写了同级脚本

```Plain Text
2. Launcher 与RLinf原生脚本的关系
RLinf官方确实有同级脚本：
examples/embodiment/
├── run_embodiment.sh                 官方通用launcher
├── run_fastwam_robotwin_ppo.sh       Fast-WAM专用launcher
├── run_fastwam_robotwin_grpo.sh      Fast-WAM专用launcher
└── train_embodied_agent.py           三者共用的官方Python入口
两条调用链本质相同：
shell launcher
→ train_embodied_agent.py
→ Hydra
→ Ray actor / rollout / env
→ EmbodiedRunner
我们的脚本不是套在官方脚本外面，而是与它并列，直接调用相同的 train_embodied_agent.py。
官方脚本：
使用当前 shell 的 python。
用户直接传 config name。
前台运行并通过 tee 写日志。
RoboTwin路径仍是通用占位方式。
Fast-WAM脚本只在其前后增加运维行为：
固定联合环境和Fast-WAM/ModelScope/RoboTwin路径。
映射 smoke/train 配置。
启动前检查并解析Hydra配置。
防止重复driver。
后台启动并准确记录训练PID。
同步启动资源监控。
保存命令、resolved config、日志、CSV和peak。
没有改动官方 run_embodiment.sh、train_embodied_agent.py，也没有通过脚本修改PPO/GRPO、FSDP、同步或DCP逻辑。选择直接调用共同Python入口，是为了避免再套一层shell后PID和资源监控跟错进程。



```





![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjFiMzVmOGNiMzhkYjc2NzZhOGZkYmI5MDA3MDEzOGFfOGQ4ODQxODI0MTRiYTQ1MDBmMmQxNjhhMzkxYTA3ZjhfSUQ6NzY2NDQ3MjE5MTE2MTE1ODYwMV8xNzg1MTYwMjAyOjE3ODUyNDY2MDJfVjM)

不work

```Plain Text
社区 Fast‑WAM 的公开正向实验是 critic-free GRPO。社区Fast‑WAM GRPO实验
它的 PPO YAML 自称“小型单GPU verification config”，只有2个env、global batch 2，没有正向学习曲线。社区PPO配置
社区实现还明确记录了 critic过浅、BF16精度和batch路径未完整验证等问题。社区Fast‑WAM适配说明
```



资源，比较理想；



其他

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MzUxNDM5MmYwNDMzN2U2YTliMDRhMzk2NjFlNjk5MWVfMzNjMjE2MWE5YWNhMzUxODIwNDBhNThhZWQ5ZGY0YmJfSUQ6NzY2NDQ3MjI0NzY2MjY5MzM0NF8xNzg1MTYwMjAyOjE3ODUyNDY2MDJfVjM)



### Todo



修fastwam ppo

增加输出？

```Bash
建议指标都可以输出到同一张表/SwanLab/TensorBoard：
Critic：全 rollout EV、样本数、return 方差；V/return 分布；Pearson 看线性相关，Spearman 看排序；成功/失败轨迹平均 V；feature batch std/effective rank。
Advantage：原始 TD residual 是“一步惊喜”；raw advantage 是多步累计训练信号；输出正负比例、p05/p50/p95，并按环境时间和 denoise k 分桶。
Actor：action expert/value head 各自 grad norm；实际 clip coefficient；update_norm/param_norm；参数、Adam state 和 checkpoint delta dtype/幅度。
Probability：首次更新前 old/new 误差；ratio 分位数；非负 k3 KL；按 k 的 clip fraction；advantage 符号与 logprob 变化是否一致。
Sync：更新前后 hash、rollout worker hash、固定观测 action 差异。
与“PPO是否真正成功”相关性从高到低是：
固定 seed 确定性 eval 成功率。
old/new parity、参数真实变化、同步 hash/action。
full-buffer EV、return相关性、GAE方向。
grad norm、clip fraction、KL、资源等诊断指标。

EV 表示 critic 能解释多少 return 的变化，但它不等于“绝对预测准确”。例如 \(V=R+10\) 时 EV 仍可能很高，因此还要同时看 bias、MAE/MSE 和校准。
当前 Fast-WAM EV 全 NaN 的原因是：
在 micro_batch_size=2 上计算；
[losses.py (line 371)](C:/Users/86136/Documents/rl/.rlinf-fastwam-worktree/rlinf/algorithms/losses.py:371) 使用默认无偏方差；
有效样本只剩 1 个或 return 方差为 0 时产生 NaN；
随后污染整步平均。
应该在完整 rollout buffer 上计算 EV，并同时输出样本数和 Var(return)。
value_clip_ratio 应表示“原始新 value 相对旧 value 的变化有多少超过 clip 边界”。当前代码却先 clamp，再检查是否超过边界，因此几乎必为 0，见 [losses.py (line 347)](C:/Users/86136/Documents/rl/.rlinf-fastwam-worktree/rlinf/algorithms/losses.py:347) 和 [line 360 (line 360)](C:/Users/86136/Documents/rl/.rlinf-fastwam-worktree/rlinf/algorithms/losses.py:360)。
这两个是确定的仪表盘实现缺陷，不会直接破坏当前 critic loss。

5. PPO 原理和你列出的链条
PPO 的核心很简单：
critic 的 \(V(s)\) 预测“从当前状态出发，未来累计回报有多少”。
advantage 表示这次动作结果比 critic 原先预期更好还是更差。
PPO 比较新旧策略对同一个样本的概率：
\[ r=\exp(\log\pi_{\text{new}}-\log\pi_{\text{old}}) \]
用裁剪目标限制一次更新不要走太远：
\[ L=-\min(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A) \]
在 Fast-WAM/π0 这种 flow policy 中，这里的“概率”主要是某个 denoise 随机转移 \(x_k\rightarrow x_{k+1}\) 的概率，不是离散机器人动作 token 的概率。
你给出的链条可以这样理解：
环节含义与探测方法
reward/done/mask 正确成功奖励和仿真 predicate 一致；terminal、time-limit truncation 分开；done 后样本不进入 loss。用手工小轨迹检查奖励和有效样本数。
return/bootstrap 正确真正 terminal 不 bootstrap；超时 truncation 加 \(\gamma V(s_{\text{final}})\)。用 2–3 步 toy trajectory 手算对照。
critic 预测未来回报\(V(s)\) 不是当前奖励，而是未来 return。看全 rollout EV、相关性、误差和成功/失败轨迹的 V 分离度。
GAE 方向和方差正确\(\delta_t=r_t+\gamma V_{t+1}-V_t\)，再按 \(\gamma\lambda\) 累积。正面惊喜应产生正 advantage。代码见 [advantages.py (line 68)](C:/Users/86136/Documents/rl/.rlinf-fastwam-worktree/rlinf/algorithms/advantages.py:68)。
old/new logprob 同一 transition必须使用同一个 \(x_k,x_{k+1},k,\text{noise},mask\)。第一次更新前 ratio 应约等于 1。历史静态探针达到过误差 0，但训练中没有逐步记录。
PPO ratio 梯度正确正 advantage 应提高该 transition 的 logprob，负 advantage 应降低。看符号一致率、ratio 分位数、clip fraction。
actor 参数有效变化有梯度不等于参数真的变化；BF16 舍入和全局裁剪都可能吞掉小更新。看 update_norm/param_norm 和 checkpoint delta。
新权重同步 rolloutactor 更新后的 hash 应等于 rollout worker 同步后的 hash；固定观测下 action 应随权重变化。
固定 seed eval 上升这是最后的因果验收。相同 seed、场景和无探索设置下比较 checkpoint 成功率及置信区间。当前所有 run 都缺这一项。
目前链条里，GAE公式、基础 replay 和静态 logprob parity 有较强代码证据；critic质量、actor各参数组的真实变化、同步后的完全一致性和固定 eval 尚未闭环。
```



实现不一致

```Plain Text
BF16 参数优化
这里确实没有和 π0 对齐：
Fast-WAM PPO/GRPO：模型参数路径为 BF16。
π0 PPO/GRPO：FP32 参数/optimizer，算子用 BF16 混合精度。
RLinf 自己也明确警告低精度参数可能影响 optimizer 收敛，见 [fsdp_model_manager.py (line 68)](C:/Users/86136/Documents/rl/.rlinf-fastwam-worktree/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py:68)。
这不是漏接代码，而是为了贴合官方 Fast-WAM BF16 和显存所做的配置选择；但从 π0 基线对齐角度看，它是一个尚未验证的偏离。尤其 actor LR 只有 5e-6，小更新可能被低精度量化削弱。现在尚未记录 Adam state 的真实 dtype 和 checkpoint 参数 delta，不能下最终结论。


```





ppo调参？

micro batch目前只有2。pi0有32。合理值应该在中间。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NmY2Yzk0OWU3YjlmMDY0NmQwMDA4YTQ1ZWU1OWI5ODdfMTM5MTQ2Y2I4ZjFhZDI3MGQxZTE2OTllOTdmMTUxMzZfSUQ6NzY2NDE4MzQyMzA1MzEwNjQ3NV8xNzg1MTYwMjAyOjE3ODUyNDY2MDJfVjM)

每步多200s 5%左右的时间。能优化但不致命。

critic效果如何查看？添加输出？补充指标？：

```Plain Text
最小、最合理的改法不是改 value head，而是补充观测指标：
在更新前利用已经存在的 prev_values 和 returns，记录：
old_value_explained_variance
value/return 的 mean、std
MAE、RMSE
Pearson correlation
修复当前 EV：跨完整 global batch 聚合 return 和 residual 的统计量，再跨两张GPU all_reduce；不要在 micro_batch_size=2 内单独计算
```



时间

robotwin角度优化？reset？

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ODQ1MDYxNDMwZDJmNWExNTY5NjQ1MzVkZjBmNTZlMjdfMmU2ZWNlYjZjOGI0MTNjNmI3MzdjMDk2MTQ3NjlhMDlfSUQ6NzY2NDE5NzA4MDI4NTY0NTgxMV8xNzg1MTYwMjAyOjE3ODUyNDY2MDJfVjM)

```Plain Text
RoboTwin 的 reset 和物理执行合计约占总时间的 84%，占 rollout 的约 92%

实际调用结构是：
一个 PPO step
└─ 顺序跑 32 个 rollout_epoch
   └─ 每轮只有 4 个并发 env
      ├─ reset 4 个场景（实际被锁串行）
      └─ 重复 8 次
         ├─ Fast-WAM 预测 24 个待执行动作
         ├─ RoboTwin TOPP 插值
         ├─ 大量内部 physics/render step
         └─ 拍摄 head + 双 wrist observation

为什么 RoboTwin reset 很慢
当前 4 个环境位于一个 Ray EnvWorker 中，使用 4 个线程，而不是 4 个进程。但 reset 持有同一个 global_lock，所以 4 个场景的销毁与初始化实际上串行执行

每个 episode reset 都会重新创建：
SAPIEN Engine、RT Renderer 和 Scene
Aloha 双臂 URDF、碰撞体和任务物体
相机、灯光、材质与渲染资源
MPLib planning world 和 TOPP
任务物体，然后重新做稳定性检查

稳定性检查本身固定执行 2000 + 500 次 scene.step()

当前每步有 4 × 32 = 128 次 episode reset，因此 reset 的 physics step 数量非常大，而且全部串行。实测每轮 4-env reset 约 25.1 秒。

为什么 action execution 很慢
表面看每条轨迹只有 192 步，但 RoboTwin 并不是简单执行 192 次 physics step。
每个 24-action chunk 会：
左右臂分别执行 TOPP 时间参数化。
每条轨迹最多下采样为 500 个内部控制点。
每个控制点都执行：
scene.step()
scene.update_render()
check_success()
循环结束后再次更新渲染。
get_obs() 又更新一次渲染，并让三台相机拍照、读回 RGB。

这里TOPP 是什么？

实测一批 4 env 执行一个 24-action chunk 约 5.45 秒；每轮有 8 个 chunk。因此“192 个动作”背后可能是数千个 physics/render tick。

Private Dirty 几乎等于 PSS，说明主要不是 Linux 文件缓存，而是 SAPIEN、MPLib、renderer 等 native heap：
PhysX scene、双臂 articulation、任务 mesh
RT renderer、加速结构、材质和纹理
planning world、碰撞几何、TOPP
相机与 render targets
native allocator 的高水位和碎片

两个明确的重复来源：
当前 planner_backend=mplib 时，每个 env 创建左右两个 MplibWrapperPlanner，又为 TOPP 创建左右两个 MplibPlanner，合计四套 planning world。[planner 创建 (line 259)](C:/Users/86136/Documents/rl/audits/20260719-robotwin-performance-analysis/source/robotwin_robot.py:259)
除 Fast-WAM 必须的 head+双腕三台 320×240 相机外，还无条件创建 observer 和两台 world camera，三台都是 640×480；当前 third_view=false、pointcloud=false，它们并未被使用。[额外相机 (line 219)](C:/Users/86136/Documents/rl/audits/20260719-robotwin-performance-analysis/source/robotwin_camera.py:219)

【qpos RL 模式下，不在每个内部控制点调用 _update_render()：
保留每次 scene.step() 和 success check
只在最终 get_obs() 前同步 renderer 并拍照
这是最可能显著缩短那 23.3 分钟 action execution 的地方。固定 seed/action 对比最终 qpos、物体 pose、reward、success 和 RGB 后即可判断是否透明等价。】这里，不_update_render，对任务执行有没有影响？对训练整体有没有影响？我简单理解，fastwam是预测一整个chunk然后开环执行，这中间模型是不需要图像的？这个能传参控制吗？还是要我们改代码传参控制？总之简要解释；

目前主要时间占用是reset和动作执行，对这两个，怎么方便有效的优化？深入看看代码？
```

4卡？实际并行多很多？实际上更省？



grpo调参，跑

```Plain Text
cd /root/autodl-tmp/RLinf_fastwam_rlinf

# 新GRPO配置
bash examples/embodiment/run_fastwam_robotwin_grpo.sh train \
  robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu_pi0_aligned
```



盯额度；感觉用量不太对？怎么检查？





### Ask











2

找优化角度

除了robotwin角度优化，你浏览所有的上下文，发散的寻找，还有哪些角度可能减少训练整体时间？

要简单有效；不影响训练结果，内存显存不爆，修改少，最好是调个参数，等等；



3

实现审查

整体再审查一下我给你的其他材料；这台机前面的所有日志，其他相关实现例如rlinf官方，我们的motus lawam并入rlinf，社区fastwam rlinf，等等；我记得你有文档记录这些上下文？你过一遍所有上下文，找找有没有关于效果问题的可能的线索；我主要担心你上下文窗口有限，实现或配置时遗漏了一些本该能利用的经验；

总之整体审查下实现，看看有没有什么坑





### Cache



ssh \-p 36406 root@connect\.bjb1\.seetacloud\.com

<REDACTED_SERVER_PASSWORD>



窗口如何用账号密码登录服务器

```Plain Text
正确行为：
收到服务器地址和密码后，直接使用 Paramiko。
密码注入当前进程，关闭 key/agent 探测。
先执行 hostname; pwd; id -u 等只读身份探针。
验证成功后继续用户授权范围内的工作。
只有 Paramiko 密码认证实际失败，才能报告登录阻塞；不得把 AskPass 或 SSH key 当作前置条件。

4. Paramiko 是什么
Paramiko 是 Python 实现的 SSH 客户端库。它没有绕过 SSH，只是直接在 Python 进程里完成密码认证、加密连接、远程命令和 SFTP 文件传输。
我这里的连接过程是：
从当前进程读取服务器密码。
连接 connect.bjb1.seetacloud.com:36406。
设置 look_for_keys=False、allow_agent=False，直接使用密码认证。
用 exec_command() 执行命令；小文件需要同步时使用 SFTP。
命令输出返回到当前任务。
```



磁盘情况统计 07191352

```Plain Text
1. 磁盘情况
数据盘 /root/autodl-tmp：
总容量：1.8 TiB
已用：913 GiB
可用：847 GiB
使用率：52%
inode：2%
主要占用：
目录大小主要内容
RLinf_fastwam_rlinf296.25 GiB几乎全是近期 Fast-WAM checkpoints
RLinf149.24 GiBπ0 logs 135.56G + venv 13.65G
RoboTwin111.84 GiBpolicy 95.35G + assets 15.52G
旧 RLinf_wamppo_backup...110.94 GiB多套旧环境和旧日志
models90.73 GiBMotus约60G、Fast-WAM约23G
conda47.99 GiB当前和历史环境
RLinf_old_20260618...30.90 GiB旧环境和旧run
cache27.54 GiBpip/uv等缓存
你指定保留的近期实验占用主要是：
Fast-WAM move_stapler_pad GRPO：188.50 GiB，7个 DCP。
Fast-WAM adjust_bottle GRPO：53.86 GiB。
Fast-WAM PPO smoke：26.95 GiB。
Fast-WAM GRPO smoke：26.93 GiB。
π0 GRPO：96.81 GiB。
π0 PPO：19.38 GiB。
这些目前都不建议动。
当前 PPO formal 每个 checkpoint 约26.95 GiB，每10步保存一次，100步最多新增约269.5 GiB；完训后预计仍剩约577 GiB，因此不用边训练边清盘。
未来推荐清理顺序：
训练结束后清理可重建的 pip/uv cache 和旧 Ray 临时文件，约30.8 GiB。
给旧快照保存代码、diff、配置、环境清单和一个代表性 checkpoint 后，处理：
旧 wamppo备份：110.94 GiB
RLinf_old：30.90 GiB
合计约141.8 GiB。
如果确定不再复现 Motus/LaWAM，可再审查其旧模型、结果和环境，约167.7 GiB。
理论上非近期内容可释放约340 GiB，但第二、三类应逐项确认后再删。
需要特别保留：
RoboTwin_RLinf/assets 15.52 GiB：这是当前训练依赖的独立副本。
Fast-WAM release及DiffSynth模型约23.13 GiB。
FastWAM-RLinf、FastWAM-official 环境。
π0 golden venv备份。



```



随机化

```Plain Text
RLinf当前基础env片段确实默认开启随机背景、杂物、桌高和光照：[基础env配置 (line 43)](C:/Users/86136/Documents/rl/.rlinf-fastwam-worktree/examples/embodiment/config/env/robotwin_adjust_bottle.yaml:43)。
但本次正式run又在顶层显式覆盖成了clean，所有主要随机化字段均关闭：[实际resolved配置 (line 36)](C:/Users/86136/Documents/rl/audits/20260718-0843-fastwam-current/resolved-config.yaml:36)。所以答案是：
RLinf示例fragment默认开。
当前训练实际没开。
可以完全通过YAML配置修改，不是代码硬编码。

```



git情况 07191416

```Plain Text
git操作文档：C:/Users/86136/Documents/rl/.rlinf-fastwam-worktree/docs/fastwam_rlinf_git_publication_20260719.md

Git上云已经完成，公开仓库是：
github.com/Yutenji-Nyamu/rlinf_fastwam
最终状态
可见性：Public
默认分支：main
本地开发分支：feat/fastwam-robotwin-grpo
官方远端：origin → RLinf/RLinf
个人远端：personal → Yutenji-Nyamu/rlinf_fastwam
工作区：干净，已跟踪 personal/main
主体commit：768e0243e4dafedea6c92b3f37b652c51efb5a2e
流水账commit：8138d6700e3838250c1139289ebfba43d48ff7de
```

