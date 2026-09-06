# π0.5 adv-DVAC [0.5,1.5] 正式启动合同

2026-09-06。用户已要求停止Control并在同两卡替换运行。新方法从相同Sidney SFT重新开始，目标200轮沿当前Control上限；不从Control Step160或其他训练权重开始。

## 已核对参数与资源

- GPU4/5：训练32环境/卡×两卡×串行4＝256条/轮；G8，C50，horizon200，M10，flow_sde noise0.5。
- 仅训练action expert；LR5e-6，micro32/global1024，U2，每轮2次Adam；原混合精度、FSDP local_shard和同步不变。
- 固定评估共32条（16/卡），每5轮；每10轮保存checkpoint。任务move_pillbottle_pad，原相机/OIDN/归一化不变。
- 方法：action_level logprob/ratio/clip；action_advantage乘权重；tail3，warmup1，历史5轮log-variance标准化、z_clip2，w=1+0.25z，范围[0.5,1.5]。沿旧fix的sum-actions再mean-queries；不是BC逐chunk均值1，不声称与chunk级Control只差一个乘法。
- 200轮总预算：51,200训练尝试，最多204,800 query，400次Adam、1,280固定评估episodes、20代checkpoint约537GiB；按Control近期速度约76—84小时，约152—168 GPU小时。
- 显存预计接近Control每卡75GiB量级，需实际启动核验；新进程初期RAM预估100—200GiB，长期环境RSS增长未定位。14:32 /data余约1.33TiB，当前预算足够，保留其他两项BC的后续保存余量。
- 正常停止于200轮，timeout345600秒；fatal/OOM/nonfinite/通信异常则停止本job并报告，不自动改参/循环重启。只核对本人精确namespace/GPU4/5，GPU6/7和共享Ray不变。
- CPU compose逐叶对照通过；14项现有DVAC单元回归通过。没有重跑完整GPU smoke，不把CPU测试称作新方法长程验证。

## 精确命令

```sh
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python /data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-grpo-dvac-adv/examples/embodiment/train_embodied_agent.py --config-path /data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-grpo-dvac-adv/examples/embodiment/config --config-name robotwin_move_stapler_pad_grpo_openpi_pi05_sidney 'cluster.component_placement={actor\, env\, rollout:"4,5"}' runner.logger.log_path=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-dvac-adv-w05to15-formal200-2gpu64x4-g8-b1024-u2-m10-noise05-h200-fixed32-phys45-20260906-v1 runner.logger.experiment_name=pi05_sidney_pillbottle_grpo_dvac_adv_w05to15_formal200_phys45_20260906 runner.only_eval=false runner.max_epochs=1000 runner.max_steps=200 runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null algorithm.group_size=8 algorithm.update_epoch=2 algorithm.adv_type=grpo algorithm.filter_rewards=true env.train.total_num_envs=64 env.train.rollout_epoch=4 env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200 env.train.assets_path=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support env.train.video_cfg.save_video=false env.train.task_config.save_path=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-dvac-adv-w05to15-formal200-2gpu64x4-g8-b1024-u2-m10-noise05-h200-fixed32-phys45-20260906-v1/robotwin_data/train env.train.task_config.task_name=move_pillbottle_pad env.train.task_config.step_lim=200 env.eval.total_num_envs=32 env.eval.rollout_epoch=1 env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200 env.eval.use_fixed_reset_state_ids=true env.eval.assets_path=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support env.eval.video_cfg.save_video=true env.eval.video_cfg.video_base_dir=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-dvac-adv-w05to15-formal200-2gpu64x4-g8-b1024-u2-m10-noise05-h200-fixed32-phys45-20260906-v1/video/eval env.eval.task_config.save_path=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-dvac-adv-w05to15-formal200-2gpu64x4-g8-b1024-u2-m10-noise05-h200-fixed32-phys45-20260906-v1/robotwin_data/eval env.eval.task_config.task_name=move_pillbottle_pad env.eval.task_config.step_lim=200 actor.global_batch_size=1024 actor.micro_batch_size=32 actor.model.model_path=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab actor.model.num_steps=10 actor.model.openpi.num_steps=10 actor.model.openpi.noise_level=0.5 ++actor.fsdp_config.checkpoint_format=local_shard +dvac_grpo=adv_half
```

输出目录：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-dvac-adv-w05to15-formal200-2gpu64x4-g8-b1024-u2-m10-noise05-h200-fixed32-phys45-20260906-v1`。

## 完整resolved配置（JSON与YAML等价）

```json
{
  "env": {
    "train": {
      "env_type": "robotwin",
      "total_num_envs": 64,
      "auto_reset": false,
      "ignore_terminations": false,
      "reward_coef": 1,
      "use_custom_reward": true,
      "use_rel_reward": true,
      "center_crop": false,
      "seed": 0,
      "group_size": 8,
      "use_fixed_reset_state_ids": false,
      "max_steps_per_rollout_epoch": 200,
      "max_episode_steps": 200,
      "is_eval": false,
      "assets_path": "/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support",
      "seeds_path": "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-grpo-dvac-adv/rlinf/envs/robotwin/seeds/train_seeds.json",
      "video_cfg": {
        "save_video": false,
        "info_on_video": true,
        "video_base_dir": "/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-dvac-adv-w05to15-formal200-2gpu64x4-g8-b1024-u2-m10-noise05-h200-fixed32-phys45-20260906-v1/video/train"
      },
      "enable_offload": false,
      "task_config": {
        "task_name": "move_pillbottle_pad",
        "step_lim": 200,
        "planner_backend": "mplib",
        "render_freq": 0,
        "episode_num": 100,
        "use_seed": false,
        "save_freq": 15,
        "embodiment": [
          "aloha-agilex"
        ],
        "language_num": 100,
        "domain_randomization": {
          "random_background": false,
          "cluttered_table": false,
          "clean_background_rate": 1,
          "random_head_camera_dis": 0,
          "random_table_height": 0,
          "random_light": false,
          "crazy_random_light_rate": 0
        },
        "camera": {
          "head_camera_type": "D435",
          "wrist_camera_type": "D435",
          "collect_head_camera": true,
          "collect_wrist_camera": true
        },
        "data_type": {
          "rgb": true,
          "third_view": false,
          "depth": false,
          "pointcloud": false,
          "observer": false,
          "endpose": false,
          "qpos": true,
          "mesh_segmentation": false,
          "actor_segmentation": false
        },
        "pcd_down_sample_num": 1024,
        "pcd_crop": true,
        "save_path": "/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-dvac-adv-w05to15-formal200-2gpu64x4-g8-b1024-u2-m10-noise05-h200-fixed32-phys45-20260906-v1/robotwin_data/train",
        "clear_cache_freq": 1,
        "collect_data": true,
        "eval_video_log": true
      },
      "rollout_epoch": 4
    },
    "eval": {
      "env_type": "robotwin",
      "total_num_envs": 32,
      "auto_reset": true,
      "ignore_terminations": true,
      "reward_coef": 1,
      "use_custom_reward": true,
      "use_rel_reward": true,
      "center_crop": false,
      "seed": 0,
      "group_size": 1,
      "use_fixed_reset_state_ids": true,
      "max_steps_per_rollout_epoch": 200,
      "max_episode_steps": 200,
      "is_eval": true,
      "assets_path": "/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support",
      "seeds_path": "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-grpo-dvac-adv/rlinf/envs/robotwin/seeds/eval_seeds.json",
      "video_cfg": {
        "save_video": true,
        "info_on_video": true,
        "video_base_dir": "/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-dvac-adv-w05to15-formal200-2gpu64x4-g8-b1024-u2-m10-noise05-h200-fixed32-phys45-20260906-v1/video/eval"
      },
      "enable_offload": false,
      "task_config": {
        "task_name": "move_pillbottle_pad",
        "step_lim": 200,
        "planner_backend": "mplib",
        "render_freq": 0,
        "episode_num": 100,
        "use_seed": false,
        "save_freq": 15,
        "embodiment": [
          "aloha-agilex"
        ],
        "language_num": 100,
        "domain_randomization": {
          "random_background": false,
          "cluttered_table": false,
          "clean_background_rate": 1,
          "random_head_camera_dis": 0,
          "random_table_height": 0,
          "random_light": false,
          "crazy_random_light_rate": 0
        },
        "camera": {
          "head_camera_type": "D435",
          "wrist_camera_type": "D435",
          "collect_head_camera": true,
          "collect_wrist_camera": true
        },
        "data_type": {
          "rgb": true,
          "third_view": false,
          "depth": false,
          "pointcloud": false,
          "observer": false,
          "endpose": false,
          "qpos": true,
          "mesh_segmentation": false,
          "actor_segmentation": false
        },
        "pcd_down_sample_num": 1024,
        "pcd_crop": true,
        "save_path": "/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-dvac-adv-w05to15-formal200-2gpu64x4-g8-b1024-u2-m10-noise05-h200-fixed32-phys45-20260906-v1/robotwin_data/eval",
        "clear_cache_freq": 1,
        "collect_data": true,
        "eval_video_log": true
      },
      "rollout_epoch": 1
    },
    "group_name": "EnvGroup",
    "enable_offload": true
  },
  "actor": {
    "model": {
      "model_type": "openpi",
      "model_path": "/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab",
      "precision": null,
      "num_action_chunks": 50,
      "action_dim": 14,
      "is_lora": false,
      "lora_rank": 32,
      "use_proprio": true,
      "num_steps": 10,
      "add_value_head": false,
      "openpi": {
        "config_name": "pi05_sidney_robotwin",
        "num_images_in_input": 3,
        "noise_level": 0.5,
        "action_chunk": 50,
        "num_steps": 10,
        "train_expert_only": true,
        "action_env_dim": 14,
        "noise_method": "flow_sde",
        "add_value_head": false,
        "value_after_vlm": false,
        "value_vlm_mode": "mean_token",
        "detach_critic_input": true,
        "use_dsrl": false,
        "dsrl_state_dim": 8,
        "dsrl_action_noise_dim": 32,
        "dsrl_num_q_heads": 10,
        "dsrl_agg_q": "mean",
        "dsrl_image_latent_dim": 64,
        "dsrl_state_latent_dim": 64,
        "dsrl_hidden_dims": [
          128,
          128,
          128
        ]
      }
    },
    "fsdp_config": {
      "strategy": "fsdp",
      "sharding_strategy": "full_shard",
      "gradient_checkpointing": false,
      "cpu_offload": false,
      "offload_pin_memory": false,
      "reshard_after_forward": true,
      "enable_gradient_accumulation": true,
      "forward_prefetch": false,
      "limit_all_gathers": false,
      "backward_prefetch": null,
      "use_orig_params": false,
      "use_liger_kernel": false,
      "mixed_precision": {
        "param_dtype": null,
        "reduce_dtype": null,
        "buffer_dtype": null
      },
      "amp_autocast": {
        "enabled": false,
        "precision": "bf16"
      },
      "grad_scaler": {
        "enabled": false
      },
      "checkpoint_format": "local_shard"
    },
    "group_name": "ActorGroup",
    "training_backend": "fsdp",
    "global_batch_size": 1024,
    "micro_batch_size": 32,
    "seed": 1234,
    "enable_offload": true,
    "optim": {
      "lr": 0.000005,
      "value_lr": 0.0001,
      "adam_beta1": 0.9,
      "adam_beta2": 0.95,
      "adam_eps": 1e-8,
      "weight_decay": 0.01,
      "clip_grad": 1,
      "critic_warmup_steps": 0
    }
  },
  "weight_syncer": {
    "type": "patch",
    "patch": {
      "snapshot_device": "cpu",
      "delta_encoding": true,
      "compression": "none",
      "init_sync": {
        "enabled": true,
        "prefixes": null,
        "bucket_size": 134217728
      }
    }
  },
  "cluster": {
    "num_nodes": 1,
    "component_placement": {
      "actor, env, rollout": "4,5"
    }
  },
  "runner": {
    "task_type": "embodied",
    "logger": {
      "log_path": "/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-dvac-adv-w05to15-formal200-2gpu64x4-g8-b1024-u2-m10-noise05-h200-fixed32-phys45-20260906-v1",
      "project_name": "rlinf",
      "experiment_name": "pi05_sidney_pillbottle_grpo_dvac_adv_w05to15_formal200_phys45_20260906",
      "logger_backends": [
        "tensorboard"
      ]
    },
    "max_epochs": 1000,
    "max_steps": 200,
    "only_eval": false,
    "val_check_interval": 5,
    "save_interval": 10,
    "resume_dir": null,
    "ckpt_path": null
  },
  "algorithm": {
    "normalize_advantages": true,
    "kl_penalty": "kl",
    "group_size": 8,
    "reward_coef": 1,
    "reward_type": "chunk_level",
    "logprob_type": "action_level",
    "entropy_type": "token_level",
    "update_epoch": 2,
    "adv_type": "grpo",
    "loss_type": "actor",
    "loss_agg_func": "token-mean",
    "kl_beta": 0,
    "entropy_bonus": 0,
    "clip_ratio_high": 0.2,
    "clip_ratio_low": 0.2,
    "clip_ratio_c": 3,
    "value_clip": 0.2,
    "huber_delta": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "filter_rewards": true,
    "rewards_lower_bound": 0.1,
    "rewards_upper_bound": 0.9,
    "dvac_gradient_weighting": {
      "mode": "apply",
      "application": "action_advantage",
      "selected_l": 3,
      "warmup_steps": 1,
      "window_steps": 5,
      "log_eps": 1e-12,
      "std_floor": 0.000001,
      "z_clip": 2,
      "strength": 0.5,
      "weight_min": 0.5,
      "weight_max": 1.5,
      "save_step_tensors": true,
      "output_dir": "/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-dvac-adv-w05to15-formal200-2gpu64x4-g8-b1024-u2-m10-noise05-h200-fixed32-phys45-20260906-v1/pi05_sidney_pillbottle_grpo_dvac_adv_w05to15_formal200_phys45_20260906/dvac_train"
    }
  },
  "rollout": {
    "group_name": "RolloutGroup",
    "backend": "huggingface",
    "recompute_logprobs": false,
    "enable_offload": true,
    "pipeline_stage_num": 1,
    "model": {
      "model_type": "openpi",
      "model_path": "/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab",
      "precision": null,
      "num_action_chunks": 50,
      "action_dim": 14,
      "is_lora": false,
      "lora_rank": 32,
      "use_proprio": true,
      "num_steps": 10,
      "add_value_head": false,
      "openpi": {
        "config_name": "pi05_sidney_robotwin",
        "num_images_in_input": 3,
        "noise_level": 0.5,
        "action_chunk": 50,
        "num_steps": 10,
        "train_expert_only": true,
        "action_env_dim": 14,
        "noise_method": "flow_sde",
        "add_value_head": false,
        "value_after_vlm": false,
        "value_vlm_mode": "mean_token",
        "detach_critic_input": true,
        "use_dsrl": false,
        "dsrl_state_dim": 8,
        "dsrl_action_noise_dim": 32,
        "dsrl_num_q_heads": 10,
        "dsrl_agg_q": "mean",
        "dsrl_image_latent_dim": 64,
        "dsrl_state_latent_dim": 64,
        "dsrl_hidden_dims": [
          128,
          128,
          128
        ]
      }
    }
  },
  "reward": {
    "use_reward_model": false
  },
  "critic": {
    "use_critic_model": false
  }
}
```

