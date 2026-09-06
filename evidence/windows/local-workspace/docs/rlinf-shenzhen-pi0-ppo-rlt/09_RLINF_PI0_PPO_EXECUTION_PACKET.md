# 深圳 latest RLinf × RoboTwin × π0 PPO：fixed-8 / one-update 执行包（已被10号4卡包取代）

> 状态：2026-08-21，用户随后改为物理GPU 4–7和official-half并发；本8卡草案不再执行。当前唯一
> 启动包是 [`10_RLINF_PI0_PPO_4GPU_OFFICIAL_HALF_EXECUTION_PACKET.md`](10_RLINF_PI0_PPO_4GPU_OFFICIAL_HALF_EXECUTION_PACKET.md)。
> 本包原计划顺序执行三段：SFT fixed-8 →
> PPO 精确一次 optimizer step（含同进程 fixed-8）→ fresh-process reload fixed-8。任一段失败即停止，
> 不自动扩大到 fixed-128、完整 PPO、pilot/formal、RLT、Git push 或系统修改。

## 1. 已闭合的前置条件

| 项目 | 锁定值 / 现场结果 |
|---|---|
| RLinf | `7d07a4212ee6858cc333e1d4fab7a37256d1f839`；2026-08-21 official `main`；PPO worktree clean |
| RoboTwin | `RLinf_support@0008ae6800df9f75fc8de7098bacb01735fd8fd2`；独立 compatibility tree |
| π0 SFT | `RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50dca1a5f75adc8d332046c4cf4fa7a3d0`；两 shard 与 norm stats 已校验并真实 load |
| tokenizer | `befaa248e4f82954b625a421658f933dfd1a97a0`；4,264,023 B；SHA256 `8986bb4f...168fc6` |
| runtime | Python 3.11.14；torch 2.11.0+cu129；Ray 2.57.0；OpenPI 0.1.1；8×H100 sm90 |
| model load | `OpenPi0ForRLActionPrediction`；3,502,061,329 parameters；`pi0_aloha_robotwin`；H/C=`50/50`；14D；3 cameras |
| server | 8×H100 空闲；`/home` 约 2.3 TiB、`/data` 约 3.2 TiB 可用；root 仍约 237 GiB 可用 |
| proxy | 100 GiB 中已用 52.962 GiB、剩余 47.038 GiB；到期 `2026-08-23 14:01:48 CST` |

环境和缓存都在 `/home`，source/assets/model/results 都在 `/data`。本阶段不再需要大模型下载，正常运行
网络增量接近 0；在 `/data` 为 DCP、full weights、视频和日志预留约 80 GiB，远低于当前余量。

两项 official installer 偏差已经按实际错误做了窄处理：

1. tokenizer helper 检查路径与实际缓存路径不一致且下载未 pin；已预置 immutable revision，并让 official
   helper skip 浮动下载；
2. installer 从浮动 CuRobo `main@8e734f3...` 安装，已与 `RLinf_support` 所需的 `curobo.types.*` API
   断裂；只用 `--no-deps` 重编并锁回最后一个兼容 tag `v0.7.8@d64c4b...`，torch/CUDA 未变化。

`uv pip check` 仍报告 installer 自身解析出的 6 项 metadata 范围冲突；H100 CUDA op、所有关键 compiled
imports、official Hydra compose 以及两片 π0 safetensors + norm stats 的真实加载均已通过，所以不在真实
故障出现前继续改依赖。

## 2. 共同路径与运行合同

```text
RLINF    = /data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
ROBOTWIN = /data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV     = /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL    = /data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
RESULTS  = /data/chenyiteng/results/rlinf-shenzhen/ppo
```

保持 official：`adjust_bottle`、ALOHA、三相机、canonical 14D、π0 H=50、每次执行 C=50、8-GPU
placement、FSDP/offload、PPO actor-critic/GAE/objective/LR/value head。唯一变化是明确缩小工程预算。

官方 `run_embodiment.sh` 只转发 `STEPS/SAVE_INTER/NODES`，不能承载这里全部 Hydra overrides。因此三段
都直接调用同一锁定仓库里的 official Python entrypoint，并复刻官方 launcher 的环境变量；没有复制或改写
runner/algorithm/model 源码。

## 3. R2：official π0 SFT fixed-8

### 3.1 Resolved diff

```yaml
runner.logger.log_path: /data/chenyiteng/results/rlinf-shenzhen/ppo/sft-fixed8-v1
env.eval:
  total_num_envs: 8
  rollout_epoch: 1
  max_episode_steps: 200
  max_steps_per_rollout_epoch: 200
  use_fixed_reset_state_ids: true
  assets_path: /data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
rollout.model.model_path: /data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
```

锁定 source 的 official seed partition（base seed 0、150 success seeds、8 workers、floor 18/worker）解析出的
首个 fixed-8 为：

```text
[100100052, 100100138, 100100025, 100100177,
 100100084, 100100172, 100100082, 100100089]
```

预算：8 episodes；最多 `8 × 200 / 50 = 32` 次 policy query；最多 1,600 primitive action slots；
0 training record、0 optimizer step、0 checkpoint。GPU 0–7；硬上限 60 分钟。

输出：`/data/chenyiteng/results/rlinf-shenzhen/ppo/sft-fixed8-v1`（compose 时确认不存在）。

精确脚本：
[`shenzhen_rlinf_r2_run_sft_fixed8_v1.sh`](../../local_scripts/remote_commands/shenzhen_rlinf_r2_run_sft_fixed8_v1.sh)，
SHA256 `04980104426360cd9f4ad0199e8b603b60ba7263ff94fd44c7dc3270b3e0a7c0`。脚本会先再次核 source、模型、
GPU 空闲和 output absence；保存当次 `resolved.yaml`、`driver.log`，并验证至少一个非空 MP4。

## 4. R3-A：official PPO 精确一次 optimizer step

### 4.1 Resolved diff

```yaml
runner:
  logger.log_path: /data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-v1
  max_epochs: 1
  max_steps: 1
  val_check_interval: 1
  save_interval: 1
algorithm.update_epoch: 1
env.train:
  total_num_envs: 8
  rollout_epoch: 1
  max_steps_per_rollout_epoch: 50
  assets_path: <ROBOTWIN>
env.eval:
  total_num_envs: 8
  rollout_epoch: 1
  max_episode_steps: 200
  max_steps_per_rollout_epoch: 200
  use_fixed_reset_state_ids: true
  assets_path: <ROBOTWIN>
actor:
  micro_batch_size: 1
  global_batch_size: 8
  model.model_path: <MODEL>
```

### 4.2 “一次 update”的精确定义

```text
train records      = 8 env × 1 rollout epoch × (50 / chunk 50) = 8
global batches     = 8 records / global batch 8                = 1
optimizer steps    = 1 global batch × update epoch 1           = 1
per actor rank     = 1 record / micro batch 1                   = 1 micro batch
```

训练交互上限为 8 trajectories、8 queries、400 primitive slots；随后同进程 fixed-8 上限为 8 episodes、
32 queries、1,600 slots。保持 official `gamma=.99`、`lambda=.95`、clip/value-clip `.2`、Huber 10、
actor LR `5.6e-6`、value LR `1.1e-4`、FSDP/patch sync/offload。GPU 0–7；整段硬上限 120 分钟。

输出：`/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-v1`（compose 时确认不存在）。成功 checkpoint：

```text
robotwin_ppo_openpi/checkpoints/global_step_1/actor/
├── dcp_checkpoint/.metadata
├── dcp_checkpoint/*.distcp
└── model_state_dict/full_weights.pt
```

精确脚本：
[`shenzhen_rlinf_r3_run_ppo_oneopt_v1.sh`](../../local_scripts/remote_commands/shenzhen_rlinf_r3_run_ppo_oneopt_v1.sh)，
SHA256 `c1dca941571486d7ffb3c57b1c463b6690449e7e690284fd6601de04d9c4be61`。脚本要求 checkpoints 下只有
`global_step_1`，并校验 DCP metadata 与 full weights 非空。

## 5. R3-B：fresh-process reload `global_step_1` fixed-8

PPO driver 自然结束后，以新的 SSH command、Python 和 Ray 生命周期启动；保持同一 SFT `model_path` 用来
构造 π0 和读取 norm stats，仅新增：

```yaml
runner.ckpt_path: /data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-v1/robotwin_ppo_openpi/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt
```

其余 resolved diff 与 R2 相同，因此复评的 8 个 seed 完全相同。成功率可以升、降或不变；本轮不设提升
阈值。预算为 8 episodes、最多 32 queries / 1,600 primitive slots；GPU 0–7；硬上限 60 分钟。

输出：`/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-reload-fixed8-v1`（compose 时确认不存在）。

精确脚本：
[`shenzhen_rlinf_r3_run_reload_fixed8_v1.sh`](../../local_scripts/remote_commands/shenzhen_rlinf_r3_run_reload_fixed8_v1.sh)，
SHA256 `a03394d9d111c3b0e5a6d4bcfa7dd04353b8c2f9d7afc8cc92cd23245b1ec330`。

## 6. 已保存的 dry-resolve 证据

服务器目录：
`/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821/resolved-packet-v1/`

| 文件 | SHA256 |
|---|---|
| `sft-fixed8.resolved.yaml` | `ff9ba684032934d79c7a36db7655cb7dc65b12df9267c9d7147e8c64979834c5` |
| `ppo-oneopt.resolved.yaml` | `4453a536d5a7d6f9acac265de7179c21e4d2251cec7d5b1b8509b85c8ad79ca4` |
| `ppo-reload-fixed8.resolved.yaml` | `420396d7655f73ab3c69c6e6ef02db086640a87628c366a185367c2d3ddeae80` |
| `budget-and-seeds.json` | `4eb7923bc19b26496bdd4a53387d279fcb18d7f1ab64c744f7c92c8a93c06787` |

三份 launch script 均已通过服务器 `bash -n`，但没有执行正文、没有创建三个 run 根。

## 7. 监控、停止和验收

监控保持简洁：Ray/worker stage、rollout/metric 进展、8 卡显存与利用率、主机 RAM、`/data`、driver
exit。出现 CUDA OOM、Ray worker fatal、Vulkan/SAPIEN fatal、NaN/Inf、checkpoint save error，或 workers
ready 后 20 分钟没有新的 rollout/metric/stage 进展时，只终止本轮 owned process；达到 60/120/60 分钟
硬上限同样停止。任一 gate 失败不进入下一段，保留当段日志，不覆盖或删除结果。

整体工程验收：

- 三段 exit 0，保存三份当次 `resolved.yaml` 与 `driver.log`；
- pre / inline post-update / fresh reload 三组有限的 `eval/success_once` 和视频，不设成功率门槛；
- PPO 只出现一次训练 metrics/update，loss 与 grad norm 有限；
- 只有 `global_step_1`，DCP 与 `full_weights.pt` 完整并记录大小/SHA256；
- fresh reload 明确读取上述 full weights，并复用同一 SFT/norm 和 fixed-8 seeds；
- wall time、GPU/RAM/磁盘峰值、PID/exit 与问题处理逐条写入服务器流水账。

总上限为 5,200 primitive slots、104 次 policy query、精确 1 次 optimizer step；串行使用 8×H100，
理论硬上限 32 GPU-hours。

## 8. 本次批准边界

批准后仅按 R2 → R3-A → R3-B 顺序各启动一次；允许在上述 fatal/no-progress/timeout 条件下停止本轮 owned
process。没有批准覆盖已有路径、删除已有数据、改系统/driver、配置 Git credential/push、fixed-128、完整
PPO、pilot/formal 或 RLT。若启动前任何 output 已存在，停止并重新展示新 run ID，不自行覆盖。
