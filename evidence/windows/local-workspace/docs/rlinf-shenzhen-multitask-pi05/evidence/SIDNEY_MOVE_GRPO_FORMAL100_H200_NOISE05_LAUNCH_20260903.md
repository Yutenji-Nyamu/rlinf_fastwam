# Sidney multi-task pi0.5 GRPO formal100 launch

更新时间：2026-09-03 16:59 CST

## 结论

按用户明确口径，已在物理 GPU 4/5 fresh 启动 `move_stapler_pad` 两卡 GRPO。进程已完成模型与
RoboTwin worker 初始化并进入第 1 步 rollout；两卡各约 16 GiB，未见 fatal/OOM。

## 锁定口径

- source：`codex/sz-sidney-pi05-current-rlinf@f50e235c5ab1`
- run：`move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1`
- GPU：4/5；outer steps：100；fresh run
- train/eval env：64/32；rollout epoch：4；共 256 trajectories、32 个 G8 group/step
- H50/C50/M10；Sidney checkpoint、norm 与 absolute 14D 合同不变
- train/eval 的 `max_episode_steps`、`max_steps_per_rollout_epoch`、`task_config.step_lim` 均为 200
- 最多 4 query records/trajectory，即 1024 records/step
- GB1024/MB32/update2：每步 2 次 optimizer call、每条 record 呈现 2 次
- `noise_level=0.5`；DVAC off；fixed32/eval5/save10；local-shard checkpoint

相对已通过的 Sidney smoke，科学行为变化仅为用户明确选择的 `horizon 400 -> 200`、
`noise 0.3 -> 0.5`，以及 formal 所需的 rollout/评估/步数预算扩展；模型身份与训练实现未变。

## 路径与启动现场

- run：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1`
- packet：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1`
- wrapper PID：`2660157`
- 启动确认：wrapper alive、exit pending、两个 rollout worker/EnvWorker/FSDP actor 均已创建，
  Sidney norm stats 已在两 rank 加载，日志进入 `Generating Rollout Epochs: 0/4`，fatal scan 为空。

单独的 SFT 训练或 fixed-seed SFT Control 不是技术前置，本轮按用户选择直接 formal；原始 Sidney
checkpoint 就是 SFT 起点。200-action held-out SFT 基线若后续需要，仍可用未更新 checkpoint 补测。
