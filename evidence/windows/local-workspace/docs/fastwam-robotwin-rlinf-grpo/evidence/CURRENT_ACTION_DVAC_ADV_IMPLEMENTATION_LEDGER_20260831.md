# Fast-WAM current RLinf Action-DVAC-Adv 实施账本（2026-08-31）

## 1. 授权与边界

- 用户授权：在已经跑通的 current RLinf Fast-WAM GRPO 上实现
  `Action-DVAC-Adv [0.5,1.5]`，做必要检查、真实 smoke、Git push。
- 不启动 formal；不干扰现有正式训练；smoke 使用现场空闲 GPU。
- Control 的采样、并行、优化和评估预算保持不变，方法配置只增加动作专家 DVAC endpoint、
  action-level advantage 与对应的 loss 聚合。

## 2. 基线与隔离

- 远端基线：`origin/codex/sz-fastwam-current-rlinf-grpo`，
  `7b2331c55d14397cfb4cb16181470ddc8afae44a`。
- 发现旧本地 worktree 同时存在历史 staged/unstaged 内容；未清理、未覆盖。
- 新建独立 worktree：`worktrees/fastwam-action-dvac-adv`。
- 新建分支：`codex/sz-fastwam-action-dvac-adv`。

## 3. 已完成的语义迁移

- 从已经跑通的 current pi0 Action-Adv Fix 线性移植四个通用提交：
  recent-5 全局统计与 sidecar、显式权重端点、action-level advantage、H 维求和恢复更新尺度。
- 排除了源分支中与 Fast-WAM 无关的 pi0 配置和 pi0 telemetry writer；保留 Fast-WAM
  current typed trajectory、FSDP2、DCP 和 RoboTwin 路线。
- Fast-WAM 模型侧只在 opt-in 时，从既有 M=10 velocity forward 记录
  `z = x - t v`，裁到实际执行 C=24，输出 `[B,10,24,14]`；无额外模型 forward。
- 新增薄方法配置，继承 plain Fast-WAM GRPO，仅覆盖方法名、独立输出目录、
  `logprob_type=action_level` 与 `[0.5,1.5]` DVAC Action-Adv 参数。

## 4. 静态审计中的窄修复

- Fast-WAM 是运行时注册的 model type，不是 `SupportedModel.FASTWAM` 类属性；actor 改为比较
  已注册 OpenPI 值与字符串 `fastwam`，避免初始化时 `AttributeError`。
- rollout 能力检查保留 OpenPI 既有 `use_dsrl/use_rlt/is_nft` 排斥逻辑；其他模型只通过显式
  `rlinf_dvac_endpoint_capable` 接入，避免回归已验证的 OpenPI 路径。
- 通用 `selected_l` 默认恢复为 pi0 的 3；Fast-WAM 薄配置显式锁定 L=5，避免配置遗漏时
  rollout 与 actor 使用不同 tensor key。

## 5. 静态检查、endpoint probe 与 Git

- 方法单元测试：`10 passed`。
- Hydra compose 与 resolved leaf 断言通过；第一版薄配置触发 Hydra 的明确限制：被 include 的
  非主配置不能声明 `hydra.searchpath`。未启动训练；修复为逐叶复制 Control YAML，仅保留
  输出命名、`logprob_type` 和 DVAC 方法块三类差异。
- GPU2 B=1 endpoint probe 通过：开关 telemetry 时 action 与 old log-prob 最大差均为 `0.0`；
  endpoint 为 `[1,10,24,14]`，L5 variance 为 `[1,24]` 且有限、非零；峰值显存约
  `23.31 GiB`。
- exact HEAD：`a6ad77ea9ee9bf0b355251324c4cf88b6e9a47e7`。
- 分支已由深圳仓库专用 SSH remote 推送到
  `personal/codex/sz-fastwam-action-dvac-adv`。本机 HTTPS 两次失败为连接 reset/timeout，
  因此传输 20 KiB + 2.5 KiB 的验证 Git bundle 到服务器，保留了相同 commit identity。

## 6. 真实 smoke

- v1 packet 在 compose 阶段因 Hydra flag 后仍放 override 而退出；没有启动 Ray actor、没有占卡。
- v2 已启动：GPU2/3，32 train env、rollout1、G8、32 trajectories、最多256 query records、
  B256/MB2/update1、fixed32、H32/C24/M10、L5 `[0.5,1.5]`。
- 目标：Step1 warm-up 全1并保存 DCP+两 rank sidecar；全新进程恢复并完成 Step2，验证权重
  有限、位于 `[0.5,1.5]` 且非均匀，然后保存第二个完整 DCP。

### 6.1 最终结果

- fresh Step1 自然完成、`exit=0`：rollout、fixed32、一次真实 backward/update 和
  `global_step_1` DCP 全部完成。warm-up 权重严格全1，`grad_norm=13.217`，所有训练量有限。
- Step1 checkpoint 完整包含 `.metadata + __0_0.distcp + __1_0.distcp`，并包含
  `dvac_state_rank0000.json` 与 `dvac_state_rank0001.json`。
- 随后由全新进程显式从上述目录恢复并完成 Step2，日志明确打印
  `Resuming training from checkpoint directory .../global_step_1`；进程自然 `exit=0`。
- Step2 已离开 warm-up：`dvac_warmup=0`，history/current 均为真实统计；权重
  `min/max/mean/std=0.56995/1.50000/0.98606/0.21731`，ESS fraction=`0.954`，证明
  recent-5 sidecar 恢复后产生了有限、非均匀且位于 `[0.5,1.5]` 内的 action advantage。
- Step2 `grad_norm=19.578`、KL=`0.0030`、policy/total loss=`0.606/0.0095`，均为有限值；
  `global_step_2` 再次产生完整的两片 DCP 与两份 rank sidecar。
- 真实并发峰值约 `64.0 GiB/card`，与 plain Fast-WAM smoke 的约 `62.5 GiB/card` 接近；
  退出后 GPU2/3 均约 `9 MiB`。GPU4--7 两条既有 PPO formal 始终存活，未被停止或改动。
- packet 最终标记 `SMOKE_OK`，worker `exit=0`；本轮没有启动 formal。

## 7. 收口判断

Fast-WAM `Action-DVAC-Adv [0.5,1.5]` 已完成实现、推送、endpoint parity、真实更新与跨进程
DCP/sidecar 恢复验证。后续若启动 formal，应逐叶继承 plain Fast-WAM Control，只放大
`rollout1→4`、`B256/update1→B1024/update2`，方法侧只保留本分支已有的 endpoint、
action-level advantage 与 `[0.5,1.5]` 配置。
