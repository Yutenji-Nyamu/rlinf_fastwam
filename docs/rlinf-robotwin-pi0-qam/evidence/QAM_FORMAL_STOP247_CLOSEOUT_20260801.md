# QAM × π0 × RoboTwin：formal stop247 收尾

日期：2026-08-01。任务：`adjust_bottle`。本报告只总结本轮
`Plain-QAM π0 online adaptation` 的运行证据，不把在线训练 rollout 当作独立评估。

## 1. 停止与可恢复状态

- 用户于 17:28 CST 明确要求停止；17:28:58 对已核对命令行的 driver PID `380841`
  发送 `SIGTERM`。
- 最后完整 cycle 为 `247/380`。driver 于 4 秒内退出，资源 monitor 随后 exit0；两卡显存
  归零，无残留 QAM/Ray 训练进程。
- runtime 记录的 driver exit code 为 `134`，末尾明确写有同一时刻的 `SIGTERM received`；
  这是本次显式停止路径，不是训练自行崩溃。
- 最新可恢复点为 `global_step_225`，`complete=true`、`schema_version=2`、
  `world_size=2`，大小 `15,030,394,539` bytes（约 14.00 GiB）。cycle 226–247 的日志
  保留，但其参数更新不在 checkpoint225 中。

## 2. 最终训练证据

| 指标 | 最终值 |
|---|---:|
| online train success | `86/494 = 17.41%` |
| collect | `8/50 = 16.00%` |
| q_only | `6/52 = 11.54%` |
| am_on | `72/392 = 18.37%` |
| 最近 10 / 20 / 50 cycles | `35.0% / 22.5% / 24.0%` |
| global macro inserts | `4,861` |
| critic updates | `4,349` |
| fine updates / policy version | `3,837 / 3,837` |
| pending update credit | `0` |

计数满足 `4861-512=4349`、`4349-512=3837`，说明 collect、q_only 和 am_on 的
schedule 连续，没有重置或欠账。

最终 critic loss 为 `0.0039`；最近 20 cycles 的 `Q mean / TD target` 为
`0.03065 / 0.03090`，Q-head std 为 `0.01815`。最近 10 cycles 的 AM loss、terminal
adjoint norm、fine pre-clip grad norm 分别为 `7.2708 / 0.00474 / 3356.71`。AM 初期前
10 cycles 的 terminal adjoint 均值为 `0.0097`，到末段约下降一半。

v4 GPU 显存峰值为 `43,567 / 43,693 MiB`，host anon 峰值约 `44.99 GiB`；
OOM/OOM-kill 为 `0/0`。资源不足或数值崩溃不是当前直接失败证据。

## 3. 为什么可能不 work

结论不是“已经证明 QAM 无效”，而是：**critic 已达到 TD 自洽，但没有直接证据证明它学到
了可靠的 action-conditioned 上坡方向。**末段成功率有回升，因此也不能据本次在线训练
曲线宣布确定失败或确定涨点。

按证据优先级，主要风险是：

1. **online-only critic 覆盖不足。**官方实验先用丰富 OGBench transition，再在线追加
   数据；本适配只有当前策略附近的 `4,861` 条 macro transition。约 86 个成功 episode
   对应的稀疏正结果只占 macro 数的约 `1.8%`，同一状态附近缺少好/坏动作对照。低 TD
   loss 与低 ensemble std 可能只是 Q 近似成 $V(s)$，并不证明 $\nabla_aQ$ 有用。
2. **N20 macro credit 过粗。**Q 一次评价 `[20,14]`，即 280 维完整计划，reward 主要在
   query 末端；这比官方 primitive replay 上重叠的固定 `H=5` window 更稀疏，也更难把
   成败归因到具体动作维和具体时刻。
3. **C1 表示可能丢失 critic 所需细节。**十个 Q head 共用 frozen π0 prefix 的四块池化
   特征再拼 proprio。若瓶子姿态、接触或细小时序在池化时丢失，十个 head 会共同看不见，
   此时低 std 反而可能是假自信。
4. **给 F1 的实际价值方向偏弱。**terminal adjoint 从约 `0.0097` 降到 `0.00474`；同时
   fine pre-clip gradient 仍远大于 global clip `1.0`。这可能让更新长期只保留归一化后的
   方向，而 reward 强弱信息被压平；但尚未测量 fine 相对 frozen π0 的动作距离，所以这
   仍是待验证解释。
5. **B1 与官方 Plain-QAM 仍有方法差异。**本轮冻结 SFT π0 behavior，不再训练 behavior
   FM/slow EMA；它保留了 behavior VJP 主干，但不是官方 B2 的联合 behavior update。
6. **缺少受控效果评估。**每 cycle 只有两个在线 train episode，策略和 replay 同时变化；
   最后 10 cycles 的 35% 是积极信号，但样本太少。没有同 seed frozen-π0 baseline、独立
   held-out evaluation，也没有真实执行的 `base / +dQ/da / -dQ/da` 对照。

下一轮若继续研究，最先补的不是延长同一训练，而是三个小诊断：同状态动作扰动 Q 排序与
真实执行、C1 feature 的 success/failure 可分性、fine 相对 frozen π0 的动作漂移。它们能
先区分“Q 没学到动作”“表示看不见”与“actor 更新太弱”。

## 4. 产物

服务器轻量收尾目录：

`/root/autodl-tmp/experiment_exports/qam_pi0_robotwin_formal_stop247_20260801`

运行包：

`/root/autodl-tmp/experiment_exports/qam_pi0_robotwin_formal_stop247_runtime_20260801.tar.gz`

SHA-256：

`6740c3e71f6b963940498cec214b7448cd483b847baa4e303d869548b44d14ab`

Windows 轻量副本：

- `exports/qam_pi0_robotwin_formal_stop247_runtime_20260801.tar.gz`；SHA-256 同上；
- `exports/qam_pi0_robotwin_formal_stop247_high_info_20260801_v1.zip`；最终 SHA-256
  记录在同名 `.sha256` sidecar，避免归档内报告自引用归档 hash。

包内包含 v1–v4 runtime、完整 driver/resource/config/command/provenance、四份 launcher、
checkpoint 文件清单和每个 QAM completion JSON。checkpoint tensor、DCP shard、rank
sidecar/replay、视频、模型、venv 和数据集均只留服务器；该包可审计和重画曲线，不能独立
resume。
