# FastWAM GRPO · RoboTwin 2.0 拨开关

原模型固定种子独评 13/32，第 130 轮达到 21/32，第 135、140、145 轮分别为 10、10、15/32。按阶段汇总，1–50、51–100、101–147 轮的独评成功率为 38.1%、44.4%、50.3%。本次登记为阶段性提升案例。

## 成绩与配置

- [逐轮训练与独立评估、曲线和画图脚本](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/fbfa3c3e925a13e188667f731f94aafb337767a9/experiments/rt2-fastwam-switch-192/trends/2026-11-28)。记录覆盖 147 轮训练、29 次独立评估及一项基线。
- [第 130 轮所附的训练实配](formal.yaml)。目录变量分别指向 RoboTwin、原版模型、种子和实验结果目录。[固定训练和评估种子](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/experiments/rt2-fastwam-switch-192/seeds)。
- 32 个环境、两卡每卡 16 路，累计 8 批形成每轮 256 局。每组 8 局，更新两遍，全局批量 1,024，学习率 5e-6，裁剪 0.2。每 5 轮独评 32 局。原版模型预测 32 步、执行 24 步、去噪 10 步，单局最多 192 步。[微批从 2 调至 1 的阶段记录](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/experiments/rt2-fastwam-switch-192/microbatch-resume)。

## 权重与续训

在深圳 1 机，DATA、DATA2 分别表示本人在两个同名盘上的目录。[保护清单](../../../docs/u_policy_switch/PROTECTED_CHECKPOINTS.json)内搜索 `sz1_fastwam_grpo_turn_switch_cp130_20261128`。

- 推理权重位于 `DATA/models/protected-eval/fastwam-grpo-turn-switch-20261128/global_step_130/full_weights.pt`。这是 RLinf 模型的完整 state_dict，共 2,089 个张量。已在 CPU 重新加载并核对键、形状和类型。用同版本 FastWAM 构造模型，再加载这些参数。权重与原版的统计归一化文件配套。
- 完整续训点位于 `DATA2/grpo-best/fastwam-resume47/global_step_130`。它保留 DCP 两个分片、索引、优化器、学习率调度和 RNG，也附原训练配置、归一化文件和基础模型标识。续训时将 `runner.resume_dir` 指向这一目录。

## 核心实现

- [rlinf/models/embodiment/fastwam/fastwam_rl.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/rlinf/models/embodiment/fastwam/fastwam_rl.py)
- [rlinf/models/embodiment/fastwam/fastwam_policy.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/rlinf/models/embodiment/fastwam/fastwam_policy.py)
- [rlinf/models/embodiment/fastwam/robotwin_adapter.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/rlinf/models/embodiment/fastwam/robotwin_adapter.py)
- [rlinf/models/embodiment/fastwam/builder.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/rlinf/models/embodiment/fastwam/builder.py)
- [rlinf/workers/actor/embodied_fsdp_actor_worker.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/rlinf/workers/actor/embodied_fsdp_actor_worker.py)
- [rlinf/algorithms/advantages.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/rlinf/algorithms/advantages.py)
- [rlinf/algorithms/losses.py](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/rlinf/algorithms/losses.py)

[部署版本相对于上述提交的差异](deployed-core.patch)。
