# FastWAM GRPO · 复用路线

## 从哪个任务开始

想先验证算法，优先复用深圳 1 机 RT2 的移订书机。想增加 Dojo 算法覆盖，优先 Dojo 标准折衣服，公开 FastWAM 完整成功率为 22.67%。丢瓶子为 40%，也适合收集有成有败的组，但原生预算 700 步；倒液体的公开 FastWAM 成功率是 0%，π0.5 在这里的成绩不能代替 FastWAM。折衣服是备选新实验，本次正式训练仍为 π0.5 倒液体。[逐任务榜单](https://robodojo-benchmark.com/leaderboard/rollouts/FastWAM?bench=sim)

## 三层各自复用什么

| 层 | 做法 | 首试依据 |
|---|---|---|
| 环境 | Dojo 复用当前原生 ARX-X5、三相机、14 维关节、固定布局、任务自己的上限及成功判定。RT2 走已有机器人接口。组内重复布局 ID，初始动作噪声独立。 | 本次 Dojo 接口与成功 RT2 π0.5 的 reset ID 分组 |
| 模型 | Dojo 用官方 joint-0 的 step_020000.pt 和配套 dataset_stats，再接入 Wan/T5/VAE 等基础组件。以 XPolicyLab 的三图拼接、关节解码、归一化、提示词、推理时间表为对照。RT2 沿用该基准自己的权重与统计。 | Dojo 官方适配与 FastWAM 作者推理 |
| 学习 | 复用现成 FastWAMPolicy 的 predict_action_batch 和 forward 重算 log-prob。冻结视觉、视频及文本组件，只更新动作分支。固定初态的 8 局分别采样，按组内成败算优势，屏蔽终止后动作。实际被采出的随机转移及旧 log-prob 一起进入更新。 | 本机 FastWAM GRPO 已跑过真实采样、actor 更新、权重同步、检查点恢复 |
| 精度 | BF16 前向，建议训练参数保有 FP32 主副本、Adam 一二阶矩用 FP32。先在相同小批上比较更新写回及峰值显存。 | 旧实验 action 参数及 Adam 矩均为 BF16，部分张量动量非零但大量权重未变 |
| 初始参数 | G8、256 局/轮、clip 0.2、LR 5e-6、两遍更新。模型预测 32 步、执行 24 步重看图、去噪 10 步、shift 5。训练探索 0.3 来自旧 RT2 FastWAM；评估沿官方模型采样器。GB1024、每卡 microbatch 2 是 RT2 起点，根据 Dojo 显存及有效动作块重核。 | RT2 的成功 π0.5 给出算法对照；模型专属参数优先用 FastWAM 旧实跑与 Dojo deploy.yml |

RT2 旧协议是 192 动作，恰好 8×24。若比较时用 200 动作，末块只执行 8 步。Dojo 的 400 或 500 动作分别剩 16 或 20 步。环境、终止掩码和 log-prob 减约要一同覆盖这个尾块，逐项验证后再扩大并行。

## 最短验证路径

1. 同一录制观测、同一初始噪声对齐官方动作，覆盖图像顺序、归一化和左右夹爪。先 B1，再比较批推理与逐个推理。
2. 原模型在所选任务做一次固定布局独评。取 4 个 G8 检查有成有败的组数，保留失败回合；没有有效组时先看任务和初始能力。
3. 一个真实采样批做一次更新。更新前在相同随机转移上重算，检查有效样本的比率接近 1；更新后检查动作分支的 FP32 主权重有变化、采样进程收到新权重、恢复后能够继续更新。
4. 正式每轮 256 局。RT2 先沿用旧 32 并行×8 批；Dojo 的并行度根据 FastWAM 自己的场景、模型和训练峰值定。固定评估按同一协议比较原始点和后续检查点。

## 已有结果

深圳 1 机移订书机 256 局版的事件文件有 17 轮训练，三次独评 11/32、11/32、13/32；当时因原生线程故障中断。128 局版也做过。旧实验验证了接口和更新链；固定评估仅增加两局，尚不足以确认稳定收益。建议先解决精度、重放和基础模型对齐，保留原有学习率，再决定是否改变探索幅度。

[FastWAM 作者仓库](https://github.com/yuantianyuan01/FastWAM) · [Dojo 官方权重](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/35efbc7dedfdbeeb6e95fb749bd885d73d483e41/ckpt/RoboDojo/FastWAM) · [Dojo 模型适配](https://github.com/XPolicyLab/XPolicyLab/blob/10ab2651a0b7cd5b8a948cb13f18ccdf818c6de4/policy/FastWAM/model.py) · [Dojo 推理设置](https://github.com/XPolicyLab/XPolicyLab/blob/10ab2651a0b7cd5b8a948cb13f18ccdf818c6de4/policy/FastWAM/deploy.yml)

