# 深圳两卡 π0 与 π0.5 GRPO 早期对照

> 口径：2026-09-01 18:34 CST；服务器只读检查。π0.5 Control 完整到 Step 12，正在 Step 13 rollout；无 fatal，Step 10 checkpoint 完整。本文只诊断，不改变运行。

## 1. 先给结论

- π0.5 当前相对自己的起点确实没有出现 π0 前十步那种上升，held-out fixed32 也暂未提高。
- 但这不等于 π0.5 或 GRPO 实现坏了：π0.5 前十步绝对成功率仍较高，所有更新、checkpoint 和数值指标都真实生效。
- 当前比较不是单一模型轴。π0.5 使用 `GB512/update5`，π0 使用 `GB1024/update2`；在同为最多 1,024 records/outer 时，前者是 10 次 optimizer call，后者是 2 次。
- 最可信解释是三项叠加：更强 SFT 起点导致 headroom 小；高成功率使 G8 中更多全成功组被过滤；`GB512/update5` 对同批数据复用更强，导致 KL/clip 明显更高。

## 2. 真实早期曲线

| 指标 | 两卡 π0 GRPO | 两卡 π0.5 GRPO |
|---|---:|---:|
| Step 1 raw | 69.14% | 88.67% |
| Step 5 raw / MA5 | 86.33% / 75.08% | 83.98% / 85.63% |
| Step 10 raw / MA5 / MA10 | 82.42% / 81.64% / 78.36% | 78.52% / 84.69% / 85.16% |
| Step 12 raw / MA5 / MA10 | 76.56% / 80.39% / 80.23% | 71.48% / 78.59% / 82.46% |
| fixed32 at Step 5 / 10 | 28/32、29/32 | 28/32、27/32 |

π0 的 Step 6--10 均值比 Step 1--5 高 6.56 个百分点；π0.5 低 0.94 个百分点。π0.5 是“高起点后没有继续涨”，不是“从一开始就比 π0 差”。

## 3. 模型与推理合同

共同部分：

- 同属 OpenPI `Pi0` 模型家族，主要骨干均为 PaliGemma/Gemma 2B VLM 与 Gemma 300M action expert。
- 输入均为三相机、任务文本和机器人 state；最终均输出连续 flow-matching 动作。
- 本任务均为 `H=C=50,D=14`：每次 query 生成 50 个 future actions，并执行 50 个环境动作；每个 200-action episode 最多 4 次 query。
- 当前均冻结 VLM，训练 action expert/projection；RTC 关闭。

π0.5 的核心结构差异：

1. π0 把连续 state 投影后放入 action suffix；π0.5 把 state 离散成 prefix/language token。
2. π0 用 action-time MLP 注入 flow timestep；π0.5 用 AdaRMSNorm 条件化 action expert。
3. π0.5 prompt 上限更长，RoboTwin 使用 M5；π0 使用 M4。M 是一次 query 内的 flow-SDE 积分步数，不是 rollout 或 episode 长度。
4. π0.5 使用自己的 `adapt_to_pi` 变换与 norm stats，名义 noise 也为 0.3，而 π0 为 0.5。

官方代码依据：

- <https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py>
- <https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py>
- <https://arxiv.org/abs/2504.16054>

π0.5 论文还讨论离散动作预训练和高层语言能力，但本次 RLinf runtime 使用的是连续 flow-matching action head，不运行高层 subtask planner。

## 4. GRPO 实现是否不同

没有单独的 π0.5 GRPO 算法。两者共用：

- current typed rollout；
- Flow-SDE log-prob；
- G8 group-relative advantage；
- actor-only、chunk-level reward/log-prob/PPO clip；
- `filter_rewards=true`；
- 同一 `advantages.py` 和训练 worker 主路径。

π0.5 分支只新增一份主 YAML，production Python 零改动。仓库中虽然还包含 Action-Adv 的 opt-in 支持，但 clean Control 是 `mode=off + chunk_level`，不会进入 DVAC 权重或 action-level loss。因此没有发现 clean π0.5 GRPO 的 no-op、错位或另一路实现。

## 5. 精确配置差异

| 项目 | 两卡 π0 | 两卡 π0.5 | 性质 |
|---|---:|---:|---|
| train/eval env | 64/32 | 64/32 | 相同 |
| rollout / trajectories | 4 / 256 | 4 / 256 | 相同 |
| group | G8，32 groups | G8，32 groups | 相同 |
| query records 上限 | 1,024 | 1,024 | 相同 |
| H/C/D | 50/50/14 | 50/50/14 | 相同 |
| MB | 32 | 32 | 相同 |
| GB × update | 1,024 × 2 | 512 × 5 | **非模型固有差异** |
| optimizer calls/outer | 2 | 10 | **5 倍** |
| record presentations | 2,048 | 5,120 | **2.5 倍** |
| flow steps M | 4 | 5 | 模型合同 |
| noise | 0.5 | 0.3 | 模型配方 |
| actor LR | 5.6e-6 | 5.0e-6 | 模型配方 |
| checkpoint | DCP | local-shard | 保存格式，不影响数学 |

π0.5 的 `update5` 来自官方 RoboTwin π0.5 PPO 配方的两卡缩放，不是官方 RoboTwin π0.5 GRPO 配方。RLinf 没有公开后者；官方 LIBERO π0.5 GRPO 示例反而使用 `update_epoch=1`，但环境不同，不能直接照抄：

- <https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml>
- <https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/libero_spatial_grpo_openpi_pi05.yaml>

## 6. 为什么目前没有明显提升

按证据强度排序：

1. **起点高、空间小。** 官方 RoboTwin `adjust_bottle/demo_clean` 中，π0 SFT/PPO 为 76.56/98.44%，π0.5 为 85.94/96.09%；π0.5 的可见提升空间本就更小。
2. **有效 GRPO group 更少。** 二值 reward、G8、过滤全同 reward 时，0/8 和 8/8 都不训练。按前五步均值作独立近似，π0 约有 28.8/32 个 mixed groups，π0.5 约 22.8/32；这只是方向性估计，当前日志没有记录真实 active-group fraction。
3. **更新壳偏强。** 前 11 步 π0.5/π0 的平均 KL 为 0.0984/0.0268，clip fraction 为 0.233/0.0853，即约 3.7 倍和 2.7 倍。π0.5 不是没学，而是每轮改得更大。
4. **模型采样分布不同。** M5、noise 0.3、离散 state prefix 与不同动作坐标变换会改变探索相关性和 log-prob 尺度；相同 clip=0.2 不保证完全相同的有效约束。
5. **样本仍早。** 当前只有 12 个 outer steps、两次 fixed32；末端下降可能包含固定 seed 难度与 rollout 随机性。

RLinf 官方 issue 也有另一位用户报告 RoboTwin π0.5 + GRPO 不稳定，并同样采用 `update_epoch=5`；任务、chunk 和 batch 均不同，因此只能作旁证，不能当本次根因：<https://github.com/RLinf/RLinf/issues/1268>。

## 7. 最干净的下一项诊断

若要回答“只换成 π0.5 后 GRPO 是否仍有效”，下一条应保留 π0.5 固有的 checkpoint、norm、M5、noise 0.3 与 LR 5e-6，只把优化壳对齐历史 π0：

```text
GB1024 / MB32 / update2
```

env、rollout、G8、records、eval/save 全部不动。这样先隔离目前最大的非模型差异；不要同时改 group size、noise 或 M。当前 formal 未被停止或修改。

## 8. 本地证据

- π0 曲线：[`six-2gpu-grpo-comparison-live-20260830-1038/curves.csv`](../../rlinf-shenzhen-grpo-dvac-action-adv/evidence/six-2gpu-grpo-comparison-live-20260830-1038/curves.csv)
- π0 原始日志：[`grpo-control-2gpu-stopped-step96-20260828/raw/runtime/driver.log`](../../rlinf-shenzhen-grpo-dvac-action-adv/evidence/grpo-control-2gpu-stopped-step96-20260828/raw/runtime/driver.log)
- π0.5 当前曲线：[`pi05-grpo-pair-live-20260901-1804/curves.csv`](pi05-grpo-pair-live-20260901-1804/curves.csv)
- π0.5 参数依据：[`FORMAL_PAIR_READONLY_PREP_20260901.md`](FORMAL_PAIR_READONLY_PREP_20260901.md)
