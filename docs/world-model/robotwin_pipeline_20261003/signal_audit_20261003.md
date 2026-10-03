# OpenDW C32 smoke 的 GRPO 信号审计

2026-10-03；只读本地固定 `2151a08ee1bd75df1bef0d8190e594bd5c7f7977` Git 对象与本轮 adapter/service。没有 SSH、测试或接口修改。结论是**奖励能够进入 GRPO，未发现 sum/mean 接线错误；单块 smoke 仍可能被原生组过滤全部屏蔽，程序跑完不代表产生了学习信号。**

## 实际计算链

- [adapter](../../../local_patches/opendw_smoke_20261003/rlinf/envs/world_model/opendw_adapter.py) 将 8 张未来图的连续成功分数 $s_1,\ldots,s_8$ 做相邻差分，只写到第 4、8、…、32 个动作位置，其余为 0。没有把每个分数重复 4 次。故一个动作块奖励和为 `coef × (s8 − previous_score)`；reset 的 previous=0、本轮 coef=1，所以首块回报就是末图 $s_8$。
- 原生 actor 先把每样本全部动作奖励求和，再对同起点的 G8 回报求均值。只有组均值落在 **[0.1, 0.9]，含两端**，才保留整个组。来源是原完整 resolved Control 的 `algorithm.filter_rewards=true / rewards_lower_bound=0.1 / rewards_upper_bound=0.9`，生成器没有改这三项。[源配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/examples/embodiment/config/sz2_can256_clean_resume_n32-20260927-v1.yaml)、[actor 收包与过滤 L371–441](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L371-L441)。
- `reward_type=chunk_level` 的优势入口也对 32 动作求 **sum**；GRPO 用每组回报的 `(score − group_mean)/(group_std + 1e−6)`，再乘有效 loss mask。[chunk 求和与回报](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/algorithms/utils.py#L89-L151)、[GRPO L90–123](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/algorithms/advantages.py#L90-L123)。因此不是先把奖励平均为 $s_8/32$ 再送 GRPO。
- 结束标记只在第 32 个动作设置；正常起点到这一块结尾的 32 个动作仍有效，随后再经过组过滤。`success_once` 使用“8 帧任一分数≥0.9”，而回报使用末图分数。可能先出现 0.95、末图回落 0.02，表现为 success=true、回报低；这是当前沿用 WorldArena 的奖励/成功语义，不是本轮新改的成功奖励。[有效 mask 计算](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/utils/metric_utils.py#L516-L538)。

因此，`success_once=0` 不等于无连续优势。若同组末图分数为 0.2～0.6，虽然均未达到成功阈值，仍可以产生有效相对优势。反过来，八条末图均值低于 0.1 会全部被过滤；均值在区间内但八条同分，也没有相对优势。N8 只有一个组，N16 只有两个组，单块早期全滤是需要如实记录的可能结果。

## 本轮验收读什么

| 证据 | 正确读法 |
|---|---|
| `service-events.jsonl` 的逐行 `score_last / score_min / score_max` | 按每个 request 的开始/结束包络分段；每连续 8 条 `score_last` 算一组均值、标准差、是否落入 [0.1,0.9]。N8/N16 的 request_id 可能同为 `seed0-call0`，不能只按这个字符串跨档合并。 |
| 真实 loss mask，或上述一块合同重建的保留组数 | 单块结尾才 done，故重建保留 chunk 数只能是 N8 的 0/8、N16 的 0/8/16。应报告“有多少组/样本有效”，不能只看 optimizer 完成。当前普通 actor 日志没有稳定的有效 mask 数标量，不能编造该字段。 |
| `rollout/advantages_min / advantages_max` | 至少有有效非零正负优势，且有限；`advantages_mean≈0` 是组内归一化的正常现象。全滤时这几个 rollout 统计可为 NaN，原因是有效集合为空，不自动等于梯度数值爆炸。 |
| `train/actor/policy_loss_abs`、`train/actor/grad_norm` | 需要有限，且梯度范数确实大于 0；组内抵消可使有符号 `policy_loss≈0`，不能据此说没学。结合 `ratio / ratio_abs / approx_kl / clip_fraction` 看重算与裁剪是否正常，不能单独靠 ratio≈1 判无更新。 |
| checkpoint 的 AdamW `step / exp_avg / exp_avg_sq`、可训权重差异 | 本轮 U2、GB=N，期望两次真实优化器调度。固定源初始化空 step 会把计数重置 0；调度次数和非零权重差异仍不足以证明奖励学习，零梯度时 AdamW weight decay 也可能改权重。应同时看有效优势、非零梯度及有限动量。 |

日志中 `env/return` 是整块末图 score 的平均；`rollout/rewards` 却对**有效样本的原始 32 个动作奖励**求平均，所以单块约等于有效样本末图 score 均值除以 32。`env/reward` 也按动作数归一化；这些图表不能直接互相当同一尺度。[rollout 统计 L416–505](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/utils/metric_utils.py#L416-L505)、[loss 与真实指标](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/algorithms/losses.py#L250-L395)、[optimizer 初始化](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/utils/utils.py)。

## 如果 N8/N16 单块都被滤掉

保留该事实与原奖励/过滤配置，接着做 **384 动作完整回合的信号验证 smoke**，不直接进入正式训练，也不通过降低阈值、关闭过滤或改 reward 制造通过。384 是 12 个完整 C32，当前部分块接口不能把它冒充精确 400 动作。新验证预算需显式写明：如果仍 GB=N、U2，一个 runner iteration 会有 24 次优化器调度；如果要求仍两次，则应 GB=12N、MB1，累积为 N8→96、N16→192。该选择由实际资源读数决定，本审计没有修改配置或接口。

当前 owner 的 smoke 校验固定 L32；下一轮 384 验证需要单独明确该新 smoke 合同。随后依真实末图分数、组均值/方差、有效 mask、梯度和轨迹证据，决定是否确有必要调整方法。不能仅因 success=0 认定连续 GRPO 没有信号，也不能仅因退出码为 0 就升级正式预算。
