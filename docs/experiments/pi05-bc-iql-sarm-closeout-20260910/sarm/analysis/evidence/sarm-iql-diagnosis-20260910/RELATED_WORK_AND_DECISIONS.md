# 在线 RynnValue＋SARM / IQL：相关工作与下一组决策

2026-09-10。仅阅读当前规划与公开一手论文/代码，未连接服务器、未改训练。当前实验的主导原因由同目录运行数据审计判断；下文的“建议”均是待验证适配，不是论文已经验证过的本任务参数。

## 先把比较对象讲清

干净 BC 已按真实最终成功筛选。SARM 再筛的是**成功轨迹内部的动作**；IQL 从 R11 起同时把 actor 数据改为成功＋失败，并改为指数优势权重。前者必须比已知成功更准确地挑出局部动作；后者必须让尚在学习的 critic 承担原来由真实成功标签完成的筛选。这两项要求都比“给混杂离线示范加权”更强。[当前固定两阶段合同](../../BC_RYNNVALUE_IQL_TWO_STAGE_IMPLEMENTATION_PLAN_20260909.md)、[SARM接入依据](../rynnvalue-rabc-plan-20260909/SARM_RABC_REVIEW.md)

假设 actor 抽到的成功/失败 query 各占一半，失败均权为成功的 1/3，失败仍取得 **25% 的加权监督份额**。一般地：

`成功权重份额 = p_success_query * mean(w_success) / mean(w_all)`。

这是监督质量的分配，不等于实际梯度占比；后者还与每条损失和梯度方向有关。不能只看成功权重大于失败，就判断已经恢复了成功 BC 的筛选强度。

## 六条最值得借鉴的线索

### 1. 先验证局部信号，而非默认“轨迹排序准 → chunk 权重准”

**论文事实。** 原 SARM 的 RA-BC 是对进度差加权，实机策略实验用 200 小时示范、π0 LoRA、40k 更新；其代表结果是平整衣服 1/12→10/12，皱衣服 0/12→8/12。它不是每轮4次尝试的成功池。更新的 SARM2/SPIRAL 则用弱 BC 的约50–100条 rollout 做一次奖励模型适配，标注快进展、慢进展、调整、错误与最终进度。[SARM §3.2/§4.2/A.7](https://arxiv.org/html/2509.25358v4)、[SARM2 Algorithm 1与§6.14](https://arxiv.org/html/2606.10305v1)

**对我们。** 冻结 RynnValue 的剩余秒差，不自动等价于“值得模仿的进展”。必要的接近、稳定抓握、双臂协调，可能即时进展小甚至为负；两次预测误差也会进入差值。已有成功标签值得作为参照，不能因模型分数相反就当作无用动作。

**最小区分。** 从已有池按“阶段×正/负delta×成功/失败”抽固定短片人工核验；另外对同一边界做合理的帧采样扰动，检查 delta 符号是否稳定。先做小规模诊断即可，论文的100条不是本项目的硬门槛。若负delta主要落在必要准备动作，优先研究温和权重或奖励适配；只调 κ 不会修复错误方向。

### 2. 保住成功监督，拆开数据切换与权重切换

**相关证据。** AWAC 的机制是 Bellman critic＋优势加权最大似然，使用已有经验后再在线追加；其 Adroit 实验先有25条示范和500条 BC rollout。它说明次优经验可以有用，不说明弱 critic 会自动保护一个强成功 BC。[AWAC §V-A与官方项目](https://arxiv.org/pdf/2006.09359)、[作者项目](https://awacrl.github.io/)

显式保留 BC 项也有标准先例：TD3+BC 在 actor 目标中同时使用 Q 项和动作模仿项，并按 Q 的平均绝对值调节两者比例；这不是让任意失败动作都获得可靠优势。[TD3+BC论文](https://arxiv.org/abs/2106.06860)、[作者 actor 更新代码](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py)

**建议。** 第一项便宜消融：**critic 仍用完整成功＋失败池，actor 只用成功池，继续重算 IQL 权重**。它不是完整标准 IQL，而是隔离 actor 数据变化的诊断。若它恢复 BC 水平而原版下降，重点查失败权重份额与状态覆盖。后续才考虑显式混合 `L=(1−ρ)L_BC_success+ρL_weighted_all`；ρ属于新方法参数，不冒称 AWAC 默认。

### 3. 指数温度和总损失强度需要分开看

**准确来源。** 原 IQL 用 `exp(βA)` 并裁到100，但 β随任务改变：AntMaze 10、MuJoCo 3、Kitchen/Adroit .5。作者 fine-tune 代码还对不同数据域使用不同奖励变换。β=10并非通用最佳值。[IQL Appendix D](https://arxiv.org/pdf/2110.06169)、[奖励处理与训练入口](https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/train_finetune.py)

AWR 官方代码提供另一种明确做法：按 replay 内有效样本统计 `(A−mean)/std`，再 `exp(A_normalized/temperature)`、裁剪；构造函数默认温度1、上限20。**AWR 的温度在分母，IQL 的 β在乘号处，方向相反。**[AWR官方代码 `_calc_adv` / `_calc_adv_weights`](https://github.com/xbpeng/awr/blob/master/learning/awr_agent.py#L361-L379)

**建议。** 先在固定 actor1024 上报告 `βA` 分位数、触顶率、均权、ESS，以及成功/失败/阶段的权重份额。将 `w/mean(w)` 和“降低β”分成两个消融：前者主要控制整个 batch 的 loss 倍率；后者改变相对分配。AWR 式标准化可以研究，但不作为默认补丁：信号很弱时除以很小的 std 也会放大噪声。Adam 的尺度近似不变性不等于梯度裁剪、ε、动量与非均匀加权完全不受影响。

### 4. 当前50次 Q/V 预热，是工程起点，不是已知充分预算

**实际设置。** 原 IQL fine-tune 入口默认先用固定离线数据训练1,000,000步、batch256，再在线采样。RynnValue 论文实机 IQL 为10k步、均权预热200步；它公开的 RoboTwin模板则为200k/预热2000、batch64。当前是从随机视觉 critic 开始，R10仅40次尝试、最多50次 Q/V 更新。[原IQL入口](https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/train_finetune.py#L24-L39)、[RynnValue B.5](https://arxiv.org/html/2608.09853v1#A2.SS5)、[锁定模板与调用审计](../rynnvalue-iql-plan-20260909/OFFICIAL_CODE_REVIEW.md)

RLPD 的高更新率配合 critic ensemble、LayerNorm 和视觉随机平移；论文同时说明重复训练可能过拟合。我们已有 GN/LN/crop，不能把这些说成遗漏。其 pixel 实验中也不是所有任务统一 UTD20。[RLPD §4.2–4.3/§5](https://arxiv.org/html/2302.02948v3)、[官方启动方式](https://github.com/ikostrikov/rlpd)

**建议。** 在同一个完整转移快照上，仅离线比较 Q/V 更新到50、200、1000步的诊断，actor保持冻结；数字是小规模预算扫描，不是作者最优设置。按 episode 划分或使用随后收集的完整新episode验证，避免相邻chunk泄漏。看成功结果、阶段内排序与TD/权重稳定性；训练loss下降不能作为充分依据。若额外训练只改善训练集，优先考虑表征复用/更多覆盖，继续加 UTD 未必有效。

### 5. 用真实结局检验价值尺度；MC可以作辅助，但别混错单位

**证据。** Cal-QL 用行为回报参考校准 CQL 的保守价值尺度；这不是对 IQL 的 A 做 z-score。其附录还报告：所测设置中把 IQL UTD增到5未改善渐近表现，更激进策略更新可能造成退化。因此“多训几次/放大β”没有普适保证。[Cal-QL §5/Appendix A、F、G](https://arxiv.org/html/2303.05479v2)、[官方代码](https://github.com/nakamotoo/Cal-QL)

SPIRAL 同时使用 dense TD 与真实结局的 sparse MC；论文 Table10 的白板结果为纯TD 9/20、混合10/20。公开代码提供 `td/mc/blend` 三种目标，便于做机制消融。[SARM2 Table10](https://arxiv.org/html/2606.10305v1#S6.SS14)、[SPIRAL官方代码](https://github.com/Qianzhong-Chen/openspiral/blob/main/src/openpi/models/residual_rl_model.py)

**建议。** 先做 Sparse-IQL：只将塑形系数 `.1→0`，其余尤其全转移 actor 池保持一致，区分奖励信号与 IQL 本身。若进一步加MC辅助，当前 PBRS 合同下有：

`G_shaped(t) = G_task(t) − λΦ(t)`（折扣、终止一致且吸收态Φ=0）。

所以对 shaped Q 的MC参考应使用右式，不能直接将 raw `G_task` 与 shaped TD 混合后仍称同目标。MC估计的是数据行为的回报，IQL趋向较好数据行为的价值，两者也不完全相等；辅助系数需要单列。诊断时比较 `Q_shaped+λΦ` 与真实回报，避免把 Q 学到剩余秒数就当作学到了任务成功价值。

### 6. 若权重信号可信而全量更新仍不稳，才考虑更大的路线变化

**两个直接VLA先例。** RynnValue论文的在线结果用 DSRL：冻结π0.5，优化latent noise上的小策略；并不是我们当前的在线IQL全量FM更新。[RynnValue B.6](https://arxiv.org/html/2608.09853v1#A2.SS6)

RECAP 用独立预训练VLM价值网络，将优势转成正/负输入条件，训练时保留两类数据；它的AWR对照成功率尚可但速度较慢，未得到同样吞吐收益。它还使用每轮任务数据重新拟合价值，并从预训练模型重做微调以减少累计漂移。[RECAP §IV-B、V-C/D、VI-C3](https://arxiv.org/html/2511.14759v1)、[作者说明](https://www.pi.website/blog/pistar06)

**建议。** 首先尝试第2条的成功BC锚定，再考虑冻结VLA的latent/residual RL或优势条件化。它们有明确先例，但会同时改变架构、部署与训练目标，不适合充当当前 IQL 的小修复。若同批数据加权学慢，而条件化能有效保留多种动作，RECAP路线才更有吸引力；不能只把 positive 文本塞进未按该条件训练的π0.5就称实现。

## 建议的最少实验顺序

| 次序 | 只改变什么 | 能回答的问题 |
|---|---|---|
| 0，只读 | 固定视频、信号和权重诊断；拆成功/失败/阶段 | 是局部信号错、失败获得过多监督，还是权重基本无效？ |
| 1 | IQL actor全池→成功池；Q/V仍全池 | R11下降是否主要来自actor数据切换？ |
| 2 | 在同actor池下，塑形系数.1→0 | RynnValue信号是否比真实结局奖励更有帮助？ |
| 3 | 固定池额外训练Q/V，actor冻结 | 是critic训练不足还是数据/表征不足？ |
| 4 | 若信号排序有效，仅均权归一或仅降低β | 是总体更新尺度，还是相对权重过尖？ |

SARM可独立加一个简单对照：`w_mix=(1−ρ)+ρ*w_SARM/mean(w_SARM)`，先取温和ρ例如.25，在完整1024确定分母。它显式保留所有成功query的学习份额；ρ是本项目探索值。与 IQL actor 全池不同，SARM此处全部已经成功，软化过滤的含义比较清楚。全零权重情形需要明确定义，不能直接除以0。

**应先读取实测诊断再选择一项，不把表中改动合成一次“大修”。** 即便得到提升，结论也要区分固定池复训、在线采样与额外计算预算；固定32评估给出配对成败与多个checkpoint趋势，单点1–2次成功变化不宜当作机制证明。

## 来源边界与公开代码注意点

- 本地 IQL/SARM 设置来自上述当前规划，动态实验结论需以本轮服务器证据为准。
- AWR、TD3+BC、SPIRAL链接是2026-09-10实读公开分支；本轮没有为其完整实现做commit锁定或部署验证。IQL/RLPD/RynnValue既有固定源码证据仍见原审计目录。
- SPIRAL 论文写 `J_TD + .5 J_MC`，公开代码 `blend` 是 `.5*MC_target+.5*TD_target` 后求MSE；梯度相对比例不相同，不能当作逐项一致。论文表为每轮5000步，当前README列10000步，也应先选定复现合同。[公式9与表3](https://arxiv.org/html/2606.10305v1)、[公开代码与说明](https://github.com/Qianzhong-Chen/openspiral)
- SARM2收益同时包含任务内奖励适配、残差限制、critic/MC设计和更多数据，不能归因为“换了一个更好的权重公式”。本项目优先借鉴能独立验证的组件。
