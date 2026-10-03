# WMRL 的 6.51%：奖励、组内比较与有效学习信号

2026-10-03。只读讨论：复核固定源码、已采集日志和公开论文/代码；本次没有 SSH、实验变更或新训练。源码固定为 RLinf `d34d4c320d08cb982de034aa9a011f08dc0fa217`、WMPO `c836d74ec6f4525c93fe980d54d0ca870118615a`。这份讨论解释现有证据，不把候选机制当成已完成因果定位。

## 1. 6.51% 到底是什么

**6.51% 是最后 10 个已完成训练轮的 `rollout/loss_mask_fraction` 平均值 0.06509277：名义 chunk 槽位中，最终 loss mask 为真的比例。** 当前每轮有 512 条名义轨迹，每条最多 40 个 C8 chunk，所以分母是 20,480；末 10 轮平均约 1,333.1 个槽位为真。

它同时经过两道处理：

1. 每条轨迹首次 done 之后的动作不计入训练；成功所在 chunk 本身保留。
2. 同一起点的 G8 组，若组平均回报不在 [.1,.9]，整组 mask 置零。

因此它既不是成功率，也不是保留组比例，更不是“只有 6.51% 的 optimizer 更新发生了”。失败轨迹也能提供训练信号：只要同组另有成功者，它就是有效的负比较。单有 mask 为真也不是每个参数梯度非零的保证。

139 轮共 2,846,720 个名义 chunk 槽位，累计 mask 折算 220,547 个，占 7.75%；136 轮有正 mask，仅第 4/5/6 轮整轮为零。这说明信号稀疏，但**现有日志没有把“提前 done 缩短”与“整组无比较信号”分开计数，不能说 93.49% 的组被丢弃**。[已有标量与真实评估](../audit_20261003/training_dynamics.md)、[mask 指标源码](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/utils/metric_utils.py#L473-L537)

## 2. 用同一起点的 8 次尝试解释

下面是说明算法的假设例子，不是本轮抽取的真实视频。每条最多 40 个 chunk；第 8 条在第 10 个 chunk 达到成功，并在该 chunk 末仍判成功；其余七条一直失败。

|尝试|首 done 前总回报|保留 chunk|组内优势，约值|学习作用|
|---|---:|---:|---:|---|
|1|0|40|−0.354|减少这次失败行为的相对概率|
|2|0|40|−0.354|同上|
|3|0|40|−0.354|同上|
|4|0|40|−0.354|同上|
|5|0|40|−0.354|同上|
|6|0|40|−0.354|同上|
|7|0|40|−0.354|同上|
|8|1|10|+2.475|增加这次成功行为的相对概率|

组均值为 1/8，固定实现用样本标准差约 0.353553 做归一化，故成功者优势约 2.475，七个失败者各约 −0.354。每条轨迹的优势广播到其有效动作；这不是只奖励最后一下，也不能精确指出哪一下造成成败。第 8 条成功后的 30 个 chunk 被屏蔽合理；其成功前 10 个仍训练。该组保留率是 100%，chunk mask 比例却为 290/320=90.625%。这已说明两个比例不能混用。[固定优势计算](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/algorithms/advantages.py#L89-L121)

若八条全部失败，回报全 0：每条回报减组均值后已经是 0，**即使关闭组过滤，也没有这类 GRPO 的方向性比较信号**。若八条全部成功且回报都为 1，同样为零。组过滤把这些组标成无效，并非把本来有正负对比的数据误删；它与成功后的时间 mask 是两件事。当前 [.1,.9] 在二元轨迹回报下正好保留 1–7 条成功的组。一般非二元回报则要按实际均值理解，不能直接称“成功率过滤”。[过滤源码](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L251-L298)

若假设每条轨迹**独立、同成功概率 p=5%、回报恰为二元成功**，G8 中：

$$
P(\text{全败})=0.95^8\approx66.34\%,\qquad
P(\text{全成})=0.05^8\approx3.91\times10^{-11},
$$

$$
P(\text{有成有败})=1-0.95^8-0.05^8\approx33.66\%.
$$

这是概率例子，不是观测到的组保留率，更不等于 6.51%。实际同组共享起点，难度和策略行为可能相关；总体约 5% 的成功还可能集中在少数容易起点。当前 `env/success_once` 又不等于 actor 用的二元回报事件，因此不能直接代入该公式反推实测组数。

## 3. 低 mask 的后果：学习量、分母、算力要分开

**合理的 done 后 mask 可以提高可信度。** 它防止首次成功之后的无关动作、状态漂移或幻觉继续被当作成功行为训练。这是 WoVR masked GRPO 明确采用的机制，不应笼统称为 bug。[WoVR §4.2](https://arxiv.org/html/2602.13977v1#S4.SS2)

**整组回报无差别意味着缺少对比。** 当前代码只把这些组的 loss mask 清零，不删除数组，也不补采新组；少数保留组会承担本轮大部分学习方向。如果它们偏向少数任务、起点或奖励误判，更新可能不稳定或偏斜。方向是否可信比单纯把 mask 百分比抬高更关键；取消 mask 本身不能制造有效奖励。

**当前损失不是简单按全部有效 token 数做分母。** 虽然 YAML 写 `token-mean`，actor 实际传入有效长度，loss 会改用 `masked_mean_ratio`。设一轮有 B 条轨迹，名义长度 T=40 个 chunk，轨迹 i 的首次 done 前有效长度为 t_i，组保留标记为 g_i，则对整轮固定样本的损失归约等价于：

$$
\mathcal L=\frac1B\sum_{i=1}^B g_i\frac1{t_i}\sum_{t=1}^{t_i}\ell_{i,t}.
$$

这里的每项可含 clipped policy loss；实际先把 chunk 槽位打散进固定 global batch，再更新。长度 t_i 是组过滤前计算的，不因整组清零改为零。

- 提前成功的轨迹按自己的有效长度归一化，较短不自动变成较小权重。
- 过滤组贡献为 0，但名义 B 仍含这些组；在固定样本上，相对“只对保留轨迹求均值”，有一个保留轨迹比例的缩放。
- 因而不能说“mask=6.51%，梯度就只有正常的6.51%”。mask 混有长度因素；Adam 的状态、裁剪和样本组成还会影响实际参数更新。

[actor 传入有效长度](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L760-L776)、[损失覆盖逻辑](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/algorithms/losses.py#L229-L280)、[实际归约函数](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/utils/utils.py#L432-L436)

**mask 不等于省掉对应计算。** 本轮 env 仍生成固定 320 动作，actor 也在前向以后才对 loss 应用 mask，名义 batch 尺寸不变；所以低 mask 不能按比例减少 rollout 费用、actor 前向或峰值显存。它解释了为什么“有效信号很少”和“显存仍满、训练仍久”能同时发生。[actor 训练循环](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L601-L671)

## 4. 为什么 proxy 成功上升，却未必增加正回报

当前环境先对每帧输出二元成功分数 s，再产生相对奖励 `r=s_current−s_previous`。同一个 C8 chunk 的三种判断如下：

|量|当前实现|
|---|---|
|success / done|8 帧中任一帧成功就为真，done 标在第 8 帧|
|chunk 净奖励|8 个相对差分相加，等于本 chunk 末帧分数减前一 chunk 末帧分数|
|actor 回报|保留首次 done 所在 chunk，截断其后的奖励|

假设上一帧为 0，本 chunk 分数为 `0,0,1,1,0,0,0,0`：日志记“曾成功”，done 为真，但相对差分 `0,0,+1,0,−1,0,0,0` 合计为 0。若此前也全 0，这条轨迹给 actor 的回报仍为 0。同组八条都这样，则 success_once 可以为真，组内仍没有正负回报对比。

这是一条**可由源码推导的边界情况**，不是本轮已经量出频次的结论。成因可能是分类器闪烁、WM 场景幻觉，也可能是实际生成行为达到目标后又移开；现有汇总不能区分。

完整 320 步的 `env/return` 则会把首次 done 后仍生成的奖励也加上，望远镜求和等于最终分数；它不是截断的 actor 训练回报。现有 `success_once` 0.859%→4.824%，完整 return 0.003125→0.003516，确实提示“某时刻命中”与“最终状态”不同，但不能据此直接认定 reward hacking。[reward 与 done 原码](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/env.py#L232-L259)、[chunk 结束与日志](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/env.py#L607-L665)、[actor 截断回报](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/algorithms/utils.py#L136-L150)

## 5. 论文、官方代码、我们的实现分别在哪一层

|问题|论文/官方设计|本轮实际与边界|
|---|---|---|
|成功后 mask|WoVR 明确屏蔽成功后的步骤并按轨迹有效长度归一化|固定 RLinf 已实现，本轮继承；合理且有方法依据|
|全成/全败组|WMPO 论文 §3.4 明确过滤后继续采样至有效 batch 满|WMPO 官方 trainer 有同样 while 循环及数组切片；RLinf 当前只置零 mask，没有补满；这是实质差异|
|奖励事件|WoVR Eq8 写下一观测的二元状态奖励|RLinf Wan 默认相对差分，any-frame done 位于 chunk 末；论文没有展开此内部帧事件细节|
|优势公式|WoVR Eq10 展示组内中心化；WMPO Eq5 展示再除组标准差|当前 RLinf 也除样本标准差；与 WoVR 展示公式不逐字一致，属于官方代码层差异|
|折扣|WoVR 回报公式含 gamma|当前 GRPO `calculate_scores` 用未折扣求和并按 done 截断，YAML gamma 没进入这条公式；不是本地适配新改|
|动作时间粒度|两篇的策略 action 都是 action chunk；WMPO 另明确区分 frame 与 state 下标|不能把论文 t 直接认作每一物理帧，也不能断言论文要求 C8 中途逐帧终止|
|策略与观测|WoVR/RLinf Wan 例程为单主图 OFT，WMPO 为离散 OFT、单主图、C8|本轮是 π0.5 连续 Flow-SDE，H10/C8、屏蔽腕图；官方真实评估为双图/H10/C5，存在输入和反馈频率差异|
|政策分布对齐|WoVR 含 PACE；WMPO 先做 policy behavior alignment，公开例程允许冻结 WM|本轮固定 Wan/RM，没有以当前 π0.5 迭代再对齐；不等于完整 WoVR，也不是原样 WMPO|

论文依据：[WoVR §3–4](https://arxiv.org/html/2602.13977v1)、[WMPO §3 与 Algorithm 1](https://arxiv.org/html/2511.09515v1)。上述仅提取与本问题直接相关的结构，不以论文最终成绩保证本组合收益。

WMPO 公开实现的直接证据：[`ray_trainer.py:566–639` 补采](https://github.com/WM-PO/WMPO/blob/c836d74ec6f4525c93fe980d54d0ca870118615a/verl/trainer/ppo/ray_trainer.py#L566-L639)、[`ray_trainer.py:793–854` 切片过滤](https://github.com/WM-PO/WMPO/blob/c836d74ec6f4525c93fe980d54d0ca870118615a/verl/trainer/ppo/ray_trainer.py#L793-L854)、[coffee 配置 G8 与 [.1,.9]](https://github.com/WM-PO/WMPO/blob/c836d74ec6f4525c93fe980d54d0ca870118615a/examples/mimicgen/coffee/train_wmpo_128.sh#L37-L44)。其视频奖励把滑窗命中转换为完整轨迹成功标记，不采用 Wan 的逐帧相对差分路径。[`robwm_rollout.py:237–286`](https://github.com/WM-PO/WMPO/blob/c836d74ec6f4525c93fe980d54d0ca870118615a/verl/workers/rollout/robwm_rollout.py#L237-L286)

**C8、相对奖励和 any-frame done 本来就在固定官方 Wan 环境；不是我们那两个 π0.5 输入适配文件新加的。** 适配只改了 `libero_policy.py` 与 `libero_dataconfig.py` 的腕图/state输入路径。官方 OFT 的 action-level 奖励也要累加至 chunk 末 done，所以“先 +1 后 −1 抵消”不能简单归因于本轮改成 chunk-level。另一方面，把 π0.5 原有 C5 改为适应 WM 的 C8，确实延长一次反馈之间的执行长度，是另外一个控制协议差异。[本地两文件补丁](../../../local_scripts/wan_goal_20260930/pi05/apply_interface.py)、[官方 Wan 配置](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/examples/embodiment/config/wan_libero_goal_grpo_openvlaoft.yaml)、[完整方法审计](../audit_20261003/official_method_audit.md)

## 6. 空 mask 是否还更新；micro 128→64 保证什么

必须区分一个 microbatch 为零、一个 global batch 全零、整轮全零：

- 某个 microbatch 全零，其本次策略梯度为零；同一 global batch 的其他 microbatch、其他 rank 仍可能有梯度。
- 当前源码没有“global 全零就跳过 optimizer”的分支；完成积累后仍调 AdamW。对于有零梯度张量的已训练参数，旧动量和 weight decay 仍可能改变参数；不能把“本批没有新奖励方向”说成“权重一定完全没变”。未参与图且 grad=None 的参数另当别论。
- 每个 runner 轮末 scheduler 仍走一步。但本轮记录的 LR 始终是 5e−6，不能说空轮已经造成学习率下降。第4/5/6轮总 mask 为零，只证明没有新策略梯度信号，不提供逐参数变化量。

[optimizer 调用与 scheduler 位置](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L649-L675)、[有限梯度即执行 optimizer_step](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py#L456-L490)

若只把 micro 128 改为 64，global=2048、actor world_size=4 保持，单 rank 每次 global 更新还是 512 个 chunk 槽位，积累次数由 4 改为 8；每轮还是 20,480/2,048=10 次 optimizer 更新。**这会降低一次前向尺寸，不增加轨迹数、不增加有效奖励，也不改变组过滤。**

当前 loss 是“名义槽位均值×预先确定的轨迹长度权重”，而非每 micro 重新除以其有效 mask 数。因此在相同 global 样本、相同前向值、固定优势/长度、等大小切分、正确积累与相同跨 rank 归约下：

$$
\frac14\sum_{j=1}^{4}\operatorname{mean}_{128}(x_j)
=\frac18\sum_{j=1}^{8}\operatorname{mean}_{64}(x_j)
=\operatorname{mean}_{512}(x).
$$

这里 x 已含 mask 和长度权重。不存在仅因 micro 内 mask 密度不同而另换分母的问题；该结论来自当前源码，不能泛化到任意 `masked_mean` 实现。实际浮点求和次序、CUDA算子选择或随机路径不保证逐位一致；若将来要验证，可在同一固定 batch 上比较累计梯度/一次参数更新，而非仅看 loss 日志。[积累次数](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L91-L101)、[按积累次数缩放后 backward](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L830-L838)

日志则未必可直接跨 micro 设置比较：`ratio/clip/approx_kl` 是每 micro 的有效 mask 均值再等权平均，空 micro 返回 0；假设 128 槽只有前64有有效 ratio=1，micro128 日志是1，切成两个64后日志为(1+0)/2=.5，真实有效 ratio 仍为1。`actor/total_loss` 还在除 accumulation 后记录，改 accumulation 会改变该日志尺度。需要跨设置比较时，应以全局有效元素的 sum/count 和统一归约的 objective 为准。此处仅解释，没有改日志或实验。[metric 均值](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/algorithms/losses.py#L295-L323)、[空 mask 返回0](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/utils/utils.py#L369-L376)

## 7. 推盘为何特别差：目前更有依据的候选顺序

已确认的是结果集中性：推盘 48/50→0/50→6/50；其余九任务合计 380→386→398。以下是按“机制与现有事实的贴合程度”排序的研究候选，不是已测出的原因概率。

|优先级|候选机制|现有依据|尚不能推出|
|---|---|---|---|
|1|**推盘接触动力学在 WM 中失真，叠加策略分布偏移**|推盘依赖接触位置、持续侧向作用、盘与台面相对运动；视频外观合理不保证动作导致的接触运动正确。当前冻结 WM 未随 π0.5 再对齐，退化高度任务集中|尚未把同动作的真实/WM 接触片段配对，不能声称已经看到“盘无接触移动”|
|2|**该任务的奖励判断与真正达标区域不一致，或成功时序闪烁**|奖励只看生成主图；binary差分与any-frame done的事件口径不同。已有负例检查不含成功终态或生成帧|不能由零初态分数推导 RM 失效，也不能由 proxy 上升断定奖励投机|
|3|**单图训练与双图部署、C8 与 C5 的输入/反馈差异**|协议差异已确认；持续推/精细位置修正可能比部分抓放动作更受反馈间隔影响，单图也可能损失接触可见性|同协议真实评估保证原始/CP比较有效；协议差异单独不能解释“为什么训练后才下降”，需要通过权重适应与迁移来解释；也不能证明腕图是唯一原因|
|4|**多任务共享参数发生负迁移，且有效组贡献不均**|统一动作专家、没有初始策略KL或SFT锚定；其他九任务改善而一个任务崩溃与此机制相容。512名义轨迹不等于每任务相近数量的有效比较|推盘reset库存79/742并不少；没有每任务有效组/梯度统计，不能说是漏采或已经证明遗忘|
|5|**稀疏终局信用分配及“成功后不再训练”的保持性缺口**|单一轨迹优势作用于所有成功前动作，不能辨别接触关键动作；首次done后无保持目标。真实曾成功但最终失败的数量44→100→58|可解释保持性变弱的一部分；推盘的success_once本身已降到0/6，不能仅靠“成功后乱动”解释其整体退化|
|6|**通用数值/动作接口错误**|应保留核验动作归一化、坐标/夹爪语义、old/new logprob的可能性|当前核心源与固定上游一致、梯度有限、九任务并未普遍垮塌，没有证据把全局数值爆炸、任务ID错位或OOM认作CP40/80退化根因|

这里第一项是可检验的物理机制推断。WMPO 作者也展示过 WM 无法捕捉细小接触扰动导致卡住的失败案例；它支持“接触误差是视频 WM 的具体风险”，不是我们的推盘已被该论文解释。[WMPO §8](https://arxiv.org/html/2511.09515v1#S8)

已有检查还缩小了几项范围：推盘库存是49 initial+30 KIR；reward任务字符串映射到内部ID2，而物理评估列表ID5属于不同命名空间，本身无误；742个实际reset图全部输出0，但没有正例，不能评价召回或生成图误判。[本轮奖励核验](../audit_20261003/reward_audit.md)

## 8. 现有证据能回答到哪一步

可以确定：低有效 mask 是真实记录；其包含合理终止截断和无组内对比两部分；我们继承了固定 RLinf 的“置零但不补采”，与 WMPO 的补满有效 batch 不同；奖励的 any-frame 成功和 chunk 末相对回报可能不一致；micro 变小解决的是单次前向容量，不解决稀疏信号。

尚缺的关键量只有几类：每任务的组回报分布/保留组数、过滤前有效长度、首done chunk的any成功与末帧分数、RM在真实成功图和生成图上的正负错误。拿到这些才能区分“策略在WM里确实难成功”“容易/困难起点造成组内无差别”“奖励漏判/闪烁”“只有少数任务在学”。继续多跑轮数或直接取消mask，都不能替代这一层区分。本轮停留在讨论，未追加实验。
