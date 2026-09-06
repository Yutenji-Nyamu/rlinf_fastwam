# RLT × DVAC：SAC 信号挂点与少数据实验

日期：2026-08-27  
范围：解释当前方法、RLT 的真实数据循环、双卡/单卡数据差异，以及怎样只减少 Stage 2 在线采集。本文是讨论稿，不批准或启动新训练。

## 1. 先给结论

1. 当前 success-episode DVAC-BC 的**算法改动很窄**：失败/普通 episode 仍模仿冻结 $\pi_0$；成功 episode 改为模仿 replay 中实际执行成功的动作，并仅在 C10 内用均值为 1 的 DVAC 权重重新分配 BC 梯度。actor-Q、critic TD、rollout action 和奖励都不改。
2. 文献中额外信号主要进入五个不同位置：replay 采样、critic TD、actor-Q、BC/自模仿、探索。信号方向取决于挂点：**高 TD/Q 不确定性通常降权学习，高 epistemic uncertainty 可用于探索，高 return/advantage 才常用于增强模仿。**
3. 当前不是“每轮 4 条数据”。matched-width 单卡每个 runner cycle 收集 **8 个完整 episode**，每个 C10 policy query 形成一条 macro transition；实际约 90--160 条 replay row/cycle。
4. 当前 `update_epoch=5` 表示每新增 1 条 macro transition，累计安排约 5 次 critic update；`critic_actor_ratio=2` 表示约每 2 次 critic update 做 1 次 actor update。
5. 历史阶段曲线表明，只把 train env 从 8 减到 4、其余门槛全不动，会把 student 接管推迟约一倍；480 cycle 中真正的稳定在线阶段会被压得很短。
6. 若目标是“在线交互减半，但保持学习阶段和 optimizer 预算大致可比”，更合适的首个候选是：`env 8→4`、`warmup replay 20k→10k`、`UTD 5→10`；30k 启动更新、20k/50k actor warmup/ramp、critic:actor、batch 和 Stage 1 artifact 保持。

## 2. 当前方法到底改了什么

RLT Stage 2 的 actor loss 可简写为：

$$
L_{actor}=-\lambda_Q Q(s,\pi(s))
+\lambda_{BC}\operatorname{mean}_{q,h} e_{q,h}.
$$

其中 $e_{q,h}$ 是 student 第 $h$ 个动作与 BC target 的均方误差。当前方法只把 BC 项改成：

$$
L_{BC}=\operatorname{mean}_{q,h}\left[c_{q,h}e_{q,h}\right].
$$

- 普通/失败 episode：target 是 frozen $\pi_0$ reference，$c_{q,h}=1$。
- 成功 episode：target 是 replay executed action；$c_{q,h}$ 来自 frozen $\pi_0$ 的 DVAC，且 $\operatorname{mean}_h c_{q,h}=1$。
- $c$ 已 detach，因此不训练 $\pi_0$；它只改变每个 C10 位置返回 student actor 的 BC 梯度。
- 原始 $-Q$ actor 路径和 critic Bellman/TD loss完全不变。

所以数学主体就是“成功 target 切换 + 一次逐 $h$ 乘权”。工程代码看起来更大，是因为必须把 DVAC、episode-success、executed action 一起送进 replay，并记录 resume/telemetry；这些是数据管线，不是更多算法项。

当前语义可写成一句话：

> 已经知道整条 episode 最终成功时，让 student 模仿这段实际成功动作；在其 C10 内，对 frozen $\pi_0$ endpoint 改口更多的位置投入更多 BC 学习量，但不改变该 query 的平均 BC 强度。

“成功动作自模仿”有直接先例；“成功 C10 内高 teacher-DVAC 增权”仍是本项目要验证的组合假设。

## 3. 额外信号在 SAC / actor-critic 中通常怎样使用

| 挂点 | 典型信号与处理 | 代表工作 | 对当前项目的启发 |
|---|---|---|---|
| replay 采样 | TD error、target reliability 或当前 policy 密度决定哪些旧 transition 多抽 | [DisCor, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d7f426ccbc6db7e235c57958c21c5dfa-Abstract.html)、[ReaPER, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/60e3d7caf8cfdaf6598467f9b0ebf36a-Abstract-Conference.html)、[likelihood-free replay weighting, L4DC 2022](https://proceedings.mlr.press/v168/sinha22a.html) | 若以后要让成功 transition 被反复利用，独立 success buffer / 固定混合比例比直接改 critic loss更清楚。DVAC 本身不是 TD error。 |
| critic TD | 高 Q-target/OOD uncertainty通常**降权** Bellman loss | [UWAC, ICML 2021](https://proceedings.mlr.press/v139/wu21i.html)、[SUNRISE, ICML 2021](https://proceedings.mlr.press/v139/lee21g.html) | teacher endpoint DVAC 尚未证明能表示 TD target 是否可靠，因此当前不把它接入 critic 是合理的。 |
| actor / BC | return、success 或 critic advantage决定历史动作是否值得模仿 | [Self-Imitation Learning, ICML 2018](https://proceedings.mlr.press/v80/oh18b.html)、[CRR, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html)、[AWAC](https://awacrl.github.io/)、[OBAC, ICML 2024](https://proceedings.mlr.press/v235/luo24e.html) | 当前 success gate + executed-action BC 属于这一成熟路线；以后可把粗 success gate升级为 query-level Q/advantage gate。 |
| 机器人成功数据 | success buffer或高质量 RL 轨迹作为额外监督 | [FAC / RLFP, CoRL 2024 proceedings](https://proceedings.mlr.press/v270/ye25a.html) | 与“成功 episode 用 executed action 做 BC”最接近；它不直接支持 per-h DVAC 权重。 |
| 探索 | 高 epistemic uncertainty作为 UCB bonus，鼓励访问未知区域 | [SUNRISE](https://proceedings.mlr.press/v139/lee21g.html)、[BRO, NeurIPS 2024 Spotlight](https://openreview.net/forum?id=fu0xdh4aEJ) | 同一类 uncertainty 在探索中可增权，在 critic 学习中却可能降权；不能只看数值高低决定方向。 |
| 数据复用 / UTD | 少收集一次环境数据后，对 replay 做更多 critic update；靠 ensemble、归一化或正则稳定 Q | [REDQ, ICLR 2021](https://openreview.net/pdf/c4e6642e1eb45768ab5d7a3291c2a0088665e5c4.pdf)、[DroQ, ICLR 2022](https://openreview.net/forum?id=xCVJMsPv3RT)、[RLPD, ICML 2023](https://proceedings.mlr.press/v202/ball23a.html)、[CrossQ, ICLR 2024](https://openreview.net/pdf/750ae12418a1dc0f2dd3d9ff5ef5013234515fe6.pdf)、[SimbaV2, ICML 2025](https://proceedings.mlr.press/v267/lee25u.html) | “少数据”不等于少训 Q；高复用通常反而提高 critic UTD，但必须同时观察 Q bias、TD target 和稳定性。 |

最重要的对应关系是：

```text
成功 / advantage       → 这个历史动作值不值得模仿
DVAC                   → 已通过质量门后，C10 内学习量放在哪里
Q-target uncertainty   → 这次 critic backup 值不值得相信
epistemic uncertainty  → 是否值得主动探索
```

## 4. 当前 RLT 每轮数据到底怎样流动

```text
1个 runner cycle
  → 8个 train env 各跑1个完整 episode
  → student 每次输出 C10；每个 query 形成1条 macro transition
  → 实际约90--160条新 row 写入统一 replay（成功提前结束会变少）
  → replay 未到20k：先收集
  → replay 到20k：从池中随机抽 global batch 512
  → 每条新 row累计安排约5次 critic update
  → 约每2次 critic update做1次 actor update
  → target network / actor权重同步
  → 新 actor参与下一轮数据采集
```

几个容易混淆的词：

- **episode**：机器人从 reset 到成功或 200 primitive steps结束的一整条执行。
- **macro transition**：一次 C10 policy query及其环境结果；一条 episode最多约 20 条。
- **replay buffer**：保存历史 macro transition 的池；训练不是只用最新一轮，而是从整个窗口重复随机抽样。
- **UTD**：每新增一条环境 transition安排多少次更新。当前 critic UTD约为 5；它不表示“每个新样本恰好被抽 5 次”。
- **actor/critic**：critic学习整段 C10 的价值；actor用 $-Q+BC$ 更新 student，并在之后改变新的采集分布。

## 5. “深圳小 batch 更明显”现在能说明什么

它目前是有价值的线索，但还不是“样本效率已证明”：

- 小 batch 的 GRPO 配对实验中，DVAC 对 training rollout 曲线的分离更明显；
- fixed evaluation 的累计差仍很小；
- 那里修改的是 $\pi_0$ GRPO log-prob credit，而这里修改的是 RLT student 的成功 BC；
- 小 batch 还会提高梯度噪声，使 reweighting 更容易在训练曲线上显现，但不自动等价于“同样效果用了更少环境数据”。

真正证明数据利用率，需要把横轴改成**累计环境 macro transitions**，比较 control 和 DVAC 在相同 transition预算下的 fixed-reset success，而不是只比较相同 runner step。

## 6. 本轮低在线数据实验：减少交互、匹配更新

### 6.1 为什么不能只改 `env8→4`

历史成功 RLT 在约 cycle 135 收满 20k replay；只把 env 减半会把同一道门推到约 cycle 270。后续 30k 启动更新和 20k/50k actor warmup/ramp 又继续占用 cycle，最终 480 步只剩约 90--100 个稳定 student cycle。这个实验会同时改变“交互量”和“训练所处阶段”，不再只是减少数据。

### 6.2 当前最清楚的候选

```yaml
env:
  train:
    total_num_envs: 4  # 当前8 → 4

algorithm:
  update_epoch: 10        # 当前5 → 10
  rlt_schedule:
    warmup_min_size: 10000  # 当前20k → 10k
```

三项合起来的含义是：

- `env8→4`：每个 runner cycle 的新环境数据约减半；
- `warmup20k→10k`：仍在约 cycle 135 开始 learner，但 learner 只获得一半 initial unique replay；
- `UTD5→10`：每条新数据被重复利用约两倍，使每 cycle 的 critic/actor update 数接近历史 env8；
- 因此阶段边界仍大致保持在 135 / 155 / 192，而最终在线交互约减半、optimizer 预算接近原实验。

其余保持：

- 同一个 Stage 1 `global_step_2000` artifact；
- `runner.max_steps=480`、每 env 最多200 primitive steps、H50/C10；
- fixed eval仍为4 env × 5 epoch = 20 episodes，每25 cycle一次；
- global/micro batch仍为512/256；
- `warmup_post_collect_updates=30000`；
- replay仍为80k；
- `train_every_transitions=1`；
- `critic_actor_ratio=2`；
- `max_updates_per_train_step=1600`、actor warmup/ramp 20k/50k、优化器、Q/BC系数和DVAC公式均不动。

这正是在测“相同学习计算量下，能否用更少的新交互”；UTD 增大不是附带变化，而是把少一半的新数据多学几遍的核心操作。control 与 DVAC 使用同一套配置，二者差值仍只来自方法。

### 6.3 两个对照含义

- `env4 + warmup10k + UTD5`：交互和总更新都约减半，回答“整个系统预算一起缩小时谁更抗退化”。
- `env8 + max_steps240`：每轮拓扑不变、只缩短训练总时长，回答“较早停止时谁更好”。

首个候选选 matched-update 版本，是因为它最贴合“提升已有数据利用率”这个问题。

## 7. 双卡与当前单卡的数据行为差异

| 项目 | 历史双卡 | 当前 matched-width 单卡 | 是否保持高层总量 |
|---|---:|---:|---|
| train episode / cycle | 2 rank × 4 = 8 | 1 rank × 8 = 8 | 是 |
| replay | 两个 rank-local pool，各自采本地4 env | 一个统一 pool，混合8 env | 总数据接近，混合方式不同 |
| global batch | 512 | 512 | 是 |
| 一波并行样本 | 2 GPU × 128 | 1 GPU × 256 | 是 |
| 梯度累积 | 每 rank 2轮，再all-reduce | 单 rank 2轮 | 目标batch相同，浮点/随机流不同 |
| warmup | 10k/rank，机器级约20k | 单pool 20k | 是 |
| replay窗口 | 历史最多2×50k | 单pool 80k | 本次480预算内通常都不截断 |

所以双卡→单卡没有把全局 episode 数或 global batch减半。真正变化的是：两份本地 replay变成一个统一 replay、随机流和采样混合改变、rollout/计算的进程拓扑改变。它们会让曲线不逐点复现，但不代表单卡少收了一半数据。

## 8. 我建议的实验顺序

1. 先用现有日志把横轴从 runner step换成累计 macro transition，重画 control / DVAC 的 fixed eval与train success；这一步不训练，就能先看当前方法在早期数据区是否更有优势。
2. 第一组低数据 A/B：两边共同使用`env4 + warmup10k + UTD10`，保持30k启动更新、20k/50k ramp、ratio2和其余matched-width配置。它回答“同等学习预算下，DVAC能否更有效利用一半在线数据”。
3. 若还要分开看“少数据”和“多复用”，再做`env4 + warmup10k + UTD5`。
4. 最后再独立测试 `critic_actor_ratio=4`，不要与第一次减数据同时混在一起。

最低必要观测：累计unique transitions、replay size/age、每 cycle critic/actor update数、成功 episode和成功 replay row比例、Q均值/极值、TD target/loss、actor Q/BC loss、DVAC权重分布，以及fixed-reset success。

这一路线比直接“8 env→4 env，同时随手改 actor/Q”更容易回答因果问题，也保留了当前方法最简单的高层语义。

## 9. Stage 1 是什么，为什么本轮不减

RLT确有两个训练阶段，但它们的数据性质不同：

```text
Stage 1：离线clean-50成功示范
  → frozen π0提取image prefix
  → 只训练RLT encoder/decoder
  → 压成z_rl并重建原prefix

Stage 2：在线RoboTwin交互
  → z_rl + reference action + robot state
  → student C10、replay、twin-Q、actor-Q+BC
```

现有Stage 1使用50个成功episode、7,188 frames、global batch 32、2,000 optimizer steps；累计64,000次sample presentation，约相当于把数据看8.9遍。两次独立完整曲线都显示约1,000步后进入慢收益区：AutoDL 的100步窗口 loss 从`3.902 → 0.590 → 0.559 → 0.555`（步骤1--100 / 901--1000 / 1401--1500 / 1901--2000），深圳复跑为`3.258 → 0.582 → 0.553 → 0.549`。

详细证据见[Stage 1正式训练结果](../rlinf-robotwin-pi0-rltoken/03_STAGE1_FORMAL_TRAINING_20260729.md)。

Stage 1完整运行只花过约29分钟（深圳复跑约13分钟），而且 artifact 会被所有 Stage 2 实验反复复用，所以它不是当前长训练的主要耗时。未来若专门压缩它，`1,250--1,500`是有曲线依据的候选；当前低数据实验直接复用已验收的2,000步 artifact即可。

## 10. “epistemic uncertainty用于探索”到底是什么意思

先分清三类量：

- **SAC actor entropy**：一个随机策略在当前状态下故意有多分散，属于行为随机性；标准SAC通过`alpha * log pi`鼓励探索，不等于“模型知识不足”。[SAC, ICML 2018](https://proceedings.mlr.press/v80/haarnoja18b.html)
- **epistemic uncertainty**：因为数据不足，不同模型/后验样本对同一输入产生分歧；更多相关数据后原则上会下降。
- **DVAC**：冻结π0在同一次flow去噪链中对clean endpoint改口多少。它是teacher-side denoising/predictive instability，不是标准actor entropy，也不是actor或Q ensemble的epistemic variance。

“用于探索奖励”的典型做法是：多个dynamics model对下一个状态预测越不一致，就给越大的intrinsic reward，策略因此主动去未知区域收新数据。[Disagreement, ICML 2019](https://proceedings.mlr.press/v97/pathak19a.html)

Q uncertainty也不是简单的“坏信号”：

- 在**采集端**，OAC用Q上置信界、SUNRISE用UCB选动作，高Q分歧会吸引探索。[OAC, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/a34bacf839b923770b2c360eefa26748-Abstract.html)、[SUNRISE, ICML 2021](https://proceedings.mlr.press/v139/lee21g.html)
- 在**学习端**，同样的Q/OOD不确定性表示Bellman target或actor-Q依据不可靠，UWAC和SUNRISE会降权。[UWAC, ICML 2021](https://proceedings.mlr.press/v139/wu21i.html)

直接使用actor epistemic uncertainty的工作通常需要多个actor或Bayesian/dropout actor，并用其分歧/后验采样产生多样轨迹，例如TEEN的policy ensemble和ReLU-to-the-Rescue的dropout Thompson sampling；它们不是给既有BC项乘一个DVAC权重。[TEEN, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/10cb15f4559b3d578b7f24966d48a137-Abstract-Conference.html)、[ReLU to the Rescue, ICML 2024](https://proceedings.mlr.press/v235/jesson24a.html)

与teacher/student最接近的[AC-Teach, CoRL 2020](https://proceedings.mlr.press/v100/kurenkov20a.html)也不是按teacher自身生成不稳定性加BC：它用Bayesian critic对student和多个teacher建议动作的outcome做后验比较，再以Thompson sampling决定执行谁。近期[ERSAC, ICML 2023](https://proceedings.mlr.press/v202/o-donoghue23a.html)把epistemic uncertainty写进risk-seeking探索目标，[KEA, ICML 2025](https://proceedings.mlr.press/v267/yang25u.html)则用novelty协调SAC随机探索；共同点仍是改变未来采集，而不是把同一条历史target机械学得更重。

当前RLT student虽然代码上构造Normal分布，但std固定为0.002、entropy alpha关闭，因此没有可学习的state-dependent actor uncertainty。当前方法最准确的定位是：

> success-conditioned、teacher-instability-guided的C10内BC credit重分配；目标是提高已获得成功数据的利用率，不是主动探索。

## 11. RLT/SAC里有没有 advantage

通用定义是：

$$
A(s,a)=Q(s,a)-V(s).
$$

它表示“在状态$s$选动作$a$，比该状态下策略通常能做到的水平好多少”。SAC的soft value还把entropy项计入$V$。

但当前RLT实现**没有显式advantage tensor**。它直接做：

```text
replay C10 → twin-Q TD学习整段价值
student当前C10 → critic给一个scalar Q
actor loss = -lambda_Q * Q + lambda_BC * BC
```

critic把完整C10×14 flatten后只输出两个chunk级Q，再取Q1用于actor；没有十个独立的$Q_h$。因此可以额外定义：

代码入口分别是[actor loss与critic调用](../../tmp/rlt_dvac_impl_source/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py)和[C10 flatten、固定std actor与scalar twin-Q](../../tmp/rlt_dvac_impl_source/rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py)。

$$
A_{exec}(s)=\min Q(s,a_{exec})-\min Q(s,\pi(s)),
$$

或把$a_{exec}$换成$\pi_0$ reference。这是从Q得到的**query/chunk级**质量差：整段executed/reference C10是否比当前student C10更值钱。它不能直接告诉我们十个$h$里哪一格更关键；要得到per-h outcome influence，需要hybrid-action counterfactual Q或分析$\partial Q/\partial a_h$。

所以当前两层职责是：episode success先做粗粒度quality gate，DVAC再在已通过的成功C10内部做per-h分配。以后若把success gate细化为$A_{exec}>0$，就是用chunk advantage决定“这段是否值得模仿”，再让DVAC决定“十格里把BC学习量放在哪里”。CRR正是用critic估计的advantage筛选或指数加权BC的代表路线。[CRR, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html)

## 12. Fixed eval：样本数与同时并发要分开

当前`4 env × 5 epoch = 20`条fixed-reset评估的主要问题是统计颗粒较粗：一个episode就改变5个百分点。32条时一个episode改变3.125个百分点，最坏情况的二项标准误也从约11.2个百分点降到8.8个百分点。

当前并不是`4 × 1`，而是`4 env × 5 epoch = 20`。改成`4 × 2`只会得到8条，统计反而更粗。

`4 × 8`不是历史实验验证出的唯一最优并发，而是一个直接换算：保持已经跑通的4个同时环境，只把串行波数从5增到8，得到32条。2026-08-27 18:00现场两条训练合计RAM已`235.6/240 GiB`、峰值`236.1 GiB`，而显存仅约`24.7/18.1 GiB`；当前限制是共享容器主存，不是显存。因此不应在正在并跑的两条训练中临时切成每任务`32 × 1`。

若目的是把fixed eval从20条提高到32条，首选：

```yaml
env:
  eval:
    total_num_envs: 4
    rollout_epoch: 8
    seed_path: <精确32-ID bank>
```

即`4 × 8 = 32`：统计样本变多，但同时常驻环境仍是4个。若当前pair结束后做独立吞吐评估，`8 × 4`、`16 × 2`和`32 × 1`都可以作为候选；过去没有做过AutoDL RLT `32 × 1`实测，因此不能把它说成不可行。对A/B最重要的是两边使用同一组精确32-ID bank并保存per-reset结果。

## 13. Stage 2的四段时间线与历史收敛证据

`20k warmup replay`和`30k warmup critic updates`是先后两道门，不是50k条数据：

| 阶段 | 触发条件 | 环境由谁执行 | learner做什么 | 历史env8 | 只改env4的估计 |
|---|---|---|---|---:|---:|
| Stage2-A 纯采集 | replay `<20k` | frozen π0 reference | critic/actor均不更新 | cycle 1--135 | 1--267/272 |
| Stage2-B 启动消化 | replay达到20k，critic update `<30k` | 仍是reference | 每cycle最多1600 critic、约800 actor | 136--154 | 约268--286/291 |
| Stage2-C student接管并ramp | critic update 30k--70k | student | BC系数下降、Q系数上升 | 155--191 | 约287--378/390 |
| Stage2-D 稳态在线 | critic update `>=70k` | student | 稳定系数，闭环采集+replay更新 | 192--480 | 约379/391--480 |

每个runner cycle的变化也不是全程同一个“减半”：

- Stage2-A：episode从8降到4，macro row约从90--160降到45--80，更新两边都是0。
- Stage2-B：达到 replay 门后，固定30k启动任务按每cycle最多1600 critic/800 actor追赶。
- Stage2-C/D：student开始闭环采集，actor由BC主导逐步转向Q主导。

历史fixed20显示，Stage2-B并未形成行为能力：g150仅`1/20`。真正快速成形发生在Stage2-C/D交界：g175=`6/20`、g200=`16/20`、g250=`18/20`，之后长期维持80%--100%。BC loss在Stage2-B已很快下降，但Q作用和行为成功率仍未成熟，因此不能只凭BC loss就删除B/C。

历史还跑过一次把三道门都大幅压短的100-cycle pilot：student在cycle27接管、cycle52完成ramp，但fixed4在60/70/80/90/100为`1/4、2/4、0/4、0/4、0/4`。它不是单因素消融，却说明“三段一起砍短”没有形成稳定策略。由此，本轮更合理的是减少unique data、增加复用，保留B/C的optimizer进度，而不是直接删掉B/C。

## 14. DACER、DIME与近期外部信号接口

发散构思阶段先看“内部或外部信号能接到哪些训练接口”，不以术语归类提前排除方案。

### 14.1 DACER与DIME具体怎样计算

**DACER（NeurIPS 2024）**对同一个状态从diffusion actor采约200个动作，用3分量GMM拟合，再以“混合模式熵 + 每个高斯内部熵”近似动作分布熵：

$$
\widehat H_s=-\sum_k w_k\log w_k+
\sum_k w_k\frac{1}{2}\log\left((2\pi e)^d|\Sigma_k|\right).
$$

它用目标熵反馈调节噪声系数；策略太集中就增大最终动作的探索噪声，已经足够分散就减小。actor主体仍最大化Q，评估时关闭额外噪声。高层上，它把“同一状态能生成多少多样化动作”变成探索强度。

**DIME（ICML 2025）**不额外采200次，也不拟合GMM。它沿正常diffusion生成路径累计正向加噪路径与反向生成路径的log-ratio，得到最终动作熵的可计算下界$\ell_H(s)$，然后直接进入：

$$
L_{actor}=\mathbb E[-Q(s,a)-\alpha\ell_H(s)],
$$

以及带熵的Bellman target。高层上，它让actor同时追求高Q与动作分布多样性；信号不只控制采集噪声，而是进入actor和critic目标。

两者给我们的共同启发是：生成式policy内部统计既可以调探索，也可以进入actor-critic目标；DVAC除了当前BC分配外，也可发散考虑replay、探索或Q信任度接口。

### 14.2 与当前RLT切口最接近的近期工作

| 工作 | 信号接到哪里 | 对当前项目的直接启发 |
|---|---|---|
| [VACO, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17ab5964d90f38b174a068646c958789-Abstract-Conference.html) | 元网络产生sample BC weight，外层用Q训练这个权重 | 与当前weighted BC挂点最接近；可把query质量与C10内分配分成两层 |
| [ACTIVE, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c06f788963f0ce069f5b2dbf83fe7822-Abstract-Conference.html) | V-ensemble进入critic；advantage指数加权模仿并做batch mean归一 | 支持`query质量 u_q × mean-one DVAC c_qh`的结构 |
| [ACE, ICML 2024 Oral](https://proceedings.mlr.press/v235/ji24b.html) | 每个action dimension对reward的因果权重进入SAC entropy | 说明continuous actor可以接收细粒度外部信号；它选择在探索端使用 |
| [RLPD, ICML 2023](https://proceedings.mlr.press/v202/ball23a.html) | 每个batch固定混合online与prior/demo数据，配合高UTD | 少在线数据时，可通过success/high-DVAC子池提高成功数据被抽中的次数 |
| [ReLo, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/48726631f87322012c6be38e00c72a47-Abstract-Conference.html) | 用可约减loss而非raw TD error设replay priority | 可把DVAC与“重复学习后loss是否真的下降”组合，区分有信息样本与纯噪声 |
| [BAC, ICML 2024](https://proceedings.mlr.press/v235/ji24d.html) | replay中的历史高价值动作进入Bellman target | 成功经验也能通过critic传播，不只通过BC |

由此得到一条很清楚的机制链：

```text
success / Q质量 u_q       → 这条target是否值得学
DVAC c_qh                 → 成功C10内部哪里多学
success replay mix        → 这条成功数据被重复抽到多少次
critic uncertainty        → Q分支应当信多少
```

若下一步仍坚持最小改动，优先级可以是：先做本轮low-interaction matched-update A/B；再考虑RLPD式的固定success replay混采。后者直接改变成功数据的复用次数，而当前`[0,2]`只改变一条样本已经被抽中后，C10内部的BC分配。

### 14.3 其他相邻线索

近期工作还包括：

1. **Flow/VLA epistemic uncertainty：跨模型分歧。** 2026预印本[UQ-VLA](https://arxiv.org/abs/2606.18043)用小型flow-policy ensemble在同一条件、同一flow time比较velocity field disagreement（VFD），用于失败检测和主动挑选task/state请求专家数据；其SAVE在LIBERO上至少减少22%专家样本。它是目前与π0 flow最接近的工作，但尚非已录用会议论文。
2. **生成policy自身的随机性：运行时筛选或告警。** [FIPER, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0b7cb3b8cc44e652761245537027db44-Abstract-Conference.html)把多次生成action chunk的entropy与视觉OOD结合做失败预警；[KDPE, CoRL 2025](https://proceedings.mlr.press/v305/rosasco25a.html)对多条diffusion trajectory做流形KDE并选高密度轨迹。两者都没有把“不确定动作”直接学得更重。
3. **内部表征到outcome。** [SAFE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/392d0d05e2f514063e6ce6f8b370834c-Abstract-Conference.html)用成功/失败rollout训练VLA内部特征上的失败检测器，并在OpenVLA、π0、π0-FAST上验证。这提示“内部信号是否有用”最好先由outcome校准，而不只看信号幅度。
4. **SAC/diffusion actor目标。** [DIME, ICML 2025](https://proceedings.mlr.press/v267/celik25a.html)解决diffusion policy的entropy难算问题，把它纳入最大熵actor-critic；它研究的是策略采样多样性，不是知识不足。[ACTIVE, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c06f788963f0ce069f5b2dbf83fe7822-Abstract-Conference.html)则用V-ensemble的不确定性抑制value误差并自适应调整imitation约束，信号放在critic/整体BC强度，而非per-h teacher instability。
5. **较新的diffusion actor-critic更强调“分工”。** [DACER, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6174c67b136621f3f2e4a6b1d3286f6b-Abstract-Conference.html)估计diffusion policy entropy并调输出噪声来控制探索；[Entropy-regularized Diffusion Policy with Q-Ensembles, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b319c0e8092ba726cb22e718d7d9a95e-Abstract-Conference.html)用policy entropy负责动作多样性、Q ensemble lower-confidence bound负责价值不确定性。两者都说明actor随机性与critic epistemic应分开处理。
6. **多模态会让“分歧=无知”失效。** [Diff-DAgger, ICRA 2025](https://arxiv.org/abs/2410.14868)指出，多个合理动作mode会让ensemble action disagreement在分布内状态也很高，因此改用diffusion训练目标的异常度触发专家请求。它直接提醒我们：高DVAC也可能来自任务本身多解，不能只由其绝对值决定模仿强弱。

发散方案时，可以把DVAC先当作一个可用的predictive-uncertainty候选信号，分别试四种用途：成功BC分配、success replay复用、探索强度和Q信任度。等某个接口显出效果，再进一步分析它主要来自数据不足、动作多解、horizon位置还是flow路径；不在构思阶段用命名问题提前收窄方案。

## 15. 方法层收束：近期工作怎样把外部信号接入actor-critic

本节暂不讨论缩减在线数据，只回答一个问题：像DVAC这样的额外信号，在RLT这类actor-critic里有哪些有论文依据的切口。

### 15.1 先把三个训练旋钮分开

RLT的BC部分可以写成：

$$
e_{q,h}=\frac1D\left\|a^{student}_{q,h}-a^{target}_{q,h}\right\|_2^2,
$$

$$
L_{BC}=\lambda_{BC}\,\mathbb E_q\left[u_q\frac1C\sum_{h=0}^{C-1}c_{q,h}e_{q,h}\right].
$$

- $u_q$：这整条query/chunk target值不值得模仿；适合由success、return、Q advantage或target-vs-student Q决定。
- $c_{q,h}$：已经决定要模仿后，C10内部哪些future action多学；这里对应DVAC。
- $\lambda_{BC}$：整条BC相对actor的$-Q$项总体有多强。

近期工作的共同结构不是把这三个量混成一个倍率，而是分别处理“样本质量”“局部分配”和“总体正则强度”。当前success-episode DVAC-BC正对应最简单的两层实现：success选择target，mean-one DVAC在成功C10内部重新分配。

### 15.2 VACO：学习“哪条BC梯度真正有助于提高Q”

[VACO（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17ab5964d90f38b174a068646c958789-Abstract-Conference.html)不是直接令“Q越高，BC权重越大”。它训练一个小型meta-scoring网络：

$$
u_q=f_\alpha(s_q,a_q,Q(s_q,a_q)),
$$

并把它放进sample级weighted BC：

$$
L_{BC}^{w}=\mathbb E_q\left[u_q\left\|\pi(s_q)-a_q\right\|^2\right].
$$

随后做双层优化：内层用该权重更新一次actor；外层观察更新后的actor能否得到更高Q，再反过来训练scoring网络。直觉是：它学习的不是“这个样本看起来价值高不高”，而是“沿着这个样本的模仿梯度走一步，对policy improvement有没有帮助”。论文还发现学到的权重与raw Q仅弱相关。

对RLT的直接意义：VACO最适合提供整条C10的$u_q$。RLT critic只给联合C10一个标量Q，所以它不能直接产生十个独立的$c_{q,h}$；DVAC正好补上chunk内部的粒度。首版无需照搬meta-gradient，但“query quality × per-h DVAC”这层分工有直接依据。

### 15.3 ACTIVE：critic可信度、target质量和BC总强度分别控制

[ACTIVE（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c06f788963f0ce069f5b2dbf83fe7822-Abstract-Conference.html)训练多个独立的$V_i(s)$：

1. critic bootstrap使用ensemble的保守统计，降低value传播中的估计误差；
2. 用$Q(s,a)$减去$V$ ensemble的一个分位数，形成sample级advantage；
3. 用$u_q=\exp(A_q)$加权actor imitation，并除以batch mean保持数值尺度；
4. 另用dual update自动调节BC总体强度$\beta$。

所以ACTIVE中的“信号”不是单一ensemble方差。ensemble先影响critic可信度和advantage基线；真正进入weighted imitation的是$Q-V$，而总体模仿强度由另一个变量控制。

对RLT最直接的移植形式是：

$$
u_q=\operatorname{normalize}\exp\left(Q(s_q,a_q^{target})-V_c(s_q)\right),
$$

$$
L_{BC}=\lambda_{BC}\,u_q\frac1C\sum_h c_{q,h}^{DVAC}e_{q,h}.
$$

也就是：$u_q$回答“target值不值得学”，$c_{q,h}$回答“C10里面哪里多学”，$\lambda_{BC}$回答“这批BC整体多强”。

### 15.4 2024--2026最相关的方法地图

| 工作 | 使用的信号 | 插入训练哪里 | 对RLT-DVAC的意义 |
|---|---|---|---|
| [FQL, ICML 2025](https://proceedings.mlr.press/v267/park25f.html) | flow teacher target与critic Q | one-step student使用`-Q + distillation MSE` | 与RLT骨架最接近；支持在MSE按$h$求和前加DVAC，而保持scalar-Q路径不动 |
| [GFP, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33f131b806a93d376cf0ce1a456464bb-Abstract-Conference.html) | 同状态下dataset target与student proposal的Q差 | sigmoid有界权重乘flow BC | 最直接的下一层$query gate$：target比student好才多模仿；再由DVAC分配C10 |
| [VACO, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17ab5964d90f38b174a068646c958789-Abstract-Conference.html) | state、action、Q，经meta-objective学出sample usefulness | sample BC weight | 说明外部信号调BC是成熟切口；更适合整条query质量 |
| [ACTIVE, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c06f788963f0ce069f5b2dbf83fe7822-Abstract-Conference.html) | V ensemble与$Q-V$ advantage | conservative critic target、weighted imitation、adaptive BC strength | 支持把critic confidence、query weight和BC总强度拆开 |
| [QIPO, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2d9c6cdb4cfe93869c090fea7375044b-Abstract-Conference.html) | 同状态候选action的Q | softmax$(\beta Q)$权重乘flow/diffusion regression | 说明Q可以直接重排生成模型训练样本；其权重仍是action/chunk级，不提供RLT的per-h Q |
| [QVPO, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6111371a868af8dcfba0f96ad9e25ae3-Abstract-Conference.html) | 单调Q权重 | Q-weighted diffusion VLB/noise regression | 与QIPO一起支持“value决定生成target学多少”，但训练的是生成actor本体 |
| [AC3, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38937) | trajectory success | actor只从成功轨迹更新；critic仍学习全部经验 | 与当前successful-episode executed-action BC最接近的机器人action-chunk依据；它的success gate比当前修改更强 |
| [ACE, ICML 2024 Oral](https://proceedings.mlr.press/v235/ji24b.html) | 每个action dimension对reward的因果强度 | 逐action-dimension调整SAC entropy | 说明连续actor可以接细粒度结构信号；其切口是探索，不是BC |
| [ReaPER, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/60e3d7caf8cfdaf6598467f9b0ebf36a-Abstract-Conference.html) | TD error × 后续TD误差估计的target reliability | replay priority | 若DVAC用于replay，应把“有信息”与“当前可可靠学习”结合，而不是只按高DVAC多抽 |
| [PGR, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/74b7956113fdf0ec87288f351a1d8a34-Abstract-Conference.html) | Q/return、TD error或curiosity relevance | 条件生成高relevance transition补入replay | 表明外部信号也可改变数据出现频率；这是比BC更大的独立分支 |
| [DAC, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bca7a9a0dd85e2a68420e5cae27eccfb-Abstract-Conference.html) | Q ensemble lower-confidence bound | critic target与diffusion actor的Q guidance | critic uncertainty通常由critic ensemble进入Q路径，不必与teacher DVAC共用同一量 |
| [DPMD/SDAC, ICML 2025](https://proceedings.mlr.press/v267/ma25d.html) | value/目标policy分布产生的importance weight | reweighted score-matching loss | 直接证明在线diffusion actor可通过重加权生成loss做policy improvement；若未来训练π0本体，比当前student BC更接近 |

这些工作可归成五条清楚的数据流：

```text
success / Q advantage / feasibility → target或query是否值得学 → BC/actor sample weight
DVAC等局部结构信号              → target内部哪里多学      → per-h BC allocation
TD reliability / learnability       → 哪条transition多复用  → replay priority
Q/V ensemble uncertainty            → Bellman target信多少  → critic target/loss
policy entropy / causal action signal→ 去哪里增加探索       → collection noise/entropy
```

### 15.5 对当前RLT最清楚的下一层候选

当前实现先保留：

```text
episode success
  → successful episode使用executed action作BC target
  → mean-one DVAC在C10内分配十个MSE
  → 原始-Q actor路径和critic TD不变
```

若只增加一个方法组件，最贴近近期工作的候选是GFP式query质量门：

$$
\Delta Q_q=Q(s_q,a_q^{target})-Q(s_q,a_q^{student}),
$$

$$
u_q=\sigma\left(\frac{\Delta Q_q}{\eta Q_{scale}}\right).
$$

然后仍使用：

$$
L_{BC}=\lambda_{BC}\,u_q\frac1C\sum_hc_{q,h}^{DVAC}e_{q,h},
\qquad \frac1C\sum_hc_{q,h}^{DVAC}=1.
$$

它把方法问题拆成两个能单独判断的假设：

1. success/Q信号能否选出值得模仿的executed chunk；
2. 在同一条值得模仿的chunk内，DVAC能否指出更值得投入BC梯度的future action。

若以后要走更强分支，优先级分别是：AC3式success-buffer actor更新、ReaPER/ReLo式replay usefulness、ACE/DACER式探索控制、以及直接训练flow actor的QIPO/SDAC。它们改变的是不同训练接口，应分别做实验，而不是叠成一个总权重。

## 16. QIPO、QVPO、GFP、FQL与AC3：逐项对应RLT

### 16.1 QIPO与QVPO：Q直接选择生成式actor该拟合哪些动作

[QIPO（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2d9c6cdb4cfe93869c090fea7375044b-Abstract-Conference.html)是离线RL。对同一个state，从当前policy取得多个完整action候选，critic为每个候选给一个scalar Q，然后在该state内部做softmax：

$$
g_j=\frac{\exp(\beta Q(s,a_j))}{\sum_k\exp(\beta Q(s,a_k))}.
$$

$g_j$作为detached权重乘flow-matching regression；定期重新从更新后的policy采候选，再做下一轮小幅Q倾斜。这里的归一化发生在“同一state的多个完整action候选”之间，不是在一个action chunk的十个$h$之间。

[QVPO（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6111371a868af8dcfba0f96ad9e25ae3-Abstract-Conference.html)是在线off-policy版本：当前diffusion actor为state生成多个动作，twin-Q形成候选间advantage，负advantage置零，保留高价值候选来训练denoising regression；critic仍用replay做TD。它另外使用diffusion entropy regularization保持探索，并在采集时用Q从候选中选动作，降低高方差采样造成的浪费。

两者的共同数据流是：

```text
state → 生成多个完整action → scalar Q排序/加权
      → weighted flow/diffusion regression → 生成式actor更新
```

因此，如果未来直接解冻并RL微调$\pi_0$ flow actor，它们非常接近；当前RLT训练的是另一个one-step student，所以更自然的迁移是让Q产生整条C10的$query gate$，而不是把一个scalar Q拆成十个per-h Q。

### 16.2 GFP：Q不只加权BC，完整算法仍有critic与直接actor-Q更新

[GFP（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33f131b806a93d376cf0ce1a456464bb-Abstract-Conference.html)同时训练三个组件：

1. critic用TD学习Q；
2. one-step student使用`-Q + distillation`训练；
3. multi-step flow policy使用value-aware weighted BC训练。

第三项比较同一state下dataset target与student proposal：

$$
u_q=\sigma\left(\frac{Q(s,a^{target})-Q(s,a^{student})}{\eta Q_{scale}}\right).
$$

target比student好，$u_q$接近1，flow BC多学；target更差，$u_q$接近0，少学。Q在这里作为detached权重，不从BC loss反传；但在student的`-Q`项中，Q仍直接给actor梯度。所以“GFP本质只是BC、Q只负责加权”不准确：**它的flow分支是Q加权BC，完整算法仍是critic TD + student直接Q提升 + distillation。**

GFP与RLT都具有“多步生成teacher + one-step student + scalar critic”的骨架；主要区别是GFP联合训练flow teacher且主要是offline dataset action，RLT冻结预训练$\pi_0$，在线replay中普通样本模仿reference、成功episode可模仿executed action。

### 16.3 FQL：与RLT最接近的`-Q + teacher MSE`骨架

[FQL（ICML 2025）](https://proceedings.mlr.press/v267/park25f.html)先用dataset训练multi-step flow teacher，再训练one-step student：

$$
L_{actor}=-Q(s,a^{student})+alpha\left\|a^{student}-a^{flow}\right\|_2^2.
$$

第一项让student朝critic认为更好的动作移动；第二项让student留在flow teacher表达的数据支持附近。critic用student生成的next action做TD。部署时只跑one-step student，避免反传和推理都穿过完整flow链。

它没有DVAC式sample weight；价值在于确认RLT现有的结构分工是合理的：保留联合C10的`-Q`路径，把DVAC放到本来就能按$h$分解的teacher/executed-action MSE中。

### 16.4 AC3：成功轨迹决定actor学哪些数据，critic仍学全部replay

[AC3（AAAI 2026）](https://ojs.aaai.org/index.php/AAAI/article/view/38937)直接输出连续action chunk，并维护successful-trajectory buffer：

```text
全部replay        → twin-Q / critic TD
successful buffer → actor BC + actor Q objective
```

因此它不是“成功动作只多乘一点”，而是actor只在成功轨迹区域更新；critic仍利用成功与失败数据学习价值。它还用intra-chunk n-step return与chunk-aligned intrinsic reward稳定稀疏奖励下的critic。

它给当前实现提供的是success gate依据；它没有DVAC、没有per-h不确定性权重。当前RLT-DVAC-BC比AC3更弱：只在成功episode把BC target换成executed action并在C10内重分配，原始actor-Q仍从普通replay训练。

### 16.5 另外三条最接近的近期线索

| 工作 | 做法 | 对RLT的直接含义 |
|---|---|---|
| [DTQL, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/59a48c111f97f2174709ea9ed8e920d1-Abstract-Conference.html) | frozen/pure-BC diffusion prior + one-step actor；actor用`-Q + diffusion trust-region loss` | 与RLT/FQL同属双策略骨架；支持保留Q路径，把teacher信号放入独立正则项 |
| [VACO, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17ab5964d90f38b174a068646c958789-Abstract-Conference.html) | meta-network产生sample BC weight；外层用更新后的Q改善训练权重网络 | 可作为未来更复杂的$query usefulness$学习器，但首版不需要meta-gradient |
| [ACTIVE, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c06f788963f0ce069f5b2dbf83fe7822-Abstract-Conference.html) | V-ensemble构造保守advantage，加权imitation；另用dual变量调BC总强度 | 直接支持把$query质量$、C10内分配和总体BC强度分成三个旋钮 |

### 16.6 当前方法与最清楚的下一层

当前方法可写成：

$$
L_{actor}=-\lambda_QQ(s,\pi(s))
+\lambda_{BC}\frac1{10}\sum_{h=0}^{9}u_qc_{q,h}
\left\|a^{student}_{q,h}-a^{target}_{q,h}\right\|_2^2.
$$

- 当前$u_q$由episode success决定target语义；
- $c_{q,h}$是成功C10内mean-one DVAC分配；
- actor-Q、critic TD、普通reference BC和总体$\lambda_{BC}$不变。

若只增加一个组件，最有依据的是GFP式连续$query gate$：比较$Q(s,a^{target})$与$Q(s,a^{student})$，决定整条target值得模仿多少；DVAC继续只回答这条C10内部哪些$h$多学。QIPO/QVPO则留给未来直接训练$\pi_0$ flow actor的分支。

### 16.7 2026-08-27 20:07现场

matched-width control / DVAC-BC分别完整到Step `434/438`并继续运行；累计train success=`52.19/53.20%`，末5步=`95.0/85.0%`，末10步=`95.0/87.5%`；Step425 fixed20=`19/20 vs 17/20`。方法权重p05/mean/p95=`.746/1/1.252`、ESS=`.977`。两条fatal、CUDA OOM、worker crash、NCCL error与cgroup OOM/OOM-kill均为0；RAM当前/峰值=`230.81/236.40 GiB`，GPU峰值=`25.17/25.14 GiB`。共同Step433曲线与原始指标见[现场证据](evidence/rlt_success_bc_matched_width_live_refresh_20260827/README.md)。

## 17. 聚焦 action 粒度：论文边界、当前幅度与最小对照

### 17.1 “action 粒度”先分成两种

1. **物理动作坐标 $d$**：例如末端位置、旋转、夹爪等14个坐标。ICML 2024 Oral 的
   [ACE](https://proceedings.mlr.press/v235/ji24b.html)确实为不同动作坐标估计对reward的因果强度，再逐坐标调整
   SAC entropy；这是明确的action-dimension外部信号，但挂点是探索熵，不是BC。
2. **chunk内future位置 $h$**：同一次query预测的C10中第0--9个未来动作。这是当前DVAC所需粒度。检索的
   2023--2026主会工作中，没有找到“外部不确定性直接逐$h$乘deterministic chunk BC MSE，同时保持scalar-Q
   不动”的原样公式。

真正让value具有逐$h$ credit的工作，通常需要改变critic结构。例如
[CQN-AS（NeurIPS 2025）](https://papers.nips.cc/paper_files/paper/2025/hash/9eecce65f1f93d7a3e7ae677c31942ab-Abstract-Conference.html)
为action-sequence step与动作坐标建立对应Q输出；
[Q-Transformer（CoRL 2023）](https://proceedings.mlr.press/v229/chebotar23a.html)和
[ARSQ（ICML 2025）](https://proceedings.mlr.press/v267/liu25s.html)也通过因子化或自回归critic取得逐动作坐标的
value credit。它们说明：若要让Q直接判断十个$h$各自值不值，需要结构化critic；不能由当前一个联合scalar-Q
自然得到。

与当前最接近的主线反而都保留整chunk Q：

| 工作 | 信号粒度 | 实际挂点 | 对当前RLT的意义 |
|---|---|---|---|
| [FQL，ICML 2025](https://proceedings.mlr.press/v267/park25f.html) | 整个action | one-step student的联合`-Q`与uniform teacher MSE | 与RLT骨架最接近；MSE可展开到$h$，但论文没有逐$h$权重 |
| [Q-Chunking，NeurIPS 2025](https://openreview.net/pdf?id=XUks1Y96NR) | 整个chunk | scalar chunk-Q与uniform distillation | 主流chunk actor-critic仍把chunk当一个RL action |
| [AC3，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38937) | 成功trajectory/chunk | actor只从成功轨迹学，critic仍用全部replay | 支持success gate；chunk内部BC仍均匀 |
| [DPPO，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html) | diffusion denoise step | inner denoising MDP的PPO | 粒度是去噪step，不是future $h$；更适合直接训练生成actor |
| [Policy Decorator，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c361d4117d598d4bb6568b407e9ac9-Abstract-Conference.html) | 环境执行时刻 | SAC训练小型residual actor | 每步有新观测，不等于同一query中的future $h$ |

因此，当前“联合Q不动，只在本来可分解的BC MSE内逐$h$加权”仍是最窄、最容易解释的工程切口；它由上述
结构共同支持，但具体DVAC公式是本项目适配，而不是某篇论文的原配方。

### 17.2 ACTIVE到底是什么粒度

[ACTIVE（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c06f788963f0ce069f5b2dbf83fe7822-Abstract-Conference.html)
并不是action-$h$方法。它对每条dataset transition的**完整连续action向量**计算一个scalar：

$$
A(s,a)=Q(s,a)-\operatorname{Quantile}_c\{V_i(s)\},
$$

再令$u(s,a)\propto\exp(\beta_A A)$，把同一个$u$乘到整条action的imitation log-prob。$V$ ensemble还用于
保守TD backup；另一个全局dual变量调节总体imitation强度。它提供的依据是把三件事分开：critic可信度、
query/target质量、BC总体强度；它没有给C10十个位置分别赋权。

### 17.3 FQL与RLT：相同骨架和关键差别

FQL的训练流是：

```text
dataset action → multi-step flow teacher
state + same noise → one-step student
student actor loss = -Q(student action) + alpha * MSE(student, flow teacher)
critic TD的next action也由one-step student产生
部署只运行one-step student
```

RLT与它相同的是：one-step student、整段动作的scalar-Q，以及`-Q + teacher MSE`。关键差别是：FQL在同一
offline dataset上持续训练flow teacher，并用同一noise做distillation；RLT冻结预训练$\pi_0$，使用在线replay、
Stage 1表征、C10 student与自己的reference/BC-Q schedule。FQL没有success target切换，也没有DVAC权重；
官方MSE对全部动作坐标直接平均。

所以FQL支持的最小RLT切口是：

$$
-Q(s,a^{student}_{0:9})
$$

保持原样，只把

$$
\frac1{10}\sum_h e_h
$$

改成

$$
\frac1{10}\sum_h c_h e_h.
$$

### 17.4 当前实验其实同时改了两件事

相对原RLT，当前branch不是单一DVAC干预：

1. 成功episode的BC target从$\pi_0$ reference换成replay executed action；
2. 只在这些成功row内，再用mean-one DVAC重分配十个$h$的BC。

2026-08-27 20:48服务器现场到actor TensorBoard Step448：成功target占batch的`40.24%`，最近20点均值
`39.67%`；整个apply阶段均值为`25.30%`。随着成功数据变多，target切换的覆盖率在增强，而DVAC权重ESS长期
约`.977--.979`，基本稳定。

因此当前control与method的差异不能单独回答“DVAC逐$h$权重是否有效”。最小拆分应为：

```text
A  原RLT：pi0 reference target + uniform BC
B  成功自模仿：success executed target + uniform BC
C  当前组合：success executed target + DVAC-weighted BC
```

`B-A`测target替换；`C-B`才测DVAC。若要测试最简“其他全部不动，只改BC内动作权重”，还可单独做
`pi0 reference target + DVAC weight`；但此时高DVAC增权的语义是“teacher自己不稳处更强复制teacher”，与
“成功轨迹在teacher困难位置值得多学”是两个不同假设。

### 17.5 当前DVAC权重到底改得多大

当前配置公式是：

$$
c_h=1+0.25\left(\operatorname{clip}(z_h,-2,2)
-\operatorname{mean}_j\operatorname{clip}(z_j,-2,2)\right).
$$

所以配置名虽写`w0to2`，它不是把现场权重直接铺满$[0,2]$。C10约束下理论范围约$[0.1,1.9]$；Step448现场：

- p05 / mean / p95：`0.746 / 1.000 / 1.251`；上下理论边界命中率均为0；
- success-row ESS：`.9794`，等效`9.79/10`个位置仍在发言；
- all-query ESS：`.9767`，对应权重RMS波动约`15.4%`、相对uniform系数角约`8.8度`；
- top-20% weight mass：`23.97%`，uniform为`20%`；
- success weighted/unweighted BC scalar loss：`.004761/.004721`，只增加约`0.85%`。

作为描述性对照，旧GRPO global-z `[0,2]`在g49的ESS为`.897`、系数角`18.22度`；R-only g49为
`.913/16.72度`。二者loss接口不同，不能直接等同训练强度，但当前RLT的纯权重重排明显更温和。

对同一批保存tensor做的局部机制探针进一步显示：

| 改动 | 全batch BC输出梯度转角 | success子集转角 |
|---|---:|---:|
| reference target → executed target | `24.32度` | `57.48度` |
| executed-uniform → executed-DVAC | `2.42度` | `7.81度` |

在最新20个trace中，两项分别约为`37.51/65.79度`与`2.97/7.91度`。这里测的是
$\partial L_{BC}/\partial a^{student}$，还不是共享MLP参数上的完整actor梯度；但它已经清楚说明：当前实验里
target替换是更大的干预，DVAC是其内部的温和重排。

### 17.6 怎样衡量“方法究竟改了多少训练”

应从外到内分四层，而不是只看配置范围：

1. **权重系数**：p05/p95、CV、top-k mass、ESS与相对uniform系数角；回答十格发言权重排多少。
2. **实际BC贡献**：weighted/unweighted loss ratio，以及
   $c_h\lVert\nabla e_h\rVert$的realized credit ESS；回答高权重是否恰好落在大误差/大梯度位置。
3. **branch梯度**：分别测reference-BC、executed-uniform BC、executed-DVAC BC的norm/cosine；回答target切换
   和DVAC各自把BC方向转了多少。
4. **完整actor更新**：把$-Q$与$\lambda_{BC}g_{BC}$合成，再测总梯度、AdamW更新后各$h$的student action
   变化；回答局部重排是否真的穿过共享MLP与优化器。

当前已有第1层、部分第2层和BC输出空间的第3层。下一项最高信息量检查不是直接继续扩权重，而是用同一
minibatch同时计算`B-A`与`C-B`的完整actor梯度norm/cosine；这样可以先判断“不明显”来自权重温和、BC被
$-Q$压过，还是不同$h$梯度互相抵消。
