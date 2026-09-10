# IQL：实现与官方来源复核

2026-09-10。只读本地源码、冻结证据及官方网页；没有改生产、连服务器、训练或运行项目测试。运行数值引用 **11:23 快照、IQL R97**，不冒充后续最终状态。官方代码锁定 `RynnValue@10e0d333f5f3811d0d130587e50f1faf48da49e5`；本地发布记录为 `6434ca89`，生产实现提交 `4d7877eb`。

## 先看判断

**本轮没有确认一个仍存在、足以解释落后 clean 的低级实现错误。最强线索是“critic 还很早，actor 已强烈改学失败池”，并伴随明显的权重尺度变化。** 这与“移植了错误的 exp 公式”是不同问题。57 项测试和 smoke 支持数值/接线正确，不能证明 critic 已会判断动作价值。

在已有快照中：

| 时点 | 可核事实 | 含义 |
|---|---|---|
| R10 切换前 | 40 次尝试，19 成功；143 个完整 query；Q/V 各 50 次更新 | 随机视觉 encoder 只见过很小的在线池 |
| R11 第一次 IQL | 成功 query 63 / 全 query 159 = **39.6%** | actor 的采样支持从全成功突然变为约 60% 失败来源 query |
| R11 权重 | mean **24.36**；cap100 占 **16.95%**；ESS **313.3/1024**；A mean **+0.2145** | 既重分配样本，也改变损失尺度；1024 有重复，ESS 不是独立数据量 |
| R97 | mean W **6.64**；cap 占 **0.66%**；ESS **220.4/1024** | 不是全程所有样本都触顶；不能把早期饱和当唯一长期原因 |

上述权重指标为一轮 5 个更新槽的平均。它们不能单独说明高权重究竟分给成功、失败、终段还是某场景；需要最终 checkpoint 的逐记录权重分组诊断。[快照](../../../server-admin/evidence/experiment-refresh-20260910-current/snapshot.json)

## 1. 先把“官方默认”分成三份

| 参考对象 | 数据 / 策略 | warmup / 训练规模 | 奖励与时间粒度 |
|---|---|---|---|
| 论文真实机器人 IQL | 离线混合示范；398 成功、12 失败，整体 **97.1% 成功**；H16 | `w=1` 预热 **200** 次；训练 10000 次；B64 | 未完成−1、完成0，加 .1 PBRS；B.5 按决策 chunk 写 γ=.99 |
| 锁定仓库 `pi05_robotwin_iql` | 离线 RoboTwin；H10、EEF16 再 pad32 | `critic_warmup_steps=2000`；200000 次；B64 | 默认 `reward_source="terminal"`；没有自动启用 RynnValue；通用 trainer 低层累计 reward，target 使用 γ^H |
| 我们当前实验 | 空池开始、在线4/U5；Sidney绝对关节14/H50 | 成功BC10轮＋Q/V50次，再全池IQL；最多500 actor/critic次 | 成功1/其他0，加 .1 PBRS；一个实际提交query为一步，Γ=.99；所有任务终点吸收 |

论文 Table9 的 **2000 是 actor 学习率 warmup**，不能误称论文 `w=1` 预热2000。仓库 RoboTwin 恰好另有2000次优势预热，二者要分别注明。论文实际混合数据中失败比例很低，不能据“使用失败数据”就推断其结果可直接迁移到我们多数 query 来自失败的池。[官方论文 B.2/B.5、表7/9](https://arxiv.org/html/2608.09853v1#A2.S2)；[RoboTwin固定配置](https://github.com/alibaba-damo-academy/RynnValue/blob/10e0d333f5f3811d0d130587e50f1faf48da49e5/pi-rl/src/openpi/training/config.py#L3193)

**论文结果是离线 IQL；其在线结果使用 DSRL/SAC 操纵冻结 VLA 的潜变量。** 本项目是直接更新 VLA 的在线 IQL 适配，不能拿官方在线提升当成该路线的已验证收益。[官方仓库方法说明](https://github.com/alibaba-damo-academy/RynnValue#policy-rl-with-pi-rl)

仓库内部还有需要保留的差异：`train_iql.py:255–325` 的 progress 分支是 `.weight*PBRS + optional terminal_reward`，没有自动补论文每步−1成本；terminal 分支按 `actions_is_pad` 恢复低层成本。注册 RoboTwin 又选择 terminal。**直接照抄一个默认文件，不等于完整复现论文奖励。**[锁定奖励代码](https://github.com/alibaba-damo-academy/RynnValue/blob/10e0d333f5f3811d0d130587e50f1faf48da49e5/pi-rl/scripts/train_iql.py#L255)

## 2. 数据到损失逐段追踪

下表本地路径均相对 `local_scripts/bc_rynnvalue_iql_packet/src/`；行号以本轮只读文件为准。

| 检查点 | 本地接点 | 判定与反证 |
|---|---|---|
| 前图/后图/真终帧 | `rlinf/workers/env/env_worker.py:1280–1310`；`rlinf/data/online_iql.py:103–155` | step 前 clone raw obs；step 后取真实 post/final；逐env结束即停追加，后续query校验前后图/state连续。未见把 next episode 首帧当末帧的路径。smoke的缓存hash绑定已保存终帧，但不是物理重放证明 |
| 完整动作 | `online_iql.py:101–104,145–154` | Q使用实际提交的50×14 proposal。unknown execution length记−1；不按事后成功裁去标签。动作内部padding32不进Q；已明确这是proposal MDP，不是50步实际运动长度都已知 |
| Actor/Q动作Normalize | `fsdp_online_iql_policy_worker.py:202–212`；`openpi_action_model.py:639–686` | 两者均由相同`prepare_dagger_sft_batch`处理raw action，再Q取50×14。collector没有保留`model_action`，不会误走“已归一动作再归一”分支 |
| 真实loss mask | `openpi_action_model.py:416–437`；`rlinf/data/online_bc.py:14–37` | 先取真实50×14，再每query按mask均值；IQL与clean的mask都是完整proposal全1。未见从32维padding偷带额外损失或使用episode bootstrap来屏蔽action loss |
| 1024权重对齐 | `fsdp_online_iql_policy_worker.py:241–279` | 对actor实际采样的1024条分64推理，拼回原顺序；不是重复广播critic的64条权重。算完1024权重后才切micro32，期间不更新Q/V |
| 分母/梯度 | `online_bc.py:24–37`；`fsdp_dagger_policy_worker.py:479–519` | 每micro loss是`mean_query(W*L_query)`；32个等大micro再各除32，等价`sum_i W_i L_i /1024`。不是除`sum W`，不是每micro均重1，也未多除一次1024 |
| FP32/stop gradient | `online_iql_critic.py:231–249,294–337,351–379`；`algorithms/online_iql.py:56–68` | Q/V独立FP32；target和A/W停止梯度；actor无损失回传critic。target参数冻结，V训练用targetQ，Q训练用新V，actor用currentQ/V；顺序与锁定官方一致 |
| 图像/增强 | `online_iql_critic.py:47–78,132–158,202–217,308–309,365–375` | raw三相机uint8、独立resize-pad224，encoder只除255一次；当前/下一张各自crop、相机共享offset；A用不crop图。GN无BN running stats，train/eval模式本身不改变归一。该增强差异来自官方，不是多除255或串用actor随机图 |
| 模型结构 | `online_iql_critic.py:91–199` | 自定义stride8/GN4/空间softmax/50latent；共享Qencoder两头、独立Vencoder；Q head LN、V head无LN。不是拿torchvision标准ResNet18替代。真实固定参数Flax parity提供独立结构反证，但不验证价值泛化 |
| 两阶段池/RNG | `fsdp_online_iql_policy_worker.py:28–33,164–175,306–324`；`data/online_iql.py:192–259` | R1–10成功池，R11全池；critic/actor各自sampler，critic初始化与crop也独立。成功池空时只跳actor。这是明确设计切换，不是phase偏一轮 |

代码入口：[worker](../../../../local_scripts/bc_rynnvalue_iql_packet/src/rlinf/workers/actor/fsdp_online_iql_policy_worker.py)、[collector/replay](../../../../local_scripts/bc_rynnvalue_iql_packet/src/rlinf/data/online_iql.py)、[critic](../../../../local_scripts/bc_rynnvalue_iql_packet/src/rlinf/models/embodiment/online_iql_critic.py)、[loss](../../../../local_scripts/bc_rynnvalue_iql_packet/src/rlinf/data/online_bc.py)。

**Confirmed implementation defect：** 早期v1恢复曾因FSDP参数仍offload在CPU而失败；`worker:446–448`已在恢复前加载原actor，v2保存/恢复通过，正式从原模型开始。不能把已修复的smoke退出问题当成本轮性能下降原因。当前只读范围没有确认仍存在的同类缺陷。[实施记录](../../BC_RYNNVALUE_IQL_IMPLEMENTATION_20260909.md)

## 3. 奖励：有意适配，不是简单“给失败奖励”的bug

当前奖励为：

```text
Φ = −剩余秒数
r = success + .1 × (Γ × mask × Φ_next − Φ_now), Γ=.99
Q_target = r + Γ × mask × V_next
```

正常、成功、失败预算终止各自mask语义在collector独立存储；所有终点mask0，原始终帧秒数仍保留。`algorithms/online_iql.py:22–46`与`data/online_iql.py:143–155,350–370`可复核。**失败末步确实可能得到正塑形项 `.1*T_before`；但任务奖励仍为0。**

对同一起始状态、同一折扣、终势函数0的完整轨迹，折扣塑形和为 `−.1 Φ_initial`，是起点常数。这个望远镜相消关系意味着：仅凭“失败终步reward为正”不能断言目标在奖励失败；也不能因为理论相消就断言小数据近似Q/V不受影响。传播过程中，边界项可能暂时压过成功1的区别。

尤其注意两个限制：

- **0/1与论文−1/0并非本任务的无害常数平移。** 每步加1、在成功时提前终止会改变不同长度轨迹的回报差；当前倾向有限预算成功概率，论文更强调完成前成本。相同Γ=.99、2-query成功的稀疏回报：当前`.99`，论文`−1`；4-query成功：当前`.99^3=.9703`，论文`−1−.99−.99²=−2.9701`。对速度/长度的偏好强度不同。这是登记过的reward适配，不是符号算错。
- **纯当前图像并非完整马尔可夫状态。** reward的RynnValue输入含历史前缀，预算mask取决于已提交query数；critic没有历史、proprio或剩余预算。相似图像在query1和query4有不同后续预算；理论PBRS中的状态应包含这些信息。缺少它们可能提高TD拟合难度，但并未证明它是主要原因。

**Γ按query统一是内部一致的。** 本地reward/target都用.99，不存在一边.99另一边.99^50的漏改。官方通用代码两边对应低层γ^H；直接把我们的target改成.99^50≈.605却不改reward会造成新的不一致。物理TOPP时间也不是固定50步，后续若改时间折扣需同时定义真实时长。[官方Bellman代码](https://github.com/alibaba-damo-academy/RynnValue/blob/10e0d333f5f3811d0d130587e50f1faf48da49e5/pi-rl/third_party/jaxrl2/jaxrl2/agents/pi_iql/pi_iql_learner.py#L280)

## 4. 更有依据的失败机制候选

### A. 高失败比例下，权重必须真能筛好动作

R11均匀query采样的成功来源比例只有39.6%；失败通常4query，因而在query池中又比episode比例更大。失败动作也有非零exp权重；IQL并不逐条知道“这条来自失败所以不能学”。若高权重更偏向失败末段、特定起始难度或某种绝对关节姿态，actor就可能从原有成功行为退步。**高权重失败也不一定错误**：失败轨迹中有好的准备动作，需结合query位置、同场景成功对比，而非简单要求失败权重全小。

最有效的诊断是总权重份额 `sum(W on successful-episode queries)/sum(W)`，同时记录成功/失败的每query均重、终段/中段占比、场景分层。不能只比较两类均重或全局ESS。最终池诊断现已完成，见第6节；它确认成功来源被富集，仍有显著失败来源系数及阶段偏好。

### B. currentQ与V的学习时滞可能被exp放大

V追踪targetQ，target τ=.005的参数半衰期约138次Q更新。50次预热后target仍保留约`.995^50=77.8%`初始化成分，而actor权重使用currentQ。**参数残留比例不是输出误差比例**，但它说明50次不保证target已跟上。

缓存日志R10中current双Q均+.265、target最小Q均−.034、V均+.064，正向分离值得检查。这些是同一更新批的不同时间/聚合口径，不能直接相减冒充actor每样本A。应在冻结同一checkpoint、同一无增强输入上计算 `min(Q_current)−V`、`min(Q_target)−V`、`min(Q_current)−min(Q_target)`，再按success/query位置分层。官方也用currentQ；风险来自此任务的数据、奖励和学习时序，而不是原式写错。

### C. β=10会把较小A误差变成大幅重分配

`A=.1/.2/.4605`对应`W≈2.72/7.39/100`。cap前，A多估.2就放大约7.4倍。R11实际meanW24.36和约17%触顶已经说明它并非微小干预。

不做均重1符合原IQL和RynnValue代码：官方policy loss同样是`mean(per_sample_loss*advs)`。因此“没除sumW”不是移植错误。[官方权重](https://github.com/alibaba-damo-academy/RynnValue/blob/10e0d333f5f3811d0d130587e50f1faf48da49e5/pi-rl/third_party/jaxrl2/jaxrl2/agents/pi_iql/pi_iql_learner.py#L80)；[官方FM归约](https://github.com/alibaba-damo-academy/RynnValue/blob/10e0d333f5f3811d0d130587e50f1faf48da49e5/pi-rl/scripts/train_iql.py#L163)

**meanW24不能直接称actor学习率变24倍。** 全批权重尺度确实改变loss和梯度，但Adam历史二阶矩及clip1会吸收部分尺度；最终还受样本梯度方向和时序影响。均重1只检验整体尺度，不修复错误排序；原论文没有要求这么做。

### D. critic预算和输入难度更苛刻

R10仅143个转移、随机初始化约千万参数级encoder；Q输入700个真实动作数，仅50维视觉latent。相对官方H10/16与大量离线重用，长proposal更难拟合；有放回1024不是增加真实数据。后期共500次Q/V更新仍明显少于官方训练预算。可以怀疑欠拟合/捷径，也可能小池过拟合；需要按episode独立分组的误差与排序，而不是只看训练loss下降。

## 5. 下一组只改一个因素：建议优先级

| 优先级 | 单因子实验 | 固定哪些 | 能回答什么 / 依据 |
|---|---|---|---|
| 1 | **actor始终只采成功池，但继续用IQL的W；Q/V仍学全池** | 奖励、critic、β、4/U5、1024、初始化均固定 | 最贴近“在干净BC上加权”的目标；隔离R11加入大量失败来源actor数据的影响。这是诊断性适配，不称原版全池IQL |
| 2 | **β10→3** | cap100、池、reward、Q/V更新不变 | 降低A误差的指数放大；IQL原作者MuJoCo配置本就用3，温度是任务相关而非统一10。[原作者配置](https://github.com/ikostrikov/implicit_q_learning/blob/master/configs/mujoco_config.py#L14) |
| 3 | **shape系数 .1→0（Sparse-IQL）** | success0/1、吸收mask、Γ、phase和数据不变 | 直接区分问题主要来自RynnValue塑形还是全池IQL/critic本身；不要同时改成−1/0，否则无法归因 |
| 4 | **保持R11切换，critic每slot 1→4次** | actor5次、数据4条、reward/β不变 | R10得到200次Q/V更新，对应论文优势预热数量级；这只增加critic预算，不增加真实数据，也不保证泛化。当前代码K1需显式扩展后再试，不能仅改yaml绕过检查 |
| 5 | **仅W除完整1024 batch的meanW** | 相对权重、pool/reward/QV均不变 | 区分全局尺度与样本分配；分母必须跨1024，不能各micro归一。仅当输出尺度诊断支持时做，属于新消融，不是修复作者默认 |

以上是可检验的候选，不建议一次叠加。直接照搬−1/0成本、添加critic剩余预算特征、减短H或换预训练encoder也合理，但会改变目标/状态/控制或模型能力，首轮不宜一起做。第6节的新证据支持继续优先隔离actor池变化；降低β应另做单因子，不作为已经证明的修复。

## 6. 冻结critic的独立数值复核：最终全池覆盖初步小样本

本节采用11:49:25+08:00现场快照对应的已完成实验：R10冻结critic检查自身全部143条转移及之后R11–14的58条；R100冻结critic检查最终**全部1456条**。此前256条均匀样本中的“失败权重52%/第四段1.2%”被全池结果替代，不能继续作为最终结论。

**输入链已对照生产。** 无参数wrapper绑定生产`prepare_dagger_sft_batch`，经过同一AlohaInputs、`physical-intelligence/robotwin`资产均值/标准差Normalize、PadStatesAndActions，再取真实50×14动作。三相机仍从raw uint8经`stack_pixels`的确定性resize-pad，critic内部仅一次/255；没有使用actor增强后的图、没有随机crop、没有加载actor大权重。冻结Q/V用CPU FP32 eval。JSON记录的前向阶段耗时R10为39.68秒、全池为171.52秒，不含整个进程所有初始化。探针`time_cst`字段实际由无时区`datetime.now()`产生，不能按字段名直接认作北京时间；报告时间锚点采用上述带时区snapshot。[探针源码](../../../../local_scripts/diagnose_iql_critic_cpu_full_20260910.py)

独立按record索引重新连接episode结局，检查round/query身份、`A=min(Q1,Q2)−V`、`Atarget=targetQ−V`及`min(exp(10A),100)`。全部身份相符，A最大误差<6e−8，权重相对误差<4.7e−7；以下份额、ESS与主分析逐项一致。[独立复算回执](critic-probe-independent-review.json)

| 冻结critic / 数据 | 成功来源query占比 → 权重份额 | meanW / ESS占比 | 第三次query的权重份额 |
|---|---|---|---|
| R10 / R1–10训练池143条 | 41.26% → **66.28%** | 22.85 / 28.53% | 63.23% |
| R10 / 未见过的R11–14共58条 | 37.93% → **56.60%** | 21.62 / 26.39% | 66.51% |
| R100 / 全部1456条训练池 | 35.16% → **57.50%** | 3.40 / 9.09% | **64.65%** |

由此能收紧三点：

1. **不是“critic完全分不清成功”。** 两个时点均富集成功来源；最终失败来源仍占42.50%系数，但失败轨迹中也有有效准备动作，这个占比不是错误动作率。统计的是loss系数份额，实际梯度还取决于FM误差及梯度方向。
2. **阶段分配很强，比全局均重更值得注意。** 最终第三次query占27.47%记录却得到64.65%系数；前两次合计29.67%，第四次5.67%。这与R10未见数据也偏第三次的现象一致。不能凭阶段位置判定分配错误：后期动作可能更影响成功；但准备动作被学得较少，是明确的可检验线索。最后一次仅包含尚未提前终止的episode，天然有幸存选择偏差。
3. **早期current/target分离被成对验证，仍不是代码bug。** R10同143条输入：current minQ均+.291、target minQ均−.015、V均+.052，对应A均+.239、targetA均−.067；current-target差+.306。最终全池该差缩至+.0567。官方也采用currentQ；targetQ不是正确答案，此结果支持学习时滞可能放大早期W，不能据此直接改成targetQ。

**全池仍不等于泛化验证。** R100是在已见过的训练池上回看，不是新的独立评估；R10未来58条对该checkpoint未见，但来自演化中的在线策略，且同episode内query相关。最终虽仅10/1456条触顶，ESS仍只有132.34/1456：重尾不要求大量cap。初步256样本的失败份额与全池相差9.51个百分点，正说明加权统计不宜只靠小样本。

**下一步优先级补充：先success-only actor，再考虑温和权重。** 固定最终全池A静态重算β3，ESS从9.09%升到76.42%、第三次query份额从64.65%降到31.96%；但失败来源份额也从42.50%升到57.42%。所以单独降β会同时弱化阶段集中和成功筛选，不能只报告ESS改善。第一组仍建议保持critic全池训练、actor回到成功池并保留原β；下一组再单独改β，或者先做Sparse-IQL以隔离塑形。上述静态重算只解释公式效果，不预测重新在线训练后的成功率。[全池探针](critic-probe-full.json)、[主数值分析](runtime-analysis.json)

另：`q_minus_behavior_mc`比较的是已记录行为轨迹的MC回报，IQL的Q则面向expectile诱导的改进策略；即使两者使用一致PBRS回报，也不能把相差直接当成Q校准误差，更不能据其正负单独宣称过估计或正确。

## 来源与范围

- [官方论文](https://arxiv.org/html/2608.09853v1)，本轮重新浏览；关键区分为表7/9、B.4/B.5。
- [锁定官方仓库](https://github.com/alibaba-damo-academy/RynnValue/tree/10e0d333f5f3811d0d130587e50f1faf48da49e5)，与本地 `evidence/rynnvalue-iql-plan-20260909/upstream/`逐段对应。
- [固定两阶段合同](../../BC_RYNNVALUE_IQL_TWO_STAGE_IMPLEMENTATION_PLAN_20260909.md)、[此前来源规划](../../BC_RYNNVALUE_IQL_ONLINE_PLAN_20260909.md)、[实现与已修复恢复问题](../../BC_RYNNVALUE_IQL_IMPLEMENTATION_20260909.md)。
- 未从本轮只读代码推出“某个参数一定是根因”；冻结池诊断已按第6节的范围复核。本子任务仅对本地源码和父任务取得的冻结JSON证据进行读审/算术复算，没有新增服务器测试，也没有改当前实验配置。
