# 在线成功BC＋DVAC：独立设计上下文

π0.5 DVAC启动确认（09-06 00:14）：GPU7于00:09:52从原Sidney SFT/空成功池启动formal100，wrapper2223376/driver2223401、三worker路径/物理GPU7核对通过，已进入首轮采集、无所查fatal/OOM；源码626825e8已push、clean。28项CPU/接口测试通过，按用户要求未跑smoke，尚无完整首轮/实际非均匀更新/长程验证。GPU6普通BC与Sidney/shared Ray原进程保持，停止主动盯守；唯一当前运行证据见[账本第9—11条](evidence/PI05_BC_DVAC_IMPLEMENTATION_LEDGER_20260905.md)。

最新π0.5增量（09-06 00:03，任务起始09-05）：用户明确授权GPU7直接同参数正式100轮、范围[0.5,1.5]，并要求**不跑smoke**。从已运行π0.5 BC `6a93605d`独立建`codex/sz-pi05-online-bc-dvac`，五个基线接线文件与旧DVAC父版本逐字一致，已直接复用`736b1416`方法增量。只新增alpha0.125薄配置及M10/范围回归；28项CPU测试/真实Hydra对照通过。方法与边界统一见§12，执行只追加[π0.5 DVAC账本](evidence/PI05_BC_DVAC_IMPLEMENTATION_LEDGER_20260905.md)。下方π0 smoke-only授权为历史，不覆盖本轮π0.5直接正式授权；GPU6/Sidney/shared Ray不动。

2026-09-05 独立实现与GPU7两轮smoke已完成：源码`736b1416`、验收证据`912808c7`均已push，22tests通过；18:53:14—19:31:36 exit0，20次Adam/64固定评估/两代checkpoint与DVAC状态读回验收通过，显存峰77.71GiB，未宣称学习收益或长程OOM解决。基础BC唯一入口仍为[00_RESEARCH_AND_PLAN](00_RESEARCH_AND_PLAN.md)，方法见§11，结果见[验收简报](evidence/BC_DVAC_IMPLEMENTATION_AND_SMOKE_20260905.md#3-单卡smoke验收)。原GPU6 BC正式在17:44第6轮OOM，后续用户另授权eval8×4并行smoke，19:27启动、独立账本维护；GPU7本次未热改配置。用户最新要求只smoke、暂不formal，不触碰Sidney/shared Ray。以下§1—10保留设计收敛过程，不覆盖本段新状态。

## 1. 方法本体可以很小

```text
当前π0照常4步ODE采样
  → 旁路算每个C50未来位置的DVAC（不额外forward）
  → 整条成功筛选；保存同query的RGB/状态/命令/V[50]
  → 累计成功回放
  → FM的未归约误差[B,50,14]乘 detached w[B,50,1]
  → 原BC的mask/平均/反传/更新/同步
```

仍学习成功episode中本策略实际提交的命令，不改成RLT reference action，不需要critic、advantage、PPO ratio/clip或ST伪梯度。权重在归约前进入，不能用标量loss乘batch平均权重替代。

## 2. 哪些旧代码真正能借

本轮只读刷新并抽取两个相关源码树，没有遍历历史或运行旧实验：

| 来源锁 | 复用 | 不搬 |
|---|---|---|
| π0 GRPO Action-Adv Fix `7006ad20a6ad7357c48e6d20e3fcabed6c5609e4` | `openpi_action_model.py` endpoint旁路；`dvac_train_weighting.py` L3总体方差、log、recent统计、clip和detached形状 | 轨迹advantage、action-level ratio/clip、GRPO H求和尺度；训练SDE的数值分布不能直接当BC ODE标定 |
| 深圳RLT Pure `30349428c37a008b95342121c1455debfeb4805e` | `rlinf/algorithms/rlt/dvac_weighting.py::centered_mean_one_weights`，先非负再每query均值1；sidecar记录标定状态 | frozen teacher baseline、C10、reference-BC target、SAC Q项、strength1.5自动移植 |
| 当前BC | SuccessEpisodeCollector/Replay及masked_fm_loss | 不新建第二套训练器 |

锁和源文件在`evidence/dvac-reference-20260905/{grpo,rlt}/`。旧RLT Pure真实目标是reference-BC（human覆盖除外），不是本BC要的成功自执行命令；更早success-executed版本也不能与Pure混称。

## 3. 信号的精确定义

去噪第m次已有带噪动作x和速度v，原实现旁路读：`endpoint_m = x_m - t_m * v_m`，只取有效D14。

4步采样得到`[B,4,50,14]`的临时endpoint；取最后L=3次，沿m算总体方差（`unbiased=False`），沿D求和，得`V[B,50]`。不是每个h独立调用3次策略、不是图像方差，也不是直接对v方差。

采集完成只需保留50个FP32值＝200字节/query；最多12800query约2.44MiB，不存完整去噪链。观测/命令/V必须来自同一次query、同policy version。BC仍用自己的ODE M4，不为了信号改回GRPO的SDE。L3计算形式可以复用，数值尺度要从本BC新数据建立，不能用旧SDE均值/std。

V表示同一次推理中endpoint估计的分歧，不直接等于价值。成功过滤先选择行为方向；DVAC只分配成功行为内部各位置的学习强度。较高V给较大权重的假设是“成功动作中尚不稳定的位置值得多学”，不是“越不确定越好”；跨episode高V可能与失败相关，并不构成这个局部假设的证明。

## 4. 建议的最小接法与需要确认的三点

### 建议方向（尚未锁参数）

1. 采集端旁路记录V；第一轮只用w=1建立本BC的统计，不增加额外交互。
2. 优先沿GRPO的`log(V+eps) → 最近5个已完成轮次的global均值/std → z裁剪±2`。每个新有效query只进入统计一次，不按replay被抽多少次重复计数；可从失败query只传count/sum/sumsq，不保存失败RGB。
3. 为保证只重分配chunk内部注意力，借RLT的非负、均值1映射：在有效H上`c=z-mean_H(z)`，`u=max(0,1+αc)`，`w=u/mean_H(u)`，最后detach；broadcast到D14并与动作mask一起归约。这保证全1退化回原BC，不把H50求和或平均不一致造成的学习率变化当DVAC收益。
4. 权重和原始V一起随query保存，标记产生它的policy version/标定版本；回放不另跑模型重算信号。若要保留动态重标定，则只存V、每轮固定一份统计重新映射，这是另一明确选择，不混称同一配方。

### 三个真实模糊点，应讨论后再实施

| 选择 | 当前倾向 | 原因/代价 |
|---|---|---|
| 高V加权还是低V加权；强度α | 沿旧内部加权方向，高V相对加权；α单独定，不直接照抄Pure04的1.5 | 保留方法直觉，但FM与RLT/Q/GRPO目标不同；强度应明确是新方法设置 |
| 采集时固定w，还是训练时用新统计重算旧V的w | 首版倾向记录V+w，回放固定w | 旧样本权重不因后续标定漂移；含义是行为策略当时的内部信号，不宣称当前策略的不确定性 |
| global-z还是按每个h独立去趋势 | 首版global-z，保持参考代码语义 | 会保留前弱后强等固有位置趋势；不能提前把收益全归于动态难度/真正credit。以后再研究残差，不先加复杂分解 |

每query均值1之后，最大值未必仍≤2；不能同时宣称“严格[0,2]”与“采用当前归一化”而不核算。mask改变时均值必须只对有效位置求；不要把D32 padding统计进去。D0若开启，默认仍等权，当前讨论不授权给示范补额外模型推理。

本轮收敛：上述是任意强度的通用边界；§6建议α=0.25时可直接证明[0,2]且均值1，无需裁负后重新扩大权重。尚未获参数确认，不是部署值。

## 5. 预计代码范围

- OpenPI采样器/rollout：只在明确开启DVAC时旁路收集endpoint并压缩为V；不改变动作或随机数调用顺序。
- Collector/Replay：V与w及必要版本记录跟随query，按同一次成功筛选一起入池。
- 一个小权重模块/actor接点：计算/保存recent统计与权重；checkpoint包含其状态。
- masked FM接点：可选`action_weights[B,H]`，在mean前作用；关闭或全1时原BC数值/梯度一致。
- 一份薄配置和对齐/权重/恢复回归。**现在没有代码改动，也没有GPU7运行**。

先完成一个连贯实现，再做少量高信息量检查：record开关不改变同噪声动作、mask与weights对齐、w=1损失/梯度等价、一次真更新和checkpoint恢复。无需先建立critic/通用奖励框架或多套实验分支。

## 6. 当前推荐配方：让模糊点变成明确选择

| 项目 | 首版建议，尚未实施 | 依据/语义 |
|---|---|---|
| 底座与预算 | 复制当前BC resolved；GPU7、独立run、原SFT/空池，32×1、micro32/global1024/U10、M4、eval16×2均不变 | 同一BC任务的方法增量，不从官方DAgger另起一套 |
| 信号 | 当前行为策略同次ODE；最后L3个endpoint，在模型归一化坐标前14维计算V[50] | 旧π0 GRPO与RLT实现的端点方差；不是物理关节单位混加，也不是速度/图像方差 |
| 标定数据 | 所有真正提交且尚未terminal的、新query；含失败episode，失败图像不入BC池 | 避免用仅成功样本改写参考分布；后终止的无效query不参与；一个query只记一次 |
| 标定时间 | 过去最多5个完整采集轮次的count/sum/sumsq，先映射本轮、再push本轮统计 | 复用GRPO recent基线的顺序；不是5次Adam或5个microbatch |
| 第一轮 | w=1，仍正常U10更新；用这一轮建立统计，不额外采样 | 仅标定冷启动，第一轮样本以后也保留等权，不事后改写 |
| 权重 | 高V相对加权；log eps1e-12/std floor1e-6/z clip±2；α=0.25 | 数值常数沿参考；强度是本次建议，非论文/旧Pure最优值 |
| 回放 | 入池时算定w；保存V+w+已有policy_version/round及calibration版本，后续固定 | 表示行为策略当时的分歧，不是当前actor的不确定性；U10重放不重复计标定 |
| 动作/图像/监督对象 | C50不截短，RGB/augmentation off不变，仍学成功提交命令；D0默认0 | 不引入adaptive chunking，不引入teacher/Q/V/奖励或GRPO目标 |

令$z_h=\mathrm{clip}((\log(V_h+\epsilon)-\mu)/\max(\sigma,\sigma_{\min}),-2,2)$。当前50个位置各有14个有效维度，直接使用：

$$
w_h=\operatorname{stopgrad}\left[1+0.25\left(z_h-\frac{1}{50}\sum_j z_j\right)\right].
$$

因为$z_h\in[-2,2]$，中心化后在$[-4,4]$，故$w_h\in[0,2]$且每query均值1。α=0时精确回到BC；α≤0.25均有此保守边界。α更大时才需要RLT的clamp后重新归一化，其最大值不再保证≤2。因此首版无需强度1.5、softmax温度或另一套权重函数。

例：同一个chunk内z为[-1,0,1,0]，权重为[0.75,1,1.25,1]：不是丢弃成功动作，而是把同一条成功经验的监督力度重新分配。均值1保持的是监督系数总量，**不是保证实际梯度范数不变**；误差与权重相关时梯度自然会变化。

一个细节：若未触发z裁剪，后续每query中心化会抵消global mean；recent统计主要决定std尺度及裁剪是否饱和。不能把两个去均值步骤包装成两种独立credit机制。

## 7. 代码如何接：五个旧文件＋一个小模块，不改训练框架

本轮16:47重新核对服务器四个BC文件、两个GRPO参考文件、一个RLT权重文件，7个SHA256与本地逐一一致；参考树HEAD及dirty也已刷新。证据在`evidence/BC_DVAC_SOURCE_AND_GIT_GAPS_20260905.txt`。

| 接点 | 改什么 | 不改什么 |
|---|---|---|
| `openpi_action_model.py::_sample_actions_with_prefix_cache / predict_action_batch / sft_forward` | opt-in旁路从已有x_t_prev/v_t/t提取L3 endpoint，压成`dvac_v[B,50]`放forward_inputs；SFT把可选action_weights交给masked_fm_loss | 不新增模型forward/参数；不改变生成动作的数值或随机调用顺序；不改FSDP |
| `huggingface_worker.py` | 仅BC训练采集开启record标志，沿原PolicyOutput.forward_inputs透传 | BC采集传给模型的mode本来是eval，不能靠`mode=='train'`判定，否则会漏采全部信号；固定评估默认不记 |
| `data/online_bc.py` | collector在finished检查后读取V，累计新query统计；成功记录保留V/w；masked_fm_loss接受可选权重 | 当前collector字段白名单会丢DVAC，因此不能只在sampler返回值里加一个键；已有Replay通用stack/save/load可复用 |
| `env_worker.py` | BC专用发送包附带本轮小统计，既有成功episode一起发送；没有成功也发送统计 | 不保存失败RGB，不改仿真/相机/动作执行，不新开数据通道 |
| `fsdp_online_bc_policy_worker.py` | 收齐本轮packet后，用上一轮recent状态一次性标定成功records，再add_episodes；保存权重标定sidecar；forward_actor传action_weights | 先完成入池标注再落archive，避免磁盘与内存权重不一致；复用U10/Adam/同步/保存壳 |
| 小模块，例如`algorithms/online_bc_dvac.py` | 只封装V计算、recent moments、带mask的均值1映射与state_dict | 不导入整个GRPO/RLT算法栈；必要张量/有限性合同保留，不加经验阈值gate |

再加一份继承基线的薄配置与集中的回归测试；不需要改runner、FSDP manager、checkpoint strategy、RoboTwin渲染代码或依赖。两类小数组V+w共400字节/query，理论12800query约4.88MiB裸FP32数据（另加元数据与序列化开销）；L3临时张量在B32下约0.256MiB，不会增加第二个VLA。实际峰值仍以未来smoke为准。

关键传递顺序是：sampler算V → collector取V/统计并成功筛选 → actor映射w → replay保存V+w → U10抽样 → masked FM。权重在actor入池时算好，**不要求rollout每次查询都等actor下发标定参数**。

## 8. FM怎么乘，为什么不容易再引入一套FSDP问题

当前`masked_fm_loss`先按每个query有效维度平均、再对batch平均；`sft_forward`已将loss切到[B,50,14]。新的目标保持同一个分母：

$$
L=\frac{1}{B}\sum_b\frac{\sum_{h,d}m_{bhd}\,w_{bh}\,\ell_{bhd}^{FM}}{\sum_{h,d}m_{bhd}}.
$$

若将来每个位置有效维度数不同，令$q_{bh}=\sum_dm_{bhd}$，权重中心化也按q加权，保证$\sum_hq_{bh}w_{bh}=\sum_hq_{bh}$。当前全H50/D14有效时正好等价普通H均值。权重detach，不通过w反传；原生FM的随机ε和t照常重新采样，**不重放行为去噪链训练，也不在单次FM前向里重估DVAC**。

这里只改变现有loss tensor的乘法与输入字段，不新增可训练模块、不跨过现有SFT调用边界；因此没有理由重写已经跑通的FSDP/保存逻辑。但必须让recent统计与已有replay/learner计数一起保存恢复，才不会恢复后换权重标尺。

另一个清理边界：现BC ODE路径仍每步抽随机噪声再乘零std。它数值上不加SDE噪声，却消耗RNG。为了与当前BC保持同一随机轨迹，**本次不顺手删除这些调用或改logprob/chain通用接口**；以后若要瘦身，应同步处理对照。记录DVAC本身不应新增或减少任何随机抽样。

## 9. 机制判断、外部依据与下一步

比较三种语义：高V多学＝成功动作中尚不稳定/可能接触敏感的位置优先；低V多学＝保守模仿更可靠的伪标签，可能跳过真正难点；去除固定h趋势＝更强调动态残差，但已是进一步配方。首版倾向第一种，保留global-z的位置趋势，不先加残差分解或多个平行变体。成功过滤只保证episode结果，不证明每个waypoint都正确或有因果功劳。

本轮重新查看一手资料：

- [DVAC原论文§3，公式3—4](https://arxiv.org/html/2606.03847v1#S3)：同次推理endpoint尾方差是可借的内部信号；原工作用于推理时执行长度/重规划，不是监督加权。其相位关联支持研究直觉，不能替我们证明高V加权有效。
- [OpenPI原生PyTorch](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)：FM训练和推理是不同路径；只在已有未归约FM误差上加权即可，部署源码仍以当前固定版本为准，不升级。
- [LeRobot RA-BC官方接口](https://huggingface.co/docs/lerobot/en/sarm#step-5-optional-train-policy-with-ra-bc)：支持π0/π0.5加权监督，但信号是外部任务进度、通常query级；只作为加权FM接法参照，不借其权重阈值/奖励模型。

推荐下一步：先确认本稿α0.25、高V方向、入池固定w、过去5轮/首轮等权这组语义；获实现授权后一次完成主体＋少量高信息回归（record不改动作及RNG；全1 loss/梯度等价；成功/失败统计与mask对齐；真实更新与sidecar恢复）。GPU7正式仍须刷新容量并按当前BC resolved逐叶继承；不动GPU6已有训练。当前没有DVAC训练、没有宣称收益，也不承诺精确代码行数。

## 10. 先选pipeline切入点，再谈信号映射

用户本轮两次明确：要像GRPO选择reward/adv/gradient那样，研究**相关工作让信号影响BC哪一环**，不是重做旧RLT/GRPO信号转换。故§9的旧下一步暂后置，先按下列位置讨论；完整作者代码与证据集中在[切入点调研](evidence/BC_DVAC_PIPELINE_ENTRY_RELATED_WORK_20260905.md#2-直接对监督项加权实际代码与粒度)。

| 位置 | 已核对的相关工作 | 本项目判断 |
|---|---|---|
| 逐动作监督loss | AttenA+ OpenPI源码在沿D求误差后、沿H归约前加权；本项目旧RLT在逐H BC误差加权后再并入−Q＋BC | **优先建议**：原生FM误差保留H50，乘同位置DVAC权重再mask/归约 |
| 整query监督loss | LeRobot RA-BC、RynnValue-IQL、GFP的VaBC；ForesightFlow论文也为query标量 | 同属监督加权，但不能先平均H再装入50个权重；不搬它们的critic或其他目标 |
| 入池／抽样 | DataMIL选轨迹；Re-Mix调数据域混合比例 | 改学哪些数据/出现多少次，不直接表达一个C50内部的逐位置监督力度 |
| 条件输入／部署 | RLinf RECAP优势条件与CFG | 改模型条件和推理规则，不是最小的连续监督加权 |
| FM训练配对 | OTQL把权重变为目标概率质量，求噪声—动作OT耦合后训练 | 真正更早的FM切入点，但要新增OT和配对逻辑；本轮论文级核验 |

首选链为：**原采集 → 原成功过滤 → 原累计池均匀回放 → 原生逐H FM误差乘w → 原mask/归约/backward/Adam**。不改奖励、执行C50、监督动作、噪声/时间分布、模型结构或优化器。AttenA+是最直接同模型家族/同粒度实现依据；RLT证明项目已有相同“BC监督项”落点；RA-BC/RynnValue提供成熟的样本加权组织。不能把这些代码先例说成已经证明我们的DVAC有效。

action-level仍为`[B,50]`：每个未来14D动作向量一个权重。`[B,50,14]`先广播乘权重，或先沿D归约到`[B,50]`再乘，在mask/分母一致时等价；沿H归约之后再加权太晚。归一化方式是另一个问题，不在本轮重开。

固定detached w时，逐H loss加权与逐H梯度在合并前同权加权数学等价，因此无需gradient hook。普通backward后的参数梯度没有独立H轴；按层缩放或投影是另一优化方法。详见[梯度及优先回放辨析](evidence/BC_DVAC_PIPELINE_ENTRY_RELATED_WORK_20260905.md#41-加loss与加gradient何时相同)。

本轮只维护设计与证据；没有改GPU6、启动GPU7、生产测试或发布。下一步继续收敛切入点，之后再按已有独立分支边界确认实现；不把讨论误当作新增执行授权。

## 11. 已授权的独立实现与GPU7验收入口

- 用户现已授权实现、简测、单卡smoke；从原BC精确HEAD`385d4e75`建立`codex/sz-pi0-online-bc-dvac`，实现提交`736b1416`已推送。原BC生产树/主配置不变；完整五文件接线沿§7实施，新权重模块136行，已有五文件合计+115/-12，另有集中测试、18行配置组与说明。
- 接点就是§10选定的逐动作监督loss：保留`[B,50]` DVAC/w，广播到原生`[B,50,14]` FM误差，在原mask/平均之前乘。旧RLT此处是动作回归BC误差，不是FM；AttenA+是同模型家族、同H粒度的直接代码依据，不搬其权重信号。FM加权与loss加权不是两类入口。
- 首版采用已展示配方：同次M4最后L3 endpoint方差；过去5轮有效新query的log统计；高V、α0.25、z裁剪±2、chunk均值1/理论范围[0,2]；首轮等权、入池后w固定；失败仅贡献小统计，RGB仍只保留成功数据。没有新增forward/RNG调用或Q/V/RL目标。
- 22/22服务器测试通过，验证record开关动作/RNG/forward次数一致、逐H loss及梯度、成功/失败边界、入池和sidecar状态等；真实`validate_cfg`与逐叶预算核对通过。配置组追加到原主YAML，未改变原模型/任务/优化器/FSDP/渲染依赖。
- [完整smoke合同](evidence/GPU7_DVAC_SMOKE_CONTRACT_20260905.md)与[resolved](evidence/DVAC_SMOKE_RESOLVED_20260905.yaml)：GPU7，train32×1，micro32/global1024/U10，M4；2轮、eval16×2每轮、save每轮，共64次采集尝试/20次Adam/64次评估/两代checkpoint。上限90分钟；失败保留证据，不擅自调参重跑。仅smoke，未启动DVAC正式长训。
- 原BC的第6轮OOM是另一个尚未解决的容量边界：Env53.11GiB＋Actor14.85GiB＋Rollout10.68GiB，推理申请816MiB时仅余469MiB。已询问是否不创建评估环境；未收到新答复，按已有授权保留原smoke评估预算，不暗改对照。即使两轮通过，也不能声称解决原长程OOM或DVAC已有学习收益。
- 当前运行路径、PID、状态与验收只追加到[实施账本](evidence/DVAC_IMPLEMENTATION_LEDGER_20260905.md)，不在本设计文件持续堆进度。

## 12. π0.5 BC上复用DVAC：GPU7直接正式

### 12.1 增量与已收敛选择

不再有需要阻塞实现的方法歧义。用户明确同参数和范围[0.5,1.5]；沿既有方向/标定/权重固定语义，模型M10沿π0.5 BC。具体唯一变化：

| 环节 | 本次选择 | 相对旧实现 |
|---|---|---|
| 基础流程 | 原Sidney SFT→32条采集→成功累计池→FM→同步 | GPU6已运行BC保持，不改其进程/文件 |
| DVAC原始信号 | 当前M10 ODE末L3 endpoint方差，模型归一化前14维→V[50] | 原方法按配置取M，非写死M4；不加forward/RNG |
| 标定 | 过去最多5个完成轮次，全新有效query含失败；首轮等权 | 直接复用；不带入旧π0/M4统计 |
| 权重 | z clip±2；alpha0.125，w=1+alpha*(z-mask加权均值z) | 仅新配置组覆盖alpha0.25，保证[0.5,1.5]/均值1 |
| 回放 | 入池固定V/w/版本；累计query均匀有放回 | 不重标旧数据，不重复计标定 |
| 监督接点 | `[B,50,1]` detached w乘`[B,50,14]`原FM误差 | 原mask/分母/更新/同步/保存不重写 |

由于z与其有效位置均值都在[-2,2]，差在[-4,4]，乘0.125后保证w在[0.5,1.5]。这是**保证范围，不是每chunk强制minmax铺满两端**；全常量信号仍全1、首轮冷启动仍全1。保持每chunk监督系数总量，但不保证实际梯度范数或训练loss尺度完全相同。

### 12.2 精确文件与测试边界

新分支基于π0.5 BC `6a93605d`。从旧`736b1416`精确移植9文件：五个既有接线文件、`online_bc_dvac.py`136行、旧方法配置组、集中测试、说明。不整合旧π0的model/task/eval配置，不改Sidney adapter、RoboTwin、runner、FSDP manager/strategy或依赖。

新增变化只有：

- `examples/embodiment/config/bc_dvac/bounded_half.yaml`，10行，继承default，仅alpha0.125；旧default仍alpha0.25。
- 原DVAC测试参数化M4/M10及alpha0.25/0.125；新`test_pi05_online_bc_dvac.py`验证组合配置只变方法/GPU/名字，以及极端信号和partial-mask下范围/均值1。
- 方法说明区分M4/M10及新范围。六个方法生产Python文件与旧DVAC相同，不新写算法。

28项服务器CPU回归通过（9.83s）；包含真实采样循环配stub velocity，证明该循环记录开关不改动作/RNG/调用次数，**不是跑了真实π0.5 GPU推理**。FM loss/梯度、成功/失败统计、权重冻结/sidecar恢复及模型配置组合通过。按用户最新要求无smoke，不声称新组合已做完整端到端或长程验证。

### 12.3 与GPU6正式baseline逐叶对齐

唯一完整配置/命令/输出/资源/停止边界：[正式合同](evidence/GPU7_PI05_BC_DVAC_FORMAL_CONTRACT_20260905.md)、[resolved](evidence/PI05_BC_DVAC_FORMAL_RESOLVED_20260905.yaml)、[17叶差异](evidence/PI05_BC_DVAC_VALIDATION_20260905.json)。只有7 DVAC叶＋GPU6→7＋9个名称/路径；种子文件、模型数据适配和基础配置内容逐字一致。

学习32×1、100轮、micro32/global1024/U10、LR2.5e-5 constant、expert-only、无图像增强/demo0；模型Sidney M10/C50/D14/224/token200/absolute-MEAN_STD；任务pillbottle/H200；eval8×4每5轮、save10全部相同。总预算3200尝试、最多1000Adam、640评估、10代checkpoint；没有借旧DVAC smoke的eval16×2。

fresh原模型/空池/空标定状态，不从GPU6已学习权重派生，也不恢复旧smoke。估算容量不冒充新DVAC实测；正式启动后检查自身worker与stage即回报，不建立新heartbeat、不持续盯守。

09-06 00:18发布收尾：生产/配置测试626825e8、轻量启动证据776ebc98均已push、clean。实际run source-head626825e8不变，00:14已开始首轮采集/无所查fatal；启动确认后结束主动检查。
