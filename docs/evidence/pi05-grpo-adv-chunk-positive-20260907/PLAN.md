# π0.5 GRPO＋DVAC：逐动作正优势、原chunk裁剪实施合同

更新：2026-09-07。状态：**用户已授权实现、测试、推送并替换GRPO；执行中，以账本为准。** 账本：[ADV_CHUNK_POSITIVE_IMPLEMENTATION_LEDGER_20260907.md](evidence/ADV_CHUNK_POSITIVE_IMPLEMENTATION_LEDGER_20260907.md)。下文“本轮仅规划”为原确认阶段记录，不限制最新明确授权。

这是该改动的唯一计划入口。此前长讨论、历史源码及代数核对留在[evidence/ADV_POSITIVE_CHUNK_CLIP_DISCUSSION_20260907.md](evidence/ADV_POSITIVE_CHUNK_CLIP_DISCUSSION_20260907.md)，不要求每次重新阅读全文。

## 1. 目标与底座

在已跑通的Sidney π0.5 RoboTwin GRPO Control上，保留原chunk ratio/clip，只让DVAC影响逐动作的有效优势；原优势A>0才加权，A<=0保持Control；不做每chunk权重均值1。

代码/配置比较基准是**无DVAC的π0.5 GRPO Control**，不是此前action-level clip的Adv，也不是把当前ST直接改名。旧源码核对入口为`sidney-pi05-current-rlinf@1d015a2a`；此前已复核的Control实际run resolved与后续200轮预算合同作为配置来源，实施前重新锁定源文件hash并逐叶检查。该SHA是只读核对的树HEAD，不冒充所有历史训练期的代码提交。

实施采用独立方法分支/worktree，生产依赖版本不变。若后续授权新实验，计划从Control同一Sidney SFT起点fresh开始，不续Control或ST训练权重；新路径/namespace单独登记，不覆盖旧产物。本轮未创建分支或授权新的运行操作。

## 2. 相对Control，明确改什么

| 环节 | 计划增量 | 不得混入的变化 |
|---|---|---|
| 信号采集 | 复用已有DVAC同次推理信号，得到每个未来动作位置的V_h | 不增加策略查询、环境交互或额外噪声采样；不改变生成动作 |
| 信号到权重 | 复用现有GRPO DVAC的logV、历史标尺、限幅及[0.5,1.5]映射 | 不引入BC的chunk中心化，不借机改窗口或强度 |
| 优势 | 原GRPO优势计算和归一化完成后，构造A_eff[h]=A*w[h]（仅A>0） | 不改reward、组过滤、GRPO估计器；不在乘权重后再次归一化adv |
| loss接口 | 显式逐动作adv，分离共同chunk裁剪数值与各位置反传入口 | 不用独立action ratio；不简单广播带全chunk计算图的ratio；不暗中回到logprob_st入口 |
| 方法状态/记录 | 保存既有DVAC历史统计、方法标识和权重指标；接现成checkpoint机制 | 不重写FSDP/同步/保存算法，不新增训练框架 |

“只作用正优势”精确含义是**原A>0**，不是直接检查reward==1；A<=0使用权重1，负反馈仍在。A已经完成Control原有归一化，再乘固定、非负、detach的w。DVAC信号是学习力度启发式，不把它当作已证明的因果信用。

## 3. 相对Control，明确不改什么

下表为已经核对过的协议，用于规划，不是本轮服务器运行快照。精确配置仍以Control实际resolved逐叶继承为准，不能只对齐表内几个数。

| 类别 | 保持的协议 |
|---|---|
| 模型/起点 | 同Sidney π0.5 SFT、权重转换、normalizer、三相机、14D动作、仅训练action expert |
| 策略采样 | C50、M10、原Flow-SDE及noise_level=0.5；原随机种子/随机数调用协议，不移植BC的seed42改动 |
| 环境/任务 | move_pillbottle_pad，episode horizon200；原初态文件与顺序、渲染/OIDN、reset、终止语义 |
| 每轮数据 | 两卡、32环境/卡、串行4，共256条尝试；GRPO group_size=8 |
| 更新预算 | micro32/global1024/U2；不减为128条，不改global512，不引入BC成功回放 |
| 优化 | LR5e-6，原优化器/调度/梯度累积及裁剪、精度、FSDP、参数同步 |
| 原GRPO目标 | reward、优势估计/归一化、组过滤、有效mask及原聚合尺度；其余实际生效的loss项/数值保护不变 |
| ratio/clip | H和D求和所得原chunk logprob；原ratio定义；clip_low/high均0.2；不做GSPO的长度平均 |
| 评估/保存 | 固定32初态，16环境/卡，每5轮评估；每10轮保存，原checkpoint格式 |
| 总预算 | 延续此前已讨论的200轮上限；它来自Control后续扩展预算，不声称最初fresh配置就是200 |
| 运行隔离 | 计划继续两卡布局，具体GPU与资源在执行前刷新；新run身份、输出和namespace是必要非算法差异 |

不改变随机协议不等于新旧独立运行自动具有完全相同的随机序列；本次没有将“重新配对所有RNG”混为DVAC增量。信号采集本身必须不额外消费RNG。

## 4. 权重合同：不再增加归一化步骤

复用现有GRPO方法设置：selected_l=3，warmup=1，history最多5轮，log_eps=1e-12，std_floor=1e-6，z_clip=2，范围[0.5,1.5]。

1. 同次rollout得到V_h；其选层/去噪阶段及方差计算原样复用，不重定义DVAC。
2. `log(V_h+1e-12)`用过去至多5轮的统计标准化，得到z_h；历史按原实现含成功和失败query，不因positive-only变成成功样本统计。
3. z_h限制到[-2,2]，`w_h=1+0.25*z_h`，范围[0.5,1.5]。首轮无历史时全1；沿用原跨rank统计/窗口更新顺序。
4. 原A>0取w_h，否则取1；权重作为固定训练系数detach。
5. **不减本chunk中心、不除本chunk平均权重、不追加batch均重1，也不再标准化加权后的adv。**

GRPO旧Adv和当前ST本来就没有BC式chunk均重1；因此这里是“明确不引入”，不是声称要从现有GRPO删掉一段实际存在的代码。整段w都1.3时，允许整段正优势更新增强；这意味着本方法同时可以改变chunk间分配和正侧总体更新强度，不再仅限段内重分配。原mask/按样本数平均不是权重均重1，仍保留。

## 5. loss语义与最小代码接点

显式目标：

$$
A^{\mathrm{eff}}_{i,h}=A_i\begin{cases}\operatorname{sg}(w_{i,h}),&A_i>0\\1,&A_i\le0.\end{cases}
$$

沿用原chunk ratio的**数值和裁剪判断**，通过GSPO-token式计算图展开让第h项沿第h动作logprob回传，再乘该项有效adv。这里是算法接口的明确设计，不把detach说成无须解释的胶水代码。

在当前Control主体目标下，设`d_h=sum_D(logp_new-logp_old)`，`L=sum_H d_h`，H=50，可采用：

$$
\widetilde r_h=\exp\{\operatorname{sg}(L)+H[d_h-\operatorname{sg}(d_h)]\},\qquad
J=\frac1H\sum_h\min(\widetilde r_h A_h^{\mathrm{eff}},\operatorname{clip}(\widetilde r_h)A_h^{\mathrm{eff}}).
$$

每个r_h前向等于原exp(L)。H抵消外层1/H，以恢复Control求和的梯度尺度；不是可调DVAC强度。该表达式省略原batch/mask聚合，实际接入必须保留；若有实际生效的额外log-ratio数值裁剪，连其梯度门控也保持，不能仅复制其前向数值。全1等价以真实Control loss为验收对象，不只比较玩具公式。

预期文件职责（规划名称不是声称已经实现）：

- `dvac_train_weighting.py`：复用统计/映射与原A符号门控；不新增信号变换。
- `utils.py`/`losses.py`：方法专用显式adv展开，保留[B,H,D]到逐h路径；Control原路径不变。原`action_advantage`绑定action_level，不悄悄改其历史含义；拟增加明确模式`chunk_clipped_action_advantage`。
- `embodied_fsdp_actor_worker.py`：选择新模式、传原logprob/adv/权重，禁用该模式下的logprob_ST乘权；方法元信息进入既有保存/恢复核验。
- 配置、针对性测试与轻量证据；模型推理/信号侧只接已有采集开关，不能新增前向计算或更改噪声流程。

实现文件数和行数待实际diff确定，不提前承诺“只改一行”。不为此重写rollout、环境、GRPO优势、FSDP或checkpoint。

## 6. 有无原则性问题，以及如何确认

**设计在语义和现有代数核对上自洽，没有尚待选择的核心方法开关；生产路径尚未实现验证。**

需要保留的三条认识：

1. 单纯广播同一chunk计算图的ratio会丢失逐动作分配；必须用§5的明确分离。
2. 不做chunk均重1会允许整段更新强度变化；这是所选方法效果，不偷偷归一化抵消，也不能预先保证更稳定。
3. 在固定非负w、相同正负门控、mask和chunk裁剪下，显式adv展开与当前ST可有相同一阶梯度。接口更符合要求，不等于已得到一个必然更有效的新学习规则；与早前逐动作clip的Adv则并不等价。

实施授权后，只做少量直接对应语义的检查，不启动多臂试验或循环smoke：

- 真实Control loss入口：w全1时loss、ratio、clip、mask及logprob梯度在浮点容差内一致；原负/零优势分支保持一致。
- 正优势逐动作效应：交换[0.5,1.5]为[1.5,0.5]应交换对应logprob的梯度系数；整段w=1.3应保留整体增强；已裁剪分支仍按原chunk条件。
- 用已有真实shape/mask张量检查位置对齐、聚合尺度与方法状态序列化；沿用已验证模型、数据和权重更新链，不把旧组件测试冒充新入口验证。
- resolved逐叶对照：仅DVAC/新loss模式和必要run路径身份可以不同；配置关闭新方法走原Control。同步/保存机制不改，方法标识不允许把旧ST状态静默当新adv配置恢复。

此前12组CPU代数结果已经支持目标公式，但不是本节新生产模式通过的报告。本轮没有执行上述新检查；是否启动训练仍按后续明确授权，届时先报告配置/命令/路径/预算/现场资源。

## 7. 依据与当前交付

- 历史Control/Adv/ST及真实源码：[讨论§8](evidence/ADV_POSITIVE_CHUNK_CLIP_DISCUSSION_20260907.md#8-旧π0到π05的实际版本对照)，原始`evidence/ADV_TUTORIAL_HISTORY_READONLY_20260907.json`。
- 逐动作adv与chunk裁剪推导、ST一阶关系：[讨论§5](evidence/ADV_POSITIVE_CHUNK_CLIP_DISCUSSION_20260907.md#5-推荐的讨论候选及与当前st的真实关系)。
- GSPO-token计算图先例：[论文§4.3](https://arxiv.org/html/2507.18071v2#S4.SS3)、[已核对TRL实现](https://github.com/huggingface/trl/blob/1ecae07b30a5189795f62c8b9a0ba23a5b04f73e/trl/experimental/gspo_token/grpo_trainer.py#L59)。只借接口，不搬长度平均或训练超参。
- 非方法预算来源：[先前逐叶启动合同](evidence/PI05_ADV_LAUNCH_CONTRACT_20260906.md)、[本轮切换参数差异](../rlinf-robotwin-pi0-online-bc/evidence/BC8_U2_GRPO_ST_CUTOVER_RESULT_20260907.md#2-参数到底改了哪些)。它们是历史配置依据，不是当前进度/资源状态。

本轮交付仅本文与短HANDOFF路由；未改生产代码/配置、未运行测试或新服务器作业、未停止或启动实验、未push。
