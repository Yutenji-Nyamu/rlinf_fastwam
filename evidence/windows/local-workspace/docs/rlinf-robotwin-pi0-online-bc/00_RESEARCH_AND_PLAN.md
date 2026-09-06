# RoboTwin × RLinf × π0：干净在线 BC 独立上下文

π0.5 BC＋DVAC最新（09-06 00:10）：按用户明确GPU7直接正式/no-smoke授权，独立626825e8已push，28项CPU/配置回归通过，单次正式launch wrapper2223376。方法与范围[0.5,1.5]/同预算仅见[DVAC§12](01_DVAC_DESIGN.md#12-π05-bc上复用dvacgpu7直接正式)及其实施账本；GPU6普通BC保持，不能重复launch或把CPU回归称GPU smoke。

π0.5 BC正式最新（09-05 23:41）：按用户新增授权，GPU6正式100轮已于23:38:34从原Sidney SFT/空成功池启动，wrapper2143105，三worker正常、首轮采集中；32×1/micro32/global1024/U10、eval8×4每5轮、save10。参数分源及唯一执行入口见[π0.5上下文§10](02_PI05_ONLINE_BC_PLAN.md#10-正式100轮参数与来源用户已确认)；不重复smoke/launch，不接管Sidney/shared Ray，不新增清理/自动保留。下方历史“formal未授权”已被覆盖。

最新π0.5 BC验收（09-05 22:42，22:45 push）：独立`codex/sz-pi05-online-bc`源码653fe0fb／证据912bc690已push；13tests与真实数据/配置检查通过。GPU6两轮smoke exit0、20Adam、eval8×4两次、两代checkpoint读回通过；train8/18、fixed11/18 /32，显存峰73.74GiB、GPU6已释放。模型／任务适配之外继承原BC，不改采集/FM/更新/保存。唯一详细结果[π0.5上下文§9](02_PI05_ONLINE_BC_PLAN.md#9-π05两轮smoke结果2242验收)。未启动正式，未动Sidney/shared Ray/其他用户；两轮不作长程或效果结论。下方“π0.5未实施”保留原时间，只是历史快照。

最新研究与授权边界（09-05 21:12）：π0.5在线BC的唯一迁移上下文已建立为[02_PI05_ONLINE_BC_PLAN.md](02_PI05_ONLINE_BC_PLAN.md)，含模型／输入协议增量、原BC时间分布、U10和保存预算讨论。未实施π0.5 BC、未启动新formal。用户随后已批准指定旧checkpoint大文件清理：101文件570.08GiB已删且独立复查通过，6文件14.53GiB暂缓，详见[清理账本](../server-admin/CHENYITENG_CHECKPOINT_PRUNE_LEDGER_20260905.md#最终结果与保留边界)。该新授权覆盖下方历史“没有删除授权”；BC v7／DVAC smoke第1代大权重已清、Step2保留，新v8两代未动。Sidney121/200正常、fixed120=19/32、ckpt120双rank/full在，GPU6/7无compute；后续仍禁止自动formal，不把下面带时间快照当当前现场。

最新验收（09-05 20:14—20:15）：两项smoke均已完成并释放GPU6/7。GPU6原BC eval8×4于20:05:53 exit0，2轮/20Adam/64固定评估/两代native＋full＋replay＋learner验收通过，峰值69.52GiB，源a8764944/证据2467d997已push、clean；train26/23、fixed24/25 /32。GPU7 DVAC已于19:31:36 exit0，完整两轮及action-level权重/状态读回通过，峰77.71GiB，源736b1416/证据912808c7已push。结果唯一入口[验收§3—4](evidence/BC_DVAC_IMPLEMENTATION_AND_SMOKE_20260905.md#3-单卡smoke验收)。两项评估并发不同，不作严格方法效果比较；短测不证明100轮稳定或完整worker恢复。用户只要求smoke、暂不formal。存储清理仍只讨论：旧GRPO/PPO残留81大文件347.73GiB，上次已删146文件1051.64GiB，二者不重复；细分与其他候选见[独立清理稿§5](../server-admin/SZ_CHEN_STORAGE_CLEANUP_DISCUSSION_20260905.md#5-追问348gib是什么上次删过后为什么还有)。没有删除/迁移授权，禁止自动formal。

最新补充（09-05 19:27）：用户指定原BC评估8×4，两个卡应并行；随后明确只smoke、暂不放正式。GPU6 `a8764944`已推送，三个eval叶值/对应测试之外无生产改动，11tests/同32固定种子/真实配置通过，19:27:23启动两轮smoke，wrapper1597471；唯一接续[8×4账本](evidence/BC_EVAL8_RESTART_LEDGER_20260905.md)。GPU7 DVAC独立继续，第二轮w范围0.602—1.780/均值1，待结束验收；它未被热改成8×4。不要重复launch或自动formal。全机用户存储只读结果[分盘表](../server-admin/SZ_STORAGE_BY_USER_20260905.md)。

最新实施（09-05 19:01现场）：用户已授权BC＋DVAC独立实现、简测及单卡smoke。从原BC `385d4e75`建`codex/sz-pi0-online-bc-dvac`，`736b1416`已推送；22tests与真实配置组合/逐叶继承通过。GPU7两轮smoke于18:53:14启动，首轮已采集并开始U10，未完整验收；不要重复launch。唯一执行接续为[DVAC实施账本](evidence/DVAC_IMPLEMENTATION_LEDGER_20260905.md)，方法为[设计§11](01_DVAC_DESIGN.md#11-已授权的独立实现与gpu7验收入口)。GPU6原BC并非仍健康：17:44:40已在第6轮rollout VLM推理OOM退出，完整5轮、train25/24/28/22/25 /32，fixed5=27/32，累计124成功episode，无正式checkpoint（save10）；未重启或修改原树。DVAC保留原smoke train32×1/micro32/global1024/U10/eval16×2；两轮通过也不覆盖第6轮OOM。Sidney/shared Ray/他人不干预；GPU7正式长训未授权。

最新切入点讨论（09-05；Fast/源锁只读现场17:21）：用户明确先看相关工作让信号影响BC pipeline哪一环，不是质疑旧GRPO/RLT的V→w映射。作者源码首选依据为AttenA+逐H FM loss、旧RLT逐H BC项，RA-BC/RynnValue为整query监督加权；DataMIL/Re-Mix/RECAP/OTQL是不同入口。建议保留采集/成功过滤/均匀回放，只在未沿H归约的FM误差加action weights，未实施；[设计§10](01_DVAC_DESIGN.md#10-先选pipeline切入点再谈信号映射)链接完整[源码调研与辨析](evidence/BC_DVAC_PIPELINE_ENTRY_RELATED_WORK_20260905.md)。此前α/标定建议仍未批准。17:21所查Fast进程匹配为空、原driver/wrapper等均不在，日志仍04:47同一旧fatal，不是新事故；后续不重复播报未变化的已结束Fast。BC HEAD385d4e75与RLT30349428 clean，无生产/训练/依赖/进程改动，本轮没有刷新所有训练进度或Git发布；下方训练状态保持各自时间。

最新只读刷新/讨论（09-05 17:01）：BC完整2/100，train25/32→24/32，累计49成功episode，U10两次完整轮次均无所查错误，尚无正式fixed/ckpt。Sidney110/200、fixed110=17/32、ckpt110双rank/full在；Fast未重启。完整现场/资源/图表/Git见[本轮综述](evidence/BC_DVAC_REVIEW_20260905.md)。生产代码、配置、进程均未改；DVAC仅在[设计稿§6—9](01_DVAC_DESIGN.md#6-当前推荐配方让模糊点变成明确选择)深化，建议高V＋α0.25＋过去5轮标定＋入池固定w，未实施/未启动GPU7。/data余460.4GiB，现两run后续ckpt估计约414GiB，第三run之前需明确存储策略。17:09 BC讨论证据385d4e75、Sidney图表证据81be3193已push，生产source-head未变；共22个codex committed tip均有远端，未把诊断dirty合入训练。Git精确结果以[本轮账本](evidence/BC_DVAC_REVIEW_AND_GIT_LEDGER_20260905.md)为准；下面启动阶段状态均保留原时间，不覆盖本条。

本轮执行（09-05 16:34）：用户确认U10后，GPU6正式100轮已于16:32:12启动，wrapper1151769/observer1151770，原SFT/空成功池、非smoke续训；16:34已进入首轮采集，尚未完整Step1，无所查错误，见[启动现场](evidence/U10_FORMAL_STARTUP_20260905.json)，不要重复launch。v7于16:29:03 exit0，完整两轮/20次Adam/两次固定32评估/两代native shard＋full＋replay验收通过，显存采样峰77.46GiB、FD1003，详见[原始验收](evidence/U10_SMOKE_VERIFICATION_20260905.json)。U10/eval16×2固定种子循环源码`cb01451f`、轻量证据`1d453fcb`均push，11tests通过。本轮生产改动仍限配置、RoboTwin opt-in种子循环及测试3文件；模型/采集/loss/其余学习参数不变。完整参数、命令、资源及停止条件见[合同](evidence/GPU6_U10_RUN_CONTRACT_20260905.md)，唯一动态接续[启动账本](evidence/GPU6_U10_LAUNCH_LEDGER_20260905.md)。GPU7/Sidney/shared Ray不干预；/data16:34余496GiB，留意共享后续保存容量。以下“待确认U”与U2为历史。

最新定参讨论（09-05，14:55刷新之后）：用户已明确锁定**micro32/global1024**，单卡累积32；本轮跨论文/作者代码核查后**建议固定U10，待用户确认**。U10为10240次chunk呈现/轮，100轮为1000次Adam/1024000次呈现；是文献约束下的项目起点，不是作者默认或最优性结论，不机械继承旧B32/U100。优先依据BCIL成功监督复用、Hi-ORS真实π0更新语义，VLAW仅作全程数量级参照；预算证据集中在[U调研与建议](evidence/BC_U_BUDGET_RECOMMENDATION_20260905.md)。数据仍为历轮成功chunk累计池、均匀有放回抽样，每次Adam重抽B条，具体见[答疑§14](evidence/BC_PARAMETER_QA_AND_UPDATE_BUDGET_20260905.md#14-用户锁定micro32global1024u是谁的数据究竟怎么用)。train32/eval16×2/FD4096及必要时取消中评已同意。本轮只更新讨论文档，无服务器访问/生产修改/启动，下面动态数值保留原时间；U与完整合同确认、同并发完整smoke通过后，原SFT/空池正式。

最新追问（09-05 14:55现场）：用户明确先降评估并发，必要时可取消中途评估、事后再评；训练32不变，global/U仍要求解释、未确认B32/U100。旧Control实际已完整96、fixed95=31/32、checkpoint90双DCP/full在，证明32train＋16eval/卡长程可运行，不是正常完成100的证据；旧两卡full_shard与BC单卡no_shard不能等同容量。FSDP组件probe及真实更新后同步已通过，完整smoke阻于评估；不重复修原断言。参数白话/来源及新现场只维护在[答疑§10—12](evidence/BC_PARAMETER_QA_AND_UPDATE_BUDGET_20260905.md#10-追问旧grpo究竟跑通到哪里bc怎么继承)。BC仍无run/生产checkpoint、HEAD700b6846 clean；π0.5完整104/200、105评估中；本轮只读，无生产配置或训练变更。

最新讨论（09-05，14:14—14:17只读核验）：用户提出训练32不动、评估改16×2，并要求重查BC更新量和继承项；本轮只讨论，未改代码/config或启动训练。当前优先方向是显式两批覆盖原32固定ID，不是直接改rollout_epoch重复16；之前环境offload降为备选。重新核对BCIL/Hi-ORS/SIME/VLAW/HABC后，建议讨论micro/global32、U100（3200样本呈现/轮），**不是已确认参数**；现部署仍GB1024/U2。FM噪声/t是原生SFT目标，不属πRL；行为logprob/chain存储可删但尚未删。精度按官方null，SFT专用wrap是按实际调用图的本任务配置，保存用既有local_shard；五文件hash本地/服务器一致。完整逐项答复及来源见[本轮问答与更新预算](evidence/BC_PARAMETER_QA_AND_UPDATE_BUDGET_20260905.md)。BC仍无活跃run，HEAD700b6846/clean；Sidney完整103/200继续，本窗口未干预。

历史执行快照（13:39）：GPU6训练32×1，v6于13:20:25退出，29/32成功、87query、64微批/2次更新后首评估camera buffer分配失败；峰值79.17/79.65GiB，完整评估/checkpoint均0。此前SFT包裹/FD窄修源码72a92604已推；六次结束smoke证据700b6846已push。用户已授权完整smoke通过后原SFT/空池正式100轮，但尚未启动；GPU7预留。详见[结果](evidence/BC_RESULT_AND_NEXT_RESOURCE_CHOICE_20260905.md)、[账本](evidence/GPU6_SMOKE_LEDGER_20260905.md)。旧环境offload建议已由上方用户新讨论方向更新，勿据旧快照直接launch。

本轮讨论入口：[逐轮流程、继承组件排序及更新预算](evidence/BC_FLOW_COMPONENT_AND_UPDATE_AUDIT_20260905.md)、[为什么SFT与旧GRPO的FSDP边界不同](evidence/BC_FLOW_COMPONENT_AND_UPDATE_AUDIT_20260905.md#6-回答追问为什么旧π0-grpo能用bc这次出问题)、[BC＋DVAC独立设计上下文](01_DVAC_DESIGN.md)。DVAC仅讨论，尚未实施；只借动作信号与加权机制，不搬GRPO或RLT目标。
本文件是唯一维护的专题上下文。新窗口读完根规则与交接后，只读本文；需要核实某项结论时再按链接读取对应证据，不遍历历史。

## 1. 当前要实现什么

首个对象是 **成功过滤在线 FM-BC**，随后才是成功过滤＋DVAC。环境仍是 RoboTwin，训练框架仍是 RLinf，策略仍是 π0。

```text
当前 π0 采集完整 episode
    → 按任务 success 筛选
    → 追加到累计成功池（决策前观察＋实际提交的动作指令 chunk）
    → π0 原生 flow-matching 监督更新
    → 同步新策略，进入下一轮采集
```

这里的 BC 是用成功经验中的动作作为监督目标，训练 π0 原生 FM 损失；不是确定性动作回归，也不是 GRPO/PPO。
“在线”指更新后的策略继续采集新经验，不是把已有数据集多训练几遍。
首版建议同步分轮、累计回放，从选定的原始 SFT checkpoint 初始化一次，后续持续更新；这些是实现建议，不是已锁定的实验配置。

不需要先接 teacher、Q/V、RynnValue、RECAP 条件标签、世界模型或额外探索网络。
不为适配数据格式增加逐动作渲染，也不改现有相机、动作映射、归一化或渲染依赖。
“干净”是数据和目标明确、只加必要组件；不是为了少几行代码，把自主数据伪装成 DAgger 专家数据。

## 2. 参考代码优先级：按复用价值，而非论文影响力

| 优先级 | 参考对象 | 具体借什么 | 不整套搬什么 |
|---|---|---|---|
| **P0：主干** | 当前项目源锁下的 RLinf／RoboTwin／π0；上游对应接口见§3 | 环境与策略采集、π0 FM、FSDP 监督更新、参数同步、评估与 checkpoint | 不为 BC 升级整套部署；不继承 GRPO loss 或 DAgger 专家筛选默认 |
| **P1：算法闭环** | [Hi-ORS 成功 BC 配置](https://github.com/hiors-project/hiors/blob/5fa4b23f80f420dcb340875daec72a051e161fae/config/algo/reject_sampling_pi0.yaml) | episode 结束后入成功池、成功池采样、π0 FM 更新、发布新权重 | 不迁入人工纠正、no-op／短轨迹过滤、Dobot 动作约定和异步基础设施 |
| **P2：分轮编排** | [SIME 多轮脚本](https://github.com/EricJin2002/SIME/blob/831824101dee6900bdee6f65a925523676920326/simulation/run_full_multi_round.py) | 采集→结果筛选→合并历史数据→训练→再采集的组织顺序 | 不迁入 DP/RoboMimic、特征探索、难度筛选、每轮重置训练或固定预算 |
| **P3：存储组件** | RLinf online LeRobot／episode builder／dataset refresh | 数据分片、episode 索引、来源标记、加载刷新等局部组件 | 不直接使用其逐动作 collector 接 RoboTwin chunk；不为了格式换掉 π0 训练框架 |

**最值得实际阅读的顺序：RLinf 对应接口 → Hi-ORS 成功池与 FM → SIME 编排。** LeRobot 只在确定持久化格式时继续读，不先建设第二套数据系统。

Hi-ORS 的具体入口是 [`examples/train_rlpd.py`](https://github.com/hiors-project/hiors/blob/5fa4b23f80f420dcb340875daec72a051e161fae/examples/train_rlpd.py#L348) 的整 episode 入池与 learner，以及 [`sac_pi0.py::sft_loss_fn`](https://github.com/hiors-project/hiors/blob/5fa4b23f80f420dcb340875daec72a051e161fae/serl_launcher/serl_launcher/agents/continuous/sac_pi0.py#L464)。文件名含 SAC，不代表这里必须训练 critic；对应配置可以只做 SFT。其成功池阈值 10 是 buffer 记录数，不能翻译成“先收集 10 个成功 episode”。

SIME 不是可直接复制的纯成功 BC 默认：公开主流程使用 `train_0.5` 难度筛选分支；另有显式 exploration 分支；生成训练配置时设置 `resume_ckpt=None`、`resume_epoch=-1`。因此只借分轮编排，不照搬续训语义。详见[补充核验](evidence/ONLINE_BC_SOURCE_AUDIT_20260904.md#6-独立上下文整理补充核验2026-09-04)。

## 3. RLinf 接口：哪些能复用，哪些必须适配

以下相对路径均以 RLinf 仓库根为准，公开源码链接锁定 `dc9b87cc49334c7516487ead68ebeb060fd7c090`。本轮独立BC树已基于该pin建立；它不代表Fast-WAM或Sidney的部署版本。

| 组件 | 文件／符号 | 判断 |
|---|---|---|
| π0 监督目标 | [`rlinf/models/embodiment/openpi/openpi_action_model.py::sft_forward`](https://github.com/RLinf/RLinf/blob/dc9b87cc49334c7516487ead68ebeb060fd7c090/rlinf/models/embodiment/openpi/openpi_action_model.py#L377) | 已有逐元素 FM MSE；最终直接 mean。有效动作 mask 和后续权重必须在归约前处理 |
| 数据到模型输入 | 同文件 [`prepare_dagger_sft_batch`](https://github.com/RLinf/RLinf/blob/dc9b87cc49334c7516487ead68ebeb060fd7c090/rlinf/models/embodiment/openpi/openpi_action_model.py#L615)、[rollout 的 forward_inputs](https://github.com/RLinf/RLinf/blob/dc9b87cc49334c7516487ead68ebeb060fd7c090/rlinf/models/embodiment/openpi/openpi_action_model.py#L911) | 已有图像、状态和 action/model_action 转换源头；须区分物理动作和模型动作，防止重复归一化 |
| 分布式监督更新 | [`rlinf/workers/actor/fsdp_dagger_policy_worker.py`](https://github.com/RLinf/RLinf/blob/dc9b87cc49334c7516487ead68ebeb060fd7c090/rlinf/workers/actor/fsdp_dagger_policy_worker.py#L441) | 可借 SFT forward、梯度累积、optimizer 组织；替换数据接收／筛选，不需要 advantage |
| 采集与部署 | `rlinf/runners/embodied_runner.py`、env worker、[HF rollout worker](https://github.com/RLinf/RLinf/blob/dc9b87cc49334c7516487ead68ebeb060fd7c090/rlinf/workers/rollout/hf/huggingface_worker.py#L494) | 复用现有职责和同步能力；为自主 BC 接通外循环，不额外造一个调度框架 |
| 成功池与持久化 | DAgger replay／online LeRobot 的局部组件 | 只借容器和索引能力；自主数据资格、chunk 粒度、成功与终止语义需要专门适配 |

### 两个已确认的接口不匹配

1. **无专家能采集，不等于自主数据能进训练。** rollout 的 expert 分支是可选的；但普通 DAgger replay 接收调用 `extract_intervene_traj(mode="all")`，只提取干预／专家片段。删除 expert 或只开 `only_success` 不是完整的 self-BC 开关。
2. **RoboTwin 返回 chunk 后观察，online LeRobot builder 按逐动作帧组装。** 当前审计的 C50 路径中，`chunk_step` 只返回一个块后观察，term/trunc 标记放在动作数组末尾；builder 按 `len(obs_list)` 逐项读取动作与终止。直接拼接会错位，甚至读不到实际结束标记。

此外，builder 的 `only_success` 要求 `is_success and done_by_term`；不能假设它天然兼容成功后继续执行、最后由 time limit 截断的环境设置。成功资格应来自目标 RoboTwin 任务的真实 success 合同，episode 结束单独处理。
以上是固定源码的接口审计，**不是本轮已运行复现的 bug**。逐文件位置和 C50 推导见[源码审计§2](evidence/ONLINE_BC_SOURCE_AUDIT_20260904.md#2-rlinf对应符号与发现)。

## 4. 最小实现边界与数据记录

只增加四块必要衔接：自主 chunk 记录 → 完整 episode 成功入池 → 成功 replay 接 SFT batch → 监督更新后同步权重。
环境、模型和分布式训练主体继续复用。首版不需要新 critic，也不必先完成一个支持所有加权算法的通用框架。

| 一条 query/chunk 记录 | 必须表达的含义 |
|---|---|
| `obs_before_query` | 策略做本次决策时实际看到的相机图像、proprio 和任务文本；不能用块后图像替代 |
| `executed_action_chunk` | 实际送给环境执行的动作指令，带明确 action space／变换版本；不是 teacher 标签，也不默认拿测得的下一状态充当动作 |
| `action_valid_mask` | 提交的命令 chunk 内有效维度及 padding；不虚构 TOPP 物理执行前缀，见下方现场更正 |
| `episode_id / query_idx` | 归属与顺序，用于整条成功筛选、重放和去重 |
| `success / terminal / truncated` | 任务结果与结束原因分开；具体成功语义沿用选定任务定义 |
| `policy_version / round_id` | 记录经验由哪个版本生成，确保“更新后再采集”可核对 |

建议先按 query/chunk 存储和采样，不虚构 C50 内不存在的逐动作图像。每轮新增成功轨迹追加到累计池；失败尝试仍计入总交互预算，但不进入首版成功 BC 的 actor 池。
现场更正：RoboTwin先将整个waypoint chunk进行TOPP／压缩／重采样，成功可在物理控制循环中提前返回；当前info没有原C50的可逆执行prefix。首版监督实际提交的完整command chunk，排除成功／终止后完全未执行的后续query；不声称每个waypoint都物理执行完成，也不改变控制方式来制造逐动作标签。
若以后接 IQL，可另保留奖励、next_obs、失败数据等；它们不是当前 FM-BC loss 的必需输入。

首版在有效监督目标上使用等权 FM 损失。后续 DVAC／AttenA+／RA-BC 的浮点权重进入未归约损失，再按确定的 mask 和归约方式计算；不能把最终标量乘一个 batch 平均权重冒充逐样本加权。
先不规定 DVAC 信号与权重公式，也不迁入旧 RLT 的冻结 π0 锚定目标。

## 5. 参数和数据配方：哪些还没决定

09-05用户已明确：正式也采用单卡32并行×1串行＝每轮32条，smoke同并发/batch、仅2轮。模型M4直接继承官方model/pi0.yaml，官方RoboTwin π0 DAgger与旧同模型GRPO也用M4；删除此前无充分依据的M10覆盖。M是推理积分预算，不声称权重是“固有4步蒸馏”。方法优化器LR2.5e-5/Adam(.9,.95)/wd1e-10/clip1来自官方**π0** DAgger；U2是本次预算选择，非其默认U1。最新逐项依据、完整resolved、精确命令和预算见[GPU6合同](evidence/GPU6_SMOKE_CONTRACT_20260905.md)，覆盖先前32×8建议。没有把未测40–60GiB/128GiB当容量依据。

| 待讨论项 | 当前建议／边界 |
|---|---|
| 初始化与续训 | 用户已确认adjust_bottle π0 SFT；现场模型路径`/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50`；从SFT初始化一次，跨轮持续更新 |
| 是否混回原始示范 D0 | 用户要求参数化；实现`algorithm.online_bc.demo_weight=0`默认关闭；正值需传本地LeRobot `demo_data_path`，目标为`(online_FM+w*demo_FM)/(1+w)`。这是loss权重，不是轨迹比例 |
| 每轮尝试 N、总轮数 R | 已定N=32、单卡32并行×1；smoke R=2，正式候选R=100另行启动。失败也计入N |
| BC 更新量 U、batch B、回放采样 | 用户已确认micro32/global1024/U10，源码cb01451f已落实。每轮从累计成功池有放回抽10240次chunk、做10次Adam；100轮最多1000次Adam/1024000次呈现。不是全池遍历10遍；旧B32/U100已覆盖。依据见[预算证据§5](evidence/BC_U_BUDGET_RECOMMENDATION_20260905.md#5-为什么最终建议-10而不是机械照搬-100) |
| 行为采样与评估 | 明确 π0 采样设置并保留固定评估；不因为借 DAgger 的 `mode=eval` 就默认采样行为相同，也不无意继承 GRPO 专用噪声 |
| 任务／多任务采样 | 先沿用选定对照的任务定义；若为多任务，需说明成功池按任务还是按记录采样，避免悄悄偏向易成功任务 |

实现时只需围绕三件事做少量高信息量检查：数据对齐与真实执行范围；成功／失败／截断的入池结果及空池处理；一次监督更新后权重同步与 checkpoint 恢复。分布式空池应一致跳过或一致更新，不能各 rank 自行决定。
已完成10项基础/图像增强/FD边界回归、Ruff、AST、worker import、Hydra与真实validate_cfg，含VectorEnv import与资产根合同。原生null精度/增强off、SFT实际模块包裹和明确local_shard已验证连续2次更新及所检参数/Adam读回；不等于生产worker重启恢复通过。v1—v6历史失败与当前容量阻点见最新账本，不重复从v4开始。[09-04实施账本](evidence/IMPLEMENTATION_LEDGER_20260904.md)与旧GPU3合同均为历史。

## 6. 其他材料怎样使用

| 材料 | 处理方式 |
|---|---|
| 另一窗口的主交接与实现证据 | 已阅读；吸收最小组件、SIME 编排、成功／终止分离和 FM 接点。方法影响力排名不当作代码复用优先级 |
| RA-BC／AttenA+ | 留作成功 BC 跑通后的加权代码参考；不作为首版采集和训练外循环 |
| RynnValue／IQL／AWR | 属于外部进度或价值学习扩展；不要求为了普通 BC 先做 Q/V。后续价值方法不能默认只用成功数据 |
| RECAP／RLT | 可借数据管理、同步等经验；目标和输入不同，不迁入当前 loss。RLT 的紧凑 latent replay 不能替代 π0 所需图像 |
| SEIL／RL-100／其他 filtered-BC 论文 | 保留为分轮数据管理和实验设计的次级依据；无需为首版再读完全部工程 |

可选背景入口：[另一窗口主交接](C:/Users/86136/Documents/seek/0904_online_bc_baseline_ranking_and_experiment_handoff.md)、[实现证据](C:/Users/86136/Documents/seek/0904_online_bc_implementation_handoff_evidence.md)、[在线 BC 综述](C:/Users/86136/Documents/seek/0904_genuine_online_bc_baselines_survey.md)。这些只作外部参考，不是本项目另一份实施计划。
此前广搜、算法比较和 DVAC 初步讨论已保存在[历史讨论](history/RESEARCH_DISCUSSION_20260904.md)，不作为新窗口必读。

## 7. 当前状态与下次接续

已建立服务器独立分支`codex/sz-pi0-online-bc`，位于`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc`，官方基线`dc9b87cc49334c7516487ead68ebeb060fd7c090`。本地`worktrees/pi0-online-bc`仅为已核对官方blob的编辑副本，Git与运行以服务器为准。
成功回放、BC actor、数据接点、配置、测试和文档初批共9个文件；两项新增核心模块共302行，首批现有4文件60行新增/7行删除，其余为配置/测试/说明。保留原生null精度/增强off、实际SFT wrap/独立local_shard、EnvWorker显式FD下限；旧额外图像cast已撤。本轮在该基线上只改U10/eval16×2 opt-in种子循环及测试，当前运行源码`cb01451fbbb01f509bde126029a0cec3d577aedb`、最新HEAD为轻量证据`1d453fcbbf7fd38d99e07323bf74a5a7c045ce9d`，已push且启动前clean。11项测试与完整U10两轮GPU smoke通过，不改共享依赖。GPU固定6、7预留。没有停止/迁移/修改其他用户任务，也不再继续定向检查他们。
上游 `dc9b87c → a3795c` 已重新查询 GitHub compare：仅中英文 README 改动，旧源码审计不因这一次提交失效；不能据此推定服务器版本或后续 main 未变化。SIME 补充锁为 `8318241`，完整锁见[证据§6](evidence/ONLINE_BC_SOURCE_AUDIT_20260904.md#6-独立上下文整理补充核验2026-09-04)。

v1—v6失败轻量证据已封存，旧run完整保留；v7完整通过。现已按[U10最终合同](evidence/GPU6_U10_RUN_CONTRACT_20260905.md)另起正式100轮，唯一run为`pi0-adjust-bottle-bc32x1-b1024-u10-eval16x2-gpu6-formal100-20260905-v1`。下一次先只读刷新该run的wrapper/日志/TB/checkpoint及GPU/RAM/磁盘，不重放任何launch，不重跑精度/FSDP/FD旧probe。短测不证明100轮累计池容量、生产worker恢复与长期原生稳定性；两轮smoke成功率也不代表正式学习效果。未创建新任务窗口或自动化；DVAC仍仅规划，BC无用logprob/chain输出未在本轮顺手改动。
