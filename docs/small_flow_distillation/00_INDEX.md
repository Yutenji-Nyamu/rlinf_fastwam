# 单任务小型流策略蒸馏：讨论工作台

**2026-10-29 12:57执行更新：** [28 旧四方法续到400预算轮](28_EXTEND400_EXECUTION_20261029.md)已排在三v2之后、RLT之前，每20轮补评128回合；[29 Q高估的具体例子](29_FQL_WHY_Q_INFLATES_20261029.md)区分真实结果与自举训练答案；[信号绘图规划](../student_teacher_signal/00_INDEX.md)已按用户答复固定为旧接管G族Clean CP140/Uτ3 CP70、pillbottle N10/H50/C20，未采集。[本轮清理及轻量补归档](../experiments/extend400-cleanup-20261029/README.md)：释放86.1GiB，Home约602、Data约181GiB；以下此前快照留作历史。

**FQL讨论续篇：** [25 原FQL对照与Q可信度](25_FQL_COMPARISON_AND_Q_MEASUREMENT_20261028.md)：随机Q启动也是原方法做法；首批19倍不代表全程，原基准奖励与当前0–1回报不同；讨论Q持续训练、真实回报校准与动作排序。本次只更新讨论资料。

**2026-10-29收敛：** [27 FQL修复方案](27_FQL_REPAIR_CONVERGED_PLAN_20261029.md)：建议先alpha=30＋既定评估记录Q与真实回报；解释离线高估反馈，Q预热/双Q取小/CQL作为分开的后续候选。本轮仍只讨论，不改训练队列。

[10月29日清理与交互图](../experiments/cleanup-chart-20261029b/README.md)：已精准释放561.6GiB，保留100/200轮末与阶段最佳；单任务主图移除FQL，支持勾选、不同点形及坐标切换。新存储候选仅推荐。

[26 α、高估反馈与轻量评估切口](26_FQL_ALPHA_OVERESTIMATION_AND_EVAL_PLAN_20261028.md)：现有α增大可降低Q相对影响；核对目标和终止路径，拟复用推理前缀记录Q与真实回报，不改模拟器；尚未实施。

**2026-10-28 21:57更新：** FQL按用户要求在740步停止，保存715/最佳28完整点；Direct续训已到1320并评估。1机67后继为旧CP→旧DMD→旧MFD→CP v2→DMD v2→MFD v2→RLT，各续至/新训200预算轮。新v2保持17评×128回合，仅保留最近2个完整点和最佳完整点，旧保存不动。见[23执行与保存方案](23_V2_FORMAL_QUEUE_20261028.md)、[24单任务FQL诊断与修复讨论](24_SINGLE_FQL_DIAGNOSIS_20261028.md)。以下早前快照保留作历史。

现场快照：2026-10-28 16:54。原生直接/CP/DMD已完成100预算轮，末评103/108/103成功（各128回合）；MFD到1260/1300，最近100/128，随后单任务FQL。Direct、旧CP、DMD new续到200轮均已依次接在原队列后、RLT前，完整状态恢复、每20轮补评。DMD保持固定学习率及5:1。大产物优先Home。

同日18:59实施更新：**CP v2、DMD v2、MFD v2已独立实现并短测通过**，见21/22。统一学生及辅助动作损失评分14维，内部H50×32保持；实质改CP两项、DMD辅助、MFD辅助。1机2/3卡各12→14完整续训、B32原生N1通过后释放；未接正式队列，4–7旧训练原样。DMD默认分配器首试OOM，以expandable_segments保持batch192重测通过。dropout仍另议。17–20保留研究与实施前计划。

## 用户目标

先用单任务、较小模型快速比较直接端点蒸馏、CP、DMD、MFD、固定老师FQL，再迁回π0.5多任务。每个实验暂按一张80GB卡规划；训练和评估都测合适吞吐，少量有依据的容量探测，不从很小并发逐档爬升。约五个原生400动作上限的任务，先一个，再五个独立任务。中间约20次评估，前密后疏。

本轮追加偏好：优先找原生流匹配、已有社区影响力、RoboTwin和RoboDojo接入成熟的基座，不预先锁死经典Diffusion Policy。进一步要求解释FM/扩散、ManiFlow/FlowPolicy，并核实旧RLinf单任务π0/π0.5能否复用。

## 当前判断

1. **整块监督、截取执行是常见做法**，原DP、CP和SmolVLA均有明确代码/论文依据；RL-100 flow端点蒸馏也提供只比较执行段的实例。不能因H50/C10就判实现错误，也不能只看整体MSE判断执行前10步是否改善。
2. **跨基准的首个候选是单任务SmolVLA**：原生FM、约450M总参数/100M动作专家，LeRobot维护，XPolicyLab已有训练、合批推理入口。Dojo有官方单任务评测，RoboTwin有统一接入路径；这不等于两边都已有高成功率老师。
3. **更纯粹的小网络备选是FMP/Flow-DP**：新增核到PointFlowMatch的原生FM、U-Net及RGB版本，默认N10，是比泛用演示更具体的参考。它研究切口干净，但需自己接RoboTwin/Dojo适配并获得合格教师，不能称已验证现成基线。
4. **ManiFlow值得保留参考，但暂不直接定为多步老师**：官方有RoboTwin2 RGB/点云训练；方法已含一致性目标，主打1–2步。未在本轮所查XPolicyLab固定树中发现其Dojo插件。改成纯FM还要处理零一致性批次和target-time接口，不能只改一个比例后声称完成。
5. **最快复用的桥梁是旧RLinf π0.5单任务**：1机权重分片和norm齐全；当前协议下默认N5为86.72%、N1为76.56%，见08。单任务π0.5优先省工程和评估，小模型仍用于降低每步计算；它仍是完整π0.5，模型大小不会随任务数缩小。
6. **五种短测已通过，正式队列已启动**。原生模型接口、方法实现和实际并行见10/11；14步结果不作为收敛比较。小型平台中，双基准接入看SmolVLA，纯FM小网络看PointFlowMatch RGB/FMP。ManiFlow、FlowPolicy已有少步训练，并不代表所有FM都是一步；3D FlowMatch Actor有明确N5/N1性能差。

## 阅读顺序

- [01 监督整块还是执行段](01_HORIZON_AND_SOURCES.md)：源码证据、为何如此、当前π0.5如何解读。
- [02 基座候选与双基准支持](02_BACKBONE_SELECTION.md)：SmolVLA、ManiFlow、FMP、经典DP，固定版本与具体接入证据。
- [03 五方法切口与参数依据](03_METHOD_ADAPTERS_AND_PARAMETERS.md)：模型/时间/动作接口，原论文与我们的适配边界。
- [04 任务、预算、并行和产物](04_TASKS_AND_THROUGHPUT.md)：五任务候选、原始多步/一步对照、少量容量探测、评估安排。
- [05 FM、扩散与少步策略](05_FM_DIFFUSION_AND_FAST_POLICIES.md)：ManiFlow/FlowPolicy机制、非一步FM例子、五方法迁移条件。
- [06 已存RLinf单任务老师](06_RLINF_SINGLE_TASK_TEACHERS.md)：现场文件、官方成绩/历史训练采样区别、动作/状态/归一化差异。
- [07 单任务适配方案](07_SINGLE_TASK_ADAPTATION_PLAN.md)：已有大模型桥梁、小模型平台、实际代码切口和参数来源。
- [08 已启动的单任务评估](08_SINGLE_TEACHER_EVALUATION_20261027.md)：默认N5与N1、32并行、条件追加C10，以及实际绑定检查。
- [09 五方法FM坐标审计](09_METHOD_COORDINATE_AUDIT_20261027.md)：原方法来源、速度/噪声/分数换算、梯度符号与适配边界。

现有π0.5背景：[索引](../pi05_distillation/00_INDEX.md)、[直接蒸馏审计](../pi05_distillation/67_ENDPOINT_CONVERGENCE_AUDIT_20261027.md)、[DMD对齐](../pi05_distillation/54_ONEDP_ENDPOINT_ALIGNMENT_20261026.md)、[MFD实现](../pi05_distillation/66_MFD_IMPLEMENTATION_20261026.md)。

候选的公开成绩是文献/榜单参考，原始模型实测见08，五方法短测见11。论文、源码阅读快照在E盘`dp_plan1027/`、`fm_followup1027/`，原始评估证据在`single_teacher_eval1027/`，本轮短测在`single_methods1027/`。1机3卡原始评估已结束；6/7短测后曾恢复原多任务FQL；最新已切换单任务正式队列。2/3机本轮未刷新内部状态。

- [10 单任务五方法实施与参数依据](10_SINGLE_METHOD_IMPLEMENTATION_20261027.md)
- [11 单任务短测与FQL返回回执](11_SINGLE_METHOD_SMOKE_RESULTS_20261027.md)
- [12 单任务正式队列](12_SINGLE_FORMAL_QUEUE_20261027.md)
- [13 直接/CP收敛审计与200轮续训](13_DIRECT_CP_EXTEND200_AUDIT_20261028.md)
- [14 各方法填充维及CP dropout审计](14_CP_PADDING_DROPOUT_AUDIT_20261028.md)
- [15 CP new最小适配规划](15_CP_NEW_PLAN_20261028.md)
- [16 DMD new200轮续训回执](16_DMD_EXTEND200_RECEIPT_20261028.md)
- [17 填充评分的跨模型分类与选择](17_PADDING_OBJECTIVES_SURVEY_20261028.md)
- [18 CP dropout计算路线与适配位置](18_CP_DROPOUT_WALKTHROUGH_20261028.md)
- [19 五方法统一真实动作评分：源码审计](19_UNIFIED_PHYSICAL_LOSS_CODE_AUDIT_20261028.md)
- [20 统一评分实施前计划](20_UNIFIED_PHYSICAL_LOSS_PLAN_20261028.md)
- [21 CP v2 / DMD v2 / MFD v2独立实施](21_PHYSICAL_V2_IMPLEMENTATION_20261028.md)
- [22 三方法短测完成与GPU释放](22_PHYSICAL_V2_SMOKE_RESULTS_20261028.md)

- [30 FQL-fix实现与短测](30_FQL_FIX_IMPLEMENTATION_20261029.md)
