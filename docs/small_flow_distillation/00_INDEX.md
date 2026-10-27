# 单任务小型流策略蒸馏：讨论工作台

更新：2026-10-27。状态：检索、源码审计和规划；本轮没有部署、占卡、训练或改动现有队列。

## 用户目标

先用单任务、较小模型快速比较直接端点蒸馏、CP、DMD、MFD、固定老师FQL，再迁回π0.5多任务。每个实验暂按一张80GB卡规划；训练和评估都测合适吞吐，少量有依据的容量探测，不从很小并发逐档爬升。约五个原生400动作上限的任务，先一个，再五个独立任务。中间约20次评估，前密后疏。

本轮追加偏好：优先找原生流匹配、已有社区影响力、RoboTwin和RoboDojo接入成熟的基座，不预先锁死经典Diffusion Policy。进一步要求解释FM/扩散、ManiFlow/FlowPolicy，并核实旧RLinf单任务π0/π0.5能否复用。

## 当前判断

1. **整块监督、截取执行是常见做法**，原DP、CP和SmolVLA均有明确代码/论文依据；RL-100 flow端点蒸馏也提供只比较执行段的实例。不能因H50/C10就判实现错误，也不能只看整体MSE判断执行前10步是否改善。
2. **跨基准的首个候选是单任务SmolVLA**：原生FM、约450M总参数/100M动作专家，LeRobot维护，XPolicyLab已有训练、合批推理入口。Dojo有官方单任务评测，RoboTwin有统一接入路径；这不等于两边都已有高成功率老师。
3. **更纯粹的小网络备选是FMP/Flow-DP**：新增核到PointFlowMatch的原生FM、U-Net及RGB版本，默认N10，是比泛用演示更具体的参考。它研究切口干净，但需自己接RoboTwin/Dojo适配并获得合格教师，不能称已验证现成基线。
4. **ManiFlow值得保留参考，但暂不直接定为多步老师**：官方有RoboTwin2 RGB/点云训练；方法已含一致性目标，主打1–2步。未在本轮所查XPolicyLab固定树中发现其Dojo插件。改成纯FM还要处理零一致性批次和target-time接口，不能只改一个比例后声称完成。
5. **最快复用的桥梁是旧RLinf π0.5单任务**：2026-10-27 14:40只读确认π0/π0.5 adjust_bottle SFT文件仍在1机，权重分片和norm齐全；官方发布SFT成绩76.56%/85.94%，但当前评估器下的N1/N10差距未知。单任务π0.5优先省工程和评估，小模型仍用于降低每步计算。
6. **不立即选定正式跑五方法**。短期建议旧RLinf π0.5先测差距；小型平台中，双基准接入看SmolVLA，纯FM小网络看PointFlowMatch RGB/FMP。ManiFlow、FlowPolicy已有少步训练，并不代表所有FM都是一步；3D FlowMatch Actor有明确N5/N1性能差。

## 阅读顺序

- [01 监督整块还是执行段](01_HORIZON_AND_SOURCES.md)：源码证据、为何如此、当前π0.5如何解读。
- [02 基座候选与双基准支持](02_BACKBONE_SELECTION.md)：SmolVLA、ManiFlow、FMP、经典DP，固定版本与具体接入证据。
- [03 五方法切口与参数依据](03_METHOD_ADAPTERS_AND_PARAMETERS.md)：模型/时间/动作接口，原论文与我们的适配边界。
- [04 任务、预算、并行和产物](04_TASKS_AND_THROUGHPUT.md)：五任务候选、原始多步/一步对照、少量容量探测、评估安排。
- [05 FM、扩散与少步策略](05_FM_DIFFUSION_AND_FAST_POLICIES.md)：ManiFlow/FlowPolicy机制、非一步FM例子、五方法迁移条件。
- [06 已存RLinf单任务老师](06_RLINF_SINGLE_TASK_TEACHERS.md)：现场文件、官方成绩/历史训练采样区别、动作/状态/归一化差异。
- [07 单任务适配方案](07_SINGLE_TASK_ADAPTATION_PLAN.md)：已有大模型桥梁、小模型平台、实际代码切口和参数来源。

现有π0.5背景：[索引](../pi05_distillation/00_INDEX.md)、[直接蒸馏审计](../pi05_distillation/67_ENDPOINT_CONVERGENCE_AUDIT_20261027.md)、[DMD对齐](../pi05_distillation/54_ONEDP_ENDPOINT_ALIGNMENT_20261026.md)、[MFD实现](../pi05_distillation/66_MFD_IMPLEMENTATION_20261026.md)。

所有候选的公开成绩均为文献/榜单参考，不是我们本轮实测。下载源码只作阅读，没有执行外部安装脚本。论文、原始源码和API快照在本线程E盘`dp_plan1027/`、`fm_followup1027/`；本目录只保留轻量分析和引用。现场只检查1机我方模型文件，未占GPU或改训练；没有把2/3机旧状态当成本轮新现场。
