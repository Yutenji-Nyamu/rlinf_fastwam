# 单任务小型流策略蒸馏：讨论工作台

更新：2026-10-27。状态：检索、源码审计和规划；本轮没有部署、占卡、训练或改动现有队列。

## 用户目标

先用单任务、较小模型快速比较直接端点蒸馏、CP、DMD、MFD、固定老师FQL，再迁回π0.5多任务。每个实验暂按一张80GB卡规划；训练和评估都测合适吞吐，少量有依据的容量探测，不从很小并发逐档爬升。约五个原生400动作上限的任务，先一个，再五个独立任务。中间约20次评估，前密后疏。

本轮追加偏好：优先找原生流匹配、已有社区影响力、RoboTwin和RoboDojo接入成熟的基座，不预先锁死经典Diffusion Policy。

## 当前判断

1. **整块监督、截取执行是常见做法**，原DP、CP和SmolVLA均有明确代码/论文依据；RL-100 flow端点蒸馏也提供只比较执行段的实例。不能因H50/C10就判实现错误，也不能只看整体MSE判断执行前10步是否改善。
2. **跨基准的首个候选是单任务SmolVLA**：原生FM、约450M总参数/100M动作专家，LeRobot维护，XPolicyLab已有训练、合批推理入口。Dojo有官方单任务评测，RoboTwin有统一接入路径；这不等于两边都已有高成功率老师。
3. **更纯粹的小网络备选是FMP/Flow-DP**：复用DP的ResNet18+1D U-Net，用FM训练教师。它研究切口干净，但需自己接RoboTwin/Dojo适配，不能称已验证现成基线。
4. **ManiFlow值得保留参考，但暂不直接定为多步老师**：官方有RoboTwin2 RGB/点云训练；方法已含一致性目标，主打1–2步。未在本轮所查XPolicyLab固定树中发现其Dojo插件。改成纯FM还要处理零一致性批次和target-time接口，不能只改一个比例后声称完成。
5. **不立即选定哪套正式跑五方法**。先用选定任务验证老师成绩与N1/N多步差距；用户本轮只要求规划。优先级建议是SmolVLA单任务，其次FMP-U-Net，ManiFlow作为架构/方法参考。

## 阅读顺序

- [01 监督整块还是执行段](01_HORIZON_AND_SOURCES.md)：源码证据、为何如此、当前π0.5如何解读。
- [02 基座候选与双基准支持](02_BACKBONE_SELECTION.md)：SmolVLA、ManiFlow、FMP、经典DP，固定版本与具体接入证据。
- [03 五方法切口与参数依据](03_METHOD_ADAPTERS_AND_PARAMETERS.md)：模型/时间/动作接口，原论文与我们的适配边界。
- [04 任务、预算、并行和产物](04_TASKS_AND_THROUGHPUT.md)：五任务候选、原始多步/一步对照、少量容量探测、评估安排。

现有π0.5背景：[索引](../pi05_distillation/00_INDEX.md)、[直接蒸馏审计](../pi05_distillation/67_ENDPOINT_CONVERGENCE_AUDIT_20261027.md)、[DMD对齐](../pi05_distillation/54_ONEDP_ENDPOINT_ALIGNMENT_20261026.md)、[MFD实现](../pi05_distillation/66_MFD_IMPLEMENTATION_20261026.md)。

所有候选的公开成绩均为文献/榜单参考，不是我们本轮实测。下载源码只作阅读，没有执行外部安装脚本。论文、原始源码和API快照在本线程E盘`dp_plan1027/`；本目录只保留轻量分析和引用。
