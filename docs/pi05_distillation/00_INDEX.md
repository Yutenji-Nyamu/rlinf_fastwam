2026-10-27：[单任务小型流策略蒸馏工作台](../small_flow_distillation/00_INDEX.md)。新增整块/执行段监督证据、SmolVLA/ManiFlow/FMP等原生FM基座比较、五方法接口与参数来源、400步任务和吞吐规划；本轮仅讨论，不占卡或改变队列。

2026-10-27：[67 直接蒸馏平台与实现审计](67_ENDPOINT_CONVERGENCE_AUDIT_20261027.md) · [68 直接蒸馏文献判断](68_ENDPOINT_LITERATURE_20261027.md)。截至17760步损失仍下降，最新11130评估31.25%；发现完整H50窗口排除末49个输入观察的覆盖限制。只读审计，正式训练与队列未改。

2026-10-26最新：[60 MFD规划索引](60_MFD_INDEX_20261026.md) · [61 来源与机制](61_MFD_SOURCES_AND_MECHANISM_20261026.md) · [62 π0.5适配计划](62_MFD_PI05_ADAPTATION_PLAN_20261026.md)。已核官方论文及源码，基于DMD new复用；本轮只规划，不启动MFD。

2026-10-26最新：[59 新DMD双卡smoke与FQL接续](59_DMD_NEW_SMOKE_AND_FQL_HANDOFF_20261026.md)。旧OneDP已收束；新DMD采用1e-5/2e-4、辅助5:1、学生14维，12→14续训与N1导出通过，稳态约6秒、峰值72.6GiB/卡。1·67转FQL真实采集→正式训练→RLT，1·45直接蒸馏保持。首次采集的缺失模块已修复，失败启动记录保留且不算回合。

此前讨论：[58 DMD修整收敛与CPU供数测试](58_DMD_DECISIONS_AND_CPU_PREFETCH_20261026.md)。8线程＋预取1批已在59号双卡smoke启用；58号“方法未切换/学习率暂保持”由本轮用户明确指令更新。

2026-10-26本轮：[54 OneDP与直接蒸馏逐项对齐](54_ONEDP_ENDPOINT_ALIGNMENT_20261026.md) · [55 DMD2/flow-DMD/PDMD稳定做法](55_DMD_STABILITY_RECIPES_20261026.md) · [56 训练时间与提速讨论](56_TRAINING_TIME_AND_SPEED_PLAN_20261026.md) · [57 FQL双卡smoke完成](57_FQL_GPU_SMOKE_20261026.md)。正式4–7训练未改；条件授权后1·23完成真实采集、6→8续训与N1导出并释放。

> 2026-10-26本轮：[51 训练H50还是前C10及相关工作](51_TRAINING_HORIZON_SOURCES_20261026.md) · [52 OneDP/DMD源码与动态审计](52_ONEDP_DMD_AUDIT_20261026.md) · [53 近期产物推送覆盖](53_RECENT_PUBLICATION_AUDIT_20261026.md)。正式训练未改；当时FQL暂不smoke；后续条件授权见57号。

> 2026-10-26：FQL固定老师适配已实现，CPU检查通过；当时用户要求暂不smoke；后续条件授权见57号。见[49 实现](49_FQL_IMPLEMENTATION_20261026.md)、[50 后续入口](50_FQL_SMOKE_RUNBOOK_20261026.md)。原多任务直接蒸馏/OneDP继续运行。

# π0.5 × Consistency Policy / OneDP

2026-10-25后续规划：[46 FQL固定老师的代码切口与参数](46_FQL_FIXED_TEACHER_PLAN_20261025.md) · [47 真实转移采集契约](47_FQL_TRANSITION_CONTRACT_20261025.md) · [48 每任务96回合、数据选择与时间/容量](48_FQL_COLLECTION_AND_DATA_MIX_20261025.md)。拟用新采成功＋失败训练Q，HF继续保留；现有评估会丢弃转移，需独立补采集接口。尚未启动采集/FQL或改现有队列。

更新：2026-10-22。本轮已按用户要求收束单任务CP（峰值CP3800=98.4%，末评CP5700=91.4%），完整恢复点已保护；2机67继续OneDP。50任务CP已获授权并在2机45完成训练/续训与导出N1检查，正在验证每卡两个任务×16环境，之后自动接全任务基线与正式训练。详见[多任务实施](14_MULTITASK_EXECUTION_20261022.md)与[收束记录](../experiments/closeout-20261022/README.md)。

用户已明确：就在常用 π0.5 / RoboTwin 2.0 基座上构建，借鉴两篇思想，可复用的搬运，冲突处按我们的模型适配。模型类型不同不是阻塞；文档记录适配项即可。

## 当前主线

- 同一个 Sidney π0.5 RoboTwin 基座作为老师，两种学生都由它初始化。先做离线蒸馏，不混入 GRPO、Q 网络或新的奖励。
- CP：保留官方 CTM 的跨阶段一致性、老师、EMA 与数据监督思路；生成时间和输出换成 π0.5 的流模型约定。
- OneDP：以随机版 OneDP-S 为主，保留“老师—学生—辅助分布网络”及交替更新；把概率方向的计算适配到流速度。
- 首轮adjust_bottle已完成CP对照，OneDP继续；当前已实施50任务联合CP。
- 首轮固定三相机、绝对关节动作、原归一化、预测 H50 / 执行 C50。N 表示每块的网络计算次数；N1 仍生成 50 个动作。
- 原始 N1 是未蒸馏对照。最终比较：原始 N10、原始 N1、CP N1、OneDP-S N1；CP N3 用于必要的少步诊断。

## 文档路由

1. [来源与可搬运内容](01_SOURCES.md)：官方、其他团队、个人实现，证据和局限。
2. [通俗机制与数据需求](02_METHODS_AND_DATA.md)：怎么训、训谁、是否在线、是否预采集。
3. [本地代码切口与数学适配](03_PI05_ADAPTATION.md)：CTM、流速度到 score、梯度路径、参数来源、续训状态。
4. [首轮任务与逐步扩大方案](04_EXPERIMENT_PLAN.md)：历史 N10 成绩、配对评估、数据拆分、1→5→50。
5. [历史任务统计](evidence/task_screen_history.json)：从已完成的四噪声采集 done.json 汇总；不是新的 N1/N10 测试。

研究原件在 `E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a/onedp_cp_research1021/`，包含论文、选定源文件和来源清单；不重复放入源码仓库。

## 下一步

两方法B64及选定B128 smoke均通过，每卡64/128容量测试完成；选择每卡64、全局B128，正在 `.105` 补齐原始评估，随后正式训练。评估迁移缺失工具模块已补齐，新事务`pi05-distill-adjust-migrated-recovery2`。新suite显式允许N1/400，旧接管约束保留。分支`codex/pi05-cp-onedp-20261021`；源码`worktrees/pi05-distill-20261021`。此前六任务首筛和3000步工程预算已由本轮指令取代。
