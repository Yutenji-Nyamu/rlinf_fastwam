# 多噪声分散度与师生动作距离

2026-10-29：规划和代码定位，尚未启动采集。目标是同一条学生执行轨迹上，两条逐动作曲线，以及全体轨迹的统计关联。只在同一观察、同一噪声配对下比较动作。

- [01 已跑通实现与最小修改](01_REUSE_AND_IMPLEMENTATION.md)
- [02 信号定义、原始数据与绘图](02_SIGNALS_DATA_AND_FIGURES.md)
- [03 任务、并行与有限采集](03_TASKS_AND_BUDGET.md)

用户已确认“和之前学生老师接管实验一样”。固定成功G族配对：学生/弱策略Clean CP140，教师/强策略Uτ3 CP70，均N10，H50/C20，move_pillbottle_pad Clean，200动作预算。两个保护权重均在1机；其“更强”结论来自这个任务，首批先做pillbottle，不将它外推10任务。

这次仅采信号：学生执行，老师在所有有效query旁路预测，不实际接管；既给全程统计，也单列前3块，和G实验的有效窗口对照。名称“学生/老师”此处指弱/强策略，不是N1/N10蒸馏对。

采集卡位及正式启动留待下一轮实施；本轮只规划，不占用当前训练卡。此前暂拟N1/N10的10任务候选保留为另一个实验的备选，不进入本次协议。

已参考其他窗口“1013 u paper”的`research/u_ca/exploratory_studies_20261028/04_SIGNAL_CHARACTERIZATION.md`。UA-OPD Fig.2比较的是教师不确定性与经验奖励误差；本轮师生动作距离是另一个量，不能标成原图的Reward Gap复现。[原文](https://openreview.net/forum?id=hHUZi8oHfo)
