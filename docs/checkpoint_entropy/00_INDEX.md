# 论文熵分析：师生配对与同run检查点

2026-11-02。对齐写作侧 `uaca/research/exploratory_planning_20261102/revision_04/03_COLLECTION_PLAN.md`。

用户希望看到：动作越不确定，师生分歧是否越大；两个策略犹豫的位置是否一致；RL训练主要改变哪类动作的不确定性。三个问题共用原始预测，但不混用模型、观测池和训练横轴。

1. [图表与统计口径](01_FIGURES_AND_PROTOCOL.md)
2. [现有数据、权重与实现切口](02_DATA_AND_CODE.md)
3. [运行与排队](03_EXECUTION.md)

已实现现有药瓶32轨迹的教师K3离线补齐与CPU分析，安排在当前45卡MFD训练后、FQL v2前；先复现原教师K1作为GPU验收，失败退出后接回原队列。没有启动新的GRPO训练。

同run六点在1机不存在；其他任务的策略配对尚待用户选择。两项问题已发出，不用不同run凑训练曲线，不把药瓶RL对悄悄换成N1/N10。

旧数据回溯：[药瓶师生信号](../student_teacher_signal/00_INDEX.md)、[多任务N1/N5/N10信号](../pi05_nstep_signal/00_INDEX.md)。新输出全部写Data；不复制权重、图像或旧视频。
