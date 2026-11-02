# 论文熵分析：师生配对与同run检查点

2026-11-02。对齐写作侧 `uaca/research/exploratory_planning_20261102/revision_04/03_COLLECTION_PLAN.md`。

用户希望看到：动作越不确定，师生分歧是否越大；两个策略犹豫的位置是否一致；RL训练主要改变哪类动作的不确定性。三个问题共用原始预测，但不混用模型、观测池和训练横轴。

1. [图表与统计口径](01_FIGURES_AND_PROTOCOL.md)
2. [现有数据、权重与实现切口](02_DATA_AND_CODE.md)
3. [运行与排队](03_EXECUTION.md)
4. [本轮澄清：中图、单任务FQL与完整GRPO](04_QUEUE_AND_FIGURE_CLARIFICATION.md)
5. [已启动：原始Clean GRPO200与六点采集](05_CLEAN_GRPO_SIX_POINT_EXECUTION.md)

现有药瓶教师K3补算已完成：10个query主噪声精确复现，32局5725位置，Top20%重合85.33%。FQL第55更新评估110/128（85.94%），完整暂停点保留。用户随后确认原始Clean GRPO200，已接45启动；顺序为Clean完整训练→末代固定池与六点采集→单任务FQL余下预算→RLT。67保持当前DMD→OT→RLT。

完整GRPO六点曲线此前仅规划，未落实训练队列；本轮用户已选择Clean，现已落实，见05。旧两个RL点仍不能混成训练过程。跨任务是可选稳健性复验，当前只做药瓶，不改成N1/N10配对。

旧数据回溯：[药瓶师生信号](../student_teacher_signal/00_INDEX.md)、[多任务N1/N5/N10信号](../pi05_nstep_signal/00_INDEX.md)。新输出全部写Data；不复制权重、图像或旧视频。
